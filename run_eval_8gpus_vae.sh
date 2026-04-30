#!/bin/bash
set -uo pipefail

# ================= Configuration =================
VAE_CKPT="./checkpoints/Express4D/rvq_Express4D/model/finest.tar"
OPT_PATH="./checkpoints/Express4D/rvq_Express4D/opt.txt"
DATASET="express4d"
DATA_MODE="arkit"
EVAL_MODEL="tex_mot_match"
EVAL_MODE="wo_mm"
EVAL_LABEL="vae_recon"
RESULT_DIR="./eval_result_vae"
NUM_GPUS=8
TOTAL_EVALS=20
REPS_PER_JOB=1

# For an external VQ-VAE implementation, fill these in if the checkpoint is
# not the project's MovementConvEncoder/MovementConvDecoder format.
MODEL_MODULE=""
MODEL_CLASS=""
MODEL_KWARGS=""
STATE_DICT_KEY=""
# =================================================

mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/eval_vae_*_seed*.log

extra_model_args=()
if [ -n "$MODEL_MODULE" ]; then
    extra_model_args+=(--model_module "$MODEL_MODULE")
fi
if [ -n "$MODEL_CLASS" ]; then
    extra_model_args+=(--model_class "$MODEL_CLASS")
fi
if [ -n "$MODEL_KWARGS" ]; then
    extra_model_args+=(--model_kwargs "$MODEL_KWARGS")
fi
if [ -n "$STATE_DICT_KEY" ]; then
    extra_model_args+=(--state_dict_key "$STATE_DICT_KEY")
fi

watch_job_log() {
    local log_path="$1"
    local job_label="$2"
    local startup_reported=0
    local generation_reported=0

    while true; do
        if [ -f "$log_path" ]; then
            if grep -q "Traceback" "$log_path"; then
                echo "  [${job_label}] detected an error early. Check: ${log_path}"
                return
            fi
            if [ "$startup_reported" -eq 0 ] && grep -q "creating data loaders" "$log_path"; then
                echo "  [${job_label}] startup is normal; data loader stage reached."
                startup_reported=1
            fi
            if [ "$generation_reported" -eq 0 ] && grep -q "Generated Dataset Loading Completed!!!" "$log_path"; then
                echo "  [${job_label}] VAE reconstruction finished; metric evaluation is starting."
                generation_reported=1
            fi
            if grep -q "!!! DONE !!!" "$log_path"; then
                echo "  [${job_label}] evaluation finished."
                return
            fi
        fi
        sleep 15
    done
}

report_batch_status() {
    local logs=("$@")
    local completed=0
    local failed=0

    for log_path in "${logs[@]}"; do
        local log_name
        log_name=$(basename "$log_path")
        if grep -q "Traceback" "$log_path"; then
            echo "  [${log_name}] failed. See traceback in the log."
            failed=$((failed + 1))
        elif grep -q "!!! DONE !!!" "$log_path"; then
            echo "  [${log_name}] completed successfully."
            completed=$((completed + 1))
        else
            echo "  [${log_name}] finished without a final marker. Please inspect the log."
        fi
    done

    echo "Batch summary: ${completed} completed, ${failed} failed."
}

echo "VAE checkpoint: $VAE_CKPT"
echo "Results will be saved to: $RESULT_DIR"
echo "Launching ${TOTAL_EVALS} VAE reconstruction evaluation jobs across ${NUM_GPUS} GPUs..."

running_jobs=0
batch_logs=()

for ((job_idx=0; job_idx<${TOTAL_EVALS}; job_idx++)); do
    gpu_id=$(( job_idx % NUM_GPUS ))
    seed=$(( (job_idx + 1) * 1000 ))
    log_name="eval_vae_${EVAL_MODE}_seed${seed}.log"
    log_path="${RESULT_DIR}/${log_name}"
    metric_log_path="${RESULT_DIR}/metrics_vae_${EVAL_MODE}_seed${seed}.log"
    job_label="job $((job_idx + 1))/${TOTAL_EVALS}, seed=${seed}, gpu=${gpu_id}"

    echo "Launching job $((job_idx + 1))/${TOTAL_EVALS} on GPU ${gpu_id} (seed=${seed})..."
    echo "  log file: ${log_path}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python -m eval.eval_vae_reconstruction \
        --vae_checkpoint "$VAE_CKPT" \
        --opt_path "$OPT_PATH" \
        --dataset "$DATASET" \
        --data_mode "$DATA_MODE" \
        --device 0 \
        --use_ema \
        --eval_mode "$EVAL_MODE" \
        --eval_model_name "$EVAL_MODEL" \
        --eval_dataset_override express4d \
        --eval_label "$EVAL_LABEL" \
        --eval_rep_times "$REPS_PER_JOB" \
        --seed "$seed" \
        --log_file "$metric_log_path" \
        "${extra_model_args[@]}" > "$log_path" 2>&1 &

    watch_job_log "$log_path" "$job_label" &

    running_jobs=$((running_jobs + 1))
    batch_logs+=("$log_path")

    if [ "$running_jobs" -eq "$NUM_GPUS" ]; then
        echo "Waiting for the current batch of ${NUM_GPUS} jobs to finish..."
        wait
        report_batch_status "${batch_logs[@]}"
        running_jobs=0
        batch_logs=()
    fi
done

wait
if [ "${#batch_logs[@]}" -gt 0 ]; then
    report_batch_status "${batch_logs[@]}"
fi
echo "All VAE evaluation jobs have completed."

echo "Aggregating results from metric log files..."

python3 -c '
import glob
import math
import os
import re
import sys
import numpy as np

result_dir = sys.argv[1]
ckpt_path = sys.argv[2]
log_files = sorted(glob.glob(os.path.join(result_dir, "metrics_vae_*_seed*.log")))

if not log_files:
    print("Error: no VAE metric log files found.")
    sys.exit(1)

metrics = {}
r_precision = {}

for log_file in log_files:
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_summary = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        summary_match = re.match(r"========== (.+) Summary ==========", line)
        if summary_match:
            current_summary = summary_match.group(1)
            continue

        if current_summary == "R_precision":
            rp_match = re.match(
                r"---> \[(.+)\]\(top 1\) Mean: ([0-9eE+\-.]+) CInt: ([0-9eE+\-.]+);"
                r"\(top 2\) Mean: ([0-9eE+\-.]+) CInt: ([0-9eE+\-.]+);"
                r"\(top 3\) Mean: ([0-9eE+\-.]+) CInt: ([0-9eE+\-.]+);",
                line,
            )
            if rp_match:
                label = rp_match.group(1)
                r_precision.setdefault(label, []).append([
                    float(rp_match.group(2)),
                    float(rp_match.group(4)),
                    float(rp_match.group(6)),
                ])
            continue

        metric_match = re.match(r"---> \[(.+)\] Mean: ([0-9eE+\-.]+) CInterval: ([0-9eE+\-.]+)", line)
        if metric_match and current_summary:
            label = metric_match.group(1)
            metrics.setdefault(current_summary, {}).setdefault(label, []).append(float(metric_match.group(2)))

def summarize(values):
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cint = 1.96 * std / math.sqrt(len(arr))
    return mean, cint

print(f"\n==================== Aggregated VAE results for {os.path.basename(ckpt_path)} from {len(log_files)} jobs ====================")

for metric_name in ("Matching Score", "R_precision", "FID", "Diversity", "MultiModality"):
    print(f"========== {metric_name} Summary ==========")
    if metric_name == "R_precision":
        for label, values in sorted(r_precision.items()):
            mean, cint = summarize(values)
            if mean is not None:
                print(
                    f"---> [{label}](top 1) Mean: {mean[0]:.4f} CInt: {cint[0]:.4f};"
                    f"(top 2) Mean: {mean[1]:.4f} CInt: {cint[1]:.4f};"
                    f"(top 3) Mean: {mean[2]:.4f} CInt: {cint[2]:.4f};"
                )
    else:
        for label, values in sorted(metrics.get(metric_name, {}).items()):
            mean, cint = summarize(values)
            if mean is not None:
                print(f"---> [{label}] Mean: {mean:.4f} CInterval: {cint:.4f}")

print("====================================================================================================")
' "$RESULT_DIR" "$VAE_CKPT"
