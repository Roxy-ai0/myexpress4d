import argparse
import importlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.networks.modules import MovementConvDecoder, MovementConvEncoder
from data_loaders.humanml.utils.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_top_k,
    euclidean_distance_matrix,
)
from utils.fixseed import fixseed


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MovementAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, motions):
        return self.decoder(self.encoder(motions))


class VAEReconstructionDataset(Dataset):
    def __init__(self, vae, gt_loader, device, num_samples_limit=None):
        self.generated_motion = []
        self.mm_generated_motion = []
        self.device = device

        vae.eval()
        with torch.no_grad():
            for batch in gt_loader:
                word_embs, pos_ohots, captions, cap_lens, motions, m_lens, tokens = batch
                recon = reconstruct_batch(vae, motions, m_lens, device)
                recon = recon.detach().cpu()

                batch_size = recon.shape[0]
                for idx in range(batch_size):
                    self.generated_motion.append({
                        "word_embs": word_embs[idx].detach().cpu().numpy(),
                        "pos_ohots": pos_ohots[idx].detach().cpu().numpy(),
                        "caption": captions[idx],
                        "cap_len": int(cap_lens[idx]),
                        "motion": recon[idx].numpy(),
                        "length": int(m_lens[idx]),
                        "tokens": tokens[idx],
                    })
                    if num_samples_limit is not None and len(self.generated_motion) >= num_samples_limit:
                        return

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        return (
            data["word_embs"],
            data["pos_ohots"],
            data["caption"],
            data["cap_len"],
            data["motion"],
            data["length"],
            data["tokens"],
        )


class EmptyMMDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item):
        raise IndexError


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    r_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print("========== Evaluating Matching Score ==========")
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        matching_score_sum = 0
        top_k_count = 0
        all_size = 0
        with torch.no_grad():
            for batch in motion_loader:
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_count += calculate_top_k(argsmax, top_k=3).sum(axis=0)
                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        r_precision = top_k_count / all_size
        match_score_dict[motion_loader_name] = matching_score
        r_precision_dict[motion_loader_name] = r_precision
        activation_dict[motion_loader_name] = all_motion_embeddings

        print(f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}")
        print(f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}", file=file, flush=True)

        line = f"---> [{motion_loader_name}] R_precision: "
        for idx in range(len(r_precision)):
            line += "(top %d): %.4f " % (idx + 1, r_precision[idx])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, r_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print("========== Evaluating FID ==========")
    with torch.no_grad():
        for batch in groundtruth_loader:
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(motions=motions, m_lens=m_lens)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f"---> [{model_name}] FID: {fid:.4f}")
        print(f"---> [{model_name}] FID: {fid:.4f}", file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating Diversity ==========")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f"---> [{model_name}] Diversity: {diversity:.4f}")
        print(f"---> [{model_name}] Diversity: {diversity:.4f}", file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating MultiModality ==========")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for batch in mm_motion_loader:
                motions, m_lens = batch
                motion_embeddings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}", file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=False):
    with open(log_file, "w", encoding="utf-8") as file:
        all_metrics = OrderedDict({
            "Matching Score": OrderedDict({}),
            "R_precision": OrderedDict({}),
            "FID": OrderedDict({}),
            "Diversity": OrderedDict({}),
            "MultiModality": OrderedDict({}),
        })
        for replication in range(replication_times):
            motion_loaders = OrderedDict({"ground truth": gt_loader})
            mm_motion_loaders = OrderedDict({})
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f"==================== Replication {replication} ====================")
            print(f"==================== Replication {replication} ====================", file=file, flush=True)
            mat_score_dict, r_precision_dict, activation_dict = evaluate_matching_score(eval_wrapper, motion_loaders, file)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, activation_dict, file)
            div_score_dict = evaluate_diversity(activation_dict, file, diversity_times)
            if run_mm:
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times)

            print("!!! DONE !!!")
            print("!!! DONE !!!", file=file, flush=True)

            for key, item in mat_score_dict.items():
                all_metrics["Matching Score"].setdefault(key, []).append(item)
            for key, item in r_precision_dict.items():
                all_metrics["R_precision"].setdefault(key, []).append(item)
            for key, item in fid_score_dict.items():
                all_metrics["FID"].setdefault(key, []).append(item)
            for key, item in div_score_dict.items():
                all_metrics["Diversity"].setdefault(key, []).append(item)
            if run_mm:
                for key, item in mm_score_dict.items():
                    all_metrics["MultiModality"].setdefault(key, []).append(item)

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print("========== %s Summary ==========" % metric_name)
            print("========== %s Summary ==========" % metric_name, file=file, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + "_" + model_name] = mean
                if isinstance(mean, (np.float64, np.float32)):
                    print(f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}")
                    print(f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}", file=file, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for idx in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (idx + 1, mean[idx], conf_interval[idx])
                    print(line)
                    print(line, file=file, flush=True)
        return mean_dict


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for key in ("recon", "reconstruction", "pred", "output", "x_recon", "motion"):
            if key in value:
                tensor = first_tensor(value[key])
                if tensor is not None:
                    return tensor
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def normalize_reconstruction_shape(recon, reference):
    if recon.dim() == 4 and recon.shape[2] == 1:
        recon = recon.squeeze(2)
    if recon.dim() == 3 and recon.shape[1] == reference.shape[-1] and recon.shape[2] == reference.shape[1]:
        recon = recon.permute(0, 2, 1)
    if recon.shape != reference.shape:
        raise RuntimeError(
            f"VAE reconstruction shape {tuple(recon.shape)} does not match input shape {tuple(reference.shape)}. "
            "Add/adjust an adapter in eval/eval_vae_reconstruction.py for this model."
        )
    return recon


def reconstruct_batch(vae, motions, lengths, device):
    motions = motions.to(device).float()

    if hasattr(vae, "encode") and hasattr(vae, "decode"):
        encoded = vae.encode(motions)
        if isinstance(encoded, (list, tuple)):
            encoded = encoded[0]
        output = vae.decode(encoded)
    else:
        try:
            output = vae(motions, lengths.to(device) if torch.is_tensor(lengths) else lengths)
        except TypeError:
            output = vae(motions)

    recon = first_tensor(output)
    if recon is None:
        raise RuntimeError("VAE forward/encode-decode did not return a tensor reconstruction.")
    return normalize_reconstruction_shape(recon, motions)


def read_opt_txt(opt_path):
    values = {}
    if not opt_path or not os.path.exists(opt_path):
        return values
    with open(opt_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("-") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            values[key.strip()] = value.strip()
    return values


def coerce_value(value):
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_json_or_inline(value):
    if not value:
        return {}
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def pick_state_dict(checkpoint, use_ema, state_dict_key):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if state_dict_key:
        if state_dict_key not in checkpoint:
            raise KeyError(f"Requested state dict key [{state_dict_key}] was not found in checkpoint.")
        return checkpoint[state_dict_key]

    if use_ema:
        for key in ("model_avg", "net_ema", "model_ema", "ema", "ema_model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                print(f"loading EMA weights from checkpoint key [{key}]")
                return checkpoint[key]

    for key in ("model", "net", "state_dict", "vqvae", "autoencoder"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            print(f"loading weights from checkpoint key [{key}]")
            return checkpoint[key]

    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def build_project_movement_vae(checkpoint, args, opt_values):
    dim_pose = int(opt_values.get("dim_pose", args.dim_pose))
    foot_contact_entries = int(opt_values.get("foot_contact_entries", args.foot_contact_entries))
    enc_hidden = int(opt_values.get("dim_movement_enc_hidden", args.dim_movement_enc_hidden))
    dec_hidden = int(opt_values.get("dim_movement_dec_hidden", args.dim_movement_dec_hidden))
    latent = int(opt_values.get("dim_movement_latent", args.dim_movement_latent))

    encoder = MovementConvEncoder(dim_pose - foot_contact_entries, enc_hidden, latent)
    decoder = MovementConvDecoder(latent, dec_hidden, dim_pose)
    encoder.load_state_dict(checkpoint["movement_enc"])
    decoder.load_state_dict(checkpoint["movement_dec"])
    print("loaded project MovementConvEncoder/MovementConvDecoder checkpoint")
    return MovementAutoEncoder(encoder, decoder)


def build_external_vae(checkpoint, args):
    if not args.model_module or not args.model_class:
        raise RuntimeError(
            "This checkpoint is not the project's MovementConv autoencoder format. "
            "Pass --model_module and --model_class for your VQ-VAE implementation, or add its adapter here."
        )

    module = importlib.import_module(args.model_module)
    cls = getattr(module, args.model_class)
    kwargs = load_json_or_inline(args.model_kwargs)
    model = cls(**kwargs)

    state_dict = strip_module_prefix(pick_state_dict(checkpoint, args.use_ema, args.state_dict_key))
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)
    if missing:
        print(f"missing keys while loading VAE: {len(missing)}")
    if unexpected:
        print(f"unexpected keys while loading VAE: {len(unexpected)}")
    return model


def load_vae(args):
    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
    opt_values = read_opt_txt(args.opt_path)
    opt_values = {key: coerce_value(value) for key, value in opt_values.items()}

    if isinstance(checkpoint, dict) and "movement_enc" in checkpoint and "movement_dec" in checkpoint:
        model = build_project_movement_vae(checkpoint, args, opt_values)
    else:
        model = build_external_vae(checkpoint, args)

    return model


def get_eval_settings(eval_mode, eval_rep_times):
    if eval_mode == "debug":
        settings = (1000, 48, 20, False, 0, 0, 0)
    elif eval_mode == "wo_mm":
        settings = (1000, 48, 3, False, 0, 0, 0)
    elif eval_mode == "mm_short":
        settings = (1000, 48, 10, True, 100, 30, 10)
    else:
        raise ValueError(f"Unsupported eval_mode [{eval_mode}]")
    num_samples_limit, diversity_times, replication_times, run_mm, mm_num_samples, mm_num_repeats, mm_num_times = settings
    if eval_rep_times > 0:
        replication_times = eval_rep_times
    return num_samples_limit, diversity_times, replication_times, run_mm, mm_num_samples, mm_num_repeats, mm_num_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--opt_path", default="")
    parser.add_argument("--dataset", default="express4d")
    parser.add_argument("--data_mode", default="arkit")
    parser.add_argument("--eval_dataset_override", default="")
    parser.add_argument("--eval_model_name", default="tex_mot_match")
    parser.add_argument("--eval_mode", default="wo_mm", choices=["debug", "wo_mm", "mm_short"])
    parser.add_argument("--eval_rep_times", type=int, default=-1)
    parser.add_argument("--eval_split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--maximum_frames", type=int, default=196)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_label", default="vae_recon")
    parser.add_argument("--log_file", default="")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--model_module", default="")
    parser.add_argument("--model_class", default="")
    parser.add_argument("--model_kwargs", default="")
    parser.add_argument("--state_dict_key", default="")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--dim_pose", type=int, default=61)
    parser.add_argument("--foot_contact_entries", type=int, default=0)
    parser.add_argument("--dim_movement_enc_hidden", type=int, default=512)
    parser.add_argument("--dim_movement_dec_hidden", type=int, default=512)
    parser.add_argument("--dim_movement_latent", type=int, default=512)
    args = parser.parse_args()

    fixseed(args.seed)
    from data_loaders.get_data import get_dataset_loader

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    eval_dataset_name = args.eval_dataset_override or args.dataset

    num_samples_limit, diversity_times, replication_times, run_mm, mm_num_samples, mm_num_repeats, mm_num_times = get_eval_settings(
        args.eval_mode, args.eval_rep_times
    )

    print(f"Loading VAE checkpoint from [{args.vae_checkpoint}]...")
    vae = load_vae(args).to(device)
    vae.eval()

    print("creating data loaders...")
    gt_loader = get_dataset_loader(
        name=eval_dataset_name,
        batch_size=args.batch_size,
        num_frames=None,
        data_mode=args.data_mode,
        max_len=args.maximum_frames,
        flip_face_on=False,
        fps=args.fps,
        split=args.eval_split,
        hml_mode="gt",
        shuffle=False,
    )

    def get_vae_loader():
        recon_dataset = VAEReconstructionDataset(vae, gt_loader, device, num_samples_limit=num_samples_limit)
        motion_loader = DataLoader(
            recon_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=args.num_workers,
        )
        mm_loader = DataLoader(EmptyMMDataset(), batch_size=1, num_workers=0)
        print("Generated Dataset Loading Completed!!!")
        return motion_loader, mm_loader

    if args.log_file:
        log_file = args.log_file
    else:
        ckpt_dir = os.path.dirname(args.vae_checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.vae_checkpoint))[0]
        log_file = os.path.join(ckpt_dir, f"eval_vae_{ckpt_name}_{args.eval_mode}_seed{args.seed}.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    print(f"Will save to log file [{log_file}]")

    eval_wrapper = EvaluatorMDMWrapper(eval_dataset_name, args.eval_model_name, device)
    eval_motion_loaders = OrderedDict({args.eval_label: get_vae_loader})
    evaluation(
        eval_wrapper,
        gt_loader,
        eval_motion_loaders,
        log_file,
        replication_times,
        diversity_times,
        mm_num_times,
        run_mm=run_mm,
    )


if __name__ == "__main__":
    main()
