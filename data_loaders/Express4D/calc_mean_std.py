import logging
import torch
import os
from tqdm import tqdm
import numpy as np

from data_loaders.data_loader_utils import get_data_by_key
from utils.data_loaders_utils import convert_to_6d, smooth_filter, get_identity, process_lmks, process_bsps


logger = logging.getLogger(__name__)


def calculate_mean_std(add_velocities=False, add_landmarks_diffs=False, filter_length=7, debug=False, data_dir='dataset/Express4D'):

    # means = []
    # stds = []
    tensors = []

    # data_mode_for_paths = data_mode if not '_full' in data_mode else data_mode.replace('_full','')
    for root, dirs, files in os.walk(data_dir):
        for fl in files:
            if '.npy' in fl:
                data = np.load(os.path.join(root, fl))
                tensors.append(data)
    if not tensors:
        raise ValueError(f'No .npy files were found under [{data_dir}] when calculating mean/std.')

    all_frames = np.vstack(tensors)

    overall_mean = all_frames.mean(axis=0)
    overall_std = all_frames.std(axis=0)
    zero_std_mask = overall_std < 1e-8
    if np.any(zero_std_mask):
        logger.warning(
            'Found %d feature dimensions with zero std under [%s]; replacing their std with 1.0 to avoid NaNs during normalization.',
            int(zero_std_mask.sum()),
            data_dir,
        )
        overall_std = overall_std.copy()
        overall_std[zero_std_mask] = 1.0
    return overall_mean, overall_std

if __name__ == "__main__":
    calculate_mean_std('landmarks_70_centralized', filter_length=7)


