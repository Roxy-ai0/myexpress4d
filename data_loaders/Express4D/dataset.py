import logging
import torch
from argparse import Namespace
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from data_loaders.data_loader_utils import get_data_by_key
from utils.arkit_utils import livelink_csv_to_sequence
from random import randint
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.Express4D.calc_mean_std import calculate_mean_std

logger = logging.getLogger(__name__)

class Express4D(Dataset):
    def __init__(self, datapath: str = './dataset',
                 split: str = "train",
                 mode: str = 'generator',
                 data_mode: str = 'arkit',
                 minimum_frames=60,
                 debug=False,
                 flip_face_on = False,
                 fps = 20,
                 max_motions = -1,
                 name='express4d',
                 **kwargs):
        assert mode in ['evaluator_train', 'eval', 'train_classifier', 'generator', 'gt']
        assert split in ['train', 'test']

        # 1. 必须先初始化 self.opt !!!
        self.opt = Namespace()
        
        # 2. 动态设置数据集名称和路径
        self.opt.dataset_name = name
        if name == 'Express4D_reaction_nofilter_v1':
            self.opt.data_root = 'dataset/Express4D_reaction_nofilter_v1'
        else:
            self.opt.data_root = 'dataset/Express4D'
            
        self.opt.motion_dir = os.path.join(self.opt.data_root, 'data')
        
        if data_mode == 'arkit_labels':
            self.opt.text_dir = os.path.join(self.opt.data_root, 'labels')
        else:
            self.opt.text_dir = os.path.join(self.opt.data_root, 'texts')

        self.opt.max_motion_length = 196
        self.opt.max_text_len = 20
        self.opt.max_motions = max_motions
        self.minimum_frames = minimum_frames

        self.opt.meta_dir = 'dataset'
        self.opt.data_rep = 'hml_vec'
        self.opt.use_cache = kwargs.get('use_cache', False)
        
        # 3. 设置索引文件路径 (不再写死)
        self.split_file = os.path.join(self.opt.data_root, f'{split}.txt')

        self.epsilon = 1E-10
        self.split = split
        self.mode = mode
        self.opt.fps = fps
        self.opt.flip_face_on = flip_face_on

        # 4. 调用原有的核心逻辑
        self.mean, self.std = calculate_mean_std()
        self.w_vectorizer = WordVectorizer('glove', 'our_vab')
        
        # 这一步会真正去加载数据，如果路径对了，就能成功
        self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, self.opt.max_motions)
        self.num_actions = 1

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
    
    def inv_transform(self, data):
        return self.t2m_dataset.inv_transform(data)
