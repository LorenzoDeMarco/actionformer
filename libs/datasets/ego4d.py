import os
import json
import numpy as np
import glob # Aggiunto per cercare i file con nomi lunghi

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("ego4d")
class EGO4DDataset(Dataset):
    def __init__(
        self,
        is_training,     
        split,           
        feat_folder,     
        json_file,       
        feat_stride,     
        num_frames,      
        default_fps,     
        downsample_rate, 
        max_seq_len,     
        trunc_thresh,    
        crop_ratio,      
        input_dim,       
        num_classes,     
        file_prefix,     
        file_ext,        
        force_upsampling 
    ):
        if not isinstance(feat_folder, (list, tuple)):
            feat_folder = (feat_folder, )

        self.feat_folder = feat_folder
        self.file_prefix = file_prefix if file_prefix is not None else ''
        self.file_ext = file_ext
        self.json_file = json_file

        self.split = [s.lower() for s in split] 
        self.is_training = is_training

        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        dict_db, label_dict = self._load_json_db(self.json_file)
        
        if len(label_dict) != num_classes:
            print(f"--> INFO: Updating num_classes from {num_classes} to {len(label_dict)}")
            self.num_classes = len(label_dict)
            
        self.data_list = dict_db
        self.label_dict = label_dict

        self.db_attributes = {
            'dataset_name': 'ego4d',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }

        print(f"--> Dataset loaded: {len(self.data_list)} videos found for split {self.split}")

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                if 'annotations' not in value: continue
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        dict_db = tuple()
        for key, value in json_db.items():

            if value['subset'].lower() not in self.split:
                continue

            actual_feat_path = None
            for folder in self.feat_folder:
                search_pattern = os.path.join(folder, f"*{key}*{self.file_ext}")
                found_files = glob.glob(search_pattern)
                if found_files:
                    actual_feat_path = found_files[0]
                    break
            
            if actual_feat_path is None:
                continue

            fps = self.default_fps if self.default_fps is not None else value.get('fps', 30)
            duration = value.get('duration', 1e8)

            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = label_dict[act['label']]
            else:
                segments, labels = None, None

            dict_db += ({'id': key,
                         'actual_path': actual_feat_path, 
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'offset': value.get('offset'),
            }, )

        return dict_db, label_dict

    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        feats = np.load(video_item['actual_path'])
        
        if isinstance(feats, np.lib.npyio.NpzFile):
            key = feats.files[0]
            feats = feats[key]
            
        feats = feats.astype(np.float32)

        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,
                     'segments'        : segments,
                     'labels'          : labels,
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames,
                     'offset'          : video_item['offset'],
        }

        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )


        return data_dict
    
    def __len__(self):
        return len(self.data_list)