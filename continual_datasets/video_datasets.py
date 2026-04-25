import pickle
import random
import numpy as np
import torch

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from torch.utils.data import Dataset
from PIL import Image

try:
    import decord
    # Tell decord to return PyTorch tensors directly
    decord.bridge.set_bridge('torch')
except ImportError:
    raise ImportError("Please install decord: pip install decord")


class UCF101(Dataset):
    """
    Online Dataset for UCF101 Class-Incremental Learning.
    
    Dynamically samples frames directly from the video files during training,
    preserving the temporal variance necessary for robust continual learning.
    """
    
    NUM_CLASSES = 101

    def __init__(
        self,
        root: str,
        train: bool = True,
        num_tasks: int = 10,
        transform: Optional[Callable] = None,
        num_segments: Optional[int] = 3,
        target_transform: Optional[Callable] = None,
    ):
        self.video_root = Path(root) / "videos"
        self.train = train
        self.num_segments = num_segments
        self.transform = transform
        self.target_transform = target_transform

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video directory not found at {self.video_root}")

        # 1. Fast O(1) Video Indexing 
        # Scans the directory once instead of calling .exists() 23,000+ times
        self._available_videos = {}
        for ext in ['.mp4', '.avi', '.mkv', '.webm']:
            # videos in {class_name}/{video_name}.{ext}
            for p in self.video_root.glob(f"*/*{ext}"):
                self._available_videos[f"{p.parent.name}/{p.stem}"] = p

        # 2. Parse the PKL file
        pkl_file = Path(root) / f"UCF101_data_{num_tasks}tasks.pkl"
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
            
        split_key = 'train' if self.train else 'test'
        if split_key not in pkl_data:
            raise ValueError(f"Split '{split_key}' not found in the provided pkl file.")
        
        # 3. Robust Class Parsing
        # Extracts classes in exact order while guaranteeing no duplicates
        self.classes = []
        seen = set()
        for task_dict in pkl_data["train"]:
            for cls_name in task_dict.keys():
                if cls_name not in seen:
                    self.classes.append(cls_name)
                    seen.add(cls_name)
                    
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_mask = [
            [self.class_to_idx[cls_name] for cls_name in task_dict.keys() if cls_name in self.class_to_idx]
            for task_dict in pkl_data["train"]
        ]
        
        if len(self.classes) != self.NUM_CLASSES:
            print(f"WARNING: Expected {self.NUM_CLASSES} classes, found {len(self.classes)} in {pkl_file.name}.")

        # 4. Build the sample index
        self.samples = []
        self.task_ids = []
        missing_count = 0
        
        for task_id, task_dict in enumerate(pkl_data[split_key]):
            for cls_name, entries in task_dict.items():
                if cls_name not in self.class_to_idx:
                    print(f"WARNING: Class {cls_name} not found in {pkl_file.name}.")
                    continue  
                
                label = self.class_to_idx[cls_name]
                
                for entry in entries:
                    video_path = self._available_videos.get(f"{cls_name}/{entry}")
                    if video_path is None:
                        missing_count += 1
                        continue
                        
                    self.samples.append({
                        'video_path': str(video_path),
                        'label': label,
                        'cls_name': cls_name
                    })
                    self.task_ids.append(task_id)
        self.targets = [sample['label'] for sample in self.samples]
        print(f"Initialized Online UCF101 ({split_key}): {len(self.samples)} annotated segments.")
        if missing_count > 0:
            print(f"WARNING: {missing_count} annotated segments skipped due to missing video files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Open video efficiently
        vr = decord.VideoReader(sample['video_path'], ctx=decord.cpu(), num_threads=1)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        
        # Bound the annotations to actual video length
        t_start = 0.0
        t_end = duration
        
        # The video is divided into equal segments
        seg_duration = max(t_end - t_start, 0.1) / self.num_segments
        
        frame_indices = []
        for i in range(self.num_segments):
            s = t_start + i * seg_duration
            e = t_start + (i + 1) * seg_duration
            
            # One frame is randomly sampled from each segment during training
            if self.train:
                t = random.uniform(s, e)
            else:
                t = (s + e) / 2.0
                
            # Convert timestamp to exact frame index
            f_idx = int(t * fps)
            f_idx = min(max(f_idx, 0), total_frames - 1)
            frame_indices.append(f_idx)
            
        # Extract the batch of frames (Shape: [num_segments, H, W, C])
        frames_tensor = vr.get_batch(frame_indices)
        
        # Convert Decord tensors to PIL Images for standard torchvision transforms
        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
        
        # Apply transforms independently or grouped
        if self.transform is not None:
            frames_transformed = [self.transform(img) for img in frames_pil]
            # Stack into (num_segments, C, H, W)
            video_tensor = torch.stack(frames_transformed)
        else:
            # Fallback format if no transform is applied: [num_segments, C, H, W]
            video_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0

        label = sample['label']
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return video_tensor, label

class ActivityNet(Dataset):
    """
    Online Dataset for ActivityNet-200 Class-Incremental Learning.
    
    Dynamically samples frames directly from the video files during training,
    preserving the temporal variance necessary for robust continual learning.
    """
    
    NUM_CLASSES = 200

    def __init__(
        self,
        root: str,
        train: bool = True,
        num_tasks: int = 10,
        transform: Optional[Callable] = None,
        num_segments: Optional[int] = 3,
        target_transform: Optional[Callable] = None,
    ):
        self.video_root = Path(root) / "AnetVideos"
        self.train = train
        self.num_segments = num_segments
        self.transform = transform
        self.target_transform = target_transform

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video directory not found at {self.video_root}")

        # 1. Fast O(1) Video Indexing 
        # Scans the directory once instead of calling .exists() 23,000+ times
        self._available_videos = {}
        for ext in ['.mp4', '.avi', '.mkv', '.webm']:
            for p in self.video_root.glob(f"*{ext}"):
                self._available_videos[p.stem] = p

        # 2. Parse the PKL file
        pkl_file = Path(root) / f"ActivityNet_data_{num_tasks}tasks.pkl"
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
            
        split_key = 'train' if self.train else 'val'
        if split_key not in pkl_data:
            raise ValueError(f"Split '{split_key}' not found in the provided pkl file.")
        
        # 3. Robust Class Parsing
        # Extracts classes in exact order while guaranteeing no duplicates
        self.classes = []
        seen = set()
        for task_dict in pkl_data["train"]:
            for cls_name in task_dict.keys():
                if cls_name not in seen:
                    self.classes.append(cls_name)
                    seen.add(cls_name)
                    
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_mask = [
            [self.class_to_idx[cls_name] for cls_name in task_dict.keys() if cls_name in self.class_to_idx]
            for task_dict in pkl_data["train"]
        ]
        
        if len(self.classes) != self.NUM_CLASSES:
            print(f"WARNING: Expected {self.NUM_CLASSES} classes, found {len(self.classes)} in {pkl_file.name}.")

        # 4. Build the sample index
        self.samples = []
        self.task_ids = []
        missing_count = 0
        
        for task_id, task_dict in enumerate(pkl_data[split_key]):
            for cls_name, entries in task_dict.items():
                if cls_name not in self.class_to_idx:
                    print(f"WARNING: Class {cls_name} not found in {pkl_file.name}.")
                    continue  
                
                label = self.class_to_idx[cls_name]
                
                for entry in entries:
                    video_path = self._available_videos.get(entry['filename'])
                    if video_path is None:
                        missing_count += 1
                        continue
                        
                    self.samples.append({
                        'video_path': str(video_path),
                        't_start': float(entry['t_start']),
                        't_end': float(entry['t_end']),
                        'label': label,
                        'cls_name': cls_name
                    })
                    self.task_ids.append(task_id)
        self.targets = [sample['label'] for sample in self.samples]
        print(f"Initialized Online ActivityNet ({split_key}): {len(self.samples)} annotated segments.")
        if missing_count > 0:
            print(f"WARNING: {missing_count} annotated segments skipped due to missing video files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Open video efficiently
        vr = decord.VideoReader(sample['video_path'], ctx=decord.cpu(), num_threads=1)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        
        # Bound the annotations to actual video length
        t_start = max(0.0, sample['t_start'])
        t_end = min(duration, sample['t_end'])
        
        # The video is divided into equal segments
        seg_duration = max(t_end - t_start, 0.1) / self.num_segments
        
        frame_indices = []
        for i in range(self.num_segments):
            s = t_start + i * seg_duration
            e = t_start + (i + 1) * seg_duration
            
            # One frame is randomly sampled from each segment during training
            if self.train:
                t = random.uniform(s, e)
            else:
                t = (s + e) / 2.0
                
            # Convert timestamp to exact frame index
            f_idx = int(t * fps)
            f_idx = min(max(f_idx, 0), total_frames - 1)
            frame_indices.append(f_idx)
            
        # Extract the batch of frames (Shape: [num_segments, H, W, C])
        frames_tensor = vr.get_batch(frame_indices)
        
        # Convert Decord tensors to PIL Images for standard torchvision transforms
        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
        
        # Apply transforms independently or grouped
        if self.transform is not None:
            frames_transformed = [self.transform(img) for img in frames_pil]
            # Stack into (num_segments, C, H, W)
            video_tensor = torch.stack(frames_transformed)
        else:
            # Fallback format if no transform is applied: [num_segments, C, H, W]
            video_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0

        label = sample['label']
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return video_tensor, label