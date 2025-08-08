# import os
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import yaml
# from pathlib import Path

# class VoicePairDataset(Dataset):
#     def __init__(self, data_dir, config_path, mode='train'):
#         self.data_dir = Path(data_dir)
#         self.mode = mode
#         with open(config_path) as f:
#             self.config = yaml.safe_load(f)
            
#         self.audio_processor = AudioProcessor(**self.config['audio'])
        
#         # Load file pairs
#         self.input_files = []
#         self.output_files = []
        
#         split_file = self.data_dir / 'splits' / f'{mode}_files.txt'
#         with open(split_file) as f:
#             for line in f:
#                 input_path, output_path = line.strip().split('|')
#                 self.input_files.append(self.data_dir / 'raw' / input_path)
#                 self.output_files.append(self.data_dir / 'raw' / output_path)
                
#     def __len__(self):
#         return len(self.input_files)
    
#     def __getitem__(self, idx):
#         input_wave = self.audio_processor.load_audio(self.input_files[idx])
#         output_wave = self.audio_processor.load_audio(self.output_files[idx])
        
#         input_wave = self.audio_processor.preprocess_waveform(input_wave)
#         output_wave = self.audio_processor.preprocess_waveform(output_wave)
        
#         if self.mode == 'train':
#             # Data augmentation
#             input_wave = self.audio_processor.add_noise(input_wave)
            
#         return {
#             'input_audio': input_wave,
#             'output_audio': output_wave,
#             'input_path': str(self.input_files[idx]),
#             'output_path': str(self.output_files[idx])
#         }
    
#     @staticmethod
#     def collate_fn(batch):
#         inputs = torch.stack([item['input_audio'] for item in batch])
#         outputs = torch.stack([item['output_audio'] for item in batch])
#         return {
#             'input_audio': inputs,
#             'output_audio': outputs,
#             'paths': [(item['input_path'], item['output_path']) for item in batch]
#         }

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
from pathlib import Path
from .audio_processing import AudioProcessor
import torch.nn.functional as F

class VoicePairDataset(Dataset):
    def __init__(self, data_dir, config_path, mode='train'):
        self.data_dir = Path(data_dir)
        self.mode = mode
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.audio_processor = AudioProcessor(**self.config['audio'])
        
        self.input_files = []
        self.output_files = []
        
        split_file = self.data_dir / 'splits' / f'{mode}_files.txt'
        with open(split_file) as f:
            for line in f:
                input_path, output_path = line.strip().split('|')
                self.input_files.append(self.data_dir / 'raw' / input_path)
                self.output_files.append(self.data_dir / 'raw' / output_path)
                
    def __len__(self):
        return len(self.input_files)
    
    # def __getitem__(self, idx):
    #     input_wave = self.audio_processor.load_audio(self.input_files[idx])
    #     output_wave = self.audio_processor.load_audio(self.output_files[idx])
        
    #     input_wave = self.audio_processor.preprocess_waveform(input_wave)
    #     output_wave = self.audio_processor.preprocess_waveform(output_wave)
        
    #     if self.mode == 'train':
    #         input_wave = self.audio_processor.add_noise(input_wave)
            
    #     return {
    #         'input_audio': input_wave,
    #         'output_audio': output_wave,
    #         'input_path': str(self.input_files[idx]),
    #         'output_path': str(self.output_files[idx])
    #     }
    def __getitem__(self, idx):
        input_wave = self.audio_processor.load_audio(self.input_files[idx])
        output_wave = self.audio_processor.load_audio(self.output_files[idx])
        
        # Ensure both input and output have same length
        min_length = min(input_wave.shape[-1], output_wave.shape[-1])
        input_wave = input_wave[..., :min_length]
        output_wave = output_wave[..., :min_length]
        
        # Pad/trim to exact target length
        target_length = self.config['audio']['max_wave_length']
        if input_wave.shape[-1] > target_length:
            start = torch.randint(0, input_wave.shape[-1] - target_length, (1,)).item()
            input_wave = input_wave[..., start:start+target_length]
            output_wave = output_wave[..., start:start+target_length]
        else:
            pad_amount = target_length - input_wave.shape[-1]
            input_wave = F.pad(input_wave, (0, pad_amount))
            output_wave = F.pad(output_wave, (0, pad_amount))
            
        return {
            'input_audio': input_wave,
            'output_audio': output_wave,
            'input_path': str(self.input_files[idx]),
            'output_path': str(self.output_files[idx])
        }
    
    @staticmethod
    def collate_fn(batch):
        inputs = torch.stack([item['input_audio'] for item in batch])
        outputs = torch.stack([item['output_audio'] for item in batch])
        return {
            'input_audio': inputs,
            'output_audio': outputs,
            'paths': [(item['input_path'], item['output_path']) for item in batch]
        }