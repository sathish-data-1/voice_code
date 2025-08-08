# import torch
# import torchaudio
# import numpy as np
# import librosa
# from scipy.signal import lfilter

# class AudioProcessor:
#     def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, 
#                  n_mels=80, f_min=0, f_max=8000, max_wave_length=48000):
#         self.sample_rate = sample_rate
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.n_mels = n_mels
#         self.f_min = f_min
#         self.f_max = f_max
#         self.max_wave_length = max_wave_length
        
#         self.mel_basis = librosa.filters.mel(
#             sr=sample_rate,
#             n_fft=n_fft,
#             n_mels=n_mels,
#             fmin=f_min,
#             fmax=f_max
#         )
        
#     def load_audio(self, path):
#         waveform, sr = torchaudio.load(path)
#         if sr != self.sample_rate:
#             resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
#             waveform = resampler(waveform)
#         return waveform
    
#     def normalize_waveform(self, waveform):
#         return waveform / (torch.max(torch.abs(waveform)) + 1e-7)
    
#     def preprocess_waveform(self, waveform):
#         # Trim silence
#         waveform, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
#         waveform = torch.from_numpy(waveform).float()
        
#         # Normalize
#         waveform = self.normalize_waveform(waveform)
        
#         # Pad or trim
#         if waveform.shape[-1] > self.max_wave_length:
#             waveform = waveform[:self.max_wave_length]
#         else:
#             pad_length = self.max_wave_length - waveform.shape[-1]
#             waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
#         return waveform.unsqueeze(0)  # Add channel dim
    
#     def waveform_to_mel(self, waveform):
#         spectrogram = torchaudio.transforms.MelSpectrogram(
#             sample_rate=self.sample_rate,
#             n_fft=self.n_fft,
#             win_length=self.win_length,
#             hop_length=self.hop_length,
#             n_mels=self.n_mels,
#             f_min=self.f_min,
#             f_max=self.f_max
#         )(waveform)
#         return torch.log(spectrogram + 1e-7)
    
#     def mel_to_waveform(self, mel):
#         griffin_lim = torchaudio.transforms.GriffinLim(
#             n_fft=self.n_fft,
#             win_length=self.win_length,
#             hop_length=self.hop_length
#         )
#         return griffin_lim(torch.exp(mel))
    
#     def add_noise(self, waveform, noise_level=0.005):
#         noise = torch.randn_like(waveform) * noise_level
#         return waveform + noise

import torch
import torchaudio
import numpy as np
import librosa
from scipy.signal import lfilter

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, 
                 n_mels=80, f_min=0, f_max=8000, max_wave_length=48000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.max_wave_length = max_wave_length
        
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max
        )
        
    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    def normalize_waveform(self, waveform):
        return waveform / (torch.max(torch.abs(waveform)) + 1e-7)
    
    def preprocess_waveform(self, waveform):
        waveform, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
        waveform = torch.from_numpy(waveform).float()
        waveform = self.normalize_waveform(waveform)
        
        if waveform.shape[-1] > self.max_wave_length:
            waveform = waveform[:self.max_wave_length]
        else:
            pad_length = self.max_wave_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
        return waveform.unsqueeze(0)
    
    def waveform_to_mel(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )(waveform)
        return torch.log(spectrogram + 1e-7)
    
    def mel_to_waveform(self, mel):
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length
        )
        return griffin_lim(torch.exp(mel))
    
    def add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise