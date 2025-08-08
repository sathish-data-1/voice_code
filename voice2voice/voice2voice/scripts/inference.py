# import torch
# import numpy as np
# import yaml
# import soundfile as sf
# from pathlib import Path

# from models.voice2voice import Voice2VoiceModel
# from utils.audio_processing import AudioProcessor

# class RealTimeVoiceProcessor:
#     def __init__(self, config_path, model_path, chunk_size=4800):
#         with open(config_path) as f:
#             self.config = yaml.safe_load(f)
            
#         self.device = torch.device(self.config['training']['device'])
#         self.model = Voice2VoiceModel(self.config).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
        
#         self.audio_processor = AudioProcessor(**self.config['audio'])
#         self.chunk_size = chunk_size
#         self.buffer = torch.zeros(
#             1, 1, self.config['audio']['max_wave_length']).to(self.device)
        
#     def process_file(self, input_path, output_path):
#         waveform = self.audio_processor.load_audio(input_path)
#         waveform = self.audio_processor.preprocess_waveform(waveform)
        
#         with torch.no_grad():
#             waveform = waveform.to(self.device)
#             output = self.model(waveform)
            
#         output = output.squeeze().cpu().numpy()
#         sf.write(output_path, output, self.config['audio']['sample_rate'])
        
#     def process_chunk(self, audio_chunk):
#         """Process real-time audio chunk (numpy array)"""
#         with torch.no_grad():
#             # Update buffer
#             self.buffer = torch.roll(self.buffer, -self.chunk_size, dims=-1)
#             self.buffer[0, 0, -self.chunk_size:] = torch.from_numpy(audio_chunk).float().to(self.device)
            
#             # Process
#             output = self.model(self.buffer)
            
#             # Return the most recent chunk
#             return output[0, 0, -self.chunk_size:].cpu().numpy()

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='configs/base_config.yaml')
#     parser.add_argument('--model', type=str, required=True)
#     parser.add_argument('--input', type=str, required=True)
#     parser.add_argument('--output', type=str, default='output.wav')
#     args = parser.parse_args()
    
#     processor = RealTimeVoiceProcessor(args.config, args.model)
#     processor.process_file(args.input, args.output)

import torch
import numpy as np
import yaml
import soundfile as sf
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.voice2voice import Voice2VoiceModel
from utils.audio_processing import AudioProcessor

class RealTimeVoiceProcessor:
    def __init__(self, config_path, model_path, chunk_size=4800):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['training']['device'])
        self.model = Voice2VoiceModel(self.config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.audio_processor = AudioProcessor(**self.config['audio'])
        self.chunk_size = chunk_size
        self.buffer = torch.zeros(
            1, 1, self.config['audio']['max_wave_length']).to(self.device)
        
    def process_file(self, input_path, output_path):
        waveform = self.audio_processor.load_audio(input_path)
        waveform = self.audio_processor.preprocess_waveform(waveform)
        
        with torch.no_grad():
            waveform = waveform.to(self.device)
            output = self.model(waveform)
            
        output = output.squeeze().cpu().numpy()
        sf.write(output_path, output, self.config['audio']['sample_rate'])
        
    def process_chunk(self, audio_chunk):
        with torch.no_grad():
            self.buffer = torch.roll(self.buffer, -self.chunk_size, dims=-1)
            self.buffer[0, 0, -self.chunk_size:] = torch.from_numpy(audio_chunk).float().to(self.device)
            
            output = self.model(self.buffer)
            
            return output[0, 0, -self.chunk_size:].cpu().numpy()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.wav')
    args = parser.parse_args()
    
    processor = RealTimeVoiceProcessor(args.config, args.model)
    processor.process_file(args.input, args.output)