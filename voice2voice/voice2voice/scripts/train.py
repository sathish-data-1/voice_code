# import os
# import yaml
# import torch
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.nn import L1Loss
# from tqdm import tqdm

# from models.voice2voice import Voice2VoiceModel
# from utils.dataset import VoicePairDataset
# from utils.helpers import save_checkpoint

# # import os
# # import sys
# # import yaml
# # import torch
# # from torch.utils.data import DataLoader
# # from torch.optim import Adam
# # from torch.nn import L1Loss
# # from tqdm import tqdm

# # # Add project root to path
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # from voice2voice.models.voice2voice import Voice2VoiceModel
# # from voice2voice.utils.dataset import VoicePairDataset
# # from voice2voice.utils.helpers import save_checkpoint, load_checkpoint

# def train(config_path):
#     # Load config
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
    
#     # Initialize model
#     device = torch.device(config['training']['device'])
#     model = Voice2VoiceModel(config).to(device)
    
#     # Optimizer and loss
#     optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
#     criterion = L1Loss()
    
#     # Datasets
#     train_dataset = VoicePairDataset(
#         config['data']['path'],
#         config_path,
#         mode='train')
    
#     val_dataset = VoicePairDataset(
#         config['data']['path'],
#         config_path,
#         mode='val')
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=True,
#         num_workers=config['training']['num_workers'],
#         collate_fn=VoicePairDataset.collate_fn)
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=False,
#         num_workers=config['training']['num_workers'],
#         collate_fn=VoicePairDataset.collate_fn)
    
#     # Training loop
#     best_val_loss = float('inf')
#     for epoch in range(config['training']['epochs']):
#         model.train()
#         train_loss = 0.0
        
#         # Training phase
#         for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
#             inputs = batch['input_audio'].to(device)
#             targets = batch['output_audio'].to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
        
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 inputs = batch['input_audio'].to(device)
#                 targets = batch['output_audio'].to(device)
                
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()
        
#         avg_val_loss = val_loss / len(val_loader)
        
#         print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
#         # Save checkpoint
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'loss': avg_val_loss,
#             }, filename=f'best_model.pth')
        
#         if (epoch + 1) % config['training']['save_every'] == 0:
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'loss': avg_val_loss,
#             }, filename=f'checkpoint_{epoch+1}.pth')

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='configs/base_config.yaml')
#     args = parser.parse_args()
    
#     train(args.config)


#################  this ok but not checking 
# import os
# import sys
# import yaml
# import torch
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.nn import L1Loss
# from tqdm import tqdm

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.getcwd())

# from models.voice2voice import Voice2VoiceModel
# from utils.dataset import VoicePairDataset
# from utils.helpers import save_checkpoint, load_checkpoint

# def train(config_path):
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
    
#     # device = torch.device(config['training']['device'])
#     device_str = config['training']['device']
#     device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
#     model = Voice2VoiceModel(config).to(device)
    
#     optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
#     criterion = L1Loss()
    
#     train_dataset = VoicePairDataset(
#         config['data']['path'],
#         config_path,
#         mode='train')
    
#     val_dataset = VoicePairDataset(
#         config['data']['path'],
#         config_path,
#         mode='val')
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=True,
#         num_workers=config['training']['num_workers'],
#         collate_fn=VoicePairDataset.collate_fn)
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=False,
#         num_workers=config['training']['num_workers'],
#         collate_fn=VoicePairDataset.collate_fn)
    
#     best_val_loss = float('inf')
#     for epoch in range(config['training']['epochs']):
#         model.train()
#         train_loss = 0.0
        
#         for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
#             inputs = batch['input_audio'].to(device)
#             targets = batch['output_audio'].to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
        
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 inputs = batch['input_audio'].to(device)
#                 targets = batch['output_audio'].to(device)
                
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()
        
#         avg_val_loss = val_loss / len(val_loader)
        
#         print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'loss': avg_val_loss,
#             }, filename='best_model.pth')
        
#         if (epoch + 1) % config['training']['save_every'] == 0:
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'loss': avg_val_loss,
#             }, filename=f'checkpoint_{epoch+1}.pth')

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='configs/base_config.yaml')
#     args = parser.parse_args()
    
#     train(args.config)




import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # NEW
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Current working directory:", os.getcwd())

from models.voice2voice import Voice2VoiceModel
from utils.dataset import VoicePairDataset
from utils.helpers import save_checkpoint, load_checkpoint

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device setup
    device_str = config['training']['device']
    device = torch.device("cuda" if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print(f"Using device: {device}")

    model = Voice2VoiceModel(config).to(device)
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = L1Loss()

    train_dataset = VoicePairDataset(config['data']['path'], config_path, mode='train')
    val_dataset = VoicePairDataset(config['data']['path'], config_path, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=VoicePairDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=VoicePairDataset.collate_fn
    )

    best_val_loss = float('inf')
    scaler = GradScaler()  # For mixed precision

    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
            inputs = batch['input_audio'].to(device, non_blocking=True)
            targets = batch['output_audio'].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_audio'].to(device, non_blocking=True)
                targets = batch['output_audio'].to(device, non_blocking=True)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, filename='best_model.pth')

        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, filename=f'checkpoint_{epoch+1}.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    args = parser.parse_args()
    train(args.config)
