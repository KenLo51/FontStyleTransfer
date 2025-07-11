import argparse
import os
import numpy as np
import tqdm

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader

import model as FG_model
import dataset as FG_dataset
from torchvision.utils import make_grid, save_image

# Argument parser for command line arguments
arg = argparse.ArgumentParser(description='Training script for font generation model')

# Args for hyperparameters
arg.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
arg.add_argument('--num-styles-per-batch', type=int, default=4, help='Number of styles per batch')
arg.add_argument('--contrastive-weight', type=float, default=0.5, help='Weight for contrastive loss')
arg.add_argument('--pixel-weight', type=float, default=1.0, help='Weight for pixel reconstruction loss')
arg.add_argument('--adversarial-weight', type=float, default=0.0, help='Weight for adversarial loss')
arg.add_argument('--style-shuffle', action='store_true', help='Whether to shuffle styles in the batch')
arg.add_argument('--content-shuffle', action='store_true', help='Whether to shuffle content in the batch')
arg.add_argument('--neg-lossfn', type=str, default='sigmoid', choices=['sigmoid', 'reciprocal', 'sech', 'csch'], help='Negative loss function for contrastive loss. Options: sigmoid, reciprocal, sech, csch')
arg.add_argument('--pos-lossfn', type=str, default='abs', choices=['square', 'abs'], help='Loss function for contrastive loss. Options: square, abs')

arg.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
arg.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu)')
arg.add_argument('--trainSet', type=str, default='train', help='Dataset to use for training (train or val)')
arg.add_argument('--testSet', type=str, default='val', help='Dataset to use for testing (train or val)')

# Args for pre-trained models
arg.add_argument('--pre-style-encoder', type=str, default='', help='Pre-trained style encoder model path')
arg.add_argument('--pre-content-encoder', type=str, default='', help='Pre-trained content encoder model path')
arg.add_argument('--pre-decoder', type=str, default='', help='Pre-trained decoder model path')

# Args for output files
arg.add_argument('--output-style-encoder', type=str, default='combined_style_encoder_final.pth', help='Output file for the style encoder model')
arg.add_argument('--output-content-encoder', type=str, default='combined_content_encoder_final.pth', help='Output file for the content encoder model')
arg.add_argument('--output-decoder', type=str, default='combined_font_decoder_final.pth', help='Path to save the trained decoder model')
arg.add_argument('--output-style-discriminator', type=str, default='combined_style_discriminator_final.pth', help='Output file for the style discriminator model')
arg.add_argument('--output-content-discriminator', type=str, default='combined_content_discriminator_final.pth', help='Output file for the content discriminator model')
arg.add_argument('--loss-logfile', type=str, default='trainCombined_loss_log.txt', help='File to log training losses')
arg.add_argument('--generate-samples-dir', type=str, default='', help='Directory to save generated samples during training')
arg.add_argument('--style-encoder-checkpoint', type=str, default='', help='Checkpoint file for style encoder')
arg.add_argument('--content-encoder-checkpoint', type=str, default='', help='Checkpoint file for content encoder')
arg.add_argument('--font-decoder-checkpoint', type=str, default='', help='Checkpoint file for font decoder')
arg.add_argument('--style-discriminator-checkpoint', type=str, default='', help='Checkpoint file for style discriminator')
arg.add_argument('--content-discriminator-checkpoint', type=str, default='', help='Checkpoint file for content discriminator')
arg.add_argument('--checkpoint-period', type=int, default=2, help='Period for saving checkpoints during training')

args = arg.parse_args()

# Configuration hyperparameters
batch_size = args.batch_size
training_epochs = args.epochs
num_styles_per_batch = args.num_styles_per_batch  # Number of styles per batch
contrastive_loss_weight = args.contrastive_weight  # Weight for contrastive loss
pixel_loss_weight = args.pixel_weight  # Weight for pixel reconstruction loss
adversarial_loss_weight = args.adversarial_weight  # Weight for adversarial loss
with_style_shuffling = args.style_shuffle  # Whether to shuffle styles in the batch
with_content_shuffling = args.content_shuffle  # Whether to shuffle content in the batch

checkpoint_period = args.checkpoint_period  # Period for saving checkpoints during training
# Get the device to use for training
device = args.device
print(f"Using device: {device}")

# Initialize the dataset
dataset_train = FG_dataset.FontDataset(font_dir = args.trainSet)
dataset_val = FG_dataset.FontDataset(font_dir = args.testSet)

# Define the transforms
transform_augmente = transforms.Compose([
    transforms.RandomAffine(degrees=10, 
                            translate=(0.1, 0.1), 
                            scale=(0.80, 1.05),
                            shear=5,
                            fill=1.0),  # Random affine transformation
])
transform_encoder = transforms.Compose([
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=1.0),  # Random erasing
])
transform_decoder = None

# Define the functions for loading and saving the model & checkpoints
def save_checkpoint(model, optimizer, scheduler=None, epoch=0, filepath='checkpoint.pth'):
    """
    Save the model, optimizer, and scheduler state to a checkpoint file.
    
    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The learning rate scheduler to save.
        epoch: The current epoch number.
        filepath: The path to save the checkpoint file.
    """
    data = dict()
    if epoch is not None:
        data['epoch'] = epoch
    if model is not None:
        data['model_state_dict'] = model.state_dict()
    if optimizer is not None:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()
        
    # Save the checkpoint before deleting the old one
    if os.path.isfile(filepath):
        os.rename(filepath, filepath + '.bak')
        torch.save(data, filepath)
        os.remove(filepath + '.bak')
    else:
        torch.save(data, filepath)
    
    print(f"Checkpoint saved to {filepath}")
    
def load_checkpoint(model, optimizer, scheduler=None, filepath='checkpoint.pth'):
    """
    Load the model, optimizer, and scheduler state from a checkpoint file.
    
    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        scheduler: The learning rate scheduler to load the state into.
        filepath: The path to the checkpoint file.
        
    Returns:
        epoch: The epoch number from the checkpoint.
    """
    checkpoint = torch.load(filepath, weights_only=True)
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = 0
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath}, starting from epoch {epoch + 1}")
    
    return epoch
    
def save_model(model, filepath):
    """
    Save the model state to a file.
    
    Args:
        model: The model to save.
        filepath: The path to save the model file.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    
def load_model(model, filepath):
    """
    Load the model state from a file.
    
    Args:
        model: The model to load the state into.
        filepath: The path to the model file.
    """
    model.load_state_dict(torch.load(filepath, weights_only=True))
    print(f"Model loaded from {filepath}")

def get_model_parameters(model):
    """
    Print the number of parameters in the model.
    Args:
        model: The model to analyze.
    Returns:
        num_params_grad: Number of parameters that require gradients.
        num_params: Total number of parameters in the model.
    """
    num_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    return num_params_grad, num_params

# Initialize the model components
style_encoder = FG_model.StyleEncoder().to(device)
content_encoder = FG_model.ContentEncoder().to(device)
font_decoder = FG_model.FontDecoder().to(device)
if adversarial_loss_weight > 0:
    style_discriminator = FG_model.StyleDiscriminator().to(device)
    content_discriminator = FG_model.ContentDiscriminator().to(device)
else:
    style_discriminator, content_discriminator = None, None

# Print the number of parameters in each model
style_num_params_grad, style_num_params = get_model_parameters(style_encoder)
print(f"Style Encoder: {style_num_params_grad} trainable parameters, {style_num_params} total parameters")
content_num_params_grad, content_num_params = get_model_parameters(content_encoder)
print(f"Content Encoder: {content_num_params_grad} trainable parameters, {content_num_params} total parameters")
font_num_params_grad, font_num_params = get_model_parameters(font_decoder)
print(f"Font Decoder: {font_num_params_grad} trainable parameters, {font_num_params} total parameters")
if adversarial_loss_weight > 0:
    style_disc_num_params_grad, style_disc_num_params = get_model_parameters(style_discriminator)
    print(f"Style Discriminator: {style_disc_num_params_grad} trainable parameters, {style_disc_num_params} total parameters")
    content_disc_num_params_grad, content_disc_num_params = get_model_parameters(content_discriminator)
    print(f"Content Discriminator: {content_disc_num_params_grad} trainable parameters, {content_disc_num_params} total parameters")

# Define optimizers for each component
optimizers = {
    'style_encoder': torch.optim.AdamW(style_encoder.parameters(), lr=5e-4),
    'content_encoder': torch.optim.AdamW(content_encoder.parameters(), lr=5e-4),
    'font_decoder': torch.optim.AdamW(font_decoder.parameters(), lr=5e-4)
}
if adversarial_loss_weight > 0:
    optimizers['style_discriminator'] = torch.optim.AdamW(style_discriminator.parameters(), lr=1e-4)
    optimizers['content_discriminator'] = torch.optim.AdamW(content_discriminator.parameters(), lr=1e-4)

# Define the scheduler for learning rate decay
schedulers = {
    'style_encoder': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['style_encoder'], T_max=training_epochs * (len(dataset_train)+batch_size-1) // batch_size),
    'content_encoder': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['content_encoder'], T_max=training_epochs * (len(dataset_train)+batch_size-1) // batch_size),
    'font_decoder': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['font_decoder'], T_max=training_epochs * (len(dataset_train)+batch_size-1) // batch_size)
}
if adversarial_loss_weight > 0:
    schedulers['style_discriminator'] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['style_discriminator'], T_max=training_epochs * (len(dataset_train)+batch_size-1) // batch_size)
    schedulers['content_discriminator'] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['content_discriminator'], T_max=training_epochs * (len(dataset_train)+batch_size-1) // batch_size)

# Load pre-trained weights if available
start_epoch = 0
style_encoder_path = args.pre_style_encoder
style_encoder_checkpoint = args.style_encoder_checkpoint
content_encoder_path = args.pre_content_encoder
content_encoder_checkpoint = args.content_encoder_checkpoint
font_decoder_path = args.pre_decoder
font_decoder_checkpoint = args.font_decoder_checkpoint
style_discriminator_checkpoint = args.style_discriminator_checkpoint
content_discriminator_checkpoint = args.content_discriminator_checkpoint

if os.path.isfile(style_encoder_path):
    print(f"Loading pre-trained style encoder from {style_encoder_path}")
    load_model(style_encoder, style_encoder_path)
elif os.path.isfile(style_encoder_checkpoint):
    print(f"Loading style encoder checkpoint from {style_encoder_checkpoint}")
    new_start_epoch = load_checkpoint(style_encoder, optimizers['style_encoder'], schedulers['style_encoder'], style_encoder_checkpoint)
    start_epoch = max(start_epoch, new_start_epoch)
if os.path.isfile(content_encoder_path):
    print(f"Loading pre-trained content encoder from {content_encoder_path}")
    load_model(content_encoder, content_encoder_path)
elif os.path.isfile(content_encoder_checkpoint):
    print(f"Loading content encoder checkpoint from {content_encoder_checkpoint}")
    new_start_epoch = load_checkpoint(content_encoder, optimizers['content_encoder'], schedulers['content_encoder'], content_encoder_checkpoint)
    start_epoch = max(start_epoch, new_start_epoch)
if os.path.isfile(font_decoder_path):
    print(f"Loading pre-trained font decoder from {font_decoder_path}")
    load_model(font_decoder, font_decoder_path)
elif os.path.isfile(font_decoder_checkpoint):
    print(f"Loading font decoder checkpoint from {font_decoder_checkpoint}")
    new_start_epoch = load_checkpoint(font_decoder, optimizers['font_decoder'], schedulers['font_decoder'], font_decoder_checkpoint)
    start_epoch = max(start_epoch, new_start_epoch)
if adversarial_loss_weight > 0:
    if os.path.isfile(style_discriminator_checkpoint):
        print(f"Loading style discriminator checkpoint from {style_discriminator_checkpoint}")
        new_start_epoch = load_checkpoint(style_discriminator, optimizers['style_discriminator'], schedulers['style_discriminator'], style_discriminator_checkpoint)
        start_epoch = max(start_epoch, new_start_epoch)
    if os.path.isfile(content_discriminator_checkpoint):
        print(f"Loading content discriminator checkpoint from {content_discriminator_checkpoint}")
        new_start_epoch = load_checkpoint(content_discriminator, optimizers['content_discriminator'], schedulers['content_discriminator'], content_discriminator_checkpoint)
        start_epoch = max(start_epoch, new_start_epoch)

def log_generate_images(style_encoder, content_encoder, font_decoder,
                        style_images, content_images, target_images,
                        save_path):
    """ Generate images from the style and content encoders and save them to a file.
    
    Args:
        style_encoder: The style encoder model
        content_encoder: The content encoder model
        font_decoder: The font decoder model
        style_images: The style images (batch_size, height, width)
        content_images: The content images (batch_size, height, width)
        target_images: The target images (batch_size, height, width)
        save_path: The path to save the generated images
    """
    style_encoder.eval()
    content_encoder.eval()
    font_decoder.eval()
    
    with torch.no_grad():
        # Ensure images are on the right device and normalize to [-1, 1]
        style_images = (style_images.to(device) * 2.0) - 1.0
        content_images = (content_images.to(device) * 2.0) - 1.0
        target_images = (target_images.to(device) * 2.0) - 1.0
        
        # Extract style and content features
        style_features, _, _ = style_encoder.encode(style_images)
        content_features, _, _ = content_encoder.encode(content_images)
        
        # Generate images
        generated_images = font_decoder(style_features, content_features)
        
        # Create a grid to display the images
        batch_size = style_images.shape[0]
        height = style_images.shape[1]
        width = style_images.shape[2]
        
        # Create the full comparison image
        grid_height = 4  # style, content, generated, target
        grid_width = batch_size
        
        # Denormalize to [0, 1] for saving
        style_display = (style_images + 1.0) / 2.0
        content_display = (content_images + 1.0) / 2.0
        generated_display = (generated_images + 1.0) / 2.0
        target_display = (target_images + 1.0) / 2.0
        
        # Create the grid
        grid = torch.ones((grid_height, grid_width, height, width), device=device)
        
        # Fill the grid
        for i in range(batch_size):
            grid[0, i] = style_display[i]
            grid[1, i] = content_display[i]
            grid[2, i] = generated_display[i]
            grid[3, i] = target_display[i]
        
        # Reshape grid for torchvision
        grid = grid.reshape(grid_height * grid_width, 1, height, width)
        
        # Save the grid
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Use torchvision to create grid image
        
        # Reshape into 4x batch_size grid, 4 rows (style, content, generated, target)
        grid_image = make_grid(grid, nrow=grid_width, padding=4, pad_value=1.0)
        
        # Save the grid image
        save_image(grid_image, save_path)
        
        return generated_images
    
# Training functions with combined loss
def train_one_epoch(style_encoder, content_encoder, font_decoder, style_discriminator, content_discriminator,
                   dataloader, optimizers, schedulers=None,
                   transform=None, transform_encoder=None, transform_decoder=None, batch_size=64):
    style_encoder.train()
    content_encoder.train()
    font_decoder.train()
    if adversarial_loss_weight > 0:
        style_discriminator.train()
        content_discriminator.train()
    
    total_loss = 0.0
    total_pixel_loss = 0.0
    total_contrastive_loss = 0.0
    total_adversarial_loss = 0.0
    num_batches = 0
    
    tqdm_pbar = tqdm.tqdm(dataloader, desc="Training", total=len(dataloader))
    for data in tqdm_pbar:
        images = data["image"]
        num_styles = images.shape[0]
        num_chars = images.shape[1]
        if num_styles < 2 or num_chars < 2:  # Need at least 2 styles for contrastive loss
            continue

        # Split images into patches if needed
        char_orders = np.arange(num_chars)
        np.random.shuffle(char_orders)
        char_orders = np.array_split(char_orders, max(1, (num_styles * num_chars) // batch_size ))
        
        for char_order in char_orders:
            batch_images = images[:, char_order, :, :].to(device)
            if len(batch_images.shape) == 4:
                num_styles, num_chars, image_width, image_height = batch_images.shape
            elif len(batch_images.shape) == 5:
                num_styles, num_chars, num_channels ,image_width, image_height = batch_images.shape
            else:
                raise ValueError(f"Unexpected image shape: {batch_images.shape}")
            
            if num_styles < 2 or num_chars < 2:  # Need at least 2 styles for contrastive loss
                continue
            
            # Apply the augmentation transform
            if transform is not None:
                source_images = transform(batch_images)
            else:
                source_images = batch_images
            
            source_images = source_images.reshape(-1, image_width, image_height)
            
            # Normalize images to [-1, 1] range
            source_images = source_images.float() * 2.0 - 1.0
            
            # Apply the specific transforms
            if transform_decoder is not None:
                target_images = transform_decoder(source_images)
            else:
                target_images = source_images
            
            if transform_encoder is not None:
                input_images = transform_encoder(source_images)
            else:
                input_images = source_images
                
            # Train discriminators if adversarial loss is enabled ###############################
            if adversarial_loss_weight > 0.0:
                # Clear gradients
                for opt in optimizers.values():
                    opt.zero_grad()
                # inference the encoders and decoder for generated images
                style_encoder.eval()
                content_encoder.eval()
                font_decoder.eval()
                style_discriminator.train()
                content_discriminator.train()
                with torch.no_grad():
                    style_features, _, _ = style_encoder.encode(input_images)
                    content_features, _, _ = content_encoder.encode(input_images)
                    #     Shuffle the content and style features
                    if with_style_shuffling:
                        content_features = content_features.reshape(num_styles, num_chars, -1)
                        
                        random_style_orders = np.arange(num_styles)
                        np.random.shuffle(random_style_orders)

                        content_features = content_features[random_style_orders, :, :]
                        content_features = content_features.reshape(-1, content_features.shape[-1])
                        
                    if with_content_shuffling:
                        style_features = style_features.reshape(num_styles, num_chars, -1)
                        
                        random_content_orders = np.arange(num_chars)
                        np.random.shuffle(random_content_orders)

                        style_features = style_features[:, random_content_orders, :]
                        style_features = style_features.reshape(-1, style_features.shape[-1])
                    
                    # Decode the features to generate images
                    generated_images = font_decoder(style_features, content_features)
                    
                #     Shuffle the images
                random_style_orders = np.arange(num_styles)
                np.random.shuffle(random_style_orders)
                random_content_orders = np.arange(num_chars)
                np.random.shuffle(random_content_orders)
                
                source_images = source_images.reshape(num_styles, num_chars, image_width, image_height)
                shuffled_style_images = source_images[random_style_orders, :, :, :].reshape(-1, image_width, image_height)
                shuffled_content_images = source_images[:, random_content_orders, :, :].reshape(-1, image_width, image_height)
                source_images = source_images.reshape(-1, image_width, image_height)
                
                #     Postive samples for discriminator
                pos_style_pred = style_discriminator(shuffled_content_images, source_images)
                pos_content_pred = content_discriminator(shuffled_style_images, source_images)
                
                #     Negative samples for discriminator
                neg_style_pred = style_discriminator(shuffled_content_images, generated_images)
                neg_content_pred = content_discriminator(shuffled_style_images, generated_images)
                
                # Calculate adversarial loss
                style_adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_style_pred, torch.ones_like(pos_style_pred)) + \
                                 torch.nn.functional.binary_cross_entropy_with_logits(neg_style_pred, torch.zeros_like(neg_style_pred))
                content_adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_content_pred, torch.ones_like(pos_content_pred)) + \
                                   torch.nn.functional.binary_cross_entropy_with_logits(neg_content_pred, torch.zeros_like(neg_content_pred))
                
                adversarial_loss = style_adv_loss + content_adv_loss
                
                # Update discriminator parameters
                adversarial_loss.backward()
                
                for name, opt in optimizers.items():
                    if name in ['style_discriminator', 'content_discriminator']:
                        opt.step()
                        
                for name, scheduler in schedulers.items():
                    if name in ['style_discriminator', 'content_discriminator']:
                        scheduler.step()
                        
                # Set the encoders and decoder back to training mode
                style_encoder.train()
                content_encoder.train()
                font_decoder.train()
                style_discriminator.eval()
                content_discriminator.eval()
            
            # Train the encoders and decoder for generated images ###############################
            # Clear gradients
            for opt in optimizers.values():
                opt.zero_grad()
            # Extract style and content features
            style_features, _, _ = style_encoder.encode(input_images)
            content_features, _, _ = content_encoder.encode(input_images)
            
            # Calculate contrastive loss
            if contrastive_loss_weight > 0:
                #     Calculate contrastive loss for style features
                style_positive_mask = torch.kron(torch.eye(num_styles, dtype=torch.bool),
                                    torch.ones((num_chars, num_chars), dtype=torch.bool)).to(device)
                
                #     Calculate contrastive loss for content features
                content_target_idx = torch.arange(num_styles * num_chars)
                content_positive_mask = (content_target_idx.unsqueeze(0) % num_chars == 
                                    content_target_idx.unsqueeze(1) % num_chars).to(device)
                
                #     Feature magnitude regularization
                feature_reg_weight = 0.01
                #     Calculate feature magnitude regularization
                style_magnitude_reg = torch.norm(style_features, p=2, dim=1).mean()
                content_magnitude_reg = torch.norm(content_features, p=2, dim=1).mean()
                
                postive_steepness = 1.0
                postive_scale = 1.0
                postive_offset = 0.0
                negative_steepness = 1.0
                negative_scale = 1.0
                #     Style contrastive loss
                def cal_negative_loss(distances, positive_mask):
                    if args.neg_lossfn == 'sigmoid':
                        neg_loss = (1 - torch.sigmoid(negative_steepness * distances[~positive_mask] + 0.0)).mean()
                    elif args.neg_lossfn == 'reciprocal':
                        neg_loss = (1 / (1e-4 + negative_steepness * distances[~positive_mask] + 1.0)).mean()
                    elif args.neg_lossfn == 'sech':
                        neg_loss = (1 / torch.cosh(negative_steepness * distances[~positive_mask] + 0.0)).mean()
                    elif args.neg_lossfn == 'coth':
                        neg_loss = (1 / torch.tanh(negative_steepness * distances[~positive_mask] + 0.549306) - 1).mean()
                    elif args.neg_lossfn == 'exp':
                        neg_loss = torch.exp(-(negative_steepness * distances[~positive_mask] + 0.0)).mean()
                    else:
                        raise ValueError(f"Unknown negative loss function: {args.neg_lossfn}")
                    return negative_scale * neg_loss
                def cal_positive_loss(distances, positive_mask):
                    if args.pos_lossfn == 'square':
                        pos_loss = (postive_steepness * distances[positive_mask] + postive_offset).pow(2).mean()
                    elif args.pos_lossfn == 'exp':
                        pos_loss = torch.exp(postive_steepness * distances[positive_mask] + postive_offset).mean()
                    elif args.pos_lossfn == 'abs':
                        pos_loss = (postive_steepness * distances[positive_mask] + postive_offset).mean()
                    else:
                        raise ValueError(f"Unknown postive loss function: {args.pos_lossfn}")
                    return postive_scale * pos_loss
                style_distances = torch.cdist(style_features, style_features, p=2).abs()
                style_pos_loss = cal_positive_loss(style_distances, style_positive_mask)
                style_neg_loss = cal_negative_loss(style_distances, style_positive_mask)
                style_loss = style_pos_loss + style_neg_loss + feature_reg_weight * style_magnitude_reg
                
                #     Content contrastive loss
                content_distances = torch.cdist(content_features, content_features, p=2).abs()
                content_pos_loss = cal_positive_loss(content_distances, content_positive_mask)
                content_neg_loss = cal_negative_loss(content_distances, content_positive_mask)
                content_loss = content_pos_loss + content_neg_loss + feature_reg_weight * content_magnitude_reg
                
                #     Combined contrastive loss
                contrastive_loss = style_loss + content_loss
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
            
            # Decode the features to generate images
            #     Shuffle the content and style features
            if with_style_shuffling:
                content_features = content_features.reshape(num_styles, num_chars, -1)
                
                random_style_orders = np.arange(num_styles)
                np.random.shuffle(random_style_orders)

                content_features = content_features[random_style_orders, :, :]
                content_features = content_features.reshape(-1, content_features.shape[-1])
                
            if with_content_shuffling:
                style_features = style_features.reshape(num_styles, num_chars, -1)
                
                random_content_orders = np.arange(num_chars)
                np.random.shuffle(random_content_orders)

                style_features = style_features[:, random_content_orders, :]
                style_features = style_features.reshape(-1, style_features.shape[-1])
                
            #     Generate images using the features
            generated_images = font_decoder(style_features, content_features)
            
            # Calculate pixel reconstruction loss
            pixel_loss = torch.nn.functional.l1_loss(generated_images, target_images)
            
            # Calculate adversarial loss if applicable
            if adversarial_loss_weight > 0.0:
                random_style_orders = np.arange(num_styles)
                np.random.shuffle(random_style_orders)
                random_content_orders = np.arange(num_chars)
                np.random.shuffle(random_content_orders)
                
                #     Shuffle the images
                source_images = source_images.reshape(num_styles, num_chars, image_width, image_height)
                shuffled_style_images = source_images[random_style_orders, :, :, :].reshape(-1, image_width, image_height)
                shuffled_content_images = source_images[:, random_content_orders, :, :].reshape(-1, image_width, image_height)
                source_images = source_images.reshape(-1, image_width, image_height)
                
                #      Postive samples for discriminator ( Do not update the discriminator parameters. So the positive samples, that compute from the source images, is not used)
                # pos_style_pred = style_discriminator(shuffled_content_images, source_images)
                # pos_content_pred = content_discriminator(shuffled_style_images, source_images)
                #      Negative samples for discriminator
                neg_style_pred = style_discriminator(shuffled_content_images, generated_images)
                neg_content_pred = content_discriminator(shuffled_style_images, generated_images)
                
                # Calculate adversarial loss
                #    Set the target as ones for adversarial the discriminator
                style_adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_style_pred, torch.ones_like(neg_style_pred))
                content_adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_content_pred, torch.ones_like(neg_content_pred))
                adversarial_loss = style_adv_loss + content_adv_loss
                
            else:
                adversarial_loss = torch.tensor(0.0, device=device)
            
            # Combined total loss
            loss = pixel_loss_weight * pixel_loss + contrastive_loss_weight * contrastive_loss + adversarial_loss_weight * adversarial_loss
            loss.backward()
        
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(style_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(content_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(font_decoder.parameters(), max_norm=1.0)
            if adversarial_loss_weight > 0.0:
                torch.nn.utils.clip_grad_norm_(style_discriminator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(content_discriminator.parameters(), max_norm=1.0)
            
            # Update parameters
            for name, opt in optimizers.items():
                if name in ['style_encoder', 'content_encoder', 'font_decoder']:
                    opt.step()
            
            # Step schedulers
            if schedulers is not None:
                for name, scheduler in schedulers.items():
                    if name in ['style_encoder', 'content_encoder', 'font_decoder']:
                        scheduler.step()
                        
            # Accumulate losses
            weighted_loss = loss.item()
            total_loss += weighted_loss
            pixel_loss = pixel_loss.item()
            total_pixel_loss += pixel_loss
            contrastive_loss = contrastive_loss.item()
            total_contrastive_loss += contrastive_loss
            if adversarial_loss_weight > 0:
                adversarial_loss = adversarial_loss.item()
                total_adversarial_loss += adversarial_loss
                        
            num_batches += 1
        
        # Update progress bar
        print_datas = {
            'l': f"{weighted_loss:.4f}",
            'pl': f"{pixel_loss:.4f}",
        }
        if contrastive_loss_weight > 0:
            print_datas['cl'] = f"{contrastive_loss:.4f}"
        if adversarial_loss_weight > 0:
            print_datas['al'] = f"{adversarial_loss:.4f}"
        if schedulers is not None:
            print_datas['elr'] = f"{schedulers['style_encoder'].get_last_lr()[0]:.2e}"
            print_datas['dlr'] = f"{schedulers['font_decoder'].get_last_lr()[0]:.2e}"
        
        tqdm_pbar.set_postfix(print_datas)
    
    # Calculate average losses over all batches
    avg_loss = total_loss / num_batches
    avg_pixel_loss = total_pixel_loss / num_batches
    avg_contrastive_loss = total_contrastive_loss / num_batches
    avg_adversarial_loss = total_adversarial_loss / num_batches if adversarial_loss_weight > 0 else 0.0
    
    return avg_loss, avg_pixel_loss, avg_contrastive_loss, avg_adversarial_loss

def eval_images_similarity(images1, images2):
    """
    Evaluate the similarity between two sets of images using multiple metrics.
    
    Args:
        images1: First set of images (batch_size, channels, height, width).
        images2: Second set of images (batch_size, channels, height, width).
        
    Returns:
        mae: Mean Absolute Error between the two sets of images.
        mse: Mean Squared Error between the two sets of images.
        psnr: Peak Signal-to-Noise Ratio.
        ssim: Structural Similarity Index Measure.
    """
    # Normalize to [0, 1]
    images1 = (images1 + 1.0) / 2.0
    images2 = (images2 + 1.0) / 2.0
    
    # Ensure images have the same number of channels
    if len(images1.shape) == 3:
        images1 = images1.unsqueeze(1)
    if len(images2.shape) == 3:
        images2 = images2.unsqueeze(1)
    
    # Flatten the images to vectors
    flat_images1 = images1.view(images1.size(0), -1)
    flat_images2 = images2.view(images2.size(0), -1)
    
    # Calculate MAE
    mae = torch.nn.functional.l1_loss(flat_images1, flat_images2, reduction='mean')
    # Calculate MSE
    mse = torch.nn.functional.mse_loss(flat_images1, flat_images2, reduction='mean')
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    # PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    # For images normalized to [0, 1], MAX = 1
    epsilon = 1e-8  # To avoid log(0)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + epsilon)
    
    # Calculate SSIM (Structural Similarity Index)
    # Reshape for window-based calculation
    batch_size = images1.size(0)
    window_size = 11
    sigma = 1.5
    
    # Create a 1D Gaussian kernel
    window_1d = torch.exp(-(torch.arange(window_size) - window_size//2)**2 / (2 * sigma**2))
    window_1d = window_1d / window_1d.sum()
    
    # Create a 2D Gaussian kernel
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    window = window_2d.expand(1, 1, window_size, window_size).to(images1.device)
    
    # Constants for stability
    C1 = (0.01)**2
    C2 = (0.03)**2
    
    # Reshape images to 4D tensors with single channel
    img1 = images1.view(batch_size, 1, images1.size(-2), images1.size(-1))
    img2 = images2.view(batch_size, 1, images2.size(-2), images2.size(-1))
    
    # Calculate means
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = torch.nn.functional.conv2d(img1**2, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2**2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = ssim_map.mean()
    
    return mae, mse, psnr, ssim

def validate_one_epoch(style_encoder, content_encoder, font_decoder,
                     dataloader, transform=None, transform_encoder=None, transform_decoder=None, split_batch=1):
    style_encoder.eval()
    content_encoder.eval()
    font_decoder.eval()
    
    # Initialize accumulators for losses
    total_loss = 0.0
    total_pixel_loss = 0.0
    total_contrastive_loss = 0.0
    # Initialize metrics
    total_mae, total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    tqdm_pbar = tqdm.tqdm(dataloader, desc="Validation", total=len(dataloader))
    with torch.no_grad():
        for data in tqdm_pbar:
            images = data["image"]
            num_styles = images.shape[0]
            num_chars = images.shape[1]
            if num_styles < 2 or num_chars < 2:  # Need at least 2 styles for contrastive loss
                continue

            # Clear gradients
            for opt in optimizers.values():
                opt.zero_grad()
            
            # Split images into patches if needed
            char_orders = np.arange(num_chars)
            np.random.shuffle(char_orders)
            char_orders = np.array_split(char_orders, (num_styles * num_chars) // batch_size)
            
            for char_order in char_orders:
                batch_images = images[:, char_order, :, :].to(device)
                if len(batch_images.shape) == 4:
                    num_styles, num_chars, image_width, image_height = batch_images.shape
                elif len(batch_images.shape) == 5:
                    num_styles, num_chars, num_channels ,image_width, image_height = batch_images.shape
                else:
                    raise ValueError(f"Unexpected image shape: {batch_images.shape}")
                
                # Apply the augmentation transform
                if transform is not None:
                    source_images = transform(batch_images)
                else:
                    source_images = batch_images
                
                source_images = source_images.reshape(-1, image_width, image_height)
                
                # Normalize images to [-1, 1] range
                source_images = source_images.float() * 2.0 - 1.0
                
                # Apply the specific transforms
                if transform_decoder is not None:
                    target_images = transform_decoder(source_images)
                else:
                    target_images = source_images
                
                if transform_encoder is not None:
                    input_images = transform_encoder(source_images)
                else:
                    input_images = source_images
                
                # Extract style and content features
                style_features, _, _ = style_encoder.encode(input_images)
                content_features, _, _ = content_encoder.encode(input_images)
                    
                # Calculate contrastive loss
                if contrastive_loss_weight > 0:
                    #     Calculate contrastive loss for style features
                    style_positive_mask = torch.kron(torch.eye(num_styles, dtype=torch.bool),
                                        torch.ones((num_chars, num_chars), dtype=torch.bool)).to(device)
                    
                    #     Calculate contrastive loss for content features
                    content_target_idx = torch.arange(num_styles * num_chars)
                    content_positive_mask = (content_target_idx.unsqueeze(0) % num_chars == 
                                        content_target_idx.unsqueeze(1) % num_chars).to(device)
                    
                    #     Feature magnitude regularization
                    feature_reg_weight = 0.01
                    #     Calculate feature magnitude regularization
                    style_magnitude_reg = torch.norm(style_features, p=2, dim=1).mean()
                    content_magnitude_reg = torch.norm(content_features, p=2, dim=1).mean()
                    
                    postive_steepness = 1.0
                    postive_scale = 1.0
                    postive_offset = 0.0
                    negative_steepness = 1.0
                    negative_scale = 1.0
                    #     Style contrastive loss
                    def cal_negative_loss(distances, positive_mask):
                        if args.neg_lossfn == 'sigmoid':
                            neg_loss = (1 - torch.sigmoid(negative_steepness * distances[~positive_mask] + 0.0)).mean()
                        elif args.neg_lossfn == 'reciprocal':
                            neg_loss = (1 / (1e-4 + negative_steepness * distances[~positive_mask] + 1.0)).mean()
                        elif args.neg_lossfn == 'sech':
                            neg_loss = (1 / torch.cosh(negative_steepness * distances[~positive_mask] + 0.0)).mean()
                        elif args.neg_lossfn == 'coth':
                            neg_loss = (1 / torch.tanh(negative_steepness * distances[~positive_mask] + 0.549306) - 1).mean()
                        else:
                            raise ValueError(f"Unknown negative loss function: {args.neg_lossfn}")
                        return negative_scale * neg_loss
                    def cal_positive_loss(distances, positive_mask):
                        if args.pos_lossfn == 'square':
                            pos_loss = (postive_steepness * distances[positive_mask] + postive_offset).pow(2).mean()
                        elif args.pos_lossfn == 'exp':
                            pos_loss = torch.exp(postive_steepness * distances[positive_mask] + postive_offset).mean()
                        elif args.pos_lossfn == 'abs':
                            pos_loss = (postive_steepness * distances[positive_mask] + postive_offset).mean()
                        else:
                            raise ValueError(f"Unknown postive loss function: {args.pos_lossfn}")
                        return postive_scale * pos_loss
                    style_distances = torch.cdist(style_features, style_features, p=2).abs()
                    style_pos_loss = cal_positive_loss(style_distances, style_positive_mask)
                    style_neg_loss = cal_negative_loss(style_distances, style_positive_mask)
                    style_loss = style_pos_loss + style_neg_loss + feature_reg_weight * style_magnitude_reg
                    
                    #     Content contrastive loss
                    content_distances = torch.cdist(content_features, content_features, p=2).abs()
                    content_pos_loss = cal_positive_loss(content_distances, content_positive_mask)
                    content_neg_loss = cal_negative_loss(content_distances, content_positive_mask)
                    content_loss = content_pos_loss + content_neg_loss + feature_reg_weight * content_magnitude_reg
                    
                    #     Combined contrastive loss
                    contrastive_loss = style_loss + content_loss
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
                
                # Decode the features to generate images
                if with_style_shuffling:
                    content_features = content_features.reshape(num_styles, num_chars, -1)
                    
                    random_style_orders = np.arange(num_styles)
                    np.random.shuffle(random_style_orders)

                    content_features = content_features[random_style_orders, :, :]
                    content_features = content_features.reshape(-1, content_features.shape[-1])
                    
                if with_content_shuffling:
                    style_features = style_features.reshape(num_styles, num_chars, -1)
                    
                    random_content_orders = np.arange(num_chars)
                    np.random.shuffle(random_content_orders)

                    style_features = style_features[:, random_content_orders, :]
                    style_features = style_features.reshape(-1, style_features.shape[-1])
                
                #     Generate images using the features
                generated_images = font_decoder(style_features, content_features)
                
                # Calculate reconstruction loss
                pixel_loss = torch.nn.functional.l1_loss(generated_images, target_images)
                
                # Combined total loss
                loss = pixel_loss_weight * pixel_loss + contrastive_loss_weight * contrastive_loss
                
                # Evaluate image similarity metrics
                mae, mse, psnr, ssim = eval_images_similarity(generated_images, target_images)

                # Accumulate losses and metrics
                total_loss += loss.item()
                total_pixel_loss += pixel_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                
                total_mae += mae.item()
                total_mse += mse.item()
                total_psnr += psnr.item()
                total_ssim += ssim.item()

                num_batches += 1
            
            # Update progress bar
            tqdm_pbar.set_postfix({
                'l': f"{loss.item():.4f}",
                'pl': f"{pixel_loss.item():.4f}",
                'cl': f"{contrastive_loss.item():.4f}"
            })
            
    # Calculate average losses
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_pixel_loss = total_pixel_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        total_mae, total_mse, total_psnr, total_ssim = \
            total_mae / num_batches, total_mse / num_batches, total_psnr / num_batches, total_ssim / num_batches
    else:
        avg_loss = avg_pixel_loss = avg_contrastive_loss = 0
    
    return avg_loss, avg_pixel_loss, avg_contrastive_loss, total_mae, total_mse, total_psnr, total_ssim

def train(style_encoder, content_encoder, font_decoder, style_discriminator, content_discriminator,
         dataloader_train, dataloader_val,
         num_epochs=10, batch_size=64, start_epoch=0,
         optimizers=None, schedulers=None,
         transform=None, transform_encoder=None, transform_decoder=None,
         loss_logfile=None, generate_samples_dir=None, 
         style_encoder_checkpoint=None, content_encoder_checkpoint=None, font_decoder_checkpoint=None,
         style_discriminator_checkpoint=None, content_discriminator_checkpoint=None,
         checkpoint_period=20):
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        train_l, train_pixel, train_cont, train_adver = train_one_epoch(
            style_encoder, content_encoder, font_decoder, style_discriminator, content_discriminator,
            dataloader_train, optimizers, schedulers, 
            transform, transform_encoder, transform_decoder, 
            batch_size
        )
        
        # Validate
        val_l, val_pixel, val_cont, val_mae, val_mse, val_psnr, val_ssim = validate_one_epoch(
            style_encoder, content_encoder, font_decoder,
            dataloader_val, transform, transform_encoder, transform_decoder, 
            batch_size
        )
        
        # Print training and validation losses
        print(f"Train loss: {train_l:.4f} (Pixel: {train_pixel:.4f}, Contrastive: {train_cont:.4f}, Adversarial: {train_adver:.4f})")
        print(f"Val loss: {val_l:.4f} (Pixel: {val_pixel:.4f}, Contrastive: {val_cont:.4f})")
        
        # Log metrics
        if loss_logfile is not None:
            if epoch == 0:
                with open(loss_logfile, 'w') as f:
                    f.write("epoch,train_loss,train_pixel,train_contrastive,train_adversarial,val_loss,val_pixel,val_contrastive,val_mae,val_mse,val_psnr,val_ssim\n")
            with open(loss_logfile, 'a') as f:
                f.write(f"{epoch + 1},{train_l},{train_pixel},{train_cont},{train_adver},{val_l},{val_pixel},{val_cont},{val_mae},{val_mse},{val_psnr},{val_ssim}\n")

        # Save samples of generated images
        if generate_samples_dir and generate_samples_dir != '':
            
            # Create directory for samples if it doesn't exist
            os.makedirs(generate_samples_dir, exist_ok=True)
            
            #
            style_indices   = (np.array([0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7], dtype=np.long),
                               np.array([0,1,2,3,4,5,6,7, 1,2,3,4,5,6,7,0], dtype=np.long))
            content_indices = (np.array([0,1,2,3,4,5,6,7, 2,3,4,5,6,7,0,1], dtype=np.long),
                               np.array([0,1,2,3,4,5,6,7, 3,4,5,6,7,0,1,2], dtype=np.long))
                
            max_style_idx = 8
            
            # Get validation data
            val_images = None
            for val_batch in dataloader_val:
                if val_images is None:
                    val_images = val_batch["image"]
                else:
                    val_images = torch.cat((val_images, val_batch["image"]), dim=0)
                if val_images.shape[0] > max_style_idx + 1:
                    break
                
            sample_style_images = val_images[style_indices[0], style_indices[1], :, :]
            sample_content_images = val_images[content_indices[0], content_indices[1], :, :]
            sample_target_images = val_images[style_indices[0], content_indices[1], :, :]

            # Generate and save visualization
            save_path = os.path.join(generate_samples_dir, f"imagesamples_epoch_{epoch+1:03d}.png")
            log_generate_images(
                style_encoder, 
                content_encoder, 
                font_decoder,
                sample_style_images, 
                sample_content_images, 
                sample_target_images,
                save_path
            )
            print(f"Generated sample images saved to {save_path}")

        # Save checkpoints
        if ((epoch + 1) % checkpoint_period == 0):
            if style_encoder_checkpoint is not None and style_encoder_checkpoint != '':
                save_checkpoint(style_encoder, optimizers['style_encoder'], schedulers['style_encoder'], 
                                epoch + 1, style_encoder_checkpoint)
            if content_encoder_checkpoint is not None and content_encoder_checkpoint != '':
                save_checkpoint(content_encoder, optimizers['content_encoder'], schedulers['content_encoder'],
                                epoch + 1, content_encoder_checkpoint)
            if font_decoder_checkpoint is not None and font_decoder_checkpoint != '':
                save_checkpoint(font_decoder, optimizers['font_decoder'], schedulers['font_decoder'],
                                epoch + 1, font_decoder_checkpoint)
            if adversarial_loss_weight > 0.0:
                if style_discriminator_checkpoint is not None and style_discriminator_checkpoint != '':
                    save_checkpoint(style_discriminator, optimizers['style_discriminator'], schedulers['style_discriminator'],
                                    epoch + 1, style_discriminator_checkpoint)
                if content_discriminator_checkpoint is not None and content_discriminator_checkpoint != '':
                    save_checkpoint(content_discriminator, optimizers['content_discriminator'], schedulers['content_discriminator'],
                                    epoch + 1, content_discriminator_checkpoint)
                    
        

# Main training process
print("Starting training process...")
# Get the os type
dataLoader_train = DataLoader(dataset_train, batch_size=num_styles_per_batch, shuffle=True, num_workers=0 if os.name == 'nt' else 8)
dataLoader_val = DataLoader(dataset_val, batch_size=num_styles_per_batch, shuffle=False, num_workers=0 if os.name == 'nt' else 8)

train(style_encoder, content_encoder, font_decoder, style_discriminator, content_discriminator,
     dataLoader_train, dataLoader_val,
     num_epochs=training_epochs, batch_size=batch_size, start_epoch=start_epoch,
     optimizers=optimizers, schedulers=schedulers,
     transform=transform_augmente, transform_encoder=transform_encoder, transform_decoder=transform_decoder,
     style_encoder_checkpoint=style_encoder_checkpoint, content_encoder_checkpoint=content_encoder_checkpoint, font_decoder_checkpoint=font_decoder_checkpoint,
     style_discriminator_checkpoint=style_discriminator_checkpoint, content_discriminator_checkpoint=content_discriminator_checkpoint,
     loss_logfile=args.loss_logfile, generate_samples_dir=args.generate_samples_dir, checkpoint_period=checkpoint_period)

# Save the final models
save_model(style_encoder, args.output_style_encoder)
save_model(content_encoder, args.output_content_encoder)
save_model(font_decoder, args.output_decoder)
if adversarial_loss_weight > 0.0:
    save_model(style_discriminator, args.output_style_discriminator)
    save_model(content_discriminator, args.output_content_discriminator)