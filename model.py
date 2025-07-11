import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with shortcut connection."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, drop_rate=0.1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    
class DownResidualBlock(nn.Module):
    """Downsampling block with residual connection."""
    def __init__(self, in_channels, out_channels, drop_rate=0.1):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
            nn.Dropout2d(drop_rate)
        )
        
        self.res_block = ResidualBlock(out_channels, out_channels, drop_rate=drop_rate)
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.res_block(x)
        return x

class UpResidualBlock(nn.Module):
    """Upsampling block with residual connection."""
    def __init__(self, in_channels, out_channels, drop_rate=0.1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
            nn.Dropout2d(drop_rate)
        )
        
        self.res_block = ResidualBlock(out_channels, out_channels, drop_rate=drop_rate)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.res_block(x)
        return x
   
class ResFiLMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, use_attn=False):
        super().__init__()
        # Fix: Use min(in_ch, 8) as num_groups to ensure divisibility
        num_groups_in = max(1, min(in_ch, 8))  # Ensure at least 1 group
        num_groups_out = max(1, min(out_ch, 8))  # Ensure at least 1 group
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.norm1 = nn.GroupNorm(num_groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(num_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.film = nn.Linear(cond_dim, 2*out_ch)
        self.use_attn = use_attn
        if use_attn:
            self.attn = nn.MultiheadAttention(out_ch, 4, batch_first=True)
        self.skip = (in_ch != out_ch)
        if self.skip:
            self.conv_skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, cond_vec):
        # Apply first normalization
        h = self.norm1(x)
        h = F.silu(h)
        
        # Get FiLM parameters and apply conditioning
        gamma_beta = self.film(cond_vec)  # (B, 2*out_ch)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each is (B, out_ch)
        
        # Make sure gamma and beta have the right shape for broadcasting
        gamma = gamma.view(-1, self.out_ch, 1, 1)
        beta = beta.view(-1, self.out_ch, 1, 1)
        
        # Apply first convolution, then FiLM conditioning
        h = self.conv1(h)  # First apply convolution to change channels
        h = h * gamma + beta  # Then apply FiLM conditioning
        
        # Apply second normalization and convolution
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        # Apply attention if needed
        if self.use_attn:
            B, C, H, W = h.shape
            h_attn = h.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
            h_attn, _ = self.attn(h_attn, h_attn, h_attn)
            h = h_attn.transpose(1, 2).view(B, C, H, W)
        
        # Apply skip connection
        if self.skip:
            x = self.conv_skip(x)
        
        return x + h
    
class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=64, patch_size=16, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim (B, 1, H, W)
            
        # (B, C, H, W) -> (B, embed_dim, grid, grid)
        x = self.proj(x)
        # (B, embed_dim, grid, grid) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x
class StyleEncoder(nn.Module):
    """ViT-based encoder for font style."""
    def __init__(self, img_size=64, patch_size=16, in_channels=1, output_dim=128, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # VAE specific: produce both mean and log variance
        self.mu_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        self.logvar_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        # Get patch embeddings
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use class token for final representation
        x = x[:, 0]  # (B, embed_dim)
        
        # Project to mean and logvar
        mu = self.mu_proj(x)  # (B, output_dim)
        logvar = self.logvar_proj(x)  # (B, output_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar).to(self.get_device())
        eps = torch.randn_like(std).to(self.get_device())
        return mu + eps * std
    
    def encode(self, x):
        """Encode input and sample from the latent distribution."""
        mu, logvar = self.forward(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    def get_device(self):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = "cpu"
        return device
        
class ContentEncoder(nn.Module):
    """ViT-based encoder for character content."""
    def __init__(self, img_size=64, patch_size=16, in_channels=1, output_dim=128, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # VAE specific: produce both mean and log variance
        self.mu_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        self.logvar_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        # Get patch embeddings
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use class token for final representation
        x = x[:, 0]  # (B, embed_dim)
        
        # Project to mean and logvar
        mu = self.mu_proj(x)  # (B, output_dim)
        logvar = self.logvar_proj(x)  # (B, output_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar).to(self.get_device())
        eps = torch.randn_like(std).to(self.get_device())
        return mu + eps * std
    
    def encode(self, x):
        """Encode input and sample from the latent distribution."""
        mu, logvar = self.forward(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    def get_device(self):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = "cpu"
        return device

class FontDecoder(nn.Module):
    """Decoder with ResFiLM conditioning for font generation from style and content features."""
    def __init__(self, img_size=64, style_embed_dim=128, content_embed_dim=128, drop_rate=0.1):
        super().__init__()
        self.img_size = img_size
        
        # Initial spatial size for starting the upsampling process
        initial_size = 4  # Start with 4x4 spatial dimension
        
        # Combined embedding dimension for style and content
        cond_dim = style_embed_dim + content_embed_dim
        
        # Combine and reshape features to initial spatial feature map
        self.initial_linear = nn.Linear(cond_dim, initial_size * initial_size * 512)
        
        # Conditioning network to process the combined features
        self.cond_net_init = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.cond_net1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU()
        )
        self.cond_net2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU()
        )
        self.cond_net3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU()
        )
        
        # Initial convolutional processing
        self.initial_res_block = ResidualBlock(512, 512, drop_rate=drop_rate)
        self.initial_res = ResFiLMBlock(512, 512, cond_dim=512)
        
        # Upsampling path with ResidualBlock before ResFiLM blocks
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.res1_block = ResidualBlock(256, 256, drop_rate=drop_rate)
        self.res1 = ResFiLMBlock(256, 256, cond_dim=512, use_attn=True)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.res2_block = ResidualBlock(128, 128, drop_rate=drop_rate)
        self.res2 = ResFiLMBlock(128, 128, cond_dim=512, use_attn=True)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.res3_block = ResidualBlock(64, 64, drop_rate=drop_rate)
        self.res3 = ResFiLMBlock(64, 64, cond_dim=512, use_attn=True)
        
        # Final layers to generate the output image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, style_feat, content_feat):
        B = style_feat.shape[0]
        
        # Combine style and content features
        combined = torch.cat([style_feat, content_feat], dim=1)  # B, style_dim + content_dim
        
        # Process combined features through conditioning network
        cond_vec = self.cond_net_init(combined)  # B, 512
        
        # Initial spatial feature map
        x = self.initial_linear(combined)  # B, initial_size * initial_size * 512
        x = x.view(B, 512, 4, 4)  # Reshape to B, C, H, W format
        
        # Apply initial ResidualBlock followed by ResFiLM block with conditioning
        x = self.initial_res_block(x)
        x = self.initial_res(x, cond_vec)  # 4x4x512
        
        # First upsampling block with ResidualBlock and conditioning
        cond_vec = self.cond_net1(cond_vec) + cond_vec
        x = self.up1(x)  # 8x8x256
        x = self.res1(x, cond_vec)
        x = self.res1_block(x)
        
        # Second upsampling block with ResidualBlock and conditioning
        cond_vec = self.cond_net2(cond_vec) + cond_vec
        x = self.up2(x)  # 16x16x128
        x = self.res2(x, cond_vec)
        x = self.res2_block(x)
        
        # Third upsampling block with ResidualBlock and conditioning
        cond_vec = self.cond_net3(cond_vec) + cond_vec
        x = self.up3(x)  # 32x32x64
        x = self.res3(x, cond_vec)
        x = self.res3_block(x)
        
        # Final processing to reach target size
        x = self.final(x)  # 64x64x1
        
        # Return grayscale image (remove channel dimension if needed)
        return x.squeeze(1)  # B, H, W (grayscale output)

class StyleDiscriminator(nn.Module):
    """Discriminator for style images. Determines if the pair of images are from the same style."""
    def __init__(self, in_channels=1, drop_rate=0.1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.GELU()
        )
        
        # Downsampling path with residual blocks
        self.down1 = DownResidualBlock(32, 64, drop_rate=drop_rate)      # 32x32x64
        self.down2 = DownResidualBlock(64, 128, drop_rate=drop_rate)     # 16x16x128
        self.down3 = DownResidualBlock(128, 256, drop_rate=drop_rate)    # 8x8x256
        self.down4 = DownResidualBlock(256, 512, drop_rate=drop_rate)    # 4x4x512
        
        # Additional processing
        self.res_block = ResidualBlock(512, 512, drop_rate=drop_rate)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature combination and classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),  # Combined features from both images
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1)
        )
    
    def extract_features(self, x):
        """Extract features from a single image."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim (B, 1, H, W)
        
        x = self.conv1(x)     # 64x64x32
        x = self.down1(x)     # 32x32x64
        x = self.down2(x)     # 16x16x128
        x = self.down3(x)     # 8x8x256
        x = self.down4(x)     # 4x4x512
        x = self.res_block(x) # 4x4x512
        x = self.global_pool(x)  # B, 512, 1, 1
        x = x.flatten(1)      # B, 512
        return x
    
    def forward(self, image1, image2):
        # Extract features from both images
        feat1 = self.extract_features(image1)  # B, 512
        feat2 = self.extract_features(image2)  # B, 512
        
        # Combine features
        combined = torch.cat([feat1, feat2], dim=1)  # B, 1024
        
        # Classify if same style
        prob = self.classifier(combined)  # B, 1
        return prob.squeeze(-1)  # B

class ContentDiscriminator(nn.Module):
    """Discriminator for content images. Determines if the pair of images are from the same content."""
    def __init__(self, in_channels=1, drop_rate=0.1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.GELU()
        )
        
        # Downsampling path with residual blocks
        self.down1 = DownResidualBlock(32, 64, drop_rate=drop_rate)      # 32x32x64
        self.down2 = DownResidualBlock(64, 128, drop_rate=drop_rate)     # 16x16x128
        self.down3 = DownResidualBlock(128, 256, drop_rate=drop_rate)    # 8x8x256
        self.down4 = DownResidualBlock(256, 512, drop_rate=drop_rate)    # 4x4x512
        
        # Additional processing
        self.res_block = ResidualBlock(512, 512, drop_rate=drop_rate)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature combination and classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),  # Combined features from both images
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1)
        )
    
    def extract_features(self, x):
        """Extract features from a single image."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim (B, 1, H, W)
        
        x = self.conv1(x)     # 64x64x32
        x = self.down1(x)     # 32x32x64
        x = self.down2(x)     # 16x16x128
        x = self.down3(x)     # 8x8x256
        x = self.down4(x)     # 4x4x512
        x = self.res_block(x) # 4x4x512
        x = self.global_pool(x)  # B, 512, 1, 1
        x = x.flatten(1)      # B, 512
        return x
    
    def forward(self, image1, image2):
        # Extract features from both images
        feat1 = self.extract_features(image1)  # B, 512
        feat2 = self.extract_features(image2)  # B, 512
        
        # Combine features
        combined = torch.cat([feat1, feat2], dim=1)  # B, 1024
        
        # Classify if same content
        prob = self.classifier(combined)  # B, 1
        return prob.squeeze(-1)  # B
