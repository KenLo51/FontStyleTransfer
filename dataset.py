from pathlib import Path
from typing   import Tuple, Union, Literal, Dict
import concurrent.futures
import os

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import string

import tqdm


def _get_bbox(char: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
    """
    Bounding box helper. Always use `textbbox`, which includes ascenders/descenders.
    Returns (left, top, right, bottom) – *may* contain negative top values.
    """
    # Pillow ≥10 requires anchor; 'lt' = left-top
    dummy_img = Image.new("L", (4, 4))
    draw      = ImageDraw.Draw(dummy_img)
    return draw.textbbox((0, 0), char, font=font)

def render_character_to_array(
        char: str,
        font_path: Union[str, Path],
        image_size: Tuple[int, int] = (64, 64),
        initial_pt: int = 256,
        out: Literal["numpy", "torch"] = "numpy",
        pad_ratio: float = 0.95,              # 1.0=touch edges, 0.9=leave margin
) -> Union[np.ndarray, torch.Tensor]:
    """
    Render a single glyph so it *fills* but never exceeds `image_size`.

    ── Parameters ───────────────────────────────────────────────────────────
    char        : Glyph to render (typically one Unicode code-point).
    font_path   : Path to the TTF/OTF file.
    image_size  : (W, H) of output canvas.
    initial_pt  : Starting trial font size (big – adjusted downward).
    out         : "numpy" → np.uint8 [0,255]; "torch" → torch.float32 in [0,1].
    pad_ratio   : 0.95 ⇒ 5 % margin on each axis after fitting.

    ── Returns ──────────────────────────────────────────────────────────────
    np.ndarray shape (H, W)  or  torch.Tensor shape (1, H, W)
    """
    w_img, h_img     = image_size
    font_size        = initial_pt
    font             = ImageFont.truetype(str(font_path), font_size)

    # ---- Scale down until the glyph fits -------------------------------
    while True:
        left, top, right, bottom = _get_bbox(char, font)
        glyph_w, glyph_h         = right - left, bottom - top

        # How much room do we have, factoring pad_ratio?
        max_w, max_h             = pad_ratio * w_img, pad_ratio * h_img
        if glyph_w <= max_w and glyph_h <= max_h:
            break  # fits!

        # Shrink proportionally and re-create font
        shrink = min(max_w / glyph_w, max_h / glyph_h)
        font_size = max(1, int(font_size * shrink))
        font = ImageFont.truetype(str(font_path), font_size)

    # ---- Create final canvas & center glyph ----------------------------
    canvas = Image.new("L", image_size, color=255)
    draw   = ImageDraw.Draw(canvas)

    # Recompute bbox with the *final* font size
    left, top, right, bottom = _get_bbox(char, font)
    glyph_w, glyph_h         = right - left, bottom - top

    # Because top may be negative, shift baseline by −top
    shift_x = (w_img - glyph_w) // 2 - left
    shift_y = (h_img - glyph_h) // 2 - top

    draw.text((shift_x, shift_y), char, fill=0, font=font)

    # ---- Format output --------------------------------------------------
    arr = np.asarray(canvas, dtype=np.uint8)
    if out == "torch":
        arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) / 255.0
    return arr

# Define this function at module level for multiprocessing
def _process_single_font(args):
    font_idx, font_path, chars, image_size, pad_ratio = args
    font_images = np.zeros((len(chars), image_size[1], image_size[0]), dtype=np.uint8)
    
    for char_idx, char in enumerate(chars):
        try:
            # Render character as numpy array (uint8)
            img_array = render_character_to_array(
                char=char,
                font_path=font_path,
                image_size=image_size,
                out="numpy",
                pad_ratio=pad_ratio
            )
            
            # Store in numpy array
            font_images[char_idx] = img_array
        except Exception as e:
            print(f"Warning: Failed to render char '{char}' for font {Path(font_path).name}: {e}")
            # Use white image (255) as fallback
            font_images[char_idx] = np.full(image_size, 255, dtype=np.uint8)
            break
    
    return font_idx, font_images

class FontDataset(Dataset):
    """Dataset for font generation that provides character renderings from multiple fonts."""
    
    def __init__(
        self, 
        font_dir: Union[str, Path],
        image_size: Tuple[int, int] = (64, 64),
        chars: str = string.ascii_letters,  # a-zA-Z
        pad_ratio: float = 0.95,
    ):
        """
        Initialize the font dataset.
        
        Parameters:
        -----------
        font_dir : str or Path
            Directory containing TTF/OTF font files
        image_size : tuple (W, H)
            Size of the output images
        chars : str
            Characters to include in the dataset (default: a-zA-Z)
        pad_ratio : float
            Padding ratio for rendering
        """
        self.font_dir = Path(font_dir)
        self.image_size = image_size
        self.chars = chars
        self.pad_ratio = pad_ratio
        
        # Find all font files
        self.font_paths = sorted(
            list(self.font_dir.glob("**/*.ttf")) + 
            list(self.font_dir.glob("**/*.otf"))
        )
        
        if not self.font_paths:
            raise ValueError(f"No font files found in {font_dir}")
            
        # Create character-to-index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        
        self.num_fonts = len(self.font_paths)
        self.num_chars = len(self.chars)
        
        # Preload all rendered characters
        self._preload_all_fonts()
        
    def _preload_all_fonts(self):
        """Preload all character renderings for all fonts using parallel processing."""
        print(f"Preloading {self.num_fonts} fonts with {self.num_chars} characters each...")
        
        h, w = self.image_size
        # Create storage tensor [num_fonts, num_chars, height, width]
        self.images = torch.zeros(
            (self.num_fonts, self.num_chars, h, w), 
            dtype=torch.uint8
        )
        
        # Prepare arguments for each font
        font_args = [
            (idx, str(self.font_paths[idx]), self.chars, self.image_size, self.pad_ratio) 
            for idx in range(self.num_fonts)
        ]
        
        # Use ProcessPoolExecutor for true parallelism across CPU cores
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Process fonts in parallel with progress bar
            for font_idx, font_images in tqdm.tqdm(
                executor.map(_process_single_font, font_args),
                total=self.num_fonts,
                desc="Preloading fonts"
            ):
                # Convert numpy array to torch tensor
                self.images[font_idx] = torch.from_numpy(font_images)
        
        print(f"Preloading complete. Storage shape: {self.images.shape}, "
              f"Memory: {self.images.element_size() * self.images.nelement() / (1024*1024):.2f}MB")
        
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.num_fonts
    
    def _index_to_char_font(self, idx: int) -> Tuple[str, Path]:
        """Convert a flat index to (character, font_path) pair."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")
            
        char_idx = idx % self.num_chars
        font_idx = idx // self.num_chars
        
        return self.chars[char_idx], self.font_paths[font_idx]
    
    def _char_font_to_index(self, char: str, font_idx: int) -> int:
        """Convert a (character, font_idx) pair to flat index."""
        if char not in self.char_to_idx:
            raise ValueError(f"Character '{char}' not in character set")
        if font_idx < 0 or font_idx >= self.num_fonts:
            raise IndexError(f"Font index {font_idx} out of range [0, {self.num_fonts-1}]")
            
        char_idx = self.char_to_idx[char]
        return font_idx * self.num_chars + char_idx
    
    def __getitem__(self, idx: int) -> Dict:
        return self.get_by_font(idx)
    
    def get_by_char_font(self, char: str, font_idx: int) -> Dict:
        """Get a specific character from a specific font."""
        return self[font_idx]['image'][self.char_to_idx[char]]
    
    def get_by_font(self, font_idx: int) -> Dict:
        """Get all characters from a specific font."""
        if font_idx < 0 or font_idx >= self.num_fonts:
            raise IndexError(f"Font index {font_idx} out of range [0, {self.num_fonts-1}]")
        
        samples = {
                "image": np.empty((self.num_chars, *self.image_size), dtype=np.float32),
                "char": self.chars,
                "font_idx": font_idx,
                "font_path": str(self.font_paths[font_idx])
            }
        for char_idx in range(self.num_chars):
            img = self.images[font_idx, char_idx].float().unsqueeze(0) / 255.0
            samples["image"][char_idx] = img.numpy().squeeze()
        return samples
    
    def get_font_name(self, font_idx: int) -> str:
        """Get the name of a font by its index."""
        if font_idx < 0 or font_idx >= self.num_fonts:
            raise IndexError(f"Font index {font_idx} out of range [0, {self.num_fonts-1}]")
        return self.font_paths[font_idx].stem

class CharDataset(Dataset):
    """Dataset for rendering characters."""
    
    def __init__(self, font_dir: Union[str, Path], chars: str = string.ascii_letters, image_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the character dataset.
        
        Parameters:
        -----------
        font_dir : str or Path
            Path to the TTF/OTF font file
        chars : str
            Characters to render (default: a-zA-Z)
        image_size : tuple (W, H)
            Size of the output images
        """
        self.font_dir = Path(font_dir)
        self.chars = chars
        self.image_size = image_size
        
        self.fontDataset = FontDataset(
            font_dir=self.font_dir,
            image_size=image_size,
            chars=chars
        )
        
    def __len__(self) -> int:
        return self.fontDataset.num_chars * self.fontDataset.num_fonts
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.fontDataset[idx // self.fontDataset.num_chars]
        return {"char": item["char"][idx % self.fontDataset.num_chars],
                "image": item["image"][idx % self.fontDataset.num_chars],
                "font_idx": item["font_idx"],
                "font_path": item["font_path"]}


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fp   = "dafonts-free-v1_letters/dafonts-free-v1/val/abasterrules.ttf"
    char = "A"

    img_np = render_character_to_array(char, fp, image_size=(64, 64), out="numpy")
    img_t  = render_character_to_array(char, fp, image_size=(64, 64), out="torch")

    print("NumPy:", img_np.shape, img_np.dtype)
    print("Torch:", img_t.shape,  img_t.dtype, img_t.min().item(), img_t.max().item())

    # Visual sanity-check
    import matplotlib.pyplot as plt
    plt.imshow(img_np, cmap="gray")
    plt.title(f"Glyph '{char}' from {Path(fp).name}")
    plt.show()
    exit()
    
    # Test the FontDataset
    print("\n--- FontDataset Example ---")
    font_dir = "dafonts-free-v1/dafonts-free-v1/fonts/"
    dataset = FontDataset(font_dir)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of fonts: {dataset.num_fonts}")
    print(f"Characters: {dataset.chars[:10]}... (total: {len(dataset.chars)})")
    
    # Get a random sample
    sample_idx = 42
    sample = dataset[sample_idx]
    print(f"Sample info: char='{sample['char']}', font={sample['font_idx']}")
    
    # Display the image
    plt.figure()
    plt.imshow(sample['image'].squeeze(), cmap="gray")
    plt.title(f"Char '{sample['char']}' from {Path(sample['font_path']).name}")
    plt.show()