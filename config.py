"""Central settings for the experiment. Every number reported in the paper
should come from this file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Model (change after running speed_test.py) ----
MODEL = "qwen3-vl:4b-instruct"     # the model installed on this laptop; if too slow, pull a smaller one
SEED = 42
TEMPERATURE = 0
MAX_NEW_TOKENS = 20        # answers are one word or a number
THINK = None           # thinking OFF, avoids empty answers

# ---- Data ----
N_IMAGES = 40              # number of 512x512 tiles in the sample
RANDOM_SEED = 42
TILE_SIZE = 512
MAX_PER_IMAGE_PER_BUCKET = 2   # limits tiles taken from one source image

# ---- Degradation levels (state these in the paper) ----
NOISE_SIGMA = 25           # Gaussian noise std (0-255 scale)
BLUR_KSIZE = 7             # Gaussian blur kernel size (odd)
BLUR_SIGMA = 2.0
CONTRAST_ALPHA = 0.5       # low-contrast: pixel = alpha*pixel + beta
CONTRAST_BETA = 40

# ---- Experimental conditions (order is used in tables and figures) ----
CONDITIONS = [
    "original",
    "noisy", "noisy_denoised",
    "blurred", "blurred_sharpened",
    "lowcon", "lowcon_enhanced",
]
# Optional: apply each fix to the CLEAN image (shows whether fixes hurt good images)
INCLUDE_FIXES_ON_CLEAN = False
CLEAN_FIX_CONDITIONS = ["clean_denoised", "clean_sharpened", "clean_enhanced"]

# ---- Paths ----
RAW_DIR = ROOT / "data" / "raw"
WORK_DIR = ROOT / "work"
RESULTS_DIR = ROOT / "results"
