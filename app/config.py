"""Central configuration. Everything tunable lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# --- Models ---
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-12-v2"
FINETUNED_CE_DIR = MODELS_DIR / "cross_encoder_finetuned"

# --- Retrieval ---
EMBED_DIM = 384
TOP_K_RETRIEVE = 40   # first-stage FAISS candidates
DEFAULT_TOP_N = 10    # default shortlist size

# --- Scoring weights (used when the learned scorer is absent) ---
DEFAULT_WEIGHTS = {
    "semantic": 0.45,
    "skills":   0.30,
    "exp":      0.15,
    "edu":      0.10,
}

# --- Files ---
DATASET_CSV = DATA_DIR / "Resume.csv"
FAISS_INDEX = MODELS_DIR / "resumes.faiss"
FAISS_META  = MODELS_DIR / "resumes.meta.parquet"
SCORER_WEIGHTS = MODELS_DIR / "scorer_weights.json"

# --- Upload limits ---
# 2 GB total across an upload batch. A typical resume PDF is ~150 KB,
# so this comfortably fits ~10,000 resumes, or a large ZIP dump.
MAX_UPLOAD_MB = 2048
# Server-side hard cap on how many resumes we will parse per request.
# The scorer keeps everything in memory during ranking; ~10k stays sane.
MAX_RESUMES_PER_REQUEST = 10000
