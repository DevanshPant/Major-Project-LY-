"""Fine-tune the ms-marco cross-encoder on synthetic JD–resume pairs.

We don't have real labels, so we manufacture them:
  positive pair =  (category-JD template,  resume from that category)
  negative pair =  (category-JD template,  resume from a different category)

This is a legitimate weak-supervision setup — it teaches the model that
"good matches share topical semantics", which is exactly the job.

Run:
    python -m scripts.train_reranker  --epochs 1 --limit 4000
Output:
    models/cross_encoder_finetuned/     (loaded automatically by reranker.py)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CROSS_ENCODER, FINETUNED_CE_DIR, DATASET_CSV


# Short JD-like prompt per category, keyed by NORMALISED name (upper-case,
# words separated by spaces — matches ingest_pdfs.normalise_category).
# Unknown categories fall back to a generic role template.
CATEGORY_JD = {
    "INFORMATION TECHNOLOGY": "IT engineer with strong programming and systems background.",
    "HR": "HR professional handling recruitment, employee relations, and policy.",
    "DESIGNER": "Visual designer with a strong portfolio in branding and UX.",
    "DESIGN": "Visual designer with a strong portfolio in branding and UX.",
    "TEACHER": "Teacher with lesson planning, classroom management, and subject expertise.",
    "ADVOCATE": "Legal advocate with litigation and case-preparation experience.",
    "BUSINESS DEVELOPMENT": "Business development with sales pipeline and client acquisition.",
    "HEALTHCARE": "Healthcare professional with patient care and clinical experience.",
    "FITNESS": "Certified fitness trainer with client programming experience.",
    "AGRICULTURE": "Agriculture specialist with crop management background.",
    "BPO": "BPO agent with customer service and process handling experience.",
    "SALES": "Sales representative with quota attainment and territory management.",
    "CONSULTANT": "Consultant with client engagement and strategy background.",
    "DIGITAL MEDIA": "Digital media professional with content and analytics experience.",
    "AUTOMOBILE": "Automobile engineer with vehicle systems background.",
    "CHEF": "Chef with kitchen operations and menu design experience.",
    "FINANCE": "Finance analyst with modeling, forecasting, and reporting experience.",
    "APPAREL": "Apparel professional with merchandising and design background.",
    "ENGINEERING": "Engineer with strong technical and analytical foundation.",
    "ACCOUNTANT": "Accountant with bookkeeping, audit, and tax experience.",
    "CONSTRUCTION": "Construction professional with site management experience.",
    "PUBLIC RELATIONS": "PR professional with media relations and campaign experience.",
    "BANKING": "Banking professional with retail or corporate banking experience.",
    "ARTS": "Arts professional with creative production experience.",
    "AVIATION": "Aviation professional with airline or airport operations experience.",
    "DATA SCIENCE": "Data scientist with machine learning, statistics, and Python experience.",
    "DATA ENGINEER": "Data engineer with ETL, Spark, and warehousing experience.",
    "DEVOPS ENGINEER": "DevOps engineer with CI/CD, Docker, Kubernetes and cloud experience.",
    "BUSINESS ANALYST": "Business analyst with requirements, reporting and SQL experience.",
    "BLOCKCHAIN": "Blockchain developer with smart contract and Solidity experience.",
    "CIVIL ENGINEER": "Civil engineer with structural design and project experience.",
    "MECHANICAL ENGINEER": "Mechanical engineer with design and manufacturing experience.",
    "ELECTRICAL ENGINEER": "Electrical engineer with power or electronics experience.",
    "PYTHON DEVELOPER": "Python developer with backend, scripting and API experience.",
    "REACT DEVELOPER": "React developer with modern JavaScript frontend experience.",
    "SAP DEVELOPER": "SAP developer with ABAP and module configuration experience.",
    "SQL DEVELOPER": "SQL developer with relational database and query experience.",
    "WEB DESIGN": "Web designer with HTML, CSS and UX portfolio.",
    "NETWORK SECURITY ENGINEER": "Network security engineer with firewall and threat experience.",
    "OPERATIONS MANAGER": "Operations manager with process and team management experience.",
    "PROJECT MANAGEMENT": "Project manager with delivery, budget and stakeholder experience.",
    "TESTING": "QA / testing professional with test planning and automation experience.",
    "BIG DATA": "Big data engineer with Hadoop, Spark and data platform experience.",
    "DATABASE": "Database administrator with backup, tuning and admin experience.",
    "ARCHITECT": "Architect with design and construction documentation experience.",
}


def _load_dataset(limit: int | None):
    from app.config import DATA_DIR
    parq = DATA_DIR / "corpus.parquet"
    if parq.exists():
        df = pd.read_parquet(parq)[["text", "category"]].dropna()
        print(f"[train] loaded corpus.parquet ({len(df):,} docs)")
    elif DATASET_CSV.exists():
        df = pd.read_csv(DATASET_CSV)
        text_col = "Resume_str" if "Resume_str" in df.columns else df.columns[0]
        cat_col = "Category" if "Category" in df.columns else None
        df = df.rename(columns={text_col: "text"})
        if cat_col:
            df = df.rename(columns={cat_col: "category"})
        else:
            df["category"] = "unknown"
        df = df[["text", "category"]].dropna()
        print(f"[train] loaded {DATASET_CSV.name} ({len(df):,} docs)")
    else:
        print("No corpus found. Run scripts.ingest_pdfs or scripts.download_data first.")
        sys.exit(1)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 150].reset_index(drop=True)
    if limit:
        df = df.sample(min(len(df), limit), random_state=42).reset_index(drop=True)
    return df


def _build_pairs(df, negatives_per_positive: int = 1):
    from sentence_transformers import InputExample
    pairs: list[InputExample] = []
    cats = df["category"].unique().tolist()
    for _, row in df.iterrows():
        cat = row["category"]
        jd = CATEGORY_JD.get(cat) or f"Professional in {cat.lower().replace('-', ' ')}."
        pairs.append(InputExample(texts=[jd, row["text"][:2000]], label=1.0))
        for _ in range(negatives_per_positive):
            neg_cat = random.choice([c for c in cats if c != cat])
            neg_row = df[df["category"] == neg_cat].sample(1).iloc[0]
            neg_jd = CATEGORY_JD.get(neg_cat) or f"Professional in {neg_cat.lower().replace('-', ' ')}."
            pairs.append(InputExample(texts=[jd, neg_row["text"][:2000]], label=0.0))
    random.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=2000,
                    help="cap resumes used to keep training realistic on a laptop")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    print("[train] loading data")
    df = _load_dataset(args.limit)
    print(f"[train] {len(df)} resumes across {df['category'].nunique()} categories")
    pairs = _build_pairs(df)
    print(f"[train] {len(pairs)} training pairs")

    print("[train] loading base cross-encoder — downloads on first run")
    model = CrossEncoder(CROSS_ENCODER, num_labels=1)
    # num_workers=0 and pin_memory=False keep the CPU-only run inside a
    # modest memory budget; the default worker processes each copy the
    # dataset and exhausted the page file on a 16 GB machine.
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    print("[train] fine-tuning …")
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=100,
        show_progress_bar=True,
    )

    FINETUNED_CE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINETUNED_CE_DIR))
    print(f"[train] saved -> {FINETUNED_CE_DIR}")


if __name__ == "__main__":
    main()
