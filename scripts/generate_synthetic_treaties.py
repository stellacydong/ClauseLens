import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# Config
# -----------------------------
RAW_DIR = "../data/raw"
DEMO_DIR = "../data/demo"

N_REINSURERS = 30
N_CEDENTS = 20
N_TREATIES = 100_000
N_DEMO = 1_000

REGIONS = ["US", "EU", "APAC", "LATAM"]
LINES_OF_BUSINESS = ["Property", "Casualty", "Specialty"]
TREATY_TYPES = ["XoL", "Quota"]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# -----------------------------
# 1. Generate Reinsurer & Cedent Metadata
# -----------------------------
def generate_reinsurer_info(n_reinsurers=30, n_cedents=20):
    rows = []

    # Reinsurers
    for i in range(1, n_reinsurers + 1):
        rows.append({
            "entity_id": f"R{i:03d}",
            "entity_type": "reinsurer",
            "entity_name": f"Reinsurer_{i:03d}",
            "region": random.choice(REGIONS),
            "incumbent_flag": int(random.random() < 0.6),  # 60% incumbents
            "rating": random.choice(["AA", "A", "BBB"])
        })

    # Cedents
    for i in range(1, n_cedents + 1):
        rows.append({
            "entity_id": f"C{i:03d}",
            "entity_type": "cedent",
            "entity_name": f"Cedent_{i:03d}",
            "region": random.choice(REGIONS),
            "incumbent_flag": "",
            "rating": ""
        })

    df = pd.DataFrame(rows)
    return df

reinsurer_df = generate_reinsurer_info(N_REINSURERS, N_CEDENTS)
reinsurer_path = os.path.join(RAW_DIR, "reinsurer_info.csv")
reinsurer_df.to_csv(reinsurer_path, index=False)
print(f"✅ Saved {reinsurer_path} ({len(reinsurer_df)} rows)")

# -----------------------------
# 2. Generate Synthetic Treaties
# -----------------------------
def generate_treaties(n_treaties=100_000):
    treaties = []

    reinsurers = reinsurer_df[reinsurer_df["entity_type"] == "reinsurer"]["entity_id"].tolist()
    cedents = reinsurer_df[reinsurer_df["entity_type"] == "cedent"]["entity_id"].tolist()

    start_date = datetime(2025, 1, 1)

    for i in range(1, n_treaties + 1):
        treaty_type = random.choice(TREATY_TYPES)
        line = random.choice(LINES_OF_BUSINESS)
        region = random.choice(REGIONS)
        cedent_id = random.choice(cedents)
        reinsurer_id = random.choice(reinsurers)

        premium = round(np.random.lognormal(mean=15, sigma=0.5), -3)  # ~3M-20M
        attachment_point = round(premium * random.uniform(2, 5), -3) if treaty_type == "XoL" else ""
        limit = round(attachment_point * random.uniform(2, 5), -3) if treaty_type == "XoL" else ""
        quota_share = round(random.uniform(0.1, 0.4), 2) if treaty_type == "Quota" else ""

        accepted = int(random.random() < 0.7)  # ~70% acceptance
        observed_loss_ratio = round(min(max(np.random.normal(0.65, 0.2), 0), 2), 2)
        cvar_95 = round(premium * random.uniform(1.5, 3.5), -3)

        submission_date = start_date + timedelta(days=random.randint(0, 180))

        treaties.append({
            "treaty_id": f"T{i:06d}",
            "cedent_id": cedent_id,
            "reinsurer_id": reinsurer_id,
            "treaty_type": treaty_type,
            "line_of_business": line,
            "region": region,
            "premium": premium,
            "attachment_point": attachment_point,
            "limit": limit,
            "quota_share": quota_share,
            "accepted": accepted,
            "observed_loss_ratio": observed_loss_ratio,
            "cvar_95": cvar_95,
            "submission_date": submission_date.strftime("%Y-%m-%d")
        })

    return pd.DataFrame(treaties)

treaties_df = generate_treaties(N_TREATIES)

treaties_path = os.path.join(RAW_DIR, "treaties_raw.csv")
treaties_df.to_csv(treaties_path, index=False)
print(f"✅ Saved {treaties_path} ({len(treaties_df)} rows)")

# -----------------------------
# 3. Create 1k Demo Sample
# -----------------------------
demo_sample = treaties_df.sample(N_DEMO, random_state=42)
demo_sample_path = os.path.join(DEMO_DIR, "sample_treaties.csv")
demo_sample.to_csv(demo_sample_path, index=False)
print(f"✅ Saved demo sample {demo_sample_path} ({len(demo_sample)} rows)")
