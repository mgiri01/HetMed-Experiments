import argparse

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Explore clinical features and write a validation sample."
)
parser.add_argument(
    "--input",
    required=True,
    help="Path to the clinical-feature Excel workbook.",
)
parser.add_argument(
    "--output",
    required=True,
    help="Path for the sampled validation CSV.",
)
args = parser.parse_args()

clin_non_image_features = pd.read_excel(args.input)

list(clin_non_image_features)

 
 
# MRI Features
mri_vars = ["ContrastAgent", "TR", "TE", "SliceThickness ", "Rows", "Columns"]

# Reverse maps: code -> label (for display only)
contrast_map = {
    0: "GADAVIST",
    1: "MAGNEVIST",
    2: "MMAGNEVIST",
    3: "MULTIHANCE",
    4: "Name of agent not stated (ContrastBolusAgent tag present)",
    5: "ContrastBolusAgent tag absent",
}
slice_map = {
    0: "0.9", 1: "0.95", 2: "1", 3: "1.04", 4: "1.06", 5: "1.1", 6: "1.12",
    7: "1.15", 8: "1.2", 9: "1.23", 10: "1.24", 11: "1.25", 12: "1.3",
    13: "1.4", 14: "1.45", 15: "1.5", 16: "1.6", 17: "1.8", 18: "2",
    19: "2.2", 20: "2.5"
}
rows_cols_map = {0: "320", 1: "448", 2: "512"}

# Display-only mapping without changing original dataframe
display_df = clin_non_image_features[mri_vars].copy()
display_df["ContrastAgent"] = display_df["ContrastAgent"].map(contrast_map).fillna("unavailable")
display_df["SliceThickness "] = display_df["SliceThickness "].map(slice_map).fillna("unavailable")
display_df["Rows"] = display_df["Rows"].map(rows_cols_map).fillna("unavailable")
display_df["Columns"] = display_df["Columns"].map(rows_cols_map).fillna("unavailable")

# One table per variable, all in one figure
n = len(mri_vars)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()

for i, col in enumerate(mri_vars):
    stats = display_df[[col]].describe(include="all").round(2)

    axes[i].axis("off")
    table = axes[i].table(
        cellText=stats.values,
        colLabels=stats.columns,
        rowLabels=stats.index,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    axes[i].set_title(col.strip())

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()




#Tumor Characteristics
cat_vars = ["ER", "PR", "HER2", "Mol_Subtype", "Histologic type"]

label_maps = {
    "ER": {0: "negative", 1: "positive"},
    "PR": {0: "negative", 1: "positive"},
    "HER2": {0: "negative", 1: "positive"},
    "Mol_Subtype": {0: "luminal-like", 1: "ER/PR pos, HER2 pos", 2: "her2", 3:"trip neg"},
    "Histologic type": {
        0: "DCIS", 1: "ductal", 2: "lobular", 3: "metastacic",
        4: "LCIS", 5: "tubular", 6: "mixed", 7: "micropapillary", 8: "colloid",
        9: 'mucinous', 10:"medullary"
    },
}

n = len(cat_vars)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()

for i, col in enumerate(cat_vars):
    counts = clin_non_image_features[col].value_counts(dropna=False)

    # Map labels for display without changing underlying data
    labels = []
    for v in counts.index:
        if pd.isna(v):
            labels.append("unavailable")
        else:
            labels.append(label_maps.get(col, {}).get(v, str(v)))
    counts.index = labels

    counts.plot(kind="bar", ax=axes[i])
    axes[i].set_title(f"{col} Counts")
    axes[i].set_ylabel("Count")

# Hide any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# Label

fig, ax = plt.subplots(figsize=(6, 4))

counts = clin_non_image_features["Nottingham_Grade_v2"].value_counts(dropna=False)

labels = []
for v in counts.index:
    if pd.isna(v):
        labels.append("No Grade Data Available")
    else:
        v_num = pd.to_numeric(v, errors="coerce")
        if pd.isna(v_num):
            labels.append(str(v))
        else:
            labels.append(str(int(round(float(v_num)))))

counts.index = labels
counts.plot(kind="bar", ax=ax)

ax.set_title("Label: Tumor Grade", pad=18)
ax.text(0.5, 1.02, "Some Values Derived", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()



#Surgery

cat_vars = ["Surgery", "Days_to_Surg_from_Dx", "Def_Surg_Type"]

label_maps = {
    "Surgery": {0: "no", 1: "yes"},
    "Def_Surg_type": {0: "BCS", 1: "mastectomy"},
}

n = len(cat_vars)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
axes = axes.flatten()

for i, col in enumerate(cat_vars):
    if col == "Days_to_Surg_from_Dx":
        # numeric distribution (convert just for plotting)
        pd.to_numeric(clin_non_image_features[col], errors="coerce").dropna().plot(
            kind="hist", bins=30, ax=axes[i]
        )
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_ylabel("Count")
    else:
        counts = clin_non_image_features[col].value_counts(dropna=False)
        labels = []
        for v in counts.index:
            if pd.isna(v):
                labels.append("unavailable")
            else:
                labels.append(label_maps.get(col, {}).get(v, str(v)))
        counts.index = labels

        counts.plot(kind="bar", ax=axes[i])
        axes[i].set_title(f"{col} Counts")
        axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

#RT
# Plots for NART and ART
vars_therapy = ["NART", "ART"]
label_map = {0: "no", 1: "yes"}

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()

for i, col in enumerate(vars_therapy):
    counts = clin_non_image_features[col].value_counts(dropna=False)

    # Map labels for display without changing underlying data
    labels = []
    for v in counts.index:
        if pd.isna(v):
            labels.append("unavailable")
        else:
            labels.append(label_map.get(v, str(v)))
    counts.index = labels

    counts.plot(kind="bar", ax=axes[i])
    axes[i].set_title(f"{col} Counts")
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

#Chemotherapy

# Fix the column name typo
clin_non_image_features = clin_non_image_features.rename(
    columns={"Neoaduvant_Chemotherapy": "Neoadjuvant_Chemotherapy"}
)

vars_chemo = ["Neoadjuvant_Chemotherapy", "Adjuvant_Chemotherapy"]
label_map = {0: "no", 1: "yes"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes = axes.flatten()

for i, col in enumerate(vars_chemo):
    counts = clin_non_image_features[col].value_counts(dropna=False)

    # Map labels for display without changing underlying data
    labels = []
    for v in counts.index:
        if pd.isna(v):
            labels.append("unavailable")
        else:
            labels.append(label_map.get(v, str(v)))
    counts.index = labels

    counts.plot(kind="bar", ax=axes[i])
    axes[i].set_title(f"{col} Counts")
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

sample_100 = clin_non_image_features.sample(
    n=min(100, len(clin_non_image_features)),
    random_state=42
)


sample_100.to_csv(args.output, index=False)
