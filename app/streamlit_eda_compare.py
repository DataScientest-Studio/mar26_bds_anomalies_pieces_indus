import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Comparative EDA MVTec vs HSS IAD", layout="wide")


MVTec_META = {
    "name": "MVTec AD",
    "source": "Official MVTec website",
    "categories_total": 15,
    "images_total": 5354,
    "train_images": 3629,
    "test_images": 1725,
    "notes": "Benchmark industrial anomaly detection dataset with train folder containing only normal images and test folder containing both good and anomalous samples."
}

HSS_META = {
    "name": "HSS IAD",
    "source": "HSS IAD paper and GitHub repository",
    "categories_total": 7,
    "images_total": 8580,
    "anomalous_pixel_ratio": 0.03,
    "defect_background_similarity":"High (text)",
    "notes": "Metallic like industrial parts with subtle and confusable defects, designed to be closer to same sort industrial production conditions."
}


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def list_image_files(root: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    root_path = Path(root)
    if not root_path.exists():
        return []
    return [str(p) for p in root_path.rglob("*") if p.suffix.lower() in exts]


@st.cache_data(show_spinner=False)
def scan_mvtec_dataset(root: str) -> pd.DataFrame:
    root_path = Path(root)
    rows = []
    if not root_path.exists():
        return pd.DataFrame()

    for category_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        category = category_dir.name

        train_dir = category_dir / "train"
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"

        for split_dir, split_name in [(train_dir, "train"), (test_dir, "test")]:
            if not split_dir.exists():
                continue
            for defect_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                defect_type = defect_dir.name
                for img_path in defect_dir.rglob("*"):
                    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                        continue
                    is_anomaly = int(split_name == "test" and defect_type != "good")
                    mask_path = None
                    if is_anomaly and gt_dir.exists():
                        stem = img_path.stem
                        candidate_dir = gt_dir / defect_type
                        candidates = list(candidate_dir.glob(f"{stem}*") ) if candidate_dir.exists() else []
                        if candidates:
                            mask_path = str(candidates[0])
                    rows.append(
                        {
                            "dataset": "MVTec AD",
                            "category": category,
                            "split": split_name,
                            "label": "anomaly" if is_anomaly else "good",
                            "defect_type": defect_type,
                            "image_path": str(img_path),
                            "mask_path": mask_path,
                        }
                    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def scan_hssiad_dataset(root: str) -> pd.DataFrame:
    root_path = Path(root)
    rows = []
    if not root_path.exists():
        return pd.DataFrame()

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    for category_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        category = category_dir.name
        split_dirs = [p for p in category_dir.iterdir() if p.is_dir()] if category_dir.exists() else []

        # Flexible parser for several possible dataset organizations
        for split_dir in split_dirs:
            split_name = split_dir.name.lower()
            if split_name not in {"train", "test", "val", "validation"}:
                continue

            subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
            if not subdirs:
                for img_path in split_dir.rglob("*"):
                    if img_path.suffix.lower() in image_exts:
                        rows.append(
                            {
                                "dataset": "HSS IAD",
                                "category": category,
                                "split": split_name,
                                "label": "unknown",
                                "defect_type": "unknown",
                                "image_path": str(img_path),
                                "mask_path": None,
                            }
                        )
                continue

            for subdir in subdirs:
                defect_type = subdir.name
                label = "good" if defect_type in {"good", "normal", "ok"} else "anomaly"
                for img_path in subdir.rglob("*"):
                    if img_path.suffix.lower() not in image_exts:
                        continue
                    rows.append(
                        {
                            "dataset": "HSS IAD",
                            "category": category,
                            "split": split_name,
                            "label": label,
                            "defect_type": defect_type,
                            "image_path": str(img_path),
                            "mask_path": None,
                        }
                    )

    # Attempt to find masks by convention
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    potential_mask_roots = [
        root_path / "ground_truth",
        root_path / "masks",
        root_path / "mask",
        root_path / "annotations",
    ]
    for i, row in df.iterrows():
        if row["label"] != "anomaly":
            continue
        stem = Path(row["image_path"]).stem
        category = row["category"]
        defect_type = row["defect_type"]
        found = None
        for mask_root in potential_mask_roots:
            if not mask_root.exists():
                continue
            candidates = list(mask_root.rglob(f"{stem}*"))
            if candidates:
                found = str(candidates[0])
                break
            cat_dir = mask_root / category / defect_type
            if cat_dir.exists():
                candidates = list(cat_dir.glob(f"{stem}*"))
                if candidates:
                    found = str(candidates[0])
                    break
        if found:
            df.at[i, "mask_path"] = found
    return df


@st.cache_data(show_spinner=False)
def image_stats(image_path: str) -> Dict[str, float]:
    img = safe_open_image(Path(image_path))
    if img is None:
        return {}
    arr = np.array(img).astype(np.float32)
    gray = arr.mean(axis=2)
    return {
        "width": float(arr.shape[1]),
        "height": float(arr.shape[0]),
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
    }


@st.cache_data(show_spinner=False)
def mask_ratio(mask_path: str) -> Optional[float]:
    if not mask_path:
        return None
    img = safe_open_image(Path(mask_path))
    if img is None:
        return None
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    binary = arr > 0
    return float(binary.mean())


@st.cache_data(show_spinner=False)
def enrich_with_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    stats = df["image_path"].apply(image_stats).apply(pd.Series)
    out = pd.concat([df.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    if "mask_path" in out.columns:
        out["anomaly_pixel_ratio"] = out["mask_path"].apply(lambda x: mask_ratio(x) if pd.notna(x) else None)
    return out



def render_meta_cards():
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MVTec AD")
        st.write(pd.DataFrame({
            "Metric": ["Categories", "Total images", "Train images", "Test images"],
            "Value": [MVTec_META["categories_total"], MVTec_META["images_total"], MVTec_META["train_images"], MVTec_META["test_images"]],
        }))
        st.caption(MVTec_META["notes"])
    with c2:
        st.subheader("HSS IAD")
        st.write(pd.DataFrame({
            "Metric": ["Categories", "Total images", "Anomalous pixel ratio", "Defect background similarity"],
            "Value": [HSS_META["categories_total"], HSS_META["images_total"], HSS_META["anomalous_pixel_ratio"], HSS_META["defect_background_similarity"]],
        }))
        st.caption(HSS_META["notes"])



def plot_bar_counts(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    count_df = df_all.groupby(["dataset", "category"]).size().reset_index(name="count")
    if count_df.empty:
        st.info("No local files were detected yet.")
        return
    pivot = count_df.pivot(index="category", columns="dataset", values="count").fillna(0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Image count by category")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Category")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)



def plot_label_distribution(df_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    label_df = df_all.groupby(["dataset", "label"]).size().reset_index(name="count")
    if label_df.empty:
        st.info("No local files were detected yet.")
        return
    for dataset in label_df["dataset"].unique():
        subset = label_df[label_df["dataset"] == dataset]
        ax.bar([f"{dataset}\n{lbl}" for lbl in subset["label"]], subset["count"])
    ax.set_title("Good versus anomaly distribution")
    ax.set_ylabel("Number of images")
    st.pyplot(fig)



def plot_image_size_distribution(df_all: pd.DataFrame):
    if df_all.empty or "width" not in df_all.columns:
        st.info("Image statistics are not available yet.")
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    for dataset in df_all["dataset"].dropna().unique():
        subset = df_all[df_all["dataset"] == dataset]
        ax.hist(subset["width"].dropna(), bins=20, alpha=0.6, label=f"{dataset} width")
    ax.set_title("Image width distribution")
    ax.set_xlabel("Width in pixels")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)



def plot_intensity_distribution(df_all: pd.DataFrame):
    if df_all.empty or "mean_intensity" not in df_all.columns:
        st.info("Image statistics are not available yet.")
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    for dataset in df_all["dataset"].dropna().unique():
        subset = df_all[df_all["dataset"] == dataset]
        ax.hist(subset["mean_intensity"].dropna(), bins=20, alpha=0.6, label=dataset)
    ax.set_title("Mean intensity distribution")
    ax.set_xlabel("Mean grayscale intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)



def plot_mask_ratio_distribution(df_all: pd.DataFrame):
    if "anomaly_pixel_ratio" not in df_all.columns:
        st.info("Mask statistics are not available yet.")
        return
    subset = df_all[df_all["anomaly_pixel_ratio"].notna()]
    if subset.empty:
        st.info("No anomaly masks were found locally yet.")
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    for dataset in subset["dataset"].dropna().unique():
        part = subset[subset["dataset"] == dataset]
        ax.hist(part["anomaly_pixel_ratio"], bins=20, alpha=0.6, label=dataset)
    ax.set_title("Anomaly pixel ratio distribution")
    ax.set_xlabel("Anomalous area ratio")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)



def sample_gallery(df: pd.DataFrame, dataset_name: str, category: Optional[str], label: str, n: int = 4):
    subset = df[df["dataset"] == dataset_name]
    if category and category != "All":
        subset = subset[subset["category"] == category]
    if label != "All":
        subset = subset[subset["label"] == label]
    subset = subset.sample(min(n, len(subset)), random_state=42) if len(subset) else subset

    if subset.empty:
        st.info(f"No local images found for {dataset_name} in the current filter.")
        return

    cols = st.columns(len(subset))
    for col, (_, row) in zip(cols, subset.iterrows()):
        img = safe_open_image(Path(row["image_path"]))
        if img is not None:
            col.image(img, caption=f"{row['category']} | {row['label']} | {row['defect_type']}", use_container_width=True)



def sample_with_mask(df: pd.DataFrame, dataset_name: str, category: Optional[str]):
    subset = df[(df["dataset"] == dataset_name) & (df["label"] == "anomaly") & (df["mask_path"].notna())]
    if category and category != "All":
        subset = subset[subset["category"] == category]
    if subset.empty:
        st.info(f"No anomaly plus mask pair found for {dataset_name} in the current filter.")
        return

    row = subset.sample(1, random_state=42).iloc[0]
    img = safe_open_image(Path(row["image_path"]))
    mask = safe_open_image(Path(row["mask_path"])) if pd.notna(row["mask_path"]) else None
    c1, c2 = st.columns(2)
    if img is not None:
        c1.image(img, caption="Image", use_container_width=True)
    if mask is not None:
        c2.image(mask, caption="Mask", use_container_width=True)


# Sidebar
st.sidebar.title("Comparative EDA")
st.sidebar.write("Compare MVTec AD and HSS IAD with a single Streamlit app.")

mvtec_root = st.sidebar.text_input("MVTec root folder", value="mvtec_anomaly_detection")
hss_root = st.sidebar.text_input("HSS IAD root folder", value="HSS-IAD\\HSS-IAD")
max_rows = st.sidebar.slider("Max rows for lightweight stats", min_value=100, max_value=20000, value=4000, step=100)

# Load full datasets first
mvtec_df_full = enrich_with_stats(scan_mvtec_dataset(mvtec_root))
hss_df_full = enrich_with_stats(scan_hssiad_dataset(hss_root))

# Sample only for lightweight display and plotting
mvtec_df = (
    mvtec_df_full.sample(min(max_rows, len(mvtec_df_full)), random_state=42)
    if not mvtec_df_full.empty else mvtec_df_full
)
hss_df = (
    hss_df_full.sample(min(max_rows, len(hss_df_full)), random_state=42)
    if not hss_df_full.empty else hss_df_full
)

# Full dataset for accurate summary statistics
all_df_full = (
    pd.concat([mvtec_df_full, hss_df_full], ignore_index=True)
    if not mvtec_df_full.empty or not hss_df_full.empty else pd.DataFrame()
)

# Sampled dataset for lightweight display and visualizations
all_df = (
    pd.concat([mvtec_df, hss_df], ignore_index=True)
    if not mvtec_df.empty or not hss_df.empty else pd.DataFrame()
)

# Main app
st.title("Comparative EDA for Industrial Anomaly Detection")
st.write(
    "This app is designed for an exploratory comparison between MVTec AD and HSS IAD before preprocessing and modeling. "
    "It helps assess dataset structure, class balance, image properties, mask availability, and business relevance for an industrial inspection use case."
)

section = st.radio(
    "Section",
    [
        "Project overview",
        "Dataset snapshot",
        "Visual inspection",
        "Data visualizations",
        "Business insights",
    ],
    horizontal=True,
)

if section == "Project overview":
    st.header("Project overview")
    st.markdown(
        """
**Use case**

Detect subtle visual defects on precision machined metallic parts in a context similar to Fidémeca.

**Goal of this EDA**

1. Understand the structure of each dataset
2. Compare realism and difficulty
3. Identify preprocessing needs
4. Decide how to position the baseline model and the comparative analysis in the final report
        """
    )
    render_meta_cards()

    st.subheader("Expected comparative angle")
    st.markdown(
        """
MVTec AD is a strong benchmark and a good baseline dataset.

HSS IAD is expected to be closer to the real industrial setting because it focuses on same sort metallic like products with subtle and confusable defects.
        """
    )

elif section == "Dataset snapshot":
    st.header("Dataset snapshot")

    st.subheader("Current root folders")
    st.write(f"MVTec root: {mvtec_root}")
    st.write(f"HSS IAD root: {hss_root}")

    st.subheader("Quick check")
    st.write("MVTec path exists:", Path(mvtec_root).exists())
    st.write("HSS IAD path exists:", Path(hss_root).exists())

    if all_df_full.empty:
        st.warning("No local dataset was found yet. Check the root folders in the sidebar.")
    else:
        summary = (
            all_df_full.groupby("dataset")
            .agg(
                images=("image_path", "count"),
                categories=("category", "nunique"),
                anomaly_images=("label", lambda s: int((s == "anomaly").sum())),
                good_images=("label", lambda s: int((s == "good").sum())),
                masks=("mask_path", lambda s: int(pd.Series(s).notna().sum())),
            )
            .reset_index()
        )

        st.info("Dataset summary is computed on full datasets. The tables below are lightweight samples for display only.")

        st.subheader("Dataset summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Detected categories")
        detected_categories = (
            all_df_full.groupby("dataset")["category"]
            .unique()
            .reset_index()
        )
        detected_categories["category"] = detected_categories["category"].apply(lambda x: ", ".join(sorted(x)))
        st.dataframe(detected_categories, use_container_width=True, hide_index=True)

        st.subheader("Detailed sample of indexed files")

        st.markdown("### MVTec AD sample")
        if mvtec_df.empty:
            st.info("No MVTec files detected.")
        else:
            st.dataframe(mvtec_df.head(50), use_container_width=True, hide_index=True)

        st.markdown("### HSS IAD sample")
        if hss_df.empty:
            st.info("No HSS IAD files detected.")
        else:
            st.dataframe(hss_df.head(50), use_container_width=True, hide_index=True)

elif section == "Visual inspection":
    st.header("Visual inspection")
    dataset_name = st.selectbox("Dataset", options=["MVTec AD", "HSS IAD"])
    current_df = mvtec_df if dataset_name == "MVTec AD" else hss_df
    categories = ["All"] + sorted(current_df["category"].dropna().unique().tolist()) if not current_df.empty else ["All"]
    category = st.selectbox("Category", options=categories)
    label = st.selectbox("Label", options=["All", "good", "anomaly", "unknown"])

    st.subheader("Sample gallery")
    sample_gallery(all_df, dataset_name, category, label, n=4)
    st.subheader("Sample image plus mask")
    sample_with_mask(all_df, dataset_name, category)


elif section == "Data visualizations":
    st.header("Data visualizations")
    if all_df.empty:
        st.warning("No local dataset was found yet. Update the root folders in the sidebar after downloading the datasets.")
    else:

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Image count by category")
            plot_bar_counts(all_df)

        with col2:
            st.subheader("2. Good versus anomaly distribution")
            plot_label_distribution(all_df)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. Image size distribution")
            plot_image_size_distribution(all_df)

        with col4:
            st.subheader("4. Mean intensity distribution")
            plot_intensity_distribution(all_df)

        col5, col6 = st.columns(2)
        with col5:
            st.subheader("5. Anomaly pixel ratio distribution")
            plot_mask_ratio_distribution(all_df)


elif section == "Business insights":
    st.header("Business insights")
    st.markdown(
        """
### Suggested narrative for the report

**MVTec AD** is useful as a benchmark baseline. It is well structured, widely used, and convenient for developing a first anomaly detection pipeline.

**HSS IAD** appears more realistic for a precision manufacturing use case because it emphasizes same sort metallic like products, subtle defects, and high similarity between anomalies and background.

### Expected project decision

Use MVTec AD to validate the first end to end pipeline and use HSS IAD to challenge the approach on a dataset that is closer to real industrial complexity.

### Why this matters for Fidémeca

A dataset closer to same sort machined parts is more relevant for a company that performs visual inspection on industrial metal components rather than on a mix of unrelated objects.
        """
    )

st.divider()
st.caption(
    "Tip: create screenshots from each section for the report, then reuse the same wording in the English write up."
)
