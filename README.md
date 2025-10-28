🫁 Clustering-XRAYS: Unsupervised Analysis of Chest X-Ray Images
🚀 1. Project Overview

This repository contains the complete pipeline and results for an unsupervised clustering analysis on the Kaggle Chest X-Ray Images (Pneumonia) dataset.

The main goal is to extract visual features (intensity, texture, and shape) from chest X-rays and analyze whether clustering algorithms can naturally separate NORMAL and PNEUMONIA cases without supervision.

The workflow integrates:

Feature extraction using traditional computer vision descriptors

Feature normalization and dimensionality reduction (PCA)

Clustering via K-Means, Agglomerative Clustering, and DBSCAN

Evaluation using intrinsic metrics and t-SNE visualizations

📊 Total Images Processed: 11,712
🏆 Best Result: K-Means (K=2) — Silhouette Score = 0.0836, Purity = 0.7297

All outputs and metrics are stored in the clustering_outputs/ directory.

🛠️ 2. Technical Pipeline Breakdown
🔹 2.1 Data & Preprocessing

Combines all images from chest_xray/train, val, and test.

Conversion: Read with OpenCV → Grayscale.

Resize: 256×256 pixels.

Normalization: Pixel values normalized to range [0, 255].

🔹 2.2 Feature Extraction
Feature	Method	Key Parameters	Dimension
Intensity Histogram (v<sub>hist</sub>)	Global pixel intensity distribution	256 bins	256
HOG (v<sub>HOG</sub>)	Histogram of Oriented Gradients	orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)	~34,596
LBP (v<sub>LBP</sub>)	Local Binary Pattern	P=8, R=1, 59 uniform bins	59
Hu Moments (v<sub>Hu</sub>)	Shape invariants (log-scaled moments)	7 invariant moments	7

➡️ Total Feature Vector Dimension: ≈ 34,918

🔹 2.3 Feature Normalization & Reduction

Standardization: StandardScaler (zero mean, unit variance).

Dimensionality Reduction (for Clustering): PCA → 100 components (explained variance ≈ 32.56%).

Visualization: PCA → 50 components → t-SNE (2D, perplexity=30).

Distance Metric: Euclidean (L₂)

🤖 3. Clustering Methods & Parameters

Clustering performed on 100D PCA-reduced feature space.

Method	Parameters	Results Summary
K-Means	Optimal K=2 via Silhouette Analysis (K∈[2,6])	Silhouette = 0.0836
Agglomerative	K=2, Ward linkage	Silhouette = 0.0756
DBSCAN	ϵ=2.5, MinPts=5	Silhouette = 0.0319
📊 3.1 Quantitative Evaluation Metrics
Method	#Clusters	Silhouette Score	Davies-Bouldin Index
K-Means	2	0.0836	3.2148
Agglomerative	2	0.0756	3.4624
DBSCAN	2	0.0319	0.8611
Method	Cluster Purity (vs. True Labels)
K-Means	0.7297
Agglomerative	0.7297
DBSCAN	0.7297

📈 Interpretation:
Silhouette ≈ 0 implies overlapping clusters.
DBSCAN has lowest Davies-Bouldin (compactness), while K-Means achieves highest Silhouette and aligns best with clinical labels.

⚙️ 4. How to Run the Code
🧩 Prerequisites

Install dependencies:

pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib tqdm

🧠 Setup
git clone https://github.com/MAHESH-REDD/Clustering-XRAYS.git
cd Clustering-XRAYS


Download the Dataset:
Get Chest X-Ray Images (Pneumonia) from Kaggle and place the chest_xray/ folder in the root.

Run the Notebook or Script:

jupyter notebook Maheshreddy.ipynb


or

python Maheshreddy.py

🗂️ Outputs

All generated files are saved in the clustering_outputs/ folder:

clustering_outputs/
 ├─ xray_features_scaled.csv
 ├─ xray_features_with_clusters.csv
 ├─ clustering_evaluation.csv
 ├─ cluster_purity.csv
 ├─ k_elbow.png
 ├─ k_silhouette.png
 ├─ kmeans_tsne.png
 ├─ agg_tsne.png
 ├─ dbscan_tsne.png
 ├─ true_labels_tsne.png

💡 5. Next Steps & Recommendations

✅ Feature Engineering:
Experiment with:

Lower HOG resolution (pixels_per_cell=(16,16))

Pretrained CNN embeddings (VGG16, ResNet) for semantic features

✅ Parameter Tuning:
Use k-distance plots to refine DBSCAN’s ε and MinPts values.

✅ Supervised Validation (Optional):
Compare cluster labels to true labels via a confusion matrix or contingency table.

📜 6. Technical Summary

Feature Extraction Parameters:

HOG: 9 orientations, 8×8 pixels/cell, 2×2 cells/block → 34,596 dims

LBP: Uniform, P=8, R=1 → 59 dims

Histogram: 256 dims

Hu Moments: 7 dims

Similarity Function: Euclidean distance — compatible with K-Means & Ward linkage.
Best Algorithm: K-Means (K=2) — most coherent clustering and highest alignment with labels.

Cluster Purity: ~73% — indicates moderate unsupervised separation between NORMAL and PNEUMONIA.

🧭 7. Directory Structure
chest_xray/
 ├─ train/
 ├─ val/
 ├─ test/
clustering_outputs/
 ├─ xray_features_scaled.csv
 ├─ xray_features_with_clusters.csv
 ├─ clustering_evaluation.csv
 ├─ cluster_purity.csv
 ├─ k_elbow.png
 ├─ k_silhouette.png
 ├─ kmeans_tsne.png
 ├─ agg_tsne.png
 ├─ dbscan_tsne.png
 ├─ true_labels_tsne.png
 ├─ dendrogram_sample.png

🧾 8. Credits

Author: Mahesh Reddy D
Date: October 2025
Course: Computer Vision & Clustering Assignment
Dataset: Chest X-Ray Images (Pneumonia) — Kaggle
