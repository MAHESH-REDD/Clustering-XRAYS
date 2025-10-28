# 🫁 Clustering-XRAYS: Unsupervised Analysis of Chest X-Ray Images

## 🚀 1. Project Overview
This repository contains the code and results for an **unsupervised clustering pipeline** applied to the **Kaggle Chest X-Ray Images (Pneumonia)** dataset.

**Objective:**  
To automatically group **NORMAL** and **PNEUMONIA** X-ray images by extracting handcrafted features and applying clustering algorithms.

The workflow integrates:
- Traditional computer vision descriptors (Histogram, HOG, LBP, Hu Moments)  
- Feature normalization and dimensionality reduction (PCA)  
- Clustering using **K-Means**, **Agglomerative Clustering**, and **DBSCAN**  
- Evaluation using intrinsic metrics (Silhouette, Davies-Bouldin) and t-SNE visualization  

📊 **Total Images Processed:** 11,712  
🏆 **Best Result:** K-Means (K=2) — *Silhouette Score = 0.0836, Purity = 0.7297*

---

## 🧱 2. Pipeline Overview

### 2.1 Data & Preprocessing

All `.jpeg` or `.png` files from `train/`, `val/`, and `test/` are combined.

```python
import cv2, glob, numpy as np
from tqdm import tqdm

def load_images(folder):
    paths = glob.glob(folder + '/**/*.jpeg', recursive=True)
    imgs = []
    for p in tqdm(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        imgs.append(img)
    return np.array(imgs)

images = load_images('chest_xray')
print('Loaded', len(images), 'images')
2.2 Feature Extraction
Feature	Method	Key Parameters	Dimension
Histogram	Global intensity distribution	256 bins	256
HOG	Histogram of Oriented Gradients	orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)	~34,596
LBP	Local Binary Pattern	P=8, R=1, 59 uniform bins	59
Hu Moments	Shape invariants	7 log-scaled moments	7

➡️ Total Feature Vector Dimension: ≈ 34,918

python
Copy code
from skimage.feature import hog, local_binary_pattern

def extract_features(img):
    # 1. Intensity Histogram
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    
    # 2. HOG
    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    
    # 3. LBP
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0,60), range=(0,59))
    
    # 4. Hu Moments
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu))  # log transform
    
    return np.hstack([hist, hog_feat, lbp_hist, hu])
2.3 Feature Normalization & Reduction
python
Copy code
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = np.array([extract_features(img) for img in images])
X_scaled = StandardScaler().fit_transform(X)

# PCA for clustering
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance (100 comps):", pca.explained_variance_ratio_.sum())
📉 Explained Variance: ≈ 32.56% (100 components)
For visualization → PCA (50 comps) → t-SNE (2D, perplexity=30)

🤖 3. Clustering Methods & Parameters
Method	Parameters	Silhouette ↑	Davies-Bouldin ↓	#Clusters
K-Means	K=2 (Elbow/Silhouette)	0.0836	3.2148	2
Agglomerative	Ward linkage, K=2	0.0756	3.4624	2
DBSCAN	ε=2.5, MinPts=5	0.0319	0.8611	2 (+ noise)

python
Copy code
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_pca)
# Agglomerative
agg = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X_pca)
# DBSCAN
db = DBSCAN(eps=2.5, min_samples=5).fit(X_pca)

for name, labels in {
    "K-Means": kmeans.labels_,
    "Agglomerative": agg.labels_,
    "DBSCAN": db.labels_
}.items():
    s = silhouette_score(X_pca, labels)
    d = davies_bouldin_score(X_pca, labels)
    print(f"{name:15s} | Silhouette = {s:.4f} | DB Index = {d:.4f}")
📊 3.1 Cluster Purity (vs True Labels)
Method	Purity (%)
K-Means	72.97
Agglomerative	72.97
DBSCAN	72.97

Interpretation: All methods roughly separate NORMAL and PNEUMONIA groups.

🎨 4. Visualization
Dimensionality reduction to 2D via t-SNE for plotting:

python
Copy code
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(6,5))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans.labels_, cmap='coolwarm', s=5)
plt.title("K-Means Clusters (t-SNE 2D)")
plt.savefig('clustering_outputs/kmeans_tsne.png')
plt.show()
Generated Figures:

Copy code
k_elbow.png | k_silhouette.png | kmeans_tsne.png | agg_tsne.png | dbscan_tsne.png | true_labels_tsne.png
⚙️ 5. How to Run the Code
🧩 Prerequisites
Install dependencies:

bash
Copy code
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib tqdm
▶️ Execution
bash
Copy code
git clone https://github.com/MAHESH-REDD/Clustering-XRAYS.git
cd Clustering-XRAYS

# Download Kaggle dataset
# Place inside ./chest_xray/train/, ./val/, ./test/

jupyter notebook Maheshreddy.ipynb
All results are saved automatically in clustering_outputs/.

📊 6. Evaluation Summary
Metric	Description	Interpretation
Silhouette Score	Cluster cohesion & separation (−1 to 1)	> 0 = better separation
Davies-Bouldin Index	Ratio of within-/between-cluster scatter	Lower = better
Purity	Match between clusters and true labels	Higher = better

Observation:
➡️ K-Means achieved best Silhouette and label alignment.

🔬 7. Recommendations
Feature Engineering: Try larger HOG cells (16×16) or CNN embeddings (VGG, ResNet).

Parameter Tuning: Use k-distance plots to optimize DBSCAN ε and MinPts.

Hybrid Approach: Combine PCA + t-SNE for robust visual clustering.

📁 8. Directory Structure
bash
Copy code
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
 └─ dendrogram_sample.png
🧾 9. Technical Notes
Distance Metric: Euclidean (L₂)

PCA Components: 100 for clustering (32.56% variance explained)

t-SNE: perplexity=30, learning rate=200

Random Seed: 42 (reproducibility)

🧠 10. Conclusion
K-Means (K=2) produced the most coherent clusters.

Agglomerative and DBSCAN achieved similar purity (~73%).

Handcrafted features (HOG/LBP/Histogram) capture some structure but not full semantics.

Future improvement: integrate deep CNN features for higher diagnostic accuracy.

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

# 🫁 Clustering-XRAYS: Unsupervised Analysis of Chest X-Ray Images

## 🚀 1. Project Overview
This repository contains the code and results for an **unsupervised clustering pipeline** applied to the **Kaggle Chest X-Ray Images (Pneumonia)** dataset.

**Goal:** build a visual-feature extraction pipeline (intensity + shape + texture) and test if clustering algorithms can naturally group **NORMAL** vs **PNEUMONIA** images.

The pipeline combines:
- classic computer-vision descriptors (Histogram + HOG + LBP + Hu Moments)
- normalization + PCA reduction
- clustering via **K-Means**, **Agglomerative**, and **DBSCAN**

**Dataset Size:** 11 712 images  
**Best Result:** K-Means (K = 2) → Silhouette = 0.0836  |  Purity = 0.7297  
All results and figures are saved in `clustering_outputs/`.

---

## 🧱 2. Pipeline Overview

### 2.1 Data & Preprocessing
Images from `train/`, `val/`, and `test/` are merged.

```python
import cv2, glob, numpy as np
from tqdm import tqdm

def load_images(folder):
    paths = glob.glob(folder + '/**/*.jpeg', recursive=True)
    imgs = []
    for p in tqdm(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256,256))
        imgs.append(img)
    return np.array(imgs)

images = load_images('chest_xray')
print('Loaded', len(images), 'images')

2.2 Feature Extraction
Descriptor	Method	Key Parameters	Dim
Histogram	Global Intensity	bins = 256	256
HOG	Histogram of Oriented Gradients	orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (2, 2)	34 596
LBP	Local Binary Pattern	P = 8, R = 1 (radius), 59 uniform bins	59
Hu Moments	Shape Invariants	7 log-scaled moments	7
Total Feature Vector ≈ 34 918 dims

from skimage.feature import hog, local_binary_pattern
import cv2, numpy as np

def extract_features(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0,60), range=(0,59))
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    hu = -np.sign(hu)*np.log10(np.abs(hu))  # log transform
    return np.hstack([hist, hog_feat, lbp_hist, hu])

from skimage.feature import hog, local_binary_pattern
import cv2, numpy as np

def extract_features(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0,60), range=(0,59))
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    hu = -np.sign(hu)*np.log10(np.abs(hu))  # log transform
    return np.hstack([hist, hog_feat, lbp_hist, hu])
2.3 Feature Normalization & Reduction
python
Copy code
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = np.array([extract_features(img) for img in images])
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance sum:", pca.explained_variance_ratio_.sum())
Explained Variance ≈ 32.56 % after 100 components
For visualization: PCA → 50 components → t-SNE (2D, perplexity = 30).

🤖 3. Clustering Algorithms
Algorithm	Key Parameters	Silhouette ↑	DB Index ↓	# Clusters
K-Means	K = 2 (best by Silhouette)	0.0836	3.2148	2
Agglomerative	Ward linkage, K = 2	0.0756	3.4624	2
DBSCAN	ϵ = 2.5, MinPts = 5	0.0319	0.8611	2 (+ noise)

python
Copy code
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

kmeans = KMeans(n_clusters=2, random_state=42).fit(X_pca)
agg = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X_pca)
db = DBSCAN(eps=2.5, min_samples=5).fit(X_pca)

for name, labels in {
    "KMeans": kmeans.labels_,
    "Agglomerative": agg.labels_,
    "DBSCAN": db.labels_
}.items():
    s = silhouette_score(X_pca, labels)
    d = davies_bouldin_score(X_pca, labels)
    print(f"{name:15s} Silhouette={s:.4f}  DB Index={d:.4f}")
📊 3.1 Cluster Purity (vs True Labels)
Method	Purity (%)
K-Means	72.97
Agglomerative	72.97
DBSCAN	72.97

Interpretation → All methods form 2 main groups roughly matching NORMAL vs PNEUMONIA.

🎨 4. Visualization
Dimensionality reduction to 2D via t-SNE for plotting.

python
Copy code
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(6,5))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans.labels_, cmap='coolwarm', s=5)
plt.title("K-Means Clusters (t-SNE 2D)")
plt.savefig('clustering_outputs/kmeans_tsne.png')
plt.show()
Generated Plots:

k_elbow.png | k_silhouette.png

kmeans_tsne.png | agg_tsne.png | dbscan_tsne.png

true_labels_tsne.png

dendrogram_sample.png

⚙️ 5. How to Run the Code
🔧 Install Dependencies
bash
Copy code
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib tqdm
▶️ Execute
bash
Copy code
git clone https://github.com/MAHESH-REDD/Clustering-XRAYS.git
cd Clustering-XRAYS

# Download Kaggle dataset and place it as:
# ./chest_xray/train/, val/, test/
jupyter notebook Maheshreddy.ipynb
Outputs appear in the clustering_outputs/ folder automatically.

🧩 6. Evaluation Summary
Metric	Description	Interpretation
Silhouette Score	cohesion & separation (−1 to 1)	> 0 = clear clusters
Davies-Bouldin Index	ratio of within-cluster scatter / between-cluster separation	lower = better
Purity	overlap with ground truth labels	higher = better

📈 Observation: K-Means gave highest Silhouette and best label alignment.

🔬 7. Recommendations
Feature Engineering: Test larger cell sizes for HOG or CNN embeddings (VGG/ResNet).

Parameter Tuning: Use k-distance plot to fine-tune DBSCAN ϵ.

Hybrid Approach: Combine PCA + t-SNE for visual analysis and supervised follow-up.

📁 8. Directory Structure
bash
Copy code
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
 └─ dendrogram_sample.png
🧾 9. Technical Notes
Distance Metric: Euclidean (L₂) after StandardScaler.

PCA Components: 100 for clustering (32.56 % variance).

t-SNE: perplexity = 30, learning rate = 200.

Random Seed: 42 for reproducibility.

🧠 10. Conclusion
K-Means (K = 2) produced the most coherent clusters.

Agglomerative and DBSCAN gave similar purity (~73 %).

Classic features (HOG/LBP/Histogram) capture some structure but not all semantic patterns.

Future work: replace handcrafted features with deep CNN embeddings for higher clinical relevance.

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
