# Clustering-XRAYS
Unsupervised clustering pipeline for Chest X-Ray images (11,712 total) using Intensity, HOG, LBP, and Hu Moment features1111.
Chest X-Ray Clustering - Pneumonia Detection
Overview
This repository implements an unsupervised clustering pipeline for chest X-ray images, focusing on grouping radiographs by visual features to separate pneumonia from normal cases. It includes preprocessing, feature extraction, clustering (K-Means, Agglomerative, DBSCAN), and cluster evaluation/visualization.

All code is in Maheshreddy.ipynb, and analysis/report is in Maheshreddy_report.docx.​

Directory Structure
text
chestxray/
  ├─ train/
  ├─ val/
  ├─ test/
clusteringoutputs/
  ├─ xrayfeaturesscaled.csv
  ├─ xrayfeatureswithclusters.csv
  ├─ clusteringevaluation.csv
  ├─ clusterpurity.csv
  ├─ kelbow.png
  ├─ ksilhouette.png
  ├─ kmeanstsne.png
  ├─ aggtsne.png
  ├─ dbscantsne.png
  ├─ truelabelstsne.png
Pipeline Steps & Scripts
1. Data Loading & Preprocessing
Loads all .png, .jpg, etc. images, merges train/val/test sets.

Each image converted to grayscale and resized to 256×256 pixels.

Normalization: pixel values scaled to uint8.

2. Feature Extraction
Each image is represented by the following concatenated vector:

Intensity Histogram: 256 bins.

HOG Descriptors: 9 orientations, 
8
×
8
8×8 pixels/cell, 
2
×
2
2×2 cells/block (dimension: 34,596).

LBP Histogram: Uniform LBP (P=8, R=1), 59 bins.

Hu Moments: 7 log-scaled invariants.

Total feature vector dimension: 34,918.

All feature extraction implemented using OpenCV and scikit-image. See notebook functions:

python
def histfeat(img, bins=256): ...
def hogfeat(img): ...
def lbphistfeat(img): ...
def humomentsfeat(img): ...
3. Feature Normalization
StandardScaler (zero mean, unit variance) applied to each feature across dataset.

For clustering, dimensionality reduced via PCA to 100 components.

For visualization, PCA to 50 components, then t-SNE to 2D.

4. Clustering Algorithms
Distance metric: Euclidean distance on normalized feature space.

K-Means Clustering

K tried from 2 to 6.

Optimal K chosen by silhouette score (usually K=2).

Agglomerative Clustering

Ward linkage, clusters matched to K-Means.

DBSCAN

Tuned with eps=2.5, min_samples=5 (on 100D PCA components).

5. Evaluation & Visualization
Intrinsic metrics:

Silhouette Score, Davies-Bouldin Index for each algorithm.

Visualization:

2D scatterplots via t-SNE and PCA.

Color by cluster, and by actual medical label for purity/comparison.

Results Table:

Method	Clusters	Silhouette Score	Davies-Bouldin Index
K-Means	2	0.0836	3.2148
Agglomerative	2	0.0756	3.4624
DBSCAN	2	0.0319	0.8611
Cluster Purity:

Approximates overlap with true label: ~72.97% for all methods.

Plotting sample code:

python
def plotlabels2d(coords, labels, title, fname): ...
Technical Report Summary
Feature Extraction Parameters:

HOG: 9, 
8
×
8
8×8 pixels/cell, 
2
×
2
2×2 cells/block, 34,596 dims.

LBP: Uniform (P=8, R=1), 59 dims.

Histogram: 256 dims.

Hu Moments: 7 dims.

Similarity Function Justification:

Euclidean works well after feature scaling—compatible with chosen algorithms.

Parameter Selections:

K-Means K=2; Agglomerative n_clusters=2 (Ward linkage); DBSCAN eps=2.5 & min_samples=5.

Cluster Visualization:

All clustering and true label t-SNE plots saved in outputs.

Metric Table:

(See above results table.)

Conclusion:

K-Means with K=2 was most effective, aligning best with true disease labels and highest silhouette score.

Feature engineering is critical; further improvements can be made by tuning HOG parameters or using learned features (CNN).

Cluster purity was ~73%, indicating reasonable unsupervised separation.

Usage
Download and place the Kaggle dataset in chestxray/.

Run Maheshreddy.ipynb sequentially (all paths/parameters are set at the top).

All outputs (CSV evaluations, cluster assignments, plots) are generated in clusteringoutputs/.

Refer to Maheshreddy_report.docx for detailed analysis and discussion.

Dependencies
Python 3.8+

OpenCV

scikit-image

scikit-learn

numpy, pandas, matplotlib

tqdm

Install via:

bash
pip install opencv-python scikit-image scikit-learn pandas matplotlib tqdm







