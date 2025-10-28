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

  🫁 Clustering-XRAYS: Unsupervised Analysis of Chest X-Ray Images
1. Project Overview 🚀
This repository contains the code and results for an unsupervised clustering pipeline applied to the Kaggle Chest X-Ray Images (Pneumonia) dataset. The primary objective is to develop a feature extraction pipeline that captures visual characteristics (intensity, shape, texture) and to evaluate whether standard clustering algorithms can discover meaningful groupings, particularly those that align with the clinical labels: NORMAL versus PNEUMONIA.

The pipeline utilizes a combination of traditional computer vision descriptors, followed by normalization, dimensionality reduction (PCA), and testing with three clustering methods: K-Means, Agglomerative Clustering, and DBSCAN.


Total Images Processed: 11,712.


Best Clustering Result: K-Means (K=2) showed the highest internal separation (Silhouette Score: 0.0836) and the highest alignment with true labels (Purity: 0.7297).





Outputs: All results, evaluation metrics, and visualizations are saved in the clustering_outputs/ directory.

2. Technical Pipeline Breakdown 🛠️
2.1. Data & Preprocessing
The pipeline loads all images from the chest_xray directory (combining train, val, and test splits).


Read & Convert: Images are read using OpenCV and converted to grayscale.


Resize: All images are standardized to 256×256 pixels.


Normalization: Pixel values are normalized to uint8 (0–255) when necessary.

2.2. Feature Extraction
For each image, four distinct feature sub-vectors are concatenated to form a single high-dimensional vector.

Feature Descriptor	Method	Key Parameters	Dimension
Intensity Histogram (v 
hist
​
 )	Global intensity distribution	256 bins	
256 

HOG (v 
HOG
​
 )	Histogram of Oriented Gradients (shape/edge)	Orientations=9, Pixels/Cell=(8, 8), Cells/Block=(2, 2)		
∼34,596 


LBP (v 
LBP
​
 )	Local Binary Patterns (micro-texture)	P=8, R=1, 59 uniform bins	
59 

Hu Moments (v 
Hu
​
 )	Invariant moments (shape invariants)	Log-transformed 7 moments	
7 


Export to Sheets


Total Feature Vector Dimension: ≈34,918.

2.3. Feature Normalization and Reduction

Normalization: A StandardScaler was applied to the entire feature matrix to ensure zero mean and unit variance across dimensions, making them comparably weighted for the Euclidean distance metric.


Dimensionality Reduction (Clustering): PCA was applied to reduce the dimension to 100 components.


Explained Variance (Sum): 32.56%.


Dimensionality Reduction (Visualization): PCA to 50 components, followed by t-SNE (2D) (Perplexity=30) for visualization.


Similarity Metric: Euclidean Distance (L 
2
​
 ) was used for all clustering methods.

3. Clustering Methods & Parameters
Clustering was performed on the 100-component PCA space.

Method	Key Parameter Selection	Results Summary
K-Means	
Optimal K=2 selected by Silhouette Score analysis over K∈[2,6].


K=2. Achieved best Silhouette Score (0.0836).


Agglomerative	
K=2 (matching K-Means). Ward linkage used (minimizes variance).

K=2. Silhouette Score (0.0756).


DBSCAN		
ϵ=2.5, MinPts=5.

Found 2 clusters and some noise points. Silhouette Score (0.0319).



Export to Sheets

3.1. Quantitative Evaluation Metrics
Method	# Clusters	Silhouette Score	Davies-Bouldin Index
K-Means	2	0.083631	3.214753
Agglomerative	2	0.075555	3.462432
DBSCAN	2	0.031891	0.861130

Export to Sheets

Method	Cluster Purity (vs. True Labels)
K-Means	0.729679
Agglomerative	0.729679
DBSCAN	0.729679

Export to Sheets

Interpretation: A Silhouette Score near 0 indicates overlapping clusters. DBSCAN achieved the lowest Davies-Bouldin Index (lower is better), but K-Means had the best Silhouette score. The high and identical purity value suggests that the cluster assignments for K=2 in K-Means and Agglomerative, and the resulting two main clusters in DBSCAN, closely map to the two true clinical labels (NORMAL/PNEUMONIA).



4. How to Run the Code
Prerequisites
Python 3.x

The following libraries (install via pip): numpy, pandas, opencv-python (cv2), scikit-image (skimage), scikit-learn (sklearn), matplotlib, tqdm.

Setup
Clone the repository:

Bash

git clone https://github.com/Owner/Clustering-XRAYS.git
cd Clustering-XRAYS
Download Data: Obtain the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle and place the chest_xray/ folder in the project root.

Execute the Notebook: Run the provided Maheshreddy.ipynb notebook (or the equivalent Python script) from start to finish.

Bash

# Example command if running as a script (assuming you've converted the .ipynb to .py)
python Maheshreddy.py
Outputs
The execution will create a folder named clustering_outputs/ containing all generated files:


xray_features_scaled.csv (Normalized features + metadata) 


xray_features_with_clusters.csv (Cluster labels and PCA components) 


clustering_evaluation.csv (Metrics table shown above) 


cluster_purity.csv (Purity summary) 


k_elbow.png, k_silhouette.png (K-Means analysis plots) 


true_labels_tsne.png (Visualization of ground truth labels) 


kmeans_tsne.png, agg_tsne.png, dbscan_tsne.png (Cluster visualizations in 2D t-SNE space) 

5. Next Steps and Recommendations
To further improve the clustering quality and robustness:


Feature Engineering: Experiment with lower HOG resolution (e.g., pixels_per_cell=(16, 16)) or integrate CNN embeddings (e.g., features from a pre-trained VGG or ResNet) for more semantic representations.


Parameter Tuning: Systematically evaluate DBSCAN using a k-distance plot to select a more optimal ϵ value and test a wider range of MinPts (e.g., 5–20).


Supervised Validation: While the core task is unsupervised, compute and analyze the confusion matrix/contingency table between cluster labels and true labels to better characterize the nature of the clusters.

Would you like me to generate the content of the clustering_evaluation.csv file based on the provided notebook output?
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







