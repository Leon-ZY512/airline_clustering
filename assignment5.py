#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:46:01 2024

@author: yizhen
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

df = pd.read_excel('/Users/yizhen/Desktop/6105hw/assignment5/EastWestAirlines.xlsx',sheet_name=1)
summary_stats = df.describe()
mode_data = df.mode().iloc[0]
summary_stats.loc['mode'] = mode_data

df_cleaned = df.drop(columns=['ID#', 'Award?'])


plt.figure(figsize=(12, 10))
correlation_matrix = df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
print(df.isnull().sum())
#all numerical value


df_norm = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()
columns_to_drop = ['Flight_trans_12', 'Bonus_trans', 'cc2_miles', 'cc3_miles']
df_cleaned = df_norm.drop(columns=columns_to_drop)
#df_cleaned = df_norm

# 1. Create evaluation function
def evaluate_clustering_models(data, k_range):
    results = {
        'k_values': [],
        'kmeans_silhouette': [],
        'kmeans_davies': [],
        'hierarchical_silhouette': [],
        'hierarchical_davies': []
    }
    
    for k in k_range:
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=k)
        hierarchical_labels = hierarchical.fit_predict(data)
        
        # Store results
        results['k_values'].append(k)
        results['kmeans_silhouette'].append(silhouette_score(data, kmeans_labels))
        results['kmeans_davies'].append(davies_bouldin_score(data, kmeans_labels))
        results['hierarchical_silhouette'].append(silhouette_score(data, hierarchical_labels))
        results['hierarchical_davies'].append(davies_bouldin_score(data, hierarchical_labels))
    
    return pd.DataFrame(results)


k_range = range(2, 11)
evaluation_results = evaluate_clustering_models(df_cleaned, k_range)

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(evaluation_results['k_values'], evaluation_results['kmeans_silhouette'], 'o-', label='K-means')
plt.plot(evaluation_results['k_values'], evaluation_results['hierarchical_silhouette'], 'o-', label='Hierarchical')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Comparison')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(evaluation_results['k_values'], evaluation_results['kmeans_davies'], 'o-', label='K-means')
plt.plot(evaluation_results['k_values'], evaluation_results['hierarchical_davies'], 'o-', label='Hierarchical')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n=== Clustering Evaluation Report ===")
print("\nBest Evaluation Metrics:")
best_kmeans_k = evaluation_results.loc[evaluation_results['kmeans_silhouette'].idxmax(), 'k_values']
best_hierarchical_k = evaluation_results.loc[evaluation_results['hierarchical_silhouette'].idxmax(), 'k_values']

print(f"\nK-means Optimal Clusters: {best_kmeans_k}")
print(f"- Best Silhouette Score: {evaluation_results['kmeans_silhouette'].max():.3f}")
print(f"- Corresponding Davies-Bouldin Index: {evaluation_results.loc[evaluation_results['kmeans_silhouette'].idxmax(), 'kmeans_davies']:.3f}")

print(f"\nHierarchical Optimal Clusters: {best_hierarchical_k}")
print(f"- Best Silhouette Score: {evaluation_results['hierarchical_silhouette'].max():.3f}")
print(f"- Corresponding Davies-Bouldin Index: {evaluation_results.loc[evaluation_results['hierarchical_silhouette'].idxmax(), 'hierarchical_davies']:.3f}")


final_kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = final_kmeans.fit_predict(df_cleaned)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df_cleaned['Cluster'] = cluster_labels


cluster_summary = df_cleaned.groupby('Cluster').mean()
print("\n=== Cluster Characteristics ===")
print(cluster_summary)

plt.figure(figsize=(10, 6))
cluster_summary.plot(kind='bar', figsize=(10, 6))
plt.title('Average Feature Values by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
cluster_summary.plot(kind='bar', figsize=(15, 8))
plt.title('Average Feature Values by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.tight_layout()
plt.show()


pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_cleaned.drop(columns=['Cluster']))
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=cluster_labels, palette='viridis', s=60)
plt.title('Clusters Visualized in 2D using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()


sample_df = df_cleaned.sample(n=300, random_state=42)  # Sample for easier visualization
sns.pairplot(sample_df, hue='Cluster', vars=sample_df.columns[:-1], palette='viridis')
plt.suptitle('Pair Plot of Features Colored by Cluster', y=1.02)
plt.tight_layout()
plt.show()


print("\n=== Cluster Comparison ===")
for feature in df_cleaned.columns[:-1]:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Cluster', y=feature, data=df_cleaned)
    plt.title(f'Boxplot of {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

