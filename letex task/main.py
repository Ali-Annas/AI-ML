import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import time
import pandas as pd

class ClusteringAnalysis:
    def __init__(self):
        self.datasets = {}
        self.results = {}
        
    def generate_datasets(self):
        """Generate synthetic datasets for clustering analysis"""
        np.random.seed(42)
        
        # Dataset 1: Spherical clusters
        X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
        self.datasets['spherical'] = {'X': X1, 'y': y1, 'name': 'Spherical Clusters'}
        
        # Dataset 2: Irregular shapes (moons)
        X2, y2 = make_moons(n_samples=300, noise=0.1, random_state=42)
        self.datasets['moons'] = {'X': X2, 'y': y2, 'name': 'Moon-shaped Clusters'}
        
        # Dataset 3: Concentric circles
        X3, y3 = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        self.datasets['circles'] = {'X': X3, 'y': y3, 'name': 'Concentric Circles'}
        
        # Dataset 4: High-dimensional data
        X4, y4 = make_blobs(n_samples=500, centers=3, n_features=10, random_state=42)
        self.datasets['high_dim'] = {'X': X4, 'y': y4, 'name': 'High-dimensional Data'}
        
    def plot_datasets(self):
        """Visualize all datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (key, data) in enumerate(self.datasets.items()):
            X, y = data['X'], data['y']
            axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
            axes[idx].set_title(data['name'])
            axes[idx].set_xlabel('Feature 1')
            axes[idx].set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('datasets.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_kmeans(self, X, n_clusters=3):
        """Run K-Means clustering"""
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        execution_time = time.time() - start_time
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'execution_time': execution_time
        }
    
    def run_dbscan(self, X, eps=0.5, min_samples=5):
        """Run DBSCAN clustering"""
        start_time = time.time()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        execution_time = time.time() - start_time
        
        return {
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': list(labels).count(-1),
            'execution_time': execution_time
        }
    
    def run_hierarchical(self, X, n_clusters=3):
        """Run Hierarchical clustering"""
        start_time = time.time()
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(X)
        execution_time = time.time() - start_time
        
        return {
            'labels': labels,
            'execution_time': execution_time
        }
    
    def evaluate_clustering(self, X, labels, true_labels=None):
        """Evaluate clustering performance"""
        metrics = {}
        
        # Silhouette Score
        if len(set(labels)) > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
        else:
            metrics['silhouette'] = 0
            
        # Calinski-Harabasz Index
        if len(set(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['calinski_harabasz'] = 0
            
        # Davies-Bouldin Index
        if len(set(labels)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics['davies_bouldin'] = float('inf')
            
        # Adjusted Rand Index (if true labels available)
        if true_labels is not None:
            metrics['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
            
        return metrics
    
    def plot_clustering_results(self, dataset_name, X, results):
        """Plot clustering results for a dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Original data
        axes[0].scatter(X[:, 0], X[:, 1], c=self.datasets[dataset_name]['y'], 
                        cmap='viridis', alpha=0.7)
        axes[0].set_title('Original Data')
        
        # K-Means results
        kmeans_result = results['kmeans']
        axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_result['labels'], 
                        cmap='viridis', alpha=0.7)
        if 'centroids' in kmeans_result:
            axes[1].scatter(kmeans_result['centroids'][:, 0], 
                           kmeans_result['centroids'][:, 1], 
                           c='red', marker='x', s=200, linewidths=3)
        axes[1].set_title(f'K-Means (Silhouette: {kmeans_result["metrics"]["silhouette"]:.3f})')
        
        # DBSCAN results
        dbscan_result = results['dbscan']
        axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_result['labels'], 
                        cmap='viridis', alpha=0.7)
        axes[2].set_title(f'DBSCAN (Silhouette: {dbscan_result["metrics"]["silhouette"]:.3f})')
        
        # Hierarchical results
        hierarchical_result = results['hierarchical']
        axes[3].scatter(X[:, 0], X[:, 1], c=hierarchical_result['labels'], 
                        cmap='viridis', alpha=0.7)
        axes[3].set_title(f'Hierarchical (Silhouette: {hierarchical_result["metrics"]["silhouette"]:.3f})')
        
        # Performance comparison
        algorithms = ['K-Means', 'DBSCAN', 'Hierarchical']
        silhouette_scores = [results['kmeans']['metrics']['silhouette'],
                           results['dbscan']['metrics']['silhouette'],
                           results['hierarchical']['metrics']['silhouette']]
        
        axes[4].bar(algorithms, silhouette_scores, color=['blue', 'green', 'orange'])
        axes[4].set_title('Silhouette Score Comparison')
        axes[4].set_ylabel('Silhouette Score')
        axes[4].tick_params(axis='x', rotation=45)
        
        # Execution time comparison
        execution_times = [results['kmeans']['execution_time'],
                         results['dbscan']['execution_time'],
                         results['hierarchical']['execution_time']]
        
        axes[5].bar(algorithms, execution_times, color=['blue', 'green', 'orange'])
        axes[5].set_title('Execution Time Comparison')
        axes[5].set_ylabel('Time (seconds)')
        axes[5].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'clustering_results_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dendrogram(self, X, dataset_name):
        """Create dendrogram for hierarchical clustering"""
        # Calculate linkage matrix
        linkage_matrix = linkage(X, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, truncate_mode='level', p=3)
        plt.title(f'Dendrogram for {dataset_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig(f'dendrogram_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_analysis(self):
        """Run comprehensive clustering analysis"""
        print("=== Clustering Algorithms Analysis ===\n")
        
        # Generate datasets
        self.generate_datasets()
        print("Generated synthetic datasets:")
        for key, data in self.datasets.items():
            print(f"- {data['name']}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
        
        # Plot datasets
        self.plot_datasets()
        
        # Analyze each dataset
        for dataset_name, data in self.datasets.items():
            print(f"\n=== Analyzing {data['name']} ===")
            X, y = data['X'], data['y']
            
            # Determine number of clusters for this dataset
            n_clusters = len(set(y))
            
            # Run all algorithms
            results = {}
            
            # K-Means
            print("Running K-Means...")
            kmeans_result = self.run_kmeans(X, n_clusters)
            kmeans_result['metrics'] = self.evaluate_clustering(X, kmeans_result['labels'], y)
            results['kmeans'] = kmeans_result
            
            # DBSCAN
            print("Running DBSCAN...")
            dbscan_result = self.run_dbscan(X)
            dbscan_result['metrics'] = self.evaluate_clustering(X, dbscan_result['labels'], y)
            results['dbscan'] = dbscan_result
            
            # Hierarchical
            print("Running Hierarchical Clustering...")
            hierarchical_result = self.run_hierarchical(X, n_clusters)
            hierarchical_result['metrics'] = self.evaluate_clustering(X, hierarchical_result['labels'], y)
            results['hierarchical'] = hierarchical_result
            
            # Store results
            self.results[dataset_name] = results
            
            # Print results
            print(f"\nResults for {data['name']}:")
            print(f"{'Algorithm':<15} {'Silhouette':<12} {'Time (s)':<10} {'Clusters':<10}")
            print("-" * 50)
            
            for alg_name, result in results.items():
                silhouette = result['metrics']['silhouette']
                time_taken = result['execution_time']
                n_clusters_found = len(set(result['labels']))
                print(f"{alg_name:<15} {silhouette:<12.3f} {time_taken:<10.3f} {n_clusters_found:<10}")
            
            # Plot results
            self.plot_clustering_results(dataset_name, X, results)
            
            # Create dendrogram for first dataset
            if dataset_name == 'spherical':
                self.create_dendrogram(X, dataset_name)
    
    def generate_summary_report(self):
        """Generate a summary report of all results"""
        print("\n=== SUMMARY REPORT ===")
        
        summary_data = []
        for dataset_name, results in self.results.items():
            for alg_name, result in results.items():
                summary_data.append({
                    'Dataset': self.datasets[dataset_name]['name'],
                    'Algorithm': alg_name,
                    'Silhouette Score': result['metrics']['silhouette'],
                    'Execution Time (s)': result['execution_time'],
                    'Clusters Found': len(set(result['labels']))
                })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save results to CSV
        df.to_csv('clustering_results_summary.csv', index=False)
        print("\nResults saved to 'clustering_results_summary.csv'")
        
        return df

def main():
    """Main function to run the clustering analysis"""
    # Create analysis object
    analysis = ClusteringAnalysis()
    
    # Run comprehensive analysis
    analysis.run_comprehensive_analysis()
    
    # Generate summary report
    summary_df = analysis.generate_summary_report()
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- datasets.png: Visualization of all datasets")
    print("- clustering_results_*.png: Results for each dataset")
    print("- dendrogram_spherical.png: Dendrogram example")
    print("- clustering_results_summary.csv: Summary of all results")

if __name__ == "__main__":
    main()
