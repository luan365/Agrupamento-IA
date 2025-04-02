from sklearn.utils._bunch import Bunch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import BisectingKMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from kneed import KneeLocator
import pandas as pd
import seaborn as sns

class BisectingKMeansAlgorithm:
    def __init__(self, data_set_title, data_set: Bunch):
        self.title = data_set_title
        self.data_set = data_set.data

    def normalize_data(self):
        scaler = StandardScaler()
        self.data_set = scaler.fit_transform(self.data_set)

    def elbow_method(self):
        self.inertias = []
        cluster_range = range(1, 10)
        for cluster in cluster_range:
            model = BisectingKMeans(n_clusters=cluster, random_state=42)
            model.fit(self.data_set)
            self.inertias.append(model.inertia_)
        k = KneeLocator(cluster_range, self.inertias, curve='convex', direction='decreasing')
        self.best_cluster = k.elbow
        
    def plot_elbow(self):
        plt.plot(range(1, 10), self.inertias, marker='o')
        plt.axvline(x=self.best_cluster, color='red', linestyle='--', label=f'Melhor cluster: {self.best_cluster}')
        plt.title(self.title)
        plt.xlabel("Clusters")
        plt.ylabel("Inércias")
        plt.show()

    def plot_clusters(self, n_cluster = None):
        if n_cluster is None:
            n_cluster = self.best_cluster
        model = BisectingKMeans(n_clusters=self.best_cluster, random_state=42)
        clusters = model.fit_predict(self.data_set)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.data_set)
        df_plot = pd.DataFrame(data_pca, columns=["PC1", "PC2"])
        df_plot["Cluster"] = clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=100)
        plt.title(f"{self.title} - PCA com k={self.best_cluster}")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def run_script(self):
        self.normalize_data()
        self.elbow_method()
        self.plot_elbow()
        self.plot_clusters()
