from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import BisectingKMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

class BisectingKMeansAlgorithm:
    def load_data(self):
        # Load das bases de dados
        iris = load_iris()
        wine = load_wine()
        self.iris = iris.data
        self.wine = wine.data
    
    def normalize_data(self):
        # Normalização dos dados
        scaler = StandardScaler()
        self.iris = scaler.fit_transform(self.iris)
        self.wine = scaler.fit_transform(self.wine)
    
    def elbow_method(self):
        # Verificar qual o melhor cluster para cada base de dados
        self.inertias_iris = []
        self.inertias_wine = []
        k_range = range(1, 10)
        for k in k_range:
            model_iris = BisectingKMeans(n_clusters=k, random_state=42)
            model_iris.fit(self.iris)
            self.inertias_iris.append(model_iris.inertia_)

            model_wine = BisectingKMeans(n_clusters=k, random_state=42)
            model_wine.fit(self.wine)
            self.inertias_wine.append(model_wine.inertia_)
    
    def plot_elbow(self, k_range: range, data, title: str):
        # Gera o grafico com do elbow referente a cada base de dados
        plt.plot(k_range, data, marker='o')
        plt.title(title)
        plt.xlabel("k")
        plt.ylabel("Inércia")
        plt.show()
    
    def plot_clusters_pca(self, data, best_k, title):
        model = BisectingKMeans(n_clusters=best_k, random_state=42)
        clusters = model.fit_predict(data)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        df_plot = pd.DataFrame(data_pca, columns=["PC1", "PC2"])
        df_plot["Cluster"] = clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=100)
        plt.title(f"{title} - PCA com k={best_k}")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    bisecting = BisectingKMeansAlgorithm()
    bisecting.load_data()
    bisecting.normalize_data()
    bisecting.elbow_method()
    bisecting.plot_elbow(range(1, 10), bisecting.inertias_iris, "Elbow(Joelho) - Iris")
    bisecting.plot_elbow(range(1, 10), bisecting.inertias_wine, "Elbow(Joelho) - Wine")
    iris_best_k = int(input("Digite o valor do cluster ideal da base de dados Iris: "))
    wine_best_k = int(input("Digite o valor do cluster ideal da base de dados Wine: "))
    bisecting.plot_clusters_pca(bisecting.iris, iris_best_k, "Iris")
    bisecting.plot_clusters_pca(bisecting.wine, wine_best_k, "Wine")
