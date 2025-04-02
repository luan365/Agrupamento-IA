import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from math import sqrt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
class ClusteringKmeans:
    def __init__(self, databases):
        self.database = databases
        self.dfDatabase = pd.DataFrame(databases.data, columns=databases.feature_names)

        # Normalizando bases com StandardScaler
        scaler = StandardScaler()
        self.dfDatabase = scaler.fit_transform(self.dfDatabase)

        self.qtdClusters = []
        self.inertia = []
        self.qtdClusterJoelho = None
        self.kmeans = None
        self.clusterOtimo = None

    # Ver as características das dimensões do dataset
    def plotDataset(self):
        df = pd.DataFrame(self.database.data, columns=self.database.feature_names)
        df["target"] = self.database.target_names[self.database.target]
        sns.pairplot(df, hue="target")
        plt.show()

    # Método do cotovelo para encontrar o número ideal de clusters
    def calculateWcss(self):
        for i in range(1, 11):
            self.qtdClusters.append(i)
            self.kmeans = KMeans(n_clusters=i, random_state=42).fit(self.dfDatabase)  # Garantindo reprodutibilidade
            self.inertia.append(self.kmeans.inertia_)  # Soma das distâncias quadradas

        plt.plot(self.qtdClusters, self.inertia, 'go-')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inertia')
        plt.title('Método do Cotovelo')
        plt.show()

    # Determinar o número ótimo de clusters usando o método do cotovelo
    def optimal_number_of_clusters(self):
        x1, y1 = 1, self.inertia[0]
        x2, y2 = len(self.inertia), self.inertia[-1]

        distances = []
        for i, inertia_val in enumerate(self.inertia):
            x0, y0 = i + 1, inertia_val
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(numerator / denominator)

        self.clusterOtimo = distances.index(max(distances)) + 1
        print(f"---> Número de clusters ideal: {self.clusterOtimo}")

    def plotKmeansQtdOtima(self):
        newkmeans = KMeans(n_clusters=self.clusterOtimo, random_state=42).fit(self.dfDatabase)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.dfDatabase)
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=newkmeans.labels_, cmap='viridis')
        plt.xlabel('Componente PCA 1')
        plt.ylabel('Componente PCA 2')
        plt.title(f'Agrupamento KMeans com {self.clusterOtimo} Clusters')
        plt.show()

if __name__ == '__main__':
    # K-Means com a base Iris
    iris = load_iris()
    kmeansIris = ClusteringKmeans(iris)
    kmeansIris.plotDataset()
    kmeansIris.calculateWcss()
    kmeansIris.optimal_number_of_clusters()
    kmeansIris.plotKmeansQtdOtima()

    # K-Means com a base Wine
    wine = load_wine()
    kmeansWine = ClusteringKmeans(wine)
    kmeansWine.plotDataset()
    kmeansWine.calculateWcss()
    kmeansWine.optimal_number_of_clusters()
    kmeansWine.plotKmeansQtdOtima()
