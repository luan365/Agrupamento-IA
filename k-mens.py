import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from math import sqrt
import pandas as pd

#wine = load_wine()
#criar um dataframe usando data e features_names

# Load the data
class ClusteringKmeans:
    def __init__(self, database):
        self.database = database
        self.qtdClusters = []
        self.inertia = []
        self.qtdClusterJoelho = None
        self.vertX = None
        self.VertY = None
        self.kmeans = None
        self.target = None
        self.clusterOtimo = None

    #Metodo do cotovelo com o objetivo de encontrar o número de clusters ideal - Kmeans
    def calcutateWcss(self):
        for i in range(1, 11):
            self.qtdClusters.append(i)
            self.kmeans = KMeans(n_clusters=i).fit(self.database) #gerando centroides (agrupamentos)
            self.inertia.append(self.kmeans.inertia_) #inertia é a soma das distâncias quadradas das amostras para o centro do cluster
            print("Inertia: ", self.inertia)#printa a inertia de cada cluster

        plt.plot(self.qtdClusters, self.inertia, 'go-')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inertia')
        plt.show()

    def optimal_number_of_clusters(self):
        x1, y1 = 1, self.inertia[0]
        x2, y2 = 10, self.inertia[len(self.inertia) - 1]

        distances = []
        for i in range(len(self.inertia)):
            x0 = i + 2
            y0 = self.inertia[i]
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(numerator / denominator)

        self.clusterOtimo = distances.index(max(distances)) + 2
        print("---> Número de clusters ideal: ", self.clusterOtimo)

    #após o metódo do cotovelo é preciso inserir a quantidade de cluster idela para plotar o agrupamento do K-means
    def plotKmeansQtdOtima(self):
        newkmeans = KMeans(n_clusters=self.clusterOtimo).fit(self.database)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.database)
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=newkmeans.labels_, cmap='viridis')
        plt.ylabel('Componente PCA 2')
        plt.title(f'Agrupamento KMeans com {self.clusterOtimo} Clusters')
        plt.show()


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    kmeans = ClusteringKmeans(df)
    kmeans.calcutateWcss()
    kmeans.optimal_number_of_clusters()
    kmeans.plotKmeansQtdOtima()