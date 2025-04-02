import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from math import sqrt
import pandas as pd
import seaborn as sns
import time

# Load the data
class ClusteringKmeans:
    def __init__(self, databases):
        self.database = databases
        self.dfDatabase = pd.DataFrame(databases.data, columns = databases.feature_names)
        self.qtdClusters = []
        self.inertia = []
        self.qtdClusterJoelho = None
        self.vertX = None
        self.VertY = None
        self.kmeans = None
        self.target = None
        self.clusterOtimo = None


    #Ver as características das dimensões do dataset
    def plotDataset(self):
        df = pd.DataFrame(self.database.data, columns=self.database.feature_names)
        df["target"] = self.database.target_names[self.database.target]
        _ = sns.pairplot(df, hue="target")

    #Metodo do cotovelo com o objetivo de encontrar o número de clusters ideal - Kmeans
    def calcutateWcss(self):
        for i in range(1, 11):
            self.qtdClusters.append(i)
            self.kmeans = KMeans(n_clusters=i).fit(self.dfDatabase) #gerando centroides (agrupamentos)
            self.inertia.append(self.kmeans.inertia_) #inertia é a soma das distâncias quadradas das amostras para o centro do cluster
            print("Inertia: ", self.inertia)#printa a inertia de cada cluster

        plt.plot(self.qtdClusters, self.inertia, 'go-')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inertia')
        plt.show()

    # após o metódo do cotovelo é preciso inserir a quantidade de cluster idela para plotar o agrupamento do K-means
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


    def plotKmeansQtdOtima(self):
        newkmeans = KMeans(n_clusters=self.clusterOtimo).fit(self.dfDatabase)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.dfDatabase)
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=newkmeans.labels_, cmap='viridis')
        plt.ylabel('Componente PCA 2')
        plt.title(f'Agrupamento KMeans com {self.clusterOtimo} Clusters')
        plt.show()


if __name__ == '__main__':
#   Kmenas com a base Iris
    iris = load_iris()
    kmeansIris = ClusteringKmeans(iris)
    kmeansIris.plotDataset() #plotar um gráfico mostrando os atributos do meu dataset
    plt.figure()
    kmeansIris.calcutateWcss()
    kmeansIris.optimal_number_of_clusters()
    kmeansIris.plotKmeansQtdOtima() #USO DO PCA
#---------------------------------------------------
#   Kmenas com a base Wine
    wine = load_wine()
    kmeansWine = ClusteringKmeans(wine)
    kmeansWine.plotDataset() #plotar um gráfico mostrando os atributos do meu dataset
    plt.figure()
    kmeansWine.calcutateWcss()
    kmeansWine.optimal_number_of_clusters()
    kmeansWine.plotKmeansQtdOtima() #USO DO PCA