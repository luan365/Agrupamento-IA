import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#wine = load_wine()
#criar um dataframe usando data e features_names

# Load the data
class ClusteringKmeans:
    def __init__(self):
        self.database = load_iris()
        self.qtdClusters = []
        self.inertia = []
        self.qtdClusterJoelho = None
        self.vertX = None
        self.VertY = None
        self.kmeans = None
        self.target = None

    def centroidesKmeans(self, n_clusters):  # chamar ela dentro do for para gerar o kmeans com n clusters.
        self.kmeans = KMeans(n_clusters).fit(self.database.data)

    def getInertia(self):
        return self.kmeans.inertia_

    #gera o gráfico do joelho para descobrir qual o número de clusters perfeito
    def geradorKmeans(self):
        for i in range(1, 11):
            self.qtdClusters.append(i)
            self.centroidesKmeans(i) #gera o kmeans com i centroídes
            self.inertia.append(self.getInertia())
            print("Inertia: ", self.inertia)#printa a inertia de cada cluster
            plt.plot(self.qtdClusters, self.inertia, 'go-')
            plt.xlabel('Número de Clusters')
            plt.ylabel('Inertia')
            plt.show()

    def plotKmeansQtdPerfeita(self):
        numeroClusters = input("Digite o número de clusters analisado no joelho: ")
        self.centroidesKmeans(numeroClusters)
        #plotar o resto do kmeans com o número de clusters perfeito


if __name__ == '__main__':
    kmeans = ClusteringKmeans()
    kmeans.geradorKmeans()

