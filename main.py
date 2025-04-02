from sklearn.datasets import load_iris, load_wine
from bisecting import BisectingKMeansAlgorithm
from k_mens import ClusteringKmeans

iris = load_iris()
wine = load_wine()
kmeansIris = ClusteringKmeans(iris)
kmeansIris.plotDataset() #plotar um gráfico mostrando os atributos do meu dataset
kmeansIris.calcutateWcss()
kmeansIris.optimal_number_of_clusters()
kmeansIris.plotKmeansQtdOtima() #USO DO PCA
# iris_bisecting = BisectingKMeansAlgorithm("Iris", iris)
# wine_bisecting = BisectingKMeansAlgorithm("Wine", wine)
