from sklearn.datasets import load_iris, load_wine
from bisecting import BisectingKMeansAlgorithm
from k_mens import ClusteringKmeans

bisecting = BisectingKMeansAlgorithm("Wine", load_wine())
bisecting.run_script()

