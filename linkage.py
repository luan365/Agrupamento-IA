from sklearn.datasets import load_iris, load_wine
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  # <-- Importa a função de silhueta
import numpy as np
import matplotlib.patches as mpatches

linkage_methods = ['complete', 'average', 'ward']

def analisar_base(dados, nome_base, n_clusters=3):
    print(f"\n=== {nome_base.upper()} ===")

    # Normaliza os dados
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados)

    # PCA para visualização em 2D
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(dados_normalizados)

    for metodo in linkage_methods:
        print(f"\n🔗 Linkage: {metodo.upper()}")

        # Aplica o agrupamento hierárquico
        linked = linkage(dados_normalizados, method=metodo)

        # Gera rótulos com base em corte do dendrograma
        labels = fcluster(linked, n_clusters, criterion='maxclust')

        # Calcula o índice de silhueta
        sil_score = silhouette_score(dados_normalizados, labels)
        print(f"📊 Silhouette Score ({metodo}): {sil_score:.4f}")

        # Dendrograma
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title(f'Dendrograma - {nome_base} ({metodo})')
        plt.xlabel('Amostras')
        plt.ylabel('Distância')
        plt.legend([f'Cores representam {n_clusters} clusters gerados'], loc='upper right')
        plt.show()
        print(f"Dendrograma ({metodo}) de {nome_base} exibido com sucesso!")

        # Visualização dos clusters em 2D
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(f'Clusters - {nome_base} ({metodo})')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True)

        # Cria legenda com as cores dos clusters
        unique_clusters = np.unique(labels)
        legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=f'Cluster {i}')
                          for i in unique_clusters]
        plt.legend(handles=legend_handles)
        plt.show()
        print(f"Visualização de clusters ({metodo}) exibida com sucesso!")

# Análise da base Iris
iris = load_iris()
analisar_base(iris.data, 'Iris')

# Análise da base Wine
wine = load_wine()
analisar_base(wine.data, 'Wine')
