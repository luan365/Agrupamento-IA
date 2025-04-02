from sklearn.datasets import load_iris, load_wine
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



# Define os métodos de linkage que serão utilizados (excluindo 'single' por escolha do grupo)
linkage_methods = ['complete', 'average', 'ward']

# Define a função que analisa uma base de dados com todos os métodos de linkage
def analisar_base(dados, nome_base, n_clusters=3):
    print(f"\n=== {nome_base.upper()} ===")  # Exibe o nome da base no terminal

    # Normaliza os dados para garantir que todos os atributos estejam na mesma escala
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados)

    # Aplica PCA para reduzir os dados para 2 dimensões (para visualização 2D)
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(dados_normalizados)

    # Loop para aplicar cada método de linkage definido
    for metodo in linkage_methods:
        print(f"\n🔗 Linkage: {metodo.upper()}")

        # Cria a hierarquia de clusters com o método atual
        linked = linkage(dados_normalizados, method=metodo)

        # Gera e exibe o dendrograma (representação da hierarquia de agrupamento)
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title(f'Dendrograma - {nome_base} ({metodo})')
        plt.xlabel('Amostras')
        plt.ylabel('Distância')
        plt.show()
        print(f"Dendrograma ({metodo}) de {nome_base} exibido com sucesso!")

        # Cria rótulos dos clusters com base no corte do dendrograma
        labels = fcluster(linked, n_clusters, criterion='maxclust')

        # Plota os clusters em 2D usando as componentes principais
        plt.figure(figsize=(8, 6))
        plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(f'Clusters - {nome_base} ({metodo})')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True)
        plt.show()
        print(f"Visualização de clusters ({metodo}) exibida com sucesso!")

# Analisa a base Iris com os métodos definidos
iris = load_iris()
analisar_base(iris.data, 'Iris')

# Analisa a base Wine com os métodos definidos
wine = load_wine()
analisar_base(wine.data, 'Wine')
