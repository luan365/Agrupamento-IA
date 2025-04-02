# Importa os datasets Iris e Wine diretamente da biblioteca scikit-learn
from sklearn.datasets import load_iris, load_wine

# Importa funções para agrupamento hierárquico e extração de clusters
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Importa o PCA para reduzir a dimensionalidade e facilitar a visualização dos dados em 2D
from sklearn.decomposition import PCA

# Importa a biblioteca de gráficos para plotar visualizações
import matplotlib.pyplot as plt



# Função que realiza todo o processo de agrupamento e visualização para uma base de dados
def analisar_base(dados, nome_base, n_clusters=3):
    print(f"\n=== {nome_base.upper()} ===")  # Mostra qual base está sendo analisada

    # Aplica o algoritmo de linkage com o método 'ward'
    # Esse método agrupa os dados minimizando a variância interna dos clusters
    linked = linkage(dados, method='ward')

    # Gera e exibe o dendrograma (estrutura hierárquica de agrupamento)
    plt.figure(figsize=(10, 7))  # Define o tamanho do gráfico
    dendrogram(linked)  # Cria o dendrograma com base no linkage calculado
    plt.title(f'Dendrograma - {nome_base}')  # Título do gráfico
    plt.xlabel('Amostras')  # Rótulo do eixo X
    plt.ylabel('Distância')  # Rótulo do eixo Y
    plt.show()  # Exibe o gráfico na tela
    print(f"Dendrograma de {nome_base} exibido com sucesso!")

    # Define os rótulos dos clusters cortando o dendrograma em 'n_clusters' grupos
    labels = fcluster(linked, n_clusters, criterion='maxclust')

    # Aplica PCA para reduzir os dados para 2 dimensões (facilita a visualização em gráfico 2D)
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(dados)

    # Cria um gráfico com os dados projetados em 2D, coloridos de acordo com os clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f'Clusters - {nome_base} (Linkage Ward)')  # Título do gráfico
    plt.xlabel('Componente Principal 1')  # Eixo X
    plt.ylabel('Componente Principal 2')  # Eixo Y
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.show()  # Exibe o gráfico na tela
    print(f"Visualização de clusters da base {nome_base} exibida com sucesso!")

# ----------- ANALISANDO A BASE IRIS -----------
iris = load_iris()  # Carrega a base Iris (150 amostras de flores com 4 atributos cada)
analisar_base(iris.data, 'Iris')  # Chama a função para realizar a análise completa

# ----------- ANALISANDO A BASE WINE -----------
wine = load_wine()  # Carrega a base Wine (178 amostras de vinho com 13 atributos químicos)
analisar_base(wine.data, 'Wine')  # Chama a função para realizar a análise completa
