from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Carregar o dataset Iris
iris = load_iris()

X = iris.data  # Características
y = iris.target  # Espécies (rótulos)
nomes_especies = iris.target_names  # Nomes das espécies

# Criar e treinar o modelo KNeighborsClassifier
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X, y)

# Solicitar as medidas ao usuário
print("Informe as características da primeira flor:")
sepal_length = float(input("Digite o comprimento da sépala (cm): "))
sepal_width = float(input("Digite a largura da sépala (cm): "))
petal_length = float(input("Digite o comprimento da pétala (cm): "))
petal_width = float(input("Digite a largura da pétala (cm): "))

print("\nAgora informe as características da segunda flor:")
sepal_length2 = float(input("Digite o comprimento da sépala (cm): "))
sepal_width2 = float(input("Digite a largura da sépala (cm): "))
petal_length2 = float(input("Digite o comprimento da pétala (cm): "))
petal_width2 = float(input("Digite a largura da pétala (cm): "))

# Fazer a predição
nova_flor = [[sepal_length, sepal_width, petal_length, petal_width],
             [sepal_length2, sepal_width2, petal_length2, petal_width2]]
predicao = modelo.predict(nova_flor)

# Imprimir o resultado
print(f"\nA primeira flor provavelmente é uma {nomes_especies[predicao[0]]}.")
print(f"A segunda flor provavelmente é uma {nomes_especies[predicao[1]]}.")
