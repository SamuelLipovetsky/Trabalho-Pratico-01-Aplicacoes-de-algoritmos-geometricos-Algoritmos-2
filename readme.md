## Trabalho Prático 01 Aplicaçõe de algoritmos geométricos -Algoritmos 2

# Organização dos arquivos

- aux_functions.py : contem as funções auxiliares que leem os dados de um dataset
keel e que divide um conjunto de pontos entre teste e treinamento

- classifications.py : arquivo que  para cada dataset executa a construção da árvore kd
e também roda o algoritmo knn gerando a precisão , revocação e acurácia para aquele modelo.

- xnn.py : arquivo com as classes que implementam a árvore kd e o knn.

- tests.py : Contem funções que testam o resultado e o tempo dos métodos knn e quickSelect implementados em comparação aos métodos de força bruta

- stats.ipynb : jupyter notebook que demonstra graficamente as estatisticas geradas
para cada dataset variando o número k de vizinhos
 
- times.ipynb:  jupyter notebook que testa os dados e gera plots que facilitam a visualicação
dos tempos de execução

- Documentação.pdf : arquivo que descreve a implementação e escolhas feitas durante o 
desenvolvimento do trabalho prático.

- /datasets : pasta com os datasets escolhidos

- /figures : todos os plot gerados 

# Execução dos arquivos

Para gerar as estatisticas para todos os datasets basta executar o arquivo classifications.py que irá imprimir no stdout o nome do dataset seguido pelas estatisticas