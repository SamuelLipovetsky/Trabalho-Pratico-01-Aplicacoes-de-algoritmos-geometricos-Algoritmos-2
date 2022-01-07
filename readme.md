## Trabalho Prático 01 Aplicaçõe de algoritmos geométricos -Algoritmos 2

# Organização dos arquivos

- aux_functions.py : contem as funções auxiliares que leem os dados de um database
keel e que divide um conjunto de pontos entre teste e treinamento

- classifications.py : arquivo que  para cada dataset executa a construção da árvore kd
e também roda o algoritmo knn gerando a precisão , revocação e acurácia para aquele modelo.

- xnn.py : arquivo com as classes que implementam a árvore kd e o knn.

- notebook.py : jupyter noteebok que demonstra graficamente as estatisticas geradas
para cada dataset variando o número k de vizinhos

- Documentação.pdf : arquivo que descreve a implementação e escolhas feitas durante o 
desenvolvimento do trabalho prático.

- /datasets : pasta com os datasets escolhidos

# Execução dos arquivos

Para gerar as estatisticas para todos os datasets basta executar o arquivo classifications.py que irá imprimir no stdout o nome do dataset seguido pelas estatisticas