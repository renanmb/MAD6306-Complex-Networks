import numpy as np
import networkx as nx

# 0 & 1 & 1 & 0 \\
# 1 & 0 & 0 & 1 \\
# 1 & 0 & 0 & 1 \\
# 0 & 1 & 1 & 0 \\

a = np.array([[0, 1, 1, 0], 
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]])
A=nx.from_numpy_array(a)

# Calculate degree centrality
degree_centrality = nx.degree_centrality(A)

# Print the results
print(degree_centrality)
# {0: 0.6666666666666666, 1: 0.6666666666666666, 2: 0.6666666666666666, 3: 0.6666666666666666}
