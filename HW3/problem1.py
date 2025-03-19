import numpy as np
from numpy.linalg import eig
import networkx as nx

###################################################
# Find the Eigenvalue and Eigenvector
###################################################

# # Eample
# a = np.array([[2, 2, 4], 
#               [1, 3, 5],
#               [2, 3, 4]])
# w,v=eig(a)
# print('E-value:', w)
# print('E-vector', v)

# 0 & 1 & 1 & 0 \\
# 0 & 0 & 0 & 1 \\
# 0 & 0 & 0 & 1 \\
# 0 & 0 & 0 & 0 \\

#  this seems wrong 

a = np.array([[0, 1, 1, 0], 
              [0, 0, 0, 1],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
w1,v1=eig(a)
print('E-value:', w1)
print('E-vector', v1)

w,v = np.linalg.eig(a)

for i in range(len(v)):
    print(w[i],v[:,i])
    print(np.allclose(w[i]*v[:,i],a@v[:,i]))

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(a)

# Print results
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# E-value: [0. 0. 0. 0.]
# E-vector [[ 1.00000000e+000 -1.00000000e+000 -1.00000000e+000  1.00000000e+000]
#  [ 0.00000000e+000  4.00833672e-292  0.00000000e+000 -2.00416836e-292]
#  [ 0.00000000e+000  0.00000000e+000  4.00833672e-292 -2.00416836e-292]
#  [ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000]]

###################################################
# Katz Centrality
###################################################
# import numpy as np
# import networkx as nx

# Example1
# A=np.array([[0, 5, 5, 5, 9, 3, 3, 3, 2, 3, 0, 0, 2, 0, 0, 0,],
#             [5, 0, 7, 9, 4, 2, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0,],
#             [5, 7, 0, 7, 4, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,],
#             [5, 9, 7, 0, 4, 2, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0,],
#             [9, 4, 4, 4, 0, 2, 2, 2, 1, 4, 0, 0, 2, 0, 0, 0,],
#             [3, 2, 1, 2, 2, 0, 5, 2, 3, 1, 0, 0, 0, 0, 0, 0,],
#             [3, 2, 1, 2, 2, 5, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0,],
#             [3, 2, 1, 2, 2, 2, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0,],
#             [2, 1, 0, 1, 1, 3, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0,],
#             [3, 0, 0, 0, 4, 1, 1, 1, 1, 0, 1, 0, 3, 1, 1, 0,],
#             [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 3, 2, 0,],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#             [2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 1, 0, 0, 1, 1, 0,],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 1, 0, 2, 0,],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0,],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]])

# eig, V = np.linalg.eig(A)
# max_eig=max(eig)

# G = nx.from_numpy_matrix(A)

# print('alpha < ', 1/max_eig)

# katz_centrality = nx.katz_centrality(G, weight= 'weight', alpha = 1/(max_eig+1),max_iter = 100000)
# katz_centrality_numpy = nx.katz_centrality_numpy(G, weight= 'weight', alpha = 1/(max_eig+1))

# print("Iteration:",katz_centrality)
# print("Analytical:",katz_centrality_numpy)

###################################################
# Finding the Katz Centrality
###################################################

# Example 2

# G = nx.Graph()  # or nx.DiGraph() for a directed graph
G = nx.DiGraph() 
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])

alpha = 0.1  # Attenuation factor, should be less than the inverse of the largest eigenvalue of the adjacency matrix
katz_centrality2 = nx.katz_centrality(G, alpha=alpha)

for node, centrality in katz_centrality2.items():
    print(f"Katz centrality of {node}: {centrality}")

###################################################
# Finding the Katz Centrality
###################################################

A=nx.from_numpy_array(a)
katz_centrality3 = nx.katz_centrality(A, alpha=alpha)
for node, centrality in katz_centrality3.items():
    print(f"Katz centrality of {node}: {centrality}")

###################################################
# Finding the PageRank
###################################################
# No module named 'scipy'
# pr = nx.pagerank(A, alpha=0.9)
# pr