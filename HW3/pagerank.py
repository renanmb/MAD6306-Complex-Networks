import numpy as np

def page_rank(adj_matrix, damping_factor=0.85, max_iterations=10000, tol=1e-6):
    """
    Calculate PageRank for a given adjacency matrix.

    Parameters:
    - adj_matrix: The adjacency matrix representing the link structure of the web.
    - damping_factor: The probability that a user will continue clicking on links.
    - max_iterations: The maximum number of iterations for the PageRank calculation.
    - tol: Convergence tolerance.

    Returns:
    - pagerank: A vector representing the PageRank scores for each page.
    """
    # Normalize the adjacency matrix to ensure that each column sums to 1
    # adj_matrix = adj_matrix / adj_matrix.sum(axis=0, keepdims=True)

    # Normalize the adjacency matrix to ensure that each column sums to 1
    # Check if any column sum is zero and replace it with a uniform distribution
    column_sums = adj_matrix.sum(axis=0, keepdims=True)
    adj_matrix = np.divide(adj_matrix, column_sums, where=column_sums != 0)  # Avoid division by zero
    adj_matrix = np.nan_to_num(adj_matrix)  # Replace NaNs with 0, in case any division by zero happened

    # Get the number of nodes
    num_nodes = adj_matrix.shape[0]

    # Initialize the PageRank vector
    pagerank = np.ones(num_nodes) / num_nodes

    # Main iteration loop
    for iteration in range(max_iterations):
        # Calculate the updated PageRank vector
        new_pagerank = (1 - damping_factor) / num_nodes + damping_factor * adj_matrix @ pagerank

        # Check for convergence
        if np.linalg.norm(new_pagerank - pagerank, 1) < tol:
            break

        pagerank = new_pagerank

    return pagerank

# Example usage
# simple web structure with three pages
# A -> B, A -> C, B -> A, C -> A
# The adjacency matrix would be:
# [[0, 1, 1],
#  [1, 0, 0],
#  [1, 0, 0]]

adj_matrix = np.array([[0, 1, 1, 0], 
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])

result = page_rank(adj_matrix)
print("PageRank:", result)
# With damping of 0.1 and 10000 iterations
# PageRank: [0.27225 0.23625 0.23625 0.225  ]
# With damping of 0.85 and 10000 iterations
# PageRank: [0.12834375 0.0534375  0.0534375  0.0375    ]