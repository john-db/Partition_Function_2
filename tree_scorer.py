import numpy as np
import pandas as pd
import treeswift as ts
from IPython import embed
import sys, time

def log_pf_cond_mat_mul(P, subtrees, cells_vec, mut_id):
    if any(np.equal(subtrees, cells_vec).all(1)):
        P_minus_m = np.delete(P, mut_id, axis=1)
        col = P[:, mut_id]
        res = log_prob_mat_mul_calc(np.log2(P_minus_m), np.log2(1 - P_minus_m), subtrees)
        return np.dot(np.log2(col), cells_vec) + np.dot(np.log2(1 - col), 1 - cells_vec) + res - log_prob_mat_mul_calc(np.log2(P), np.log2(1 - P), subtrees)
    else:
        return np.float64('-inf')

def log_pf_cond_on_one_tree(P, subtrees, cells_vec, cond_m):
    r"""
    Prob_{A\sim P}[\subtree(c, R, A)\cap A\in G| A\in T] in O(n^2).

    :param P:
    :param subtrees: cell lineage tree  n x (2n+1)
    :param cells_vec: set of cells
    :param cond_m: one mutation
    :return: log 2 of the probability conditioned on the given tree...
    """

    denominator = np.float64(0)
    numerator = np.float64(0)
    col = P[:, cond_m]
    for v in subtrees:
        prob = np.float64(np.prod(col * v + (1 - col) * (1 - v)))
        denominator += prob
        if np.array_equal(v, cells_vec):
            numerator = prob
    if numerator == 0:
        return np.float64('-inf')
    else:
        return np.log2(numerator) - np.log2(denominator)

def log_pf_cond_numpy(P, subtrees, cells_vec, mutation):
    r"""
    Prob_{A\sim P}[\subtree(c, R, A)\cap A\in G| A\in T] in O(n^2).

    :param P:
    :param subtrees: cell lineage tree  n x (2n+1)
    :param cells_vec: set of cells
    :param cond_m: one mutation
    :return: log 2 of the probability conditioned on the given tree...
    """
    if any(np.equal(subtrees, cells_vec).all(1)):
        col = P[:, mutation]
        denominator = np.matmul(subtrees, np.log2(col)) + np.matmul(1 - subtrees, np.log2(1 - col))
        denominator = np.exp2(denominator)
        denominator = np.sum(denominator)
        return np.dot(np.log2(col), cells_vec) + np.dot(np.log2(1 - col), 1 - cells_vec) - np.log2(denominator)
    else:
        return np.float64('-inf')

def log_prob_mat_mul_calc(logP1, logP0, subtrees):
    """
    Parameters:
        logP1: Matrix containing the log base 2 of the probability that the corresponding entry in the genotype matrix is 1
            i.e. logP_i,j = log_2(Pr[X_i,j = 1])
        logP0: Similar to above, but it is the log base 2 of probability that the corresponding entry in the genotype matrix is 0

            Both of these matrices have dimension n rows (number of cells) by m columns (number of mutations)

        subtrees: 2D-Numpy array where each row of the array/matrix corresponds to a subtree (1 indicates that leaf is present in the subtree, 0 else)
            The dimension of this matrix varies depending on the tree, but it has as many rows as there are nodes of the tree,
            and it has n (number of cells/leaves) columns

            If the tree which the subtrees matrix is representing is binary, then it has n - 1 internal nodes,
            in which case the dimension of the subtrees matrix would be
            2n - 1 rows by n columns. if it is a nonbinary tree it will have fewer rows and the same number of columns.

    Returns:
        The log base 2 of the probability that a matrix drawn from the distribution of the ground truth (represented in logP1 and logP0)
        is consistent with the tree represented by the subtrees matrix
    """

    res = np.matmul(subtrees, logP1) + np.matmul(1 - subtrees, logP0)
    # multiplies the [(less or equal to 2n) by n] matrix against the [n by m] matrix
    # Afterwards, res_i,j = log_2 of the probability that the ith subtree is equal to the j^th mutation
    # Now, we need to sum these mutation-wise (i.e. for each mutation, sum the probability that it equals the first mutations plus the second etc...)
    # To sum probabilities, we will need to exponentiate res
    # Now np.exp2(res)_i,j = probability that the ith subtree is equal to the j^th mutation
    # res has dimensions (# of subtrees by # of mutations)
    # np.matmul documentation: https://numpy.org/doc/2.1/reference/generated/numpy.matmul.html
    #   "If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
    #   After matrix multiplication the prepended 1 is removed."
    
    res = np.exp2(res).sum(axis=0)
    res = np.log2(res).sum()

    # embed()
    # sys.exit()

    return res

# def main():
#     path_df = "./input_genotype_matrix"
#     path_trees = "./trees"

#     df = pd.read_csv(path_df, sep="\t", index_col=[0])               # read the input genotype matrix in as a dataframe
#     cells_to_indices = {df.index[i]:i for i in range(len(df.index))} # create a dict that maps leaf/cell labels to
#                                                                      #      which index/row they are in the matrix

#     alpha=0.001 # False positive rate
#     beta=0.1    # False negative rate

#     # Here we create the matrix representing the probability distribution of the ground truth,
#     #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
#     #   given the observed input genotype matrix stored in df
#     I_mtr = df.values # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
#     t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
#     t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
#     P = t1 + t2
#     P[I_mtr == 3] = 0.5 # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1


#     print(f"Parsing.", end=" ", flush=True)
#     t = time.time()
#     trees_represented_as_subtree_matrices = []
#     with open(path_trees,"r") as file:
#         for line in file:
#             # Each line of the file is a Newick string representing a leaf labelled tree topology
#             # We will convert these strings into numpy arrays representing the subtrees of the tree
#             # As we convert each string to an array, we append it to a list of all the subtree matrices to be scored
#             trees_represented_as_subtree_matrices += [newick_to_subtrees(line, cells_to_indices)]
#     print(f"Got {len(trees_represented_as_subtree_matrices)} trees in {time.time()-t} seonconds")

#     logP1 = np.log2(P)      # log2 of the probability matrix. This represents the log of the
#                            #    probability that an entry is 1
#     logP0 = np.log2(1 - P) # log2 of the complement (1 minus) of the probability matrix (). This represents the log of the
#                            #    probability that an entry is 0

#     # Create a list of the log probabilities of each tree
#     print(f"Computation.", end=" ", flush=True)
#     consistent_matrix_log_probabilities = np.empty(len(trees_represented_as_subtree_matrices))
#     for i,subtree_matrix in enumerate(trees_represented_as_subtree_matrices):
#             consistent_matrix_log_probabilities[i] = prob_mat_mul_calc(logP1, logP0, subtree_matrix)
#     print(f"Completed in {time.time()-t} seconds")

#     # # Prints out the probability values of each tree
#     # for log_prob in consistent_matrix_log_probabilities:
#     #     print(2 ** log_prob)

# if __name__ == "__main__":
#     main()
