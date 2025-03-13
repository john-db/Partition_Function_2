import treeswift as ts
import numpy as np
import pandas as pd
from phylo2vec import vecToNewick, phylovecdomain
from cf_mat_to_newick import cf_to_newick
from itertools import chain, combinations

def is_df_in_list(df, df_list):
    for df_item in df_list:
        if df.equals(df_item):
            return True
    return False

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def consistent_subtrees(nwk, n):
    sts = newick_to_subtrees(nwk)
    trivial = sts[0:n + 1] + [sts[-1]]
    nontrivial = sts[n + 1:-1]
    pset = list(powerset(nontrivial))

    ret = []
    for p in pset:
        mat = trivial[:-1] + list(p) + [trivial[-1]]
        ret += [pd.DataFrame(mat).transpose()]
    return ret

def binary_trees(n):
    vecs = phylovecdomain(n)
    newicks = list(map(lambda x: vecToNewick(x) + ";", vecs))

    return newicks

def all_trees(n):
    binary = binary_trees(n)
    ls = []
    for tree in binary:
        ls += [tree]
        for cf_mat in consistent_subtrees(tree, n)[:-1]:
            # Sort cf_mat to be safe?
            new_tree = cf_to_newick(cf_mat)
            if new_tree not in ls:
                ls += [new_tree]
    return(ls)

def newick_to_subtrees(newick, cells_to_indices):
    """
    Parameters:
             newick string representing a tree https://en.wikipedia.org/wiki/Newick_format that has leaves corresponding to
                cells in the genotype matrix
             dict that maps leaf labels to their index in the matrix
                 (i.e. dict that tells you which row of the genotype matrix corresponds to that leaf)
    Returns:
            Numpy array where each row represents a subtree, and each column represents a leaf of the tree.
    """

    tree = ts.read_tree_newick(newick) # get a tree from the newick string using the TreeSwift package
    num_leaves = len(cells_to_indices) # how many leaf nodes the tree has

    #preallocate full matrix
    subtrees_ls = np.zeros((tree.num_nodes(),num_leaves), dtype='bool')

    for i,node in enumerate(tree.traverse_preorder()):
        # This for loop visits each node of the tree. At each node, we want to create a vector that represents
        # the subtree that is rooted at that node (i.e. the vector indicates which leaves are present in that subtree)
        # We will add each of these vectors to a list, which we will return as a numpy array

        # initialize a vector that will represent the subtree of the current node
        # subtree = np.zeros(num_leaves, dtype='bool')

        for leaf in node.traverse_leaves():
            # for each leaf below this node, we will change the corresponding index in the subtree vector to 1
            #   to indicate that the leaf is present in that subtree
            subtrees_ls[i, cells_to_indices[leaf.get_label()]] = 1

    # We return the numpy matrix where row i = the ith vector in our list of subtrees
    return subtrees_ls

#def newick_to_subtrees_no_labels(newick):
#     # input: newick string of tree with n leaves labelled 0,1,2,...,n-1

#     tree = ts.read_tree_newick(newick)
#     leaves = tree.labels(leaves=True, internal=False)
#     n = len(list(leaves))

#     subtrees = []
#     for node in tree.traverse_preorder():
#         subtree = np.zeros(n, dtype=np.int8)
#         leaves = [int(leaf.get_label()) for leaf in node.traverse_leaves()]
#         subtree[leaves] = 1
#         subtrees += [subtree]

#     subtrees += [np.zeros(n, dtype=np.int8)]
#     subtrees = [np.array(x) for x in {(tuple(e)) for e in subtrees}]
#     subtrees.sort(key=lambda x : (sum(x), int(''.join(map(str,x)), 2)))
#     return subtrees
