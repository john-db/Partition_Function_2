import argparse
import pandas as pd
import numpy as np
from decimal import Decimal
from tree_scorer import log_prob_mat_mul_calc, log_pf_cond_mat_mul, log_pf_cond_numpy, log_pf_cond_on_one_tree

def partition_function(path_matrix, path_trees, alpha, beta, clade=None, mutation=None, path_list=None, path_scoring_matrix=None, gpu=False):
    df = pd.read_csv(path_matrix, sep="\t", index_col=[0])

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = df.values # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1
    
    pairs = None
    # clade,mutation pairs to be used as input for the partition function can be inputted either as:
    # (1) a matrix that scores the mutations present in the column labels with the clade being the cells (rows) with 1s for that mutation
    # (2) a text file with clade,mutation pairs (TODO implement this)
    # (3) a paired argument of clade and mutation for scoring only a single mutation against a single clade
    if path_scoring_matrix != None:
        scoring_df = pd.read_csv(path_scoring_matrix, sep="\t", index_col=[0])
        pairs = np.empty(len(scoring_df.columns), dtype=object)
        for i in range(len(scoring_df.columns)):
            col = scoring_df[scoring_df.columns[i]]
            pairs[i] = ([c for c in col.keys() if col[c] == 1], scoring_df.columns[i])
    elif path_list != None:
        pass #implement later
    else:
        pairs = [clade.split(','), mutation]

    numerators = np.full(len(pairs), Decimal(0), dtype=object)
    denominator = Decimal(0)
    num_cells = P.shape[0]
    logP1 = np.log2(P)
    logP0 = np.log2(1 - P)

    # The input file of trees contains on each line a sampling probability corresponding to a tree, 
    # and the nontrival clades/subtrees of that tree concatenated row-wise to a single line.
    # The two are separated by a space character
    # We must reconstruct the subtrees matrix, do to that we need to include the trivial clades/subtrees 
    # (i.e. the singletons and the subtree containing all leaves), and also a row representing the possibility
    # that a given mutation is not present in any cell of the tree (row of all zeros)
    
    trivial_subtrees = np.concatenate((np.diag(np.ones(num_cells, dtype='bool')), np.ones((1, num_cells), dtype='bool'), np.zeros((1, num_cells), dtype='bool')))
    with open(path_trees, 'r') as file:
        for line in file:
            line = line.strip()
            split = line.split(" ") # the probability and the subtrees are separated by a space
            log_sampling_prob = np.log2(np.float64(split[0])) # read the sampling probability

            # Now we read the nontrivial subtrees into a 2d array
            nontrivial_subtrees = np.zeros(shape=(num_cells - 2, num_cells), dtype='bool')
            for idx in range(len(split[1])):
                nontrivial_subtrees[np.unravel_index(idx, nontrivial_subtrees.shape)] = bool(int(split[1][idx]))
            # We put the nontrivial and trivial subtrees together to form the subtrees array for the given tree
            subtrees = np.concatenate((nontrivial_subtrees, trivial_subtrees))

            # the denominator is the same for each tree clade/mutation pair with the given sample of trees,
            # so we only compute this once
            log_p1 = log_prob_mat_mul_calc(logP1, logP0, subtrees)
            denominator += 2 ** Decimal(log_p1 - log_sampling_prob)

            # for each clade/mutation pair to be evaluated, we compute its numerator
            for i in range(len(numerators)):
                cell_ids = [list(df.index).index(cell) for cell in pairs[i][0]]
                cells_vec = np.zeros(P.shape[0], dtype='bool')
                cells_vec[cell_ids] = 1
                mut_id = list(df.columns).index(pairs[i][1])

                # Below are three different ways of computing p2 (the numerator is the sum of p2 * p1 over all trees)
                # The first (log_pf_cond_on_one_tree) is a slightly modified version of the way that p2 was computed for the RECOMB submission
                # The second (log_pf_cond_mat_mul) computes p2 as a ratio between two values computed using
                #           the function that computes p1 (log_prob_mat_mul_calc)
                # The third (log_pf_cond_numpy) is a translation of the first into Numpy

                # log_p2 = log_pf_cond_on_one_tree(P, subtrees, cells_vec, mut_id)
                # log_p2 = log_pf_cond_mat_mul(P, subtrees, cells_vec, mut_id)
                log_p2 = log_pf_cond_numpy(P, subtrees, cells_vec, mut_id)

                numerators[i] += 2 ** Decimal(log_p1 + log_p2 - log_sampling_prob)

    # output partition function value for each clade,mutation pair, along with the inputted arguments
    print("\t".join(["matrix","trees","fp_rate","fn_rate","clade","mutation","numerator","denominator","p"]))
    for i,numerator in enumerate(numerators):
        info = map(str, [path_matrix, path_trees, alpha, beta, ",".join(sorted(pairs[i][0])), pairs[i][1], numerator, denominator, np.float64(numerator / denominator)])
        print("\t".join(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partition function')

    #required args
    parser.add_argument("-i", "--input_matrix", type=str,                                                        
                        help="Path to input genotype matrix where rows correspond to cells/sublines and columns correspond to mutations. See repo examples for formatting.", required=True)
    parser.add_argument("-t", "--trees", type=str,                                                        
                        help="Path to file of trees generated by sampler.py", required=True)
    parser.add_argument("-fp", "--alpha", type=float,                                                        
                        help="False-positive rate (alpha in the paper)", required=True)
    parser.add_argument("-fn", "--beta", type=float,                                                        
                        help="False-negative rate (beta in the paper)", required=True)
    
    #optional args
    parser.add_argument("-g", "--gpu",
                        help="Runs probability computations on GPUs", action='store_true')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", "--list", type=str,   
                        help="Path to file containing clades and mutations to compute partition function for. See GitHub repo example for format")
    group.add_argument("-sm", "--scoring_matrix", type=str,   
                        help="Path to binary matrix of same dimensions, row labels, and column labels as the input matrix. The genotype of each ")
    group.add_argument("-cm", "--clade_mut_pair", nargs=2, metavar=("CLADE", "MUTATION"), help="Clade and mutation")
    
    args = parser.parse_args()
    if args.scoring_matrix is not None:
        partition_function(args.input_matrix, args.trees, args.alpha, args.beta, path_scoring_matrix=args.scoring_matrix, gpu=args.gpu)
    else:
        pass # TODO implement this