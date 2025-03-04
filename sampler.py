import numpy as np
import numpy.linalg as la
from scipy.special import softmax
from sklearn.metrics.pairwise import pairwise_distances
import argparse
import pandas as pd

def main(path, num_samples, alpha, beta, seed):
    df = pd.read_csv(path, sep="\t", index_col=[0])

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = df.values # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1

    # print(P.shape[0]) # Print the number of cells so that the subtree matrix can be reconstructed
    rng = np.random.default_rng(seed=seed)
    for _ in range(num_samples):
        subtrees, prob_sequence, correction = draw_sample_bclt(P, rng=rng)
        # Print the probability that the tree was sampled followed by the subtrees
        print(str(prob_sequence / correction) + " " + "".join(map(lambda x: str(int(x)), subtrees.flatten()))) # There is probably a better way to do this 
        # Optimizations: The first n rows are always the same (they are the singleton clades) 
        # and the last two rows are always the same. (2nd to last row is all 1s, last row is all 0s)
        # So these don't need to be printed

def draw_sample_bclt(P, eps=0.1, delta=0.5, coef=2, rng=None):
    """
    Parameters:
        P: the matrix of n rows/cells and m mutations/columns with
            P_i,j = Pr[X_i,j = 1]
            eps, delta, coef: hyperparameters (TODO: describe these)
            rng: Numpy random number generator object
    Returns:
        subtrees: matrix of subtrees representing the sampled BCLT
        norm_factor: the correction factor (probability of that tree topology being sampled)
    """

    n_cells = P.shape[0]
    init_subtrees = np.zeros((2 * n_cells, n_cells), dtype='bool')
    for i in range(n_cells): # replace this with np.eye(...) to generate 1s on diagonal
        init_subtrees[i][i] = 1 #initializes singleton clades/subtrees
    return sample_rec(P.copy(), init_subtrees, n_cells, 0, list(range(n_cells)), eps=eps, delta=delta, coef=coef, rng=rng)

def sample_rec(P, subtrees, n_cells, iter, rows_to_subtrees, eps=0.1, delta=0.5, coef=2, rng=None):
    
    if P.shape[0] == 1:
        return subtrees, np.float64(1), np.float64(1)
    
    dist = pairwise_distances(P, metric=(lambda a,b : la.norm(a - b) - row_leafness_score(a, b) * coef))
    dist = dist.astype(np.float64) #do we need to cast?
    np.fill_diagonal(dist, np.inf)

    dist = -dist
    dist = dist - np.max(dist.flat) # normalize
    dist = -dist * np.log(1 + eps) # this effectively changes the base of the softmax from e to 1+eps

    prob = softmax(-dist).astype(np.float64) #do we need to cast?
    
    ind = None
    pair = None
    if rng == None:
        np.random.seed(seed=None)
        ind = np.random.choice(len(prob.flat), p=prob.flat)
        pair = np.unravel_index(ind, prob.shape)
    else:
        ind = rng.choice(len(prob.flat), p=prob.flat)
        pair = np.unravel_index(ind, prob.shape)

    new_row = delta * np.minimum(P[pair[0]], P[pair[1]]) + (1 - delta) * np.maximum(P[pair[0]], P[pair[1]])
    P = np.delete(P, pair, axis=0)  # remove two rows
    # P = np.concatenate((P[:np.min(pair)], P[np.min(pair) + 1 :np.max(pair)], P[np.max(pair) + 1:]), axis=0)
    
    # P_new = np.append(
    #     P_new, new_row.reshape(1, -1), axis=0
    # )
    P = np.append(
        P, new_row.reshape(1, -1), axis=0
    )

    subtrees[n_cells + iter] = subtrees[rows_to_subtrees[np.min(pair)]] + subtrees[rows_to_subtrees[np.max(pair)]] #merge the 2 subtrees
    
    rows_to_subtrees_copy = [x for x in rows_to_subtrees]

    for i in range(len(rows_to_subtrees) - 1):
        if i >= np.min(pair):
            rows_to_subtrees[i] = rows_to_subtrees[i + 1]
    for i in range(len(rows_to_subtrees) - 1):
        if i >= np.max(pair) - 1:
            rows_to_subtrees[i] = rows_to_subtrees[i + 1]
    rows_to_subtrees = rows_to_subtrees[:-1]
    rows_to_subtrees[-1] = n_cells + iter
    subtrees, prior_prob, norm_factor = sample_rec(P, subtrees, n_cells, iter + 1, rows_to_subtrees, eps, delta, coef, rng)
    
    
    prior_prob = prior_prob * prob[pair]
    
    accum = 0
    for i in range(len(prob.flat)):
        pair_i = np.unravel_index(i, prob.shape)
        st1 = subtrees[rows_to_subtrees_copy[np.min(pair_i)]]
        st2 = subtrees[rows_to_subtrees_copy[np.max(pair_i)]]

        if any(np.equal(subtrees, st1 + st2).all(1)):
            accum += prob.flat[i]

    q_i = prob.flat[ind] / accum
    norm_factor = norm_factor * q_i

    return subtrees, prior_prob, norm_factor

def row_leafness_score(row_a, row_b):
    return np.sum(np.minimum(row_a, row_b))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run.py')

    parser.add_argument("-i", "--input_matrix", type=str,                                                        
                        help="Path to input genotype matrix where rows correspond to cells/sublines and columns correspond to mutations. See repo examples for formatting.", required=True)
    parser.add_argument("-n", "--num_samples", type=int,                                                        
                        help="Number of trees to be sampled", required=True)
    parser.add_argument("-fp", "--alpha", type=float,                                                        
                        help="False-positive rate (alpha in the paper)", required=True)
    parser.add_argument("-fn", "--beta", type=float,                                                        
                        help="False-negative rate (beta in the paper)", required=True)
    parser.add_argument("-s", "--seed", type=int,                                                        
                        help="random seed", required=False, default=None)
    
    args = parser.parse_args()
    main(args.input_matrix, args.num_samples, args.alpha, args.beta, args.seed)