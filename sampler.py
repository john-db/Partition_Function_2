import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import argparse
import pandas as pd

def main(path, num_samples, alpha, beta, out_path, seed=None):
    df = pd.read_csv(path, sep="\t", index_col=[0])

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = df.values # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1

    rng = np.random.default_rng(seed=seed)
    try:
        with open(out_path, "x") as f:
            for _ in range(num_samples):
                subtrees, prob_sequence, correction = draw_sample_bclt(P, eps=0.1, delta=0.8, coef=10, divide=True, divide_factor=10, rng=rng)
                # Print the probability that the tree was sampled followed by the (nontrivial) subtrees
                f.write(str(prob_sequence / correction) + " " + "".join(map(lambda x: str(int(x)), subtrees.flatten())) + "\n") # There is probably a better way to do this 
    except FileExistsError:
        print("The path provided for the output file already exists.")
    

def draw_sample_bclt(P, eps=0.1, delta=0.8, coef=10, divide=False, divide_factor=10, rng=None):
    """
    Parameters:
        P: the matrix of n rows/cells and m mutations/columns with
            P_i,j = Pr[X_i,j = 1]
            eps, delta, coef, divide/divide_factor: hyperparameters (TODO: describe these)
            rng: Numpy random number generator object
    Returns:
        subtrees: matrix of subtrees representing the sampled BCLT
        norm_factor: the correction factor (probability of that tree topology being sampled)
    """

    n_cells = P.shape[0]
    # We only need the subtree
    init_subtrees = np.zeros((2 * n_cells - 1, n_cells), dtype='bool')
    init_P = np.concatenate((P, np.zeros((n_cells - 1, P.shape[1]), dtype = P.dtype)))
    np.fill_diagonal(init_subtrees, True) # Add 1s for the singleton clades/subtrees

    subtrees, prob, corr = sample_rec(init_P, init_subtrees, n_cells, 0, np.arange(n_cells, dtype=int), eps=eps, delta=delta, coef=coef, divide=divide, divide_factor=divide_factor, rng=rng)
    return subtrees[n_cells:-1], prob, corr

def sample_rec(P, subtrees, n_cells, iter, rows_to_subtrees, eps, delta, coef, divide, divide_factor, rng=None):
    
    if rows_to_subtrees.size == 1:
        return subtrees, np.float64(1), np.float64(1)
    
    # TODO: implement pairwise_distances so that it does not require P to be copied
    dist = pairwise_distances(P[rows_to_subtrees], metric=(lambda a,b : np.linalg.norm(a - b) - np.sum(np.minimum(a, b)) * coef))
    dist = dist.astype(np.float64) #do we need to cast?
    np.fill_diagonal(dist, np.inf)

    dist = -dist
    dist = dist - np.max(dist.flat) # normalize
    if divide:
        #divide to normalize to max:0 min: -divide_factor

        dabs_max = 0
        for entry in dist.flat:
            if entry != float('-inf') and entry != float('inf') and abs(entry) > dabs_max:
                dabs_max = abs(entry)

        if dabs_max != 0:
            dist = -dist * (divide_factor / dabs_max)
        else:
            dist = -dist
    else:
        dist = -dist * np.log(1 + eps) # this effectively changes the base of the softmax from e to 1+eps

    prob = np.exp(-dist) / np.sum(np.exp(-dist)) # softmax
    ind = rng.choice(len(prob.flat), p=prob.flat)
    pair = np.unravel_index(ind, prob.shape)

    P[n_cells + iter] = delta * np.minimum(P[rows_to_subtrees[np.min(pair)]], P[rows_to_subtrees[np.max(pair)]]) + (1 - delta) * np.maximum(P[rows_to_subtrees[np.min(pair)]], P[rows_to_subtrees[np.max(pair)]]) #add the new row to the matrix P

    subtrees[n_cells + iter] = subtrees[rows_to_subtrees[np.min(pair)]] + subtrees[rows_to_subtrees[np.max(pair)]] #merge the 2 subtrees
    
    rows_to_subtrees_copy = rows_to_subtrees.copy()

    for i in range(len(rows_to_subtrees) - 1):
        if i >= np.min(pair):
            rows_to_subtrees[i] = rows_to_subtrees[i + 1]
    for i in range(len(rows_to_subtrees) - 1):
        if i >= np.max(pair) - 1:
            rows_to_subtrees[i] = rows_to_subtrees[i + 1]
    rows_to_subtrees = rows_to_subtrees[:-1]
    rows_to_subtrees[-1] = n_cells + iter
    subtrees, prior_prob, norm_factor = sample_rec(P, subtrees, n_cells, iter + 1, rows_to_subtrees, eps, delta, coef, divide, divide_factor, rng)
    
    
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
    parser.add_argument("-o", "--output", type=str,                                                        
                        help="Desired path for output", required=True)


    parser.add_argument("-s", "--seed", type=int,                                                        
                        help="random seed", required=False, default=None)
    #parser.add_argument("-c", "--coef", type=float,                                                        
    #                     help="coef", required=False, default=10.0)           
    #parser.add_argument("-e", "--epsilon", type=float,                                                        
    #                     help="epsilon", required=False, default=0.1) 
    #parser.add_argument("-d", "--delta", type=float,                                                        
    #                     help="delta", required=False, default=0.8)                    
    # parser.add_argument("-di", "--divide", type=bool,                                                        
    #                     help="If True, use divide normalizaion with parameter divide_factor", required=False, default=False)
    # parser.add_argument("-df", "--divide_factor", type=float,                                                        
    #                     help="divide_factor", required=False, default=10)


    args = parser.parse_args()
    main(args.input_matrix, args.num_samples, args.alpha, args.beta, args.output, args.seed)