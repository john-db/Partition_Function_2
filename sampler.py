import argparse, os, time
import numpy as np
import pandas as pd

def main(path, num_samples, alpha, beta, out_path, seed=None):
    start = time.time()

    df = pd.read_csv(path, sep="\t", index_col=[0])
    rng = np.random.default_rng(seed=seed)
    eps=0.1
    delta=0.8
    coef=10
    divide=True
    divide_factor=10

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = df.values # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1

    n_cells = P.shape[0]
    init_subtrees = np.zeros((2 * n_cells, n_cells), dtype=np.bool_)
    #init_subtrees = np.zeros((2 * n_cells - 1, n_cells), dtype=np.bool_)
    P = np.concatenate((P, np.zeros((n_cells - 1, P.shape[1]), dtype=P.dtype)))
    np.fill_diagonal(init_subtrees, True) # Add 1s for the singleton subtrees

    # Creation of initial distance matrix
    dist = np.full((P.shape[0], P.shape[0]), np.nan, dtype=np.float64)
    for row in range(n_cells):
        dist[row] = np.sqrt(np.sum((np.broadcast_to(P[row], shape=P.shape) - P) ** 2, axis=1)) - coef * np.sum(np.minimum(np.broadcast_to(P[row], shape=P.shape), P), axis=1)
    dist[:, np.isin(np.arange(dist.shape[1]), np.arange(n_cells), assume_unique=True, invert=True)] = np.nan # check if assume_unique=True and invert (as opposed to ~) speed this up? it seems that they dont from a small test
    np.fill_diagonal(dist[:n_cells], np.nan)

    lines = [None] * num_samples
    for i in range(num_samples):
        subtrees, prob_sequence, correction = draw_sample_bclt(P.copy(), init_subtrees.copy(), dist.copy(), n_cells, eps=eps, delta=delta, coef=coef, divide=divide, divide_factor=divide_factor, rng=rng)
        # Stores the probability that the tree was sampled followed by the number of subtrees in the array, and then the subtrees linearized
        lines[i] = [str(prob_sequence / correction), str(subtrees.shape[0]), "".join(map(lambda x: str(int(x)), subtrees.flatten()))]
                
    with open(out_path, "x") as f:
        f.write(str(num_samples))
        for ls in lines:
            f.write("\n" + " ".join(ls))

    print("Sampling finished in: " + str(start - time.time()))
    

def draw_sample_bclt(P, subtrees, dist, n_cells, eps=0.1, delta=0.8, coef=10, divide=False, divide_factor=10, rng=None):
    """
    Parameters:
        P: the matrix of n rows/cells and m mutations/columns with
            P_i,j = Pr[X_i,j = 1]
            eps, delta, coef, divide/divide_factor: hyperparameters (TODO: describe these)
            rng: Numpy random number generator object
    Returns:
        subtrees: matrix of subtrees representing the sampled BCLT
        prior_prob: the probability of that sequence of subtree merges in the sampling
        norm_factor: the correction factor (to be used with prior_prob to find the probability of that tree topology being sampled)
    """

    subtrees, prior_prob, norm_factor = sample_rec(P, subtrees, dist, n_cells, 0, np.arange(n_cells, dtype=int), eps=eps, delta=delta, coef=coef, divide=divide, divide_factor=divide_factor, rng=rng)
    return subtrees, prior_prob, norm_factor
    #return subtrees[n_cells:-1], prior_prob, norm_factor # the [n_cells:-1] removes the trivial subtrees (singletons) at the beginning and the trivial subtrees (root) at the end

def sample_rec(P, subtrees, dist, n_cells, iter, current_indices, eps, delta, coef, divide, divide_factor, rng=None):
    """
    Recursive function for the bottom-up sampling of trees. The arrays (P, subtrees, dist) have been initialized to have enough space to contain
    all values that they will during the execution of the function, and current_indices keeps track of indices of these arrays
    are currently being used.

    Returns:
        subtrees: matrix of subtrees representing the sampled BCLT
        prior_prob: the probability of that sequence of subtree merges in the sampling
        norm_factor: the correction factor (to be used with prior_prob to find the probability of that tree topology being sampled)
    """
    
    #base case
    if current_indices.size == 1:
        return subtrees, np.float64(1), np.float64(1)

    
    # Depending on the number of mutations, the distance values can be quite large in magnitude. This causes the
    # probability distribution created by taking the softmax to be extremely skewed (i.e. one value from the distance matrix
    # may have probability almost equal to 1)
    # We have two approaches normalizing the matrix to make the resulting distribution less skewed below:
    prob = dist - np.nanmin(dist)
    if divide:
        #divide to normalize distances to have max equal to divide_factor
        if np.nanmax(prob) != 0:
            prob *= -(divide_factor / np.nanmax(prob))
    else:
        # this effectively changes the base of the softmax to 1+eps
        prob *= -np.log(1 + eps)
    
    # Softmax
    np.nan_to_num(prob, nan= -np.inf, copy=False)
    prob = np.exp(prob)
    prob *= 1 / np.sum(prob)

    # Draw a pair from the distribution, and find which "current_indices" value it corresponds to
    ind = rng.choice(len(prob.flat), p=prob.flat)
    pair = np.unravel_index(ind, prob.shape)
    pair = np.searchsorted(current_indices, pair)

    # Update the arrays: 
    # create a new subtree that is the result of merging the pair of subtrees that were selected
    subtrees[n_cells + iter] = subtrees[current_indices[np.min(pair)]] + subtrees[current_indices[np.max(pair)]] #merge the 2 subtrees
    # create a new row of probabilities that each cell in the new subtree has each mutation
    P[n_cells + iter] = delta * np.minimum(P[current_indices[np.min(pair)]], P[current_indices[np.max(pair)]]) + (1 - delta) * np.maximum(P[current_indices[np.min(pair)]], P[current_indices[np.max(pair)]]) #add the new row to the matrix P
    # create a new row and column in the distance matrix for this new subtree
    dist[n_cells + iter] = np.sqrt(np.sum((np.broadcast_to(P[n_cells + iter], shape=P.shape) - P) ** 2, axis=1)) - coef * np.sum(np.minimum(np.broadcast_to(P[n_cells + iter], shape=P.shape), P), axis=1)
    dist[:, n_cells + iter] = dist[n_cells + iter]
    
    # store the values we are removing from current indices for later
    removed = (current_indices[np.min(pair)], current_indices[np.max(pair)]) #, current_indices[-2])

    # shift current indices in order to remove the indices of the pair of selected subtrees
    current_indices[np.min(pair):-1] = current_indices[np.min(pair) + 1:]
    current_indices[np.max(pair) - 1:-1] = current_indices[np.max(pair):]
    # add a new entry for the new subtree
    current_indices[-2] = n_cells + iter

    # remove the entries in the new row/column that correspond to subtrees that are no longer "current"
    dist[n_cells + iter, np.isin(np.arange(dist.shape[1]), current_indices[:-1], assume_unique=True, invert=True)] = np.nan
    dist[np.isin(np.arange(dist.shape[1]), current_indices[:-1], assume_unique=True, invert=True), n_cells + iter] = np.nan
    dist[n_cells + iter, n_cells + iter] = np.nan # remove diagonal entry

    # remove the entries corresponding to the selected subtrees from the distance matrix
    dist[removed[0], :] = np.nan
    dist[:, removed[0]] = np.nan
    dist[removed[1], :] = np.nan
    dist[:, removed[1]] = np.nan

    subtrees, prior_prob, norm_factor = sample_rec(P, subtrees, dist, n_cells, iter + 1, current_indices[:-1], eps, delta, coef, divide, divide_factor, rng)
    
    # Reconstructs the state of current_indices from before the recursive call so we can accumulate probability
    # of making progress towards the final tree
    # current_indices[-2] = removed[2] # it works without this line, why?
    current_indices[np.max(pair):] = current_indices[np.max(pair) - 1:-1]
    current_indices[np.min(pair) + 1:] = current_indices[np.min(pair):-1]
    current_indices[np.min(pair)] = removed[0]
    current_indices[np.max(pair)] = removed[1]
    
    # Sum the probability of choosing any pair of subtrees to merge that would make progress towards the final tree
    accum = 0
    for i in range(len(prob.flat)):
        if prob.flat[i] == 0:
            continue

        pair_i = np.unravel_index(i, prob.shape)
        pair_i = np.searchsorted(current_indices, pair_i)
        st1 = subtrees[current_indices[np.min(pair_i)]]
        st2 = subtrees[current_indices[np.max(pair_i)]]

        if any(np.equal(subtrees, st1 + st2).all(1)):
            accum += prob.flat[i]

    q_i = prob.flat[ind] / accum

    return subtrees, prior_prob * prob.flat[ind], norm_factor * q_i
    
if __name__ == "__main__":
    #TODO Add checks to make sure inputted args make sense
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
    if os.path.exists(args.output):
        raise FileExistsError("There is already a file at the output path")
    if args.alpha < 0 or args.alpha > 1:
        raise Exception("False positive probability must be in the interval [0.0, 1.0]")
    if args.beta < 0 or args.beta > 1:
        raise Exception("False negative probability must be in the interval [0.0, 1.0]")
    if args.num_samples < 1:
        raise Exception("Number of samples must be a positive integer")
    main(args.input_matrix, args.num_samples, args.alpha, args.beta, args.output, args.seed)