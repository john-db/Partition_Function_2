import argparse, os, time, sys
import numpy as np
import pandas as pd
import cupy as cp
from IPython import embed
import pyarrow as pa
import pyarrow.parquet as pq


#Global constants
rng = np.random.default_rng(seed=42)
rng_cp = cp.random.default_rng(seed=42)

eps=0.1
delta=0.8
coef=10
divide=True
divide_factor=10
batch_size=100

def main(path, num_samples, alpha, beta, out_path, seed=None):

    start = time.time()

    df = pd.read_csv(path, sep="\t", index_col=[0])

    # Create a Table schema with appropriate field names and data types
    bt_size = 2 * df.shape[0] ** 2
    schema = pa.schema([
        pa.field("sampling_prob", pa.float64()),
        pa.field("num_subtrees", pa.int32()),
        pa.field("binary_tree", pa.list_(pa.bool_(), bt_size))
    ])

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

    i=0
    with pq.ParquetWriter(out_path, schema=schema) as writer:

        P_stacked = cp.tile(P[:, :, np.newaxis], (1, 1, batch_size))
        init_subtrees_stacked = cp.tile(init_subtrees[:, :, np.newaxis], (1, 1, batch_size))
        dist_stacked = cp.tile(dist[:, :, np.newaxis], (1, 1, batch_size))

        while i < num_samples:

            #call recursive function
            subtrees, prob_sequence, correction = draw_sample_bclt_cp(P_stacked.copy(), init_subtrees_stacked.copy(), dist_stacked.copy(), n_cells)

            #write to parquet
            table = pa.Table.from_arrays([
                                            pa.array((prob_sequence/correction).get(), type=pa.float64()),                     #sampling prob
                                            pa.array(np.full(batch_size, subtrees.shape[0]), type=pa.int32()),                 #number of subtrees
                                            pa.FixedSizeListArray.from_arrays(pa.array(subtrees.flatten().get()), bt_size)     #binary trees
                                        ], schema=schema)
            writer.write_table(table)

            i+= batch_size


    print(f"Sampled {i} subtrees: " + str(time.time() - start) + " seconds")

    cp.cuda.Device(0).synchronize()

def draw_sample_bclt_cp(P, subtrees, dist, n_cells):
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

    temp_rng = [np.random.default_rng(seed=i) for i in range(batch_size)]

    subtrees, prior_prob, norm_factor = sample_rec_cp(P, subtrees, dist, n_cells, 0, cp.tile(np.arange(n_cells, dtype=int), (batch_size, 1)))
    return subtrees, prior_prob, norm_factor
    #return subtrees[n_cells:-1], prior_prob, norm_factor # the [n_cells:-1] removes the trivial subtrees (singletons) at the beginning and the trivial subtrees (root) at the end


def sample_rec_cp(P, subtrees, dist, n_cells, iter, current_indices):
    """
    Recursive function for the bottom-up sampling of trees. The arrays (P, subtrees, dist) have been initialized to have enough space to contain
    all values that they will during the execution of the function, and current_indices keeps track of indices of these arrays
    are currently being used.

    Returns:
        subtrees: matrix of subtrees representing the sampled BCLT
        prior_prob: the probability of that sequence of subtree merges in the sampling
        norm_factor: the correction factor (to be used with prior_prob to find the probability of that tree topology being sampled)
    """

    print(iter, end=" ", flush=True)

    #base case
    if current_indices.shape[1] == 1:
        return subtrees, cp.full(current_indices.shape[0], 1, dtype=np.float64), cp.full(current_indices.shape[0], 1, dtype=np.float64)


    # Depending on the number of mutations, the distance values can be quite large in magnitude. This causes the
    # probability distribution created by taking the softmax to be extremely skewed (i.e. one value from the distance matrix
    # may have probability almost equal to 1)
    # We have two approaches normalizing the matrix to make the resulting distribution less skewed below:
    prob = dist - cp.nanmin(dist, axis=(0, 1))[cp.newaxis, cp.newaxis, :]
    if divide:
        #divide to normalize distances to have max equal to divide_factor
        prob *= cp.where(
            (m := cp.nanmax(prob, axis=(0, 1))) != 0,
            -(divide_factor / m),
            1
        )[cp.newaxis, cp.newaxis, :]

    else:
        # this effectively changes the base of the softmax to 1+eps
        prob *= -cp.log(1 + eps)

    # Softmax
    cp.nan_to_num(prob, nan= -np.inf, copy=False) # TODO: try removing this by replacing numpy functions with their nan friendly counterparts
    prob = cp.exp(prob)
    prob *= 1 / cp.sum(prob, axis=(0,1))



    # Draw a pair from the distribution, and find which "current_indices" value it corresponds to

    #flatten each batch
    prob_flat = prob.reshape(prob.shape[0] * prob.shape[1], prob.shape[2])

    #apply choise batch wise

    ind = cp.apply_along_axis(lambda x: cp.random.choice(len(x), size=1, p=x), axis=0, arr=prob_flat)
    #### TEMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
    # ind = cp.array([[temp_rng[i].choice(len(prob_flat.get()[:,i]), p=prob_flat.get()[:,i]) for i in range(prob_flat.get().shape[1])]])
    #### TEMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

    pair = cp.unravel_index(ind, (prob.shape[0],prob.shape[1]))
    pair = cp.stack([pair[0][0], pair[1][0]]).T
    pair = cp.sort(pair, axis=1)

    # pair_current = cp.apply_along_axis(lambda x: cp.searchsorted(current_indices, x), axis=0, arr=pair)
    pair_current = cp.array([cp.searchsorted(current_indices[i], pair[i]) for i in range(current_indices.shape[0])])

    # Update the arrays:
    # create a new subtree that is the result of merging the pair of subtrees that were selected
    subtrees[n_cells + iter, :, cp.arange(subtrees.shape[2])] = subtrees[pair[:, 0], :, cp.arange(subtrees.shape[2])] + subtrees[pair[:, 1], :, cp.arange(subtrees.shape[2])] #merge the 2 subtrees

    # create a new row of probabilities that each cell in the new subtree has each mutation
    # P[n_cells + iter] = delta * np.minimum(P[pair[0]], P[pair[1]]) + (1 - delta) * np.maximum(P[pair[0]], P[pair[1]]) #add the new row to the matrix P
    P[n_cells + iter, :, cp.arange(subtrees.shape[2])] = delta * cp.minimum( P[pair[:, 0], :, cp.arange(P.shape[2])] ,  P[pair[:, 1], :, cp.arange(P.shape[2])]) + (1 - delta) * cp.maximum( P[pair[:, 0], :, cp.arange(P.shape[2])] ,  P[pair[:, 1], :, cp.arange(P.shape[2])]) #add the new row to the matrix P

    # create a new row and column in the distance matrix for this new subtree
    dist[n_cells + iter] = cp.sqrt(cp.sum((P[n_cells + iter] - P) ** 2, axis=1)) #- coef * cp.sum(cp.minimum(P[n_cells + iter], P), axis=1)
    dist[n_cells + iter] -=coef * cp.sum(cp.minimum(P[n_cells + iter], P), axis=1)
    dist[:, n_cells + iter, :] = dist[n_cells + iter, :, :]

    # store the values we are removing from current indices for later
    removed = (pair[:, 0], pair[:, 1]) #, current_indices[-2])

    # shift current indices in order to remove the indices of the pair of selected subtrees
    # current_indices[pair_current[0]:-1] = current_indices[pair_current[0] + 1:]
    for i in range(current_indices.shape[0]): #currently limited to for loop due to "ragged indexing" or write custom kernel in the future
        current_indices[i, pair_current[i, 0]:-1] = current_indices[i, pair_current[i, 0] + 1:]
        current_indices[i, pair_current[i, 1] - 1:-1] = current_indices[i, pair_current[i, 1]:]

    # add a new entry for the new subtree
    current_indices[:,-2] = n_cells + iter

    # remove the entries in the new row/column that correspond to subtrees that are no longer "current"
    # dist[n_cells + iter, np.isin(np.arange(dist.shape[1]), current_indices[:-1], assume_unique=True, invert=True)] = np.nan

    for i in range(dist.shape[2]):
        mask = cp.isin(cp.arange(dist.shape[1]), current_indices[i, :-1], assume_unique=True, invert=True)
        dist[n_cells + iter, mask, i] = cp.nan
        dist[mask, n_cells + iter, i] = cp.nan

    dist[n_cells + iter, n_cells + iter,:] = np.nan # remove diagonal entry

    # remove the entries corresponding to the selected subtrees from the distance matrix
    dist[removed[0], :, cp.arange(dist.shape[2])] = np.nan
    dist[:, removed[0], cp.arange(dist.shape[2])] = np.nan
    dist[removed[1], :, cp.arange(dist.shape[2])] = np.nan
    dist[:, removed[1], cp.arange(dist.shape[2])] = np.nan



    subtrees, prior_prob, norm_factor = sample_rec_cp(P, subtrees, dist, n_cells, iter + 1, current_indices[:, :-1])
    start_full = time.time()

    # Reconstructs the state of current_indices from before the recursive call so we can accumulate probability
    # of making progress towards the final tree
    # current_indices[-2] = removed[2] # it works without this line, why?
    for i in range(dist.shape[2]):
        current_indices[i, pair_current[i,1] : ] = current_indices[i, pair_current[i,1] - 1 : -1]
        current_indices[i, pair_current[i,0] + 1 : ] = current_indices[i, pair_current[i,0] : -1]

    current_indices[cp.arange(dist.shape[2]), pair_current[:,0] ] = removed[0]
    current_indices[cp.arange(dist.shape[2]), pair_current[:,1] ] = removed[1]


    ################## BEGIN WORKING BUT SLOW ###################

    # Sum the probability of choosing any pair of subtrees to merge that would make progress towards the final tree
    # start = time.time()
    for i in range(dist.shape[2]):

        rows, cols = cp.where(prob[:,:,i]!=0)
        structured_coords = cp.stack((rows, cols), axis=1) #np.array(list(zip(rows, cols)))

        def merge_helper(x):
            pair_i = cp.searchsorted(current_indices[i, :], x)
            st1 = subtrees[:,:,i][current_indices[i, :][pair_i[0]]]
            st2 = subtrees[:,:,i][current_indices[i, :][pair_i[1]]]

            return st1+st2

        # x_time = time.time()
        x = cp.apply_along_axis(merge_helper, axis=1, arr=structured_coords)
        # print(f"X {time.time() - x_time} seconds")

        # print(rows.shape, cols.shape, structured_coords.shape, x.shape, ind)

        y = cp.any(cp.all(x[:, cp.newaxis, :] == subtrees[:,:,i][cp.newaxis, :, :], axis=2), axis=1)
        structured_coords_masked = structured_coords[y]
        accum = cp.sum(prob[structured_coords_masked[:,0],structured_coords_masked[:, 1],i])

        norm_factor[i] *= prob[:,:,i][tuple(pair[i])]  / accum
        prior_prob[i]  *= prob[:,:,i][tuple(pair[i])]

    # print(f"{time.time() - start} seconds")
    # print(f"Full {time.time() - start_full} seconds")

################## END WORKING BUT SLOW ###################

    print(iter, end=" ", flush=True)

    # print(prior_prob, norm_factor)
    return subtrees, prior_prob, norm_factor


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
    # if os.path.exists(args.output):
    #     raise FileExistsError("There is already a file at the output path")
    if args.alpha < 0 or args.alpha > 1:
        raise Exception("False positive probability must be in the interval [0.0, 1.0]")
    if args.beta < 0 or args.beta > 1:
        raise Exception("False negative probability must be in the interval [0.0, 1.0]")
    if args.num_samples < 1:
        raise Exception("Number of samples must be a positive integer")
    main(args.input_matrix, args.num_samples, args.alpha, args.beta, args.output, args.seed)
