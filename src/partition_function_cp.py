import argparse, time, os, sys
import pandas as pd
import numpy as np
import cupy as cp
from decimal import Decimal
from tree_scorer import log_prob_mat_mul_calc_cp, log_pf_cond_numpy_cp # log_pf_cond_mat_mul, log_pf_cond_on_one_tree
from utilities.cf_mat_to_newick import sts_to_newick
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from IPython import embed
from pathlib import Path
import concurrent.futures

batch_size = None

def compute_estimates_orig(df, pairs, trees, log_sampling_probabilities, alpha, beta):

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = df.values # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1
    logP1 = np.log2(P)
    logP0 = np.log2(1 - P)


    best_tree = None
    best_score = None
    for i in range(len(trees)):
        tree = trees[i]
        log_sampling_prob = log_sampling_probabilities[i]

        numerators = np.full(len(pairs), Decimal(0), dtype=object)
        denominator = Decimal(0)


        # the denominator is the same for each tree clade/mutation pair with the given sample of trees,
        # so we only compute this once
        log_p1 = log_prob_mat_mul_calc(logP1, logP0, tree)
        if best_score == None or log_p1 > best_score:
            best_score = log_p1
            best_tree = i
        denominator += 2 ** Decimal(log_p1 - log_sampling_prob)

        # for each clade/mutation pair to be evaluated, we compute its numerator
        for j in range(len(numerators)):
            cell_ids = [list(df.index).index(cell) for cell in pairs[j][0]] #vectorize this with Numpy?
            cells_vec = np.zeros(P.shape[0], dtype=np.bool_)
            cells_vec[cell_ids] = 1
            mut_id = list(df.columns).index(pairs[j][1])

            log_p2 = log_pf_cond_numpy(logP1, logP0, tree, cells_vec, mut_id)

            numerators[j] += 2 ** Decimal(log_p1 + log_p2 - log_sampling_prob)

    return numerators, denominator, trees[best_tree], best_score

def batcher(iterables, n=1):
    l = set([len(it) for it in iterables])
    if len(l) != 1:
        print("Iterables are not of same size. Exiting.")
        sys.exit(1)
    l = list(l)[0]
    for ndx in range(0, l, n):
        yield [iterable[ndx:min(ndx + n, l)] for iterable in iterables], (ndx,min(ndx + n, l))

def distributed_denominator(log_p1s, log_sampling_probabilities):
    return sum(2 ** Decimal(log_p1s[i] - log_sampling_probabilities[i]) for i in range(len(log_p1s)))

def compute_log_p1_cp(df, trees, log_sampling_probabilities, alpha, beta):

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = cp.array(df.values) # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1
    logP1 = cp.log2(P)
    logP0 = cp.log2(1 - P)


    #preallocate
    log_p1s = cp.empty(trees.shape[0])
    best_tree = None
    best_score = None
    for start in range(0, trees.shape[0], batch_size):
        tree_batch = cp.array(trees[start:start+batch_size])

        # # the denominator is the same for each tree clade/mutation pair with the given sample of trees,
        # # so we only compute this once
        log_p1s[start:start+batch_size] = log_prob_mat_mul_calc_cp(logP1, logP0, tree_batch)


    best_score = log_p1s.max()
    best_tree  = log_p1s.argmax()

    log_p1s_cpu = log_p1s.get()

    #preocess denominator computation in parallel
    num_cpus=max(1, int(int(os.getenv('SLURM_CPUS_PER_TASK', 1)*0.75)))

    chunk_size = int(len(log_p1s) / num_cpus )+1
    #print(f"Using {num_cpus} threads and a chunk size of {chunk_size}")

    tasks = []
    denominator = Decimal(0)
    with concurrent.futures.ProcessPoolExecutor(num_cpus) as executor:
        #process in batches
        for [batch_log_p1s, batch_log_sampling_probabilities], idxs in batcher([log_p1s_cpu, log_sampling_probabilities], chunk_size):
            tasks.append( executor.submit(distributed_denominator, batch_log_p1s, batch_log_sampling_probabilities) )

        for i,task in enumerate(concurrent.futures.as_completed(tasks)):
            if i%1 == 0:
                # clear_output()
                print(f"\rProcessed {i+1}/{len(tasks)} batches", end='', flush=True)

            denominator += task.result()

        print()

    return log_p1s_cpu, denominator, trees[best_tree.item()], best_score

def distributed_numerator(log_p1s, log_p2s, log_sampling_probabilities):
    return sum(2 ** Decimal(log_p1s[i] + log_p2s[i] - log_sampling_probabilities[i]) for i in range(len(log_p2s)))

def compute_estimates(df, pairs, trees, log_p1s, log_sampling_probabilities, alpha, beta):

    # Here we create the matrix representing the probability distribution of the ground truth,
    #   i.e. the entry i,j is the probability that entry i,j of the ground truth matrix equals 1,
    #   given the observed input genotype matrix stored in df
    I_mtr = cp.array(df.values) # I_mtr is the observed genotype matrix (0 if a mutation was called as absent, 1 if a mutation was present, 3 represents missing data)
    t1 = I_mtr * (1 - beta) / (alpha + 1 - beta) # If a 1 was observed, then the probability that the ground truth is 1 is equal to (1 - beta) / (alpha + 1 - beta)
    t2 = (1 - I_mtr) * beta / (beta + 1 - alpha) # If a 0 was observed, then the probability that the ground truth is 1 is equal to beta / (beta + 1 - alpha)
    P = t1 + t2
    P[I_mtr == 3] = 0.5                          # if a 3 (N/A entry) is observed we assume that there is a 50% probability that the entry is a 1
    logP1 = cp.log2(P)
    logP0 = cp.log2(1 - P)


    numerators = np.full(len(pairs), Decimal(0), dtype=object)

    # for each clade/mutation pair to be evaluated, we compute its numerator
    for j in range(len(numerators)):
        cell_ids = [list(df.index).index(cell) for cell in pairs[j][0]] #vectorize this with Numpy?
        cells_vec = cp.zeros(P.shape[0], dtype=np.bool_)
        cells_vec[cell_ids] = 1
        mut_id = list(df.columns).index(pairs[j][1])


        log_p2s = cp.empty(trees.shape[0])
        batch_size = 1000
        for start in range(0, trees.shape[0], batch_size):
            tree_batch = cp.array(trees[start:start+batch_size])

            log_p2s[start:start+batch_size] = log_pf_cond_numpy_cp(logP1, logP0, tree_batch, cells_vec, mut_id)

        #preocess denominator computation in parallel
        num_cpus = max(1, int(int(os.getenv('SLURM_CPUS_PER_TASK', 1)) * 0.75))

        chunk_size = int(len(log_p2s) / num_cpus )+1
        print(f"Using {num_cpus} threads and a chunk size of {chunk_size}")

        tasks = []
        with concurrent.futures.ProcessPoolExecutor(num_cpus) as executor:
            #process in batches
            for [batch_log_p1s, batch_log_p2s, batch_log_sampling_probabilities], idxs in batcher([log_p1s, log_p2s.get(), log_sampling_probabilities], chunk_size):
                tasks.append( executor.submit(distributed_numerator, batch_log_p1s, batch_log_p2s, batch_log_sampling_probabilities) )

            for i,task in enumerate(concurrent.futures.as_completed(tasks)):
                if i%1 == 0:
                    # clear_output()
                    print(f"\rProcessed {i+1}/{len(tasks)} batches", end='', flush=True)

                numerators[j] += task.result()
            print()

        print(f"Done for clade {j}")

    return numerators

def read_trees(path_trees, num_cells):
    # The input file of trees contains on each line a sampling probability corresponding to a tree,
    # and the clades/subtrees of that tree concatenated to a single line.
    # The two are separated by a space character
    # We must reconstruct the subtrees matrix, do to that we need to include the trivial clades/subtrees
    # (i.e. the singletons and the subtree containing all leaves), and also a row representing the possibility
    # that a given mutation is not present in any cell of the tree (row of all zeros)

    #filter corrupt parquet files
    valid_files = []
    for file in Path(path_trees).iterdir():
        if file.is_file():
            try:
                pq.read_metadata(file)
                valid_files.append(file)
            except Exception as e:
                print(f"Skipping {file}")

    #read tables in as one large dataset
    dataset = ds.dataset(valid_files, format="parquet")
    batches = dataset.to_batches(batch_size=100)

    #preallocate
    num_rows = dataset.count_rows()
    binary_tree_size = dataset.schema.field("binary_tree").type.list_size
    tree_rows = int(binary_tree_size/num_cells)
    tree_cols = num_cells

    trees = np.empty([num_rows, tree_rows, tree_cols])
    log_sampling_probabilities = np.empty(num_rows)

    current_i = 0
    for batch in batches:

        log_sampling_probabilities[current_i:current_i+len(batch)] = np.log2(batch['sampling_prob'].to_numpy())

        binary_trees = np.stack(batch['binary_tree'].to_numpy(zero_copy_only=False))
        # binary_trees = binary_trees.reshape(binary_trees.shape[0], tree_rows, tree_cols)

        ###############TEMPORARY FIX TO GET THE RIGHT TREES FROM BUG IN SAMPLER #####################
        binary_trees = binary_trees.reshape(tree_rows,tree_cols,100).transpose(2, 0, 1)
        ###############TEMPORARY FIX TO GET THE RIGHT TREES FROM BUG IN SAMPLER #####################

        trees[current_i:current_i+len(batch)] = binary_trees

        current_i += len(batch)

    return trees, log_sampling_probabilities

def partition_function(path_matrix, path_trees, alpha, beta, output, clade=None, mutation=None, path_list=None, path_scoring_matrix=None, gpu=False):
    df = pd.read_csv(path_matrix, sep="\t", index_col=[0])
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

    start = time.time()
    trees, log_sampling_probabilities = read_trees(path_trees, df.shape[0])
    middle = time.time()
    log_p1s, denominator, tree, score = compute_log_p1_cp(df, trees, log_sampling_probabilities, alpha, beta)
    numerators = compute_estimates(df, pairs, trees, log_p1s, log_sampling_probabilities, alpha, beta)
    end = time.time()

    # output partition function value for each clade,mutation pair, along with the inputted arguments
    try:
        with open(args.output, "x") as file:
            file.write("\t".join(["matrix","trees","fp_rate","fn_rate","clade","mutation","numerator","denominator","p"]))
            for i,numerator in enumerate(numerators):
                info = map(str, [path_matrix, path_trees, alpha, beta, ",".join(sorted(pairs[i][0])), pairs[i][1], numerator, denominator, np.float64(numerator / denominator)])
                file.write("\n" + "\t".join(info))
    except FileExistsError:
        print("The path provided for the output file already exists.")
    print("The best tree scored was: " + str(sts_to_newick(tree, df.index)))
    print("with a score of: " + str(score))
    print("Parsing trees finished in: " + str(middle - start) + " seconds")
    print("Computing estimates finished in: " + str(end - middle) + " seconds")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partition function')

    #required args
    parser.add_argument("-i", "--input_matrix", type=str,
                        help="Path to input genotype matrix where rows correspond to cells/sublines and columns correspond to mutations. See repo examples for formatting.", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="Path for output file", required=True)
    parser.add_argument("-t", "--trees", type=str,
                        help="Path to file of trees generated by sampler.py", required=True)
    parser.add_argument("-fp", "--alpha", type=float,
                        help="False-positive rate (alpha in the paper)", required=True)
    parser.add_argument("-fn", "--beta", type=float,
                        help="False-negative rate (beta in the paper)", required=True)
    parser.add_argument("-b", "--batch_size", type=int,
                        help="batch size", required=True)

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
        batch_size = args.batch_size
        partition_function(args.input_matrix, args.trees, args.alpha, args.beta, args.output, path_scoring_matrix=args.scoring_matrix, gpu=args.gpu)
    else:
        pass # TODO implement this
