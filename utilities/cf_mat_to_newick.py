import numpy as np
import pandas as pd

def sts_to_newick(subtrees):
    subtrees = list(subtrees)
    n_cells = len(subtrees[0])
    subtrees.append(np.ones(n_cells, dtype=np.int8))
    # for i in range(n_cells):
    #     st = np.zeros(n_cells, dtype=np.int8)
    #     st[i] = 1
    #     subtrees.append(st)
    subtrees = [np.array(x) for x in {(tuple(e)) for e in subtrees} if (sum(x) > 1 and sum(x) < n_cells)]
    subtrees.sort(key=sum)
    
    return subtrees_to_newick(subtrees, list(range(len(subtrees[0]))))

def cf_to_newick(df):
    subtrees = list(df.values.transpose())
    n_cells = len(subtrees[0])
    subtrees.append(np.ones(n_cells, dtype=np.int8))
    # for i in range(n_cells):
    #     st = np.zeros(n_cells, dtype=np.int8)
    #     st[i] = 1
    #     subtrees.append(st)
    subtrees = [np.array(x) for x in {(tuple(e)) for e in subtrees} if (sum(x) > 1 and sum(x) < n_cells)]
    subtrees.sort(key=sum)
    
    return subtrees_to_newick(subtrees, df.index)

def s_t_n_rec(ints_to_cell_id, subtrees, cells):
    if len(cells) == 1:
        return str(ints_to_cell_id[cells[0]])
    if len(subtrees) == 0:
        return ",".join([str(ints_to_cell_id[c]) for c in cells])
    
    subtrees = sorted(subtrees, key=lambda x: np.linalg.norm(x), reverse=True)
    split = subtrees[0]
    if np.all(split):
        return "(" + s_t_n_rec(ints_to_cell_id, subtrees[1:], cells) + ")"
    

    ins = []
    cells_ins = []
    outs = []
    cells_outs = []
    for i in range(len(split)):
        if split[i] == 1:
            ins += [i]
            cells_ins += [cells[i]]
        else:
            outs += [i]
            cells_outs += [cells[i]]

    subtrees_ins = []
    subtrees_outs = []
    for i in range(len(subtrees)):
        st_ins = [subtrees[i][[ins]][0]]
        st_outs = [subtrees[i][[outs]][0]]
        if np.count_nonzero(st_ins) > 1:
            subtrees_ins += st_ins
        if np.count_nonzero(st_outs) > 1:
            subtrees_outs += st_outs
    

    str_ins = s_t_n_rec(ints_to_cell_id, subtrees_ins, cells_ins)
    str_outs = s_t_n_rec(ints_to_cell_id, subtrees_outs, cells_outs)
    if str_ins == "()":
        return str_outs
    elif str_outs == "()":
        return str_ins
    else:
        return str_ins + "," + str_outs

def subtrees_to_newick(subtrees, ints_to_cell_id):
    cells = list(range(len(ints_to_cell_id)))
    return "(" + s_t_n_rec(ints_to_cell_id, subtrees, cells) + ");"

# def main():
#     path = "/Users/john/Desktop/matrices/Sep9Matrices/E-C19_and_C2C5_corrected-fp_0.0001-fn_0.075.tsv"
    
#     df = pd.read_csv(path, sep="\t", index_col=[0]).sort_values(by=["cell_id_x_mut_id"])
#     print(cf_to_newick(df))

# if __name__ == "__main__":
#     main()