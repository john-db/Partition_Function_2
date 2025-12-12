
This implementation uses relies on CuPy (<https://cupy.dev/>), which implements NumPy to run on GPUs using CUDA. Make sure that CUDA has been loaded otherwise the programs will not run. On our compute cluster we must run:

`module load CUDA/12.1`

Here is an example of using the implementation to compute bipartition function estimates. We will run commands from the top directory.

Sampling trees:

`mkdir samples`
We will tell the sampler to output our samples into the samples directory.

`python ./src/sampler_cp.py -i ./example/input_genotype_matrix.tsv -n 500 -fp 0.001 -fn 0.1 -s 0 -b 100 -o ./samples/sample1.parquet`

We can collect additional samples by changing the random seed:

`python ./src/sampler_cp.py -i ./example/input_genotype_matrix.tsv -n 500 -fp 0.001 -fn 0.1 -s 1 -b 100 -o ./samples/sample2.parquet`

The implementation in `partition_function_cp.py` will read the samples from the parquet files in the samples directory and compute estimates based off of them:

`python ./src/partition_function_cp.py -i ./example/input_genotype_matrix.tsv -o ./partf_output.tsv -t ./samples -fp 0.001 -fn 0.1 -b 100 -sm ./example/mutations_to_score_matrix.tsv`
