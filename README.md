Run commands from top directory

Command to sample trees

`python src/sampler.py -i ./example/input_genotype_matrix.tsv -n 100 -fp 0.001 -fn 0.1 -s 42 -o ./example_sample.sample`

Command to compute partition function estimates

`python src/partition_function.py -i ./example/input_genotype_matrix.tsv -o ./partf_output.tsv -t example_sample.sample -fp 0.001 -fn 0.1 -sm ./example/mutations_to_score_matrix.tsv`
