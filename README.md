python sampler.py -i ./example/input_genotype_matrix.tsv -n 1000 -fp 0.001 -fn 0.4 -s 42 > example_sample.sample

python partition_function.py -i ./example/input_genotype_matrix.tsv -t example_sample.sample -fp 0.001 -fn 0.4 -sm ./example/mutations_to_score_matrix.tsv