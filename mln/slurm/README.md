# Run the STRING pipeline with SLURM

The batch script follows the DEI cluster's SLURM requirements and runs the full workflow from
`mln/configs/clustering_MLN_STRING_net_pagerank.yaml`: multiplex network
generation, multilayer PageRank, and clustering. It expects a micromamba
environment named `muxvizpy` (or the name set in `MUXVIZ_ENV`) and the input
files under `mln/data/`. By default it looks for environments under
`$HOME/micromamba`, matching the micromamba setup instructions. If you chose a
different root prefix, submit with `MAMBA_ROOT_PREFIX=/your/path`.

From the repository root, submit it with:

```bash
sbatch mln/slurm/run_string_pipeline.sbatch
```

To run the raw RNA and DNA methylation clustering baseline (test 1), submit:

```bash
sbatch mln/slurm/run_string_baseline.sbatch
```

To calculate single-layer PageRank from the STRING network for each matrix and
cluster both results separately (test 2), submit:

```bash
sbatch mln/slurm/run_string_pagerank.sbatch
```

The baseline results go under `output/string_net/clustering/baseline/`. Test 2
writes PageRank CSVs under `output/string_net/pagerank/single/{RNA,DNA}/` and
clustering results under `output/string_net/clustering/pagerank/{RNA,DNA}/`.

If network generation and PageRank have already completed, submit only the
clustering stage with:

```bash
sbatch mln/slurm/run_string_clustering.sbatch
```

The clustering-only job reads the existing PageRank CSVs under
`output/string_net/pagerank/mln/` and skips the first two stages.

SLURM writes `string-mln-<job-id>.out` and `.err` in the directory from which
you run `sbatch`. Check the job with `squeue -u "$USER"`; inspect those files
after submission. Pipeline results go under `output/string_net/`.

The script requests the `allgroups` partition, one task, one CPU, 16 GB of
memory, and 24 hours. The DEI documentation lists `--ntasks`, `--partition`,
`--time`, and `--mem` as required directives, and advises requesting only the
resources the job needs. The memory and time requests are initial estimates
because this STRING run has not yet been benchmarked on the cluster; check job
efficiency after it finishes and adjust future requests. Add your DEI email
and uncomment the mail directives if you want notifications.

The input data are git-ignored, so make sure they have been copied to the
server clone before submitting. See the [DEI SLURM basics](https://docs.dei.unipd.it/en/CLUSTER/Slurm_basics)
and [SLURM examples](https://docs.dei.unipd.it/en/CLUSTER/Slurm_examples).

