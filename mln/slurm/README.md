# Run the clustering pipelines with SLURM

All clustering configurations explore `k=2` through `k=25`. Their output
directories include `_k2-25` so these runs remain separate from earlier
`k=2..5` outputs. Run the commands below from the repository root on the
server. The jobs expect the `muxvizpy` micromamba environment and the input
files under `mln/data/`. Override `MUXVIZ_ENV` or `MAMBA_ROOT_PREFIX` when your
server setup uses different values.

## PathwayInCancer configurations

These five configs run clustering against existing data. In particular, the
RNA/DNA PageRank and MLN PageRank configs require their PageRank CSVs to have
already been generated in the configured input folders.

```bash
sbatch --export=ALL,CONFIG=mln/configs/clustering_RNA_original.yaml mln/slurm/run_config.sbatch
sbatch --export=ALL,CONFIG=mln/configs/clustering_DNA_original.yaml mln/slurm/run_config.sbatch
sbatch --export=ALL,CONFIG=mln/configs/clustering_RNA_pagerank.yaml mln/slurm/run_config.sbatch
sbatch --export=ALL,CONFIG=mln/configs/clustering_DNA_pagerank.yaml mln/slurm/run_config.sbatch
sbatch --export=ALL,CONFIG=mln/configs/clustering_MLN_pagerank.yaml mln/slurm/run_config.sbatch
```

`run_config.sbatch` runs clustering only by default. To honor every stage
enabled in a config, submit with `RUN_FULL_PIPELINE=1`; note that these
PathwayInCancer configs currently disable network generation and PageRank.

## Direct RNA and DNA baselines

This runs raw-matrix clustering for both RNA and DNA in sequence:

```bash
sbatch mln/slurm/run_string_baseline.sbatch
```

Results are written under
`output/string_net/clustering/baseline/{RNA,DNA}_k2-25/`.

## Single-layer PageRank for RNA and DNA

This computes PageRank from the STRING network separately for RNA and DNA,
then clusters both PageRank result sets:

```bash
sbatch mln/slurm/run_string_pagerank.sbatch
```

PageRank files go under `output/string_net/pagerank/single/{RNA,DNA}/` and
clustering results under
`output/string_net/clustering/pagerank/{RNA,DNA}_k2-25/`.

## Multilayer PageRank (MLN)

To generate the STRING multiplex network, calculate multilayer PageRank, and
cluster its results in one job:

```bash
sbatch mln/slurm/run_string_pipeline.sbatch
```

This runs the full config at
`mln/configs/clustering_MLN_STRING_net_pagerank.yaml`. If the multiplex
network and PageRank files already exist, run clustering only:

```bash
sbatch mln/slurm/run_string_clustering.sbatch
```

## Monitor jobs

SLURM writes log files in the directory where you submit the job. Check job
status with `squeue -u "$USER"`, then inspect the matching `.out` and `.err`
files. The existing resource requests are starting estimates; review actual
elapsed time and memory use after the first `k=2..25` run before adjusting
them. Ensure the git-ignored input data have been copied to the server clone.
