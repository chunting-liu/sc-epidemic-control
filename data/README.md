# Datasets

Both graphs are redistributed from the Stanford Network Analysis Project (SNAP)
collection, which makes them publicly available for research use. They are
included here so that the pipeline runs without a network fetch; the upstream
sources remain authoritative.

## `email-Eu-core.txt`

Email exchange records from a large European research institution.

- Source: <https://snap.stanford.edu/data/email-Eu-core.html>
- Format: one directed edge per line, whitespace-separated node ids.
- Raw file: 25,571 directed records over 1,005 nodes, of which 642 are
  self-addressed.

As loaded by `src/networks.py` and `scripts/compute_table6.py`: self-loops are
discarded, the graph is symmetrised by treating any directed record as an
undirected dependency, and the largest connected component is retained. This
gives **N = 986**, **|E| = 16,064**, mean degree **32.58**, spectral radius
**76.27**.

Retaining the self-loops instead would give |E| = 16,687 and a spectral radius
of 77.17. They are excluded because a self-loop would represent a node
propagating to itself, which the contagion process does not admit.

## `amazon0302.txt`

Amazon product co-purchasing network, March 2003.

- Source: <https://snap.stanford.edu/data/amazon0302.html>
- Format: one directed edge per line, whitespace-separated node ids.

Used only as a secondary structural comparator. The loader in
`scripts/r2_experiments.py` (`load_amazon_directed`) discards self-loops, takes
the largest strongly connected component, and then grows an 800-node sample by
breadth-first search **starting from the maximum-degree node**. The procedure is
fully deterministic — there is no random seed — but it is deliberately
hub-centred, so the sample is not representative of the full graph.

Two consequences are worth stating plainly. The sampled subgraph has mean degree
4.60 and spectral radius 22.58, a ratio of 4.9; that concentration is partly a
property of the sampling rule rather than of the underlying co-purchase network,
and any conclusion resting on the sample's degree heterogeneity should be read
with that in mind. Separately, co-purchase edges are product associations, not
inter-firm dependencies, so the graph serves here as a structurally contrasting
topology rather than as a dependency network.

## Generated topologies

The layered multi-echelon graphs are not distributed as files. They are produced
on demand by `multi_echelon()` in `scripts/r2_experiments.py` from a fixed seed,
with a tunable share of recirculating edges, and are reproducible exactly.

## Citation

If you use these datasets, cite SNAP:

> J. Leskovec and A. Krevl. SNAP Datasets: Stanford Large Network Dataset
> Collection. <http://snap.stanford.edu/data>, 2014.
