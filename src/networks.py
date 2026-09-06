"""
Network loading and synthetic generation utilities.

Provides functions to:
  - Load the Email-EU communication network from SNAP
  - Load the Amazon co-purchasing network from SNAP
  - Generate Barabási–Albert (scale-free) networks
  - Generate Watts–Strogatz (small-world) networks
  - Generate Erdős–Rényi random networks
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


def load_email_network(filepath: str | Path) -> tuple[np.ndarray, nx.Graph]:
    """Load and symmetrise the Email-EU core network from SNAP.

    The raw dataset contains directed email exchanges.  We symmetrise by
    treating any directed edge as an undirected dependency and extract the
    largest connected component.

    Parameters
    ----------
    filepath : path to ``email-Eu-core.txt``

    Returns
    -------
    A : ndarray, shape (N, N) – symmetric adjacency matrix
    G : networkx.Graph
    """
    edges: list[tuple[int, int]] = []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))

    G = nx.DiGraph()
    # Self-loops are discarded: the raw log contains self-addressed records,
    # which under the dependency reading would mean a node propagating to
    # itself.  They are not edges of the contagion process and they inflate
    # the spectral radius (77.17 -> 76.27 on the Email-EU core).
    G.add_edges_from([(u, v) for u, v in edges if u != v])
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G)

    A = nx.to_numpy_array(G)
    return A, G


def load_amazon_network(
    filepath: str | Path,
    sample_size: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, nx.Graph]:
    """Load Amazon co-purchasing network and sample a connected subgraph.

    Parameters
    ----------
    filepath : path to ``amazon0302.txt``
    sample_size : number of nodes to keep via BFS from the highest-degree node
    seed : random seed (unused; BFS is deterministic)

    Returns
    -------
    A : ndarray, shape (N, N)
    G : networkx.Graph
    """
    edges: list[tuple[int, int]] = []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))

    G_full = nx.DiGraph()
    G_full.add_edges_from(edges)

    largest_scc = max(nx.strongly_connected_components(G_full), key=len)
    G_scc = G_full.subgraph(largest_scc).copy()

    if G_scc.number_of_nodes() > sample_size:
        degrees = dict(G_scc.degree())
        start_node = max(degrees, key=degrees.get)
        sampled: set[int] = set()
        queue = [start_node]
        while len(sampled) < sample_size and queue:
            node = queue.pop(0)
            if node not in sampled:
                sampled.add(node)
                neighbours = list(G_scc.successors(node)) + list(
                    G_scc.predecessors(node)
                )
                queue.extend(n for n in neighbours if n not in sampled)
        G_sample = G_scc.subgraph(sampled).copy()
    else:
        G_sample = G_scc

    G = G_sample.to_undirected()
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G)
    return A, G


# ── Synthetic network generators ──────────────────────────────────────────

def generate_scale_free(
    n: int, m: int = 3, seed: int = 42
) -> tuple[np.ndarray, nx.Graph]:
    """Barabási–Albert scale-free network."""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return nx.to_numpy_array(G), G


def generate_small_world(
    n: int, k: int = 6, p: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, nx.Graph]:
    """Watts–Strogatz small-world network."""
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return nx.to_numpy_array(G), G


def generate_erdos_renyi(
    n: int, p: float = 0.05, seed: int = 42
) -> tuple[np.ndarray, nx.Graph]:
    """Erdős–Rényi random network."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return nx.to_numpy_array(G), G
