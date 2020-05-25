import torch
from scipy.sparse.csgraph import minimum_spanning_tree

def create_mst(distances):
    distances = torch.triu(distances).detach().numpy()

    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.

    return mst

def edges(mst):
    """
    Get all edges from a minimum spanning tree `mst`
    """

    all_edges = list(zip(mst.nonzero()[0], mst.nonzero()[1]))

    # Ensure (A, B) and (B, A) only return one combination
    all_edges = set(tuple(frozenset(sub)) for sub in set(all_edges))

    return all_edges
