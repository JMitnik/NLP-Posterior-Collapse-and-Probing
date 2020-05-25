from typing import Tuple, List
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from ete3 import Tree as EteTree

class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)

    def __str__(self):
        return self.get_ascii(show_internal=True)

    def __repr__(self):
        return str(self)

def rec_tokentree_to_ete(tokentree):
    """
    Recursively maps Conllu tokentrees to Ete-readable format.
    """
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx

def tokentree_to_ete(tokentree):
    """
    Maps Conllu tokentrees to Ete-readable format.
    """
    newick_str = rec_tokentree_to_ete(tokentree)
    return FancyTree(f"{newick_str};")


def rec_tokentree_to_nltk(tokentree):
    """
    Recursively maps Conllu tokentrees to NLTK-readable format.
    """
    token = tokentree.token["form"]
    tree_str = f"({token} {' '.join(rec_tokentree_to_nltk(t) for t in tokentree.children)})"

    return tree_str


def tokentree_to_nltk(tokentree):
    """
    Maps Conllu tokentrees to NLTK-readable format.
    """
    from nltk import Tree as NLTKTree

    tree_str = rec_tokentree_to_nltk(tokentree)

    return NLTKTree.fromstring(tree_str)

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


def get_parent_children_combo_from_tree(tokentree) -> List[Tuple[str, str]]:
    """
    Get all nodes that have a parent along with their parent
    """
    result = []

    ete_tree_nodes = tokentree_to_ete(tokentree).search_nodes()

    for node in ete_tree_nodes:
        # If parent
        if node.up is not None:
            parent_node = node.up.name
            child_node = node.name

            result.append(
                (parent_node, child_node)
            )

    return result

def parentchild_ids_to_idx(id_idx_map, combo_tuple_list):
    """
    Maps parent-child ids based on CONLLU to indices.
    """
    result = []

    for parent_child_combo in combo_tuple_list:
        parent_node_idx = id_idx_map[int(parent_child_combo[0])]
        child_node_idx = id_idx_map[int(parent_child_combo[1])]

        result.append((parent_node_idx, child_node_idx))

    return result
