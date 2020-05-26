import torch
import numpy as np
from torch import Tensor
from tools.tree_tools import get_parent_children_combo_from_tree, parentchild_ids_to_idx, tokentree_to_ete
from typing import List, Callable, Dict, Tuple, Optional, DefaultDict
from conllu import TokenList
import os
from collections import defaultdict

get_ids_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['id'] for word in sent]
get_pos_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['upostag'] for word in sent]
get_tokens_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['form'] for word in sent]

def fetch_pos_tags(
    ud_parses: List[TokenList],
    pos_vocab: Optional[DefaultDict[str, int]] = None,
    corrupted=False,
    corrupted_pos_tags: Optional[Dict[str, str]] = None
)-> Tuple[List[Tensor], DefaultDict]:
    """
    Converts `ud_parses` into a tensor of POS tags.
    """
    # If `pos_vocab` is not known, make one based on all POS tokens in `ud_parses`
    if (pos_vocab is None):
        print('get all tokens')
        all_pos_tokens = set([pos for sent in ud_parses for pos in get_pos_from_sent(sent)])
        print('get all pos2i')
        pos2i: dict = {'<pad>': 0, '<unk>': 1, **{pos.strip(): i + 2 for i, pos in enumerate(all_pos_tokens)}}
        pos_vocab = defaultdict(lambda: pos2i["<unk>"])
        pos_vocab.update(pos2i)

    pos_tokens_result: List[Tensor] = []
    sent: TokenList

    for sent in ud_parses:
        # If corrupted, let the target value be the corrupted tokens
        if corrupted:
            pos_tokens = torch.tensor([pos_vocab[corrupted_pos_tags[word]] for word in get_tokens_from_sent(sent)])
        else:
            pos_tokens = torch.tensor([pos_vocab[pos] for pos in get_pos_from_sent(sent)])
        pos_tokens_result.append(pos_tokens)


    return pos_tokens_result, pos_vocab


def create_struct_gold_distances(corpus) -> List[Tensor]:
    """
    Creates gold distances for the strucutal task
    """
    all_distances: List[Tensor] = []

    for sent in corpus:
        tokentree = sent.to_tree()
        ete_tree = tokentree_to_ete(tokentree)

        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))

        # 🏁
        token_ids = get_ids_from_sent(sent)

        # Go over all the token ids, as row and as columns
        for i, token_id_A in enumerate(token_ids):
            for j, token_id_B in enumerate(token_ids):

                # Set distance to 0 if they are equal
                if token_id_A == token_id_B:
                    distances[i, j] = 0
                else:
                    # Else use `.get_distance` to calculate the distance for row A and column B
                    A = str(token_id_A)
                    B = str(token_id_B)
                    distances[i, j] = ete_tree.get_distance(A, B)

        all_distances.append(distances)

    return all_distances

def create_dep_parent_gold_distances(
    corpus: List[TokenList],
    corrupted: bool = False,
    vocab: Dict[str, int] = None
) -> List[Tensor]:
    """
    Assigns for each sentence the index of the parent node.
    If node is -1, then that node has no parent (should be ignored).

    Input:
        - corpus: list of TokenLists
    Output:
        List of Tensors(tensor is )
    """
    all_edges: List[Tensor] = []

    for sent in corpus:
        sent_tree = sent.to_tree()
        sent_id2idx = {i['id']: idx for idx, i in enumerate(sent)}

        # Calculate tuples of (parent, child), and then map their ID to their respective indices
        parent_child_tuples = get_parent_children_combo_from_tree(sent_tree)
        parent_child_tuples = parentchild_ids_to_idx(sent_id2idx, parent_child_tuples)

        # Initialize our matrix
        sen_len = len(sent)
        edges = torch.zeros((sen_len))
        root_idx = sent_id2idx[sent_tree.token['id']]

        # For each edge, assign in the corresponding index the parent index, and -1 for root
        for parent_edge in parent_child_tuples:
            parent_idx, child_idx = parent_edge

            if corrupted:
                corrupted_choices = [0, child_idx, sen_len]
                child_token = sent[child_idx]['form']
                corrupted_choice_idx = vocab[child_token]
                edges[child_idx] = corrupted_choices[corrupted_choice_idx]
            else:
                edges[child_idx] = parent_idx

            edges[root_idx] = -1

        all_edges.append(edges)

    return all_edges

def create_corrupted_dep_vocab(corpora: List[TokenList]) -> Dict[str, str]:
    # Get sentences from all data sets
    possible_targets_distr = [0, 1, 2]

    corrupted_word_type = defaultdict(lambda: 0)
    # Get a corrupted POS tag for each word
    for sentence in corpora:
        for token in sentence:
            corrupted_behaviour = np.random.choice(possible_targets_distr, 1).item()

            if token['form'] not in corrupted_word_type:
                corrupted_word_type[token['form']] = corrupted_behaviour

    return corrupted_word_type
