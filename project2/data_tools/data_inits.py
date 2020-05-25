from data_tools.target_extractors import create_dep_parent_gold_distances, create_struct_gold_distances
from typing import List
from torch import Tensor
from conllu import TokenList, parse_incr

from data_tools.feature_extractors import fetch_sen_reps

def parse_corpus(filename: str) -> List[TokenList]:
    """
    Parses a file into a collection of TokenLists
    """
    data_file = open(filename, encoding="utf-8")
    ud_parses = list(parse_incr(data_file))

    return ud_parses

def init_tree_corpus(
    path,
    feature_model,
    feature_tokenizer,
    concat=False,
    cutoff=None,
    use_dependencies=False,
    use_corrupted=False,
    dep_vocab=None
):
    """
    Initialises the data of a corpus, with features as X and tree-based structual targets.

    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating
        memory usage.
    """
    corpus: List[TokenList] = parse_corpus(path)[:cutoff]
    embs: List[Tensor] = fetch_sen_reps(corpus, feature_model, feature_tokenizer, concat=concat)

    if not use_dependencies:
        gold_distances = create_struct_gold_distances(corpus)
    else:
        if dep_vocab is None:
            raise Exception('Need to pass `dep_vocab` as well!')

        gold_distances = create_dep_parent_gold_distances(corpus, use_corrupted, dep_vocab)

    return embs, gold_distances
