from data_tools.target_extractors import create_dep_parent_gold_distances, create_struct_gold_distances, fetch_pos_tags
from typing import List, Dict, Optional
import os
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

def parse_all_corpora(use_sample):
    ud_parses = []
    for set_type in ['train', 'dev', 'test']:
        filename = os.path.join('data', 'sample' if use_sample else '', f'en_ewt-ud-{set_type}.conllu')

        ud_parses += (parse_corpus(filename))

    return ud_parses

def init_pos_data(
    filename: str,
    model,
    w2i: Dict[str, int],
    pos_vocab=None,
    corrupted=False,
    corrupted_pos_tags: Optional[Dict[str, str]] = None
):
    print('Parsing the corpus')
    ud_parses = parse_corpus(filename)

    print('Fetching the POS tags')
    pos_tags, pos_vocab = fetch_pos_tags(
        ud_parses,
        pos_vocab=pos_vocab,
        corrupted=corrupted,
        corrupted_pos_tags=corrupted_pos_tags
    )
    print(f'Fetching sen reps using {type(model).__name__} in `create_data`')
    sen_reps = fetch_sen_reps(ud_parses, model, w2i)

    return sen_reps, pos_tags, pos_vocab

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
        if use_corrupted and dep_vocab is None:
            raise Exception('You should provide a vocabulary with these indices.')

        gold_distances = create_dep_parent_gold_distances(corpus, use_corrupted, dep_vocab)

    return embs, gold_distances
