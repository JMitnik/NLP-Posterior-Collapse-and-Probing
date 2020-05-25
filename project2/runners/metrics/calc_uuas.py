from tools.tree_tools import create_mst, edges
check_edge_in_edgeset = lambda edge, edgeset: (edge[0], edge[1]) in edgeset or (edge[1], edge[0]) in edgeset

def calc_uuas(pred_distances, gold_distances):
    uuas = None

    # Get MSTs from distances
    pred_mst = create_mst(pred_distances)
    gold_mst = create_mst(gold_distances)

    # Convert MSTs to edges
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)

    # Calculate UUAS
    nr_correct_edges =len(
        [pred_edge for pred_edge in pred_edges
         if check_edge_in_edgeset(pred_edge, gold_edges)]
    )

    uuas = nr_correct_edges / len(gold_edges)

    return uuas
