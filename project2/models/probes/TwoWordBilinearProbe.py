import torch
import torch.nn as nn

class TwoWordBilinearLabelProbe(nn.Module):
  """ Computes a bilinear function of pairs of vectors.
  For a batch of sentences, computes all n^2 pairs of scores
  for each sentence in the batch.
  """
  def __init__(
      self,
      max_rank: int,
      feature_dim: int,
      dropout_p: float
  ):
    super().__init__()
    self.maximum_rank = max_rank
    self.feature_dim = feature_dim
    self.proj_L = nn.Parameter(data = torch.zeros(self.feature_dim, self.maximum_rank), requires_grad=True)
    self.proj_R = nn.Parameter(data = torch.zeros(self.maximum_rank, self.feature_dim), requires_grad=True)
    self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=True)

    nn.init.uniform_(self.proj_L, -0.05, 0.05)
    nn.init.uniform_(self.proj_R, -0.05, 0.05)
    nn.init.uniform_(self.bias, -0.05, 0.05)

    self.dropout = nn.Dropout(p=dropout_p)
    self.softmax = nn.LogSoftmax(2)

  def forward(self, batch):
    """ Computes all n^2 pairs of attachment scores
    for each sentence in a batch.
    Computes h_i^TAh_j for all i,j
    where A = LR, L in R^{model_dim x maximum_rank}; R in R^{maximum_rank x model_rank}
    hence A is rank-constrained to maximum_rank.
    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
    """
    batchlen, seqlen, rank = batch.size()
    batch = self.dropout(batch)
    # A/Proj = L * R
    proj = torch.mm(self.proj_L, self.proj_R)

    # Expand matrix to allow another instance of all cols/rows
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)

    # Add another instance of seqlen (do it on position 1 for some reason)
    # => change view so that it becomes rows of 'rank/hidden-dim'
    batch_transposed = batch.unsqueeze(1).expand(batchlen, seqlen, seqlen, rank).contiguous().view(batchlen*seqlen*seqlen,rank,1)

    # Multiply the `batch_square` matrix with the projection
    psd_transformed = torch.matmul(batch_square.contiguous(), proj).view(batchlen*seqlen*seqlen,1, rank)

    # Multiple resulting matrix `psd_transform` with each j
    logits = (torch.bmm(psd_transformed, batch_transposed) + self.bias).view(batchlen, seqlen, seqlen)

    probs = self.softmax(logits)
    return probs
