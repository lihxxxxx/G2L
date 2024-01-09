import torch
from torch.nn.utils.rnn import pad_sequence
from g2l.structures import TLGBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        feats, queries, wordlens, ious2d, moments, num_sentences, idxs = transposed_batch
        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=queries,
            wordlens=wordlens,
            all_iou2d=ious2d,
            moments=moments,
            num_sentences=num_sentences,
        ), idxs
