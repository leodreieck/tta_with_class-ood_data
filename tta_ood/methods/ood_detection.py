import torch
import logging

logger = logging.getLogger(__name__)

def keep_threshold_n_images(logits, threshold):

    # norm logits
    confidences = logits.softmax(-1)

    # find maximum confidence per image
    max_outputs = torch.max(confidences,1)[0]

    # get batch size
    bsize = max_outputs.shape[0]

    # select top threshold*100% images with highest confidence
    _, inds = max_outputs.topk(int(bsize*threshold), 0, True, False)

    # create + fill tensor with necessary pattern ([1,1,...,1] or [0,0,...,0])
    #pattern = torch.zeros_like(logits)
    pattern = torch.full_like(logits, 1e-9)
    pattern[inds.unsqueeze(0)] = 1

    total = 1-pattern.mean().cpu().numpy()
    cifar = 1-pattern[:200].mean().cpu().numpy()
    ood = 1-pattern[200:].mean().cpu().numpy()
    logger.info("N_Images OOD detection filtered out [total,cifar,ood]: [{:.1%},{:.1%},{:.1%}] of the data".format(total, cifar, ood))

    return pattern


def keep_threshold_confidence(logits, threshold):
    
    # get confidences
    confidences = logits.softmax(-1)

    # get maximum confidence per image
    max_outputs = torch.max(confidences,1)[0]
    
    # get indices of images with maximum confidence > threshold
    inds = torch.where(max_outputs>threshold)[0]
    
    # create + fill tensorwith necessary pattern ([1,1,...,1] or [0,0,...,0])
    pattern = torch.zeros_like(logits)
    pattern[inds.unsqueeze(0)] = 1

    total = 1-pattern.mean().cpu().numpy()
    cifar = 1-pattern[:200].mean().cpu().numpy()
    ood = 1-pattern[200:].mean().cpu().numpy()
    logger.info("Confidence OOD detection filtered out [total,cifar,ood]: [{:.1%},{:.1%},{:.1%}] of the data".format(total, cifar, ood))

    return pattern
    