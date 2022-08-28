from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from methods.ood_detection import keep_threshold_n_images, keep_threshold_confidence


class PL(nn.Module):
    """
    Allows three options:
    - tent (adapts a model by entropy minimization during testing. Once tented, a model adapts itself by updating on every forward.)
    - hard pl
    - soft pl
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, threshold=0.7, pl_type="tent", detect_oods="no", ood_threshold=0.5):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "pl requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.threshold = threshold
        self.pl_type = pl_type
        self.detect_oods = detect_oods
        self.ood_threshold = ood_threshold

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=True):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            if adapt:
                outputs = forward_and_adapt(x, self.model, self.optimizer,
                                            self.threshold, self.pl_type,
                                            self.detect_oods, self.ood_threshold)
            else:
                outputs = forward_wo_adapt(x, self.model)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor, ood_detection_pattern) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(ood_detection_pattern * x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def hardlabeling(x: torch.Tensor, hard_labels, ood_detection_pattern) -> torch.Tensor:
    """Loss on labels above hard threshold only"""
    return -(ood_detection_pattern * hard_labels * x.log_softmax(1)).sum(1)

@torch.jit.script
def softlabeling(x: torch.Tensor, soft_labels, ood_detection_pattern) -> torch.Tensor:
    """Weigh loss with probability of soft_labels"""
    return -(ood_detection_pattern * soft_labels.softmax(1) * x.log_softmax(1)).sum(1)

def gce(logits, q, pattern):
    p = F.softmax(logits, dim=1) * pattern
    y_pl = logits.max(1)[1]
    Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
    tta_loss = (1- (Yg**q))/q
    return tta_loss


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, threshold, pl_type, detect_oods, ood_threshold):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    torch.autograd.set_detect_anomaly(True)
    # forward
    outputs = model(x)

    with torch.no_grad():
        ood_detection_pattern = torch.ones_like(outputs)

    if detect_oods == "threshold_n_images":
        # detect images with highest confidence
        with torch.no_grad():
            outputs_cloned = outputs.clone().detach()
            ood_detection_pattern = keep_threshold_n_images(outputs_cloned, ood_threshold)
        #outputs = torch.mul(outputs, pattern)

    elif detect_oods == "threshold_confidence":
        # if highest confidence in row < threshold, delete row ([0,...,0] in matrix)
        with torch.no_grad():
            outputs_cloned = outputs.clone().detach()
            ood_detection_pattern = keep_threshold_confidence(outputs_cloned, ood_threshold)
        #outputs = torch.mul(outputs, pattern)

    # adapt
    if pl_type == "tent":
        loss = softmax_entropy(outputs, ood_detection_pattern).mean(0)

    elif pl_type == "softpl":
        with torch.no_grad():
            #pseudo_labels = model.forward(x)
            pseudo_labels = outputs.clone().detach()
            #pseudo_labels = pseudo_labels
        loss = softlabeling(outputs, pseudo_labels, ood_detection_pattern).mean(0)

    elif pl_type == "hardpl":
        with torch.no_grad():
            pseudo_labels = outputs.clone().detach()

            confidences = pseudo_labels.softmax(-1)
            indices = (confidences > threshold)

            idx_max_per_row = pseudo_labels.max(1)[1]
            hard_labels_max = 1 * F.one_hot(idx_max_per_row, num_classes=pseudo_labels.shape[-1])

            pattern = indices * hard_labels_max

        loss = hardlabeling(outputs, pattern, ood_detection_pattern).mean(0)

    elif pl_type == "gce":
        loss = gce(outputs, threshold, ood_detection_pattern).mean(0)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def forward_wo_adapt(x, model):

    outputs = model(x)

    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with pl."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with pl."""
    is_training = model.training
    assert is_training, "pl needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "pl needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "pl should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "pl needs normalization for its optimization"
