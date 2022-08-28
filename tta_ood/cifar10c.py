from locale import normalize
import logging

import torch
import torch.optim as optim
import torchvision
import math
import datetime
import numpy as np

from robustbench.data import load_cifar10c, load_cifar100c, load_cifar100, load_svhn, load_svhnc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

#import methods.tent as tent
import methods.norm as norm
import methods.gce as gce
#import methods.hardpl as hardpl
import methods.pl as pl

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def accuracy(model: torch.nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   use_ood=False,
                   x_test_ood=None,
                   y_test_ood=None,
                   n_ood_samples=None,
                   create_tsne=False,
                   epoch=1):

    if device is None:
        device = x.device

    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    counter_ood = 0

    if create_tsne:
        # define forward hooks to get intermediate activations:
        activation = dict()
        def get_activation(name):
            def hook(model, input, output):
                out = output.detach()
                out = torch.nn.functional.avg_pool2d(out, 8)
                activation[name] = out
            return hook
        model.model.bn1.register_forward_hook(get_activation('hook'))
        #model.model.fc.register_forward_hook(get_activation('hook1'))

    with torch.no_grad():
        for counter_batch in range(n_batches):
            x_curr = x[counter_batch * batch_size:(counter_batch + 1) *
                       batch_size].to(device)
            y_curr = y[counter_batch * batch_size:(counter_batch + 1) *
                       batch_size].to(device)

            if use_ood:
                x_curr = torch.cat((x_curr, x_test_ood[counter_ood:counter_ood+n_ood_samples].cuda()), dim=0)
                y_curr = torch.cat((y_curr, (101 + y_test_ood[counter_ood:counter_ood+n_ood_samples]).cuda()), dim=0)
                counter_ood += n_ood_samples

            if cfg.MODEL.ADAPTATION in ["tent", "gce", "hardpl", "softpl"]:
                output = model(x_curr, adapt=True)
            else:
                output = model(x_curr)

            if create_tsne:
                intermediate_activations = activation['hook']
                #logits = activation['hook1']
                if counter_batch == 0:
                    embeddings = intermediate_activations.cpu()
                    #embeddings1 = logits.cpu()
                    labels = y_curr.cpu()
                else:
                    embeddings = torch.cat((embeddings, intermediate_activations.cpu()))
                    #embeddings1 = torch.cat((embeddings1, logits.cpu()))
                    labels = torch.cat((labels, y_curr.cpu()))
                if counter_batch == 4:
                    logger.info("Embeddings Shape: {}".format(embeddings.shape))
                    timestamp = datetime.datetime.now()
                    folder = '97_embeddings/tent_3epochs/'
                    np.save('{}x_{}_{}.npy'.format(folder, cfg.LOG_DEST[:-16], epoch), np.array(embeddings))
                    np.save('{}y_{}_{}.npy'.format(folder, cfg.LOG_DEST[:-16], epoch), np.array(labels))

            if use_ood:
                output = output[:len(y_curr)]

            acc += (output.max(1)[1] == y_curr).float().sum()

        # rerun forward pass on fully adopted model
        if cfg.MODEL.ADAPTATION in ["tent", "gce", "hardpl", "softpl"]:
            acc = 0.
            counter_ood = 0
            for counter_batch in range(n_batches):
                x_curr = x[counter_batch * batch_size:(counter_batch + 1) *
                        batch_size].to(device)
                y_curr = y[counter_batch * batch_size:(counter_batch + 1) *
                        batch_size].to(device)

                if use_ood:
                    x_curr = torch.cat((x_curr, x_test_ood[counter_ood:counter_ood+n_ood_samples].cuda()), dim=0)
                    counter_ood = counter_ood + n_ood_samples

                output = model(x_curr, adapt=False)

                if use_ood:
                    output = output[:len(y_curr)]

                acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

def evaluate(description):

    start_time = datetime.datetime.now()

    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

    load_time = datetime.datetime.now()
    logger.info("LOADING_TIME: loading cfg and model took {}s".format(round((load_time-start_time).total_seconds(),4)))

    x_test_ood = None
    y_test_ood = None
    use_ood = False
    n_ood_samples = 0

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_pl(base_model)
    if cfg.MODEL.ADAPTATION == "gce":
        logger.info("test-time adaptation: GCE")
        #model = setup_gce(base_model)
        model = setup_pl(base_model)
    if cfg.MODEL.ADAPTATION == "hardpl":
        logger.info("test-time adaptation: HARDPL")
        model = setup_pl(base_model)
    if cfg.MODEL.ADAPTATION == "softpl":
        logger.info("test-time adaptation: SOFTPL")
        model = setup_pl(base_model)

    # evaluate on each severity and type of corruption in turn
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:

            start_time = datetime.datetime.now()

            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")

            reset_time = datetime.datetime.now()
            logger.info("RESET_TIME: resetting model took {}s".format(round((reset_time-start_time).total_seconds(),4)))

            for epoch in range(1,cfg.N_EPOCHS+1):
                print("starting epoch ", epoch)
                epoch_start_time = datetime.datetime.now()

                x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                            severity, cfg.DATA_DIR, shuffle=True,
                                            corruptions=[corruption_type])

                if cfg.CORRUPTION.SVHN_samples > 0:
                    x_test_ood, y_test_ood = load_svhn(data_dir=cfg.DATA_DIR+"/SVHN/", shuffle=True) #, batch_size=cfg.CORRUPTION.SVHN_samples
                    use_ood = True
                    n_ood_samples = cfg.CORRUPTION.SVHN_samples

                if cfg.CORRUPTION.CIFAR100_samples > 0:
                    x_test_ood, y_test_ood = load_cifar100(data_dir=cfg.DATA_DIR+"/CIFAR100/", shuffle=True)
                    use_ood = True
                    n_ood_samples = cfg.CORRUPTION.CIFAR100_samples

                if cfg.CORRUPTION.SVHNC_samples > 0:
                    x_test_ood, y_test_ood = load_svhnc(n_examples=cfg.CORRUPTION.NUM_EX,
                                                severity=severity, data_dir=cfg.DATA_DIR, shuffle=True,
                                                corruptions=[corruption_type])
                    use_ood = True
                    n_ood_samples = cfg.CORRUPTION.SVHNC_samples

                if cfg.CORRUPTION.CIFAR100C_samples > 0:
                    x_test_ood, y_test_ood = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                                severity, cfg.DATA_DIR, shuffle=True,
                                                corruptions=[corruption_type])
                    use_ood = True
                    n_ood_samples = cfg.CORRUPTION.CIFAR100C_samples

                x_test, y_test = x_test.cuda(), y_test.cuda()

                ood_time = datetime.datetime.now()
                logger.info("OOD_TIME: loading ood data took {}s".format(round((ood_time-epoch_start_time).total_seconds(),4)))

                acc = accuracy(model=model, x=x_test, y=y_test, batch_size=cfg.TEST.BATCH_SIZE,
                                use_ood=use_ood,
                                x_test_ood=x_test_ood,
                                y_test_ood=y_test_ood,
                                n_ood_samples=n_ood_samples,
                                create_tsne = cfg.MODEL.CREATE_EMBEDDINGS)
        
                err = 1. - acc

                epoch_time = datetime.datetime.now()
                logger.info("EPOCH_TIME: running epoch took {}s".format(round((epoch_time-ood_time).total_seconds(),4)))
                logger.info(f"epoch {epoch} error % [{corruption_type}{severity}]: {err:.2%}")

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_pl(model):
    """Set up pl adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then pl the model.
    """
    model = pl.configure_model(model)
    params, param_names = pl.collect_params(model)
    optimizer = setup_optimizer(params)
    pl_model = pl.PL(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           threshold=cfg.MODEL.PL_THRESHOLD,
                           pl_type=cfg.MODEL.ADAPTATION,
                           detect_oods=cfg.MODEL.OOD_METHOD,
                           ood_threshold=cfg.MODEL.OOD_THRESHOLD)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return pl_model


def setup_gce(model):
    """Set up test-time rpl/gce adaptation.
    """
    model = gce.configure_model(model)
    params, names = gce.collect_params(model)
    optimizer = setup_optimizer(params)
    gce_model = gce.GCE(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        qval=cfg.MODEL.PL_THRESHOLD,
                        detect_oods=cfg.MODEL.OOD_METHOD,
                        ood_threshold=cfg.MODEL.OOD_THRESHOLD)
    gce.check_model(gce_model)
    logger.info(f"model for adaptation: %s", gce_model)
    logger.info(f"params for adaptation: %s", names)
    logger.info(f"optimizer for adaptation: %s", optimizer)

    return gce_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
