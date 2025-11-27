import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else 'cpu'

l1 = nn.L1Loss(reduction='none')
bce = nn.BCEWithLogitsLoss(reduction='none')
ce = nn.CrossEntropyLoss(reduction='none')


def weighting_maker(targets, target_type, ignore_value):
    """
    Calculate weightings for any imbalanced datasets
    targets - a tensor of target values (i.e. labels)
    target_type - a string indicating the type of target i.e. 'regression', 'binary_cls' etc.
    ignore_value - an integer. If a target matches ignore_value, it will be ignored
    @returns weighting, a torch tensor containing the weightings for each class
    """
    if target_type == 'regression':
        weighting = (targets != ignore_value).type(torch.float)
    elif target_type == 'binary_cls':
        num_pos = (targets == 1.).type(torch.float).sum()
        num_neg = (targets == 0.).type(torch.float).sum()
        if num_pos == 0 or num_neg == 0:
            weighting = (targets != ignore_value).type(torch.float)
        else:
            weight_pos = 1. / num_pos
            weight_neg = 1. / num_neg

            weighting = (targets != ignore_value).type(torch.float)
            weighting[targets == 1.] = weighting[targets == 1.] * weight_pos
            weighting[targets == 0.] = weighting[targets == 0.] * weight_neg
            weighting /= 2.

    elif target_type == 'cls':
        weighting = targets[:, 0] != ignore_value
        valid_cases = targets[weighting, :]
        weighting_out = weighting.view(-1, 1).type(torch.float)
        weighting_out[weighting] = weighting_out[weighting] * (valid_cases / torch.clamp(valid_cases.sum(dim=0, keepdim=True), min=1.)).sum(dim=1, keepdim=True)

        num_presenting_classes = (valid_cases.sum(dim=0) > 0.).type(torch.float).sum()
        weighting = weighting_out / torch.clamp(num_presenting_classes, min=1.)
    elif target_type == 'seg':
        weighting = weighting_maker(targets, 'binary_cls', ignore_value)
    elif target_type == 'ignore':
        weighting = (targets[:, 0:1] != ignore_value).type(torch.float)
    else:
        raise ValueError('Unknown target type: {}'.format(target_type))
    return weighting


def ce(logits, targets):
    """
    Calculate cross-entropy loss
    logits - a tensor containing the logits retrieved from the model
    targets - a tensor containing the target values (labels)
    @returns the cross entropy loss
    """
    loss = -targets * torch.log_softmax(logits, dim=1)
    loss = loss.sum(dim=1, keepdim=True)
    return loss

def normalize(loss, weight):
    """
    Normalize loss according to a weighting.
    loss - a tensor containing the loss values
    weight - a tensor containing weightings for each class
    @returns a tensor containing normalized loss values
    """
    weight_sum = weight.sum(dim=0)
    num_weight = torch.clamp(weight_sum, 1.)
    loss = (loss * weight).sum(dim=0) / num_weight
    return loss


def b_accu(binary_cls_logits, binary_cls_targets, is_logits=True):
    """
    Calculates the binary classification accuracy
    binary_cls_logits - a tensor containing the logits from the binary classification module of a model
    binary_cls_targets - a tensor containing the target values for the binary classification task(s)
    is_logits - True if binary_cls_logits are logits, False if they have been converted to probabilities
    @returns a tensor containing accuracies for each binary classification task
    """
    return (b_pred(binary_cls_logits, is_logits=is_logits) == binary_cls_targets).type(torch.float)


def b_pred(binary_cls_logits, is_logits=True):
    """
    Calculates binary classification predictions
    binary_cls_logits - a tensor containing the logits from the binary classification module of a model
    is_logits - True if binary_cls_logits are logits, False if they have been converted to probabilities
    @returns a tensor containing binary classification predictions for each binary classification task
    """
    if is_logits:
        return (binary_cls_logits > 0).type(torch.float)
    else:
        return (binary_cls_logits > 0.5).type(torch.float)


def accu(logits, targets):
    """
    Calculates the classification accuracy from logits
    logits - a tensor containing the classification logits from the classification module of the model
    targets - a tensor containing the target values for the classification task
    @returns a tensor containing classification accuracy for each classification task
    """
    max_value, max_cls = pred(logits)
    _, target_cls = targets.max(dim=1, keepdim=True)
    return (max_cls == target_cls).type(torch.float)


def pred(logits):
    """
    Identifies the predicted classification task.
    logits - a tensor containing the classification logits from the classification module of the model
    @returns a tensor containing predicted classes for each classification task
    """
    max_value, max_cls = logits.max(dim=1, keepdim=True)
    return max_value, max_cls.float()


def target_maker(targets, config):
    """
    Splits labels into targets for regression, binary classification and classification tasks
    targets - a tensor containing all targets
    config - a dictionary containing configuration settings for this project
    @returns a tuple containing regression targets, binary classification targets and classification targets
    """
    start = 0
    end = len(config['regression_tasks'])
    regression_targets = list()
    if end:
        regression_targets = targets[:, start:end]
        start = end
    end += len(config['binary_cls_tasks'])
    binary_cls_targets = list()
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    binary_cls_targets = targets[:, start:end]

    cls_targets = list()
    for t_name in config['cls_tasks']:
        start = end
        end += config['cls_tasks'][t_name]
        cls_targets.append(targets[:, start:end])

    return regression_targets, binary_cls_targets, cls_targets


def loss_maker(regression_logits, binary_cls_logits, cls_logits,
               regression_targets, binary_cls_targets, cls_targets,
               config):
    """
    Calculates loss values from targets and logits for all tasks
    regression_logits - a tensor containing the logits from the regression module of the model
    binary_cls_logits - a tensor containing the logits from the binary classification module of the model
    cls_logits - a tensor containing the logits from the classification module of the model
    regression_targets - a tensor containing the target values for all regression tasks
    binary_cls_targets - a tensor containing the target values for all binary classification tasks
    cls_targets -  a tensor containing the target values for all classification tasks
    config - a dictionary containing configuration settings for this project
    @returns a dictionary containing labelled losses for all tasks
    """
    if config['regression_tasks']:
        loss_regression = l1(regression_logits, regression_targets)
        loss_regression = normalize(loss_regression, weighting_maker(regression_targets,
                                                                 target_type='regression',
                                                                 ignore_value=config['ignore_value']))

    loss_binary_cls = bce(binary_cls_logits, binary_cls_targets)
    loss_binary_cls = normalize(loss_binary_cls, weighting_maker(binary_cls_targets,
                                                                 target_type='binary_cls',
                                                                 ignore_value=config['ignore_value']))

    loss_cls = list()
    for l, t in zip(cls_logits, cls_targets):
        ls = ce(l, t)
        ls = normalize(ls, weighting_maker(t, target_type='ignore', ignore_value=config['ignore_value']))
        loss_cls.append(ls)

    losses = dict()
    tag = 'loss_'
    for c_idx, c_name in enumerate(config['regression_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        losses[k] = loss_regression[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['binary_cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        losses[k] = loss_binary_cls[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        losses[k] = loss_cls[c_idx]

    return losses


def accuracy_maker(regression_logits, binary_cls_logits, cls_logits,
                   regression_targets, binary_cls_targets, cls_targets,
                   config, is_logits=True):
    """
    Calculates accuracy values from targets and logits for all tasks
    regression_logits - a tensor containing the logits from the regression module of the model
    binary_cls_logits - a tensor containing the logits from the binary classification module of the model
    cls_logits - a tensor containing the logits from the classification module of the model
    regression_targets - a tensor containing the target values for all regression tasks
    binary_cls_targets - a tensor containing the target values for all binary classification tasks
    cls_targets -  a tensor containing the target values for all classification tasks
    config - a dictionary containing configuration settings for this project
    is_logits (optional) - True if binary_cls_logits are logits, False if they have been converted to probabilities
    @returns a dictionary containing labelled accuracies for all tasks and the number of valid cases
    """
    if config['regression_tasks']:
        accu_regression = l1(regression_logits, regression_targets)
        valid_cases_regression = (regression_targets != config['ignore_value']).type(torch.float)
        num_valid_cases_regression = valid_cases_regression.sum(dim=0)
        accu_regression = normalize(accu_regression, valid_cases_regression)

    accu_binary_cls = b_accu(binary_cls_logits, binary_cls_targets, is_logits=is_logits)
    valid_cases_binary_cls = (binary_cls_targets != config['ignore_value']).type(torch.float)
    num_valid_cases_binary_cls = valid_cases_binary_cls.sum(dim=0)
    accu_binary_cls = normalize(accu_binary_cls, valid_cases_binary_cls)

    accu_cls = list()
    num_valid_cases_cls = list()
    for l, t in zip(cls_logits, cls_targets):
        au = accu(l, t)
        vc = (t[:, 0:1] != config['ignore_value']).type(torch.float)
        num_valid_cases_cls.append(vc.sum(dim=0))

        au = normalize(au, vc)
        accu_cls.append(au)

    accus = dict()
    num_valid_cases = dict()
    tag = 'accu_'
    tag_vc = 'nvac_'
    for c_idx, c_name in enumerate(config['regression_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        accus[k] = accu_regression[c_idx:c_idx + 1]

        k_vc = tag_vc + c_name[:min(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_regression[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['binary_cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        accus[k] = accu_binary_cls[c_idx:c_idx + 1]

        k_vc = tag_vc + c_name[:min(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_binary_cls[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        accus[k] = accu_cls[c_idx]

        k_vc = tag_vc + c_name[:min(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_cls[c_idx]

    return accus, num_valid_cases


def prediction_maker(regression_logits, binary_cls_logits, cls_logits,
                     regression_targets, binary_cls_targets, cls_targets,
                     config, is_logits=True):
    """
    Calculates predications from targets and logits for all tasks
    regression_logits - a tensor containing the logits from the regression module of the model
    binary_cls_logits - a tensor containing the logits from the binary classification module of the model
    cls_logits - a tensor containing the logits from the classification module of the model
    regression_targets - a tensor containing the target values for all regression tasks
    binary_cls_targets - a tensor containing the target values for all binary classification tasks
    cls_targets -  a tensor containing the target values for all classification tasks
    config - a dictionary containing configuration settings for this project
    is_logits (optional) - True if binary_cls_logits are logits, False if they have been converted to probabilities
    @returns a dictionary containing labelled predictions for all tasks, a dictionary containing labelled logits for all
    tasks, and a dictionary containing labelled targets for all tasks
    """
    pred_regression = regression_logits
    prob_binary_cls = binary_cls_logits
    pred_binary_cls = b_pred(binary_cls_logits, is_logits=is_logits)

    pred_cls = list()
    prob_cls = cls_logits
    for l in cls_logits:
        _, p = pred(l)
        pred_cls.append(p)

    preds = dict()
    probs = dict()
    targets = dict()
    tag = 'pred_'
    for c_idx, c_name in enumerate(config['regression_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        preds[k] = pred_regression[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['binary_cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        preds[k] = pred_binary_cls[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        preds[k] = pred_cls[c_idx]

    tag = 'prob_'
    for c_idx, c_name in enumerate(config['regression_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        probs[k] = pred_regression[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['binary_cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        probs[k] = prob_binary_cls[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        probs[k] = prob_cls[c_idx]

    tag = 'tar_'
    for c_idx, c_name in enumerate(config['regression_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        targets[k] = regression_targets[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['binary_cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        targets[k] = binary_cls_targets[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        max_v, max_idx = cls_targets[c_idx].max(dim=1, keepdim=True)
        max_idx[max_v == config['ignore_value']] = config['ignore_value']
        targets[k] = max_idx.float()

    return preds, probs, targets


def map_maker(regression_maps, binary_cls_maps, cls_maps, config):
    """
    Create a dictionary of labelled maps (e.g. class activation maps) associated with each task
    regression_maps - maps associated with regression tasks
    binary_cls_maps - maps associated with binary classification tasks
    cls_maps - maps associated with classification tasks
    config - a dictionary containing configuration settings for this project
    @returns a dictionary containing labelled maps for all tasks
    """
    tag = 'map_'
    maps = dict()
    for c_idx, c_name in enumerate(config['regression_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        maps[k] = regression_maps[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['binary_cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]
        maps[k] = binary_cls_maps[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(config['cls_tasks']):
        k = tag + c_name[:min(4, len(c_name))]

        maps[k] = cls_maps[c_idx]

    return maps
