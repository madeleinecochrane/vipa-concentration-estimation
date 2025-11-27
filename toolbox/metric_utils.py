import io

import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def normalized_accuracy(y_true, y_pred):

    # unique_labels = np.array([0, 1])
    unique_labels = np.unique(y_true)
    assert np.all([p in unique_labels for p in np.unique(y_pred)])

    unique_labels = np.sort(unique_labels)

    accu_dict = dict()

    for l in unique_labels:
        sel = y_true == l
        correct = np.sum((y_pred == y_true)[sel])
        total = np.sum(sel)
        accu_l = correct/total
        # print('[{}] correct: {:d}, total: {:d}, accuracy: {:0.3f}'.format(l, correct, total, accu_l))
        accu_dict[l] = accu_l
    cmt = cm(y_true, y_pred)

    # print(cmt)

    cmt_norm = cmt / np.expand_dims(np.sum(cmt, axis=1), axis=1)

    # print(cmt_norm)
    # mat_pretty_print(cmt_norm)

    return cmt_norm


def confusion_matrix(y_true, y_pred, normalize=None):
    cmt = cm(y_true, y_pred, normalize=normalize)
    return cmt


def binary_classification_metrics(y_true, y_pred, y_prob=None):
    assert set(np.unique(y_true)) == {0, 1}
    cmt = confusion_matrix(y_true, y_pred, normalize=None)
    tp = cmt[1, 1]
    tn = cmt[0, 0]
    fp = cmt[0, 1]
    fn = cmt[1, 0]
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)
    tpr = sensitivity
    recall = sensitivity

    specificity = tn / (tn + fp + 1e-10)
    tnr = specificity
    selectivity = specificity

    precision = tp / (tp + fp + 1e-10)
    npv = tn / (tn + fn + 1e-10)

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        auc = float('nan')
        ap = float('nan')

    f1 = 2 * precision * sensitivity / (precision + recall + 1e-10)
    # f2 = 2 / (1/recall + 1/precision)
    # f3 = 2 * tp / (2 * tp + fp + fn)
    cmt_norm = confusion_matrix(y_true, y_pred, normalize='true')
    return {'confusion matrix (unnormalized)': cmt, 'confusion matrix (normalized)': cmt_norm,
            'true positive': tp, 'true negative': tn,
            'false positive': fp, 'false negative': fn,
            'true positive rate': tpr,
            'true negative rate': tnr,
            'false positive rate': 1 - tnr,
            'false negative rate': 1 - tpr,
            'selectivity': selectivity,
            'precision': precision,
            'recall': recall,
            'ppv': precision,
            'npv': npv,
            'f1': f1,
            'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity,
            'auc': auc,
            'roc_curve': roc_curve(y_true, y_prob) if y_prob is not None else None,
            'ap': ap
            }


def multi_classification_metrics(y_true, y_pred):
    assert len(set(np.unique(y_true))) != 1
    cmt = confusion_matrix(y_true, y_pred, normalize=None)

    accuracy = np.sum(y_true == y_pred) / y_true.shape[0]

    cmt_norm = confusion_matrix(y_true, y_pred, normalize='true')
    return {'confusion matrix (unnormalized)': cmt, 'confusion matrix (normalized)': cmt_norm,
            'accuracy': accuracy
            }


def mat_pretty_print(m):

    print(' |', end=' ')
    for r_idx, r in enumerate(m):
        print('{:4d}'.format(r_idx), end=' ')
    print('')
    print('-'*(2+len(m)*5))

    for r_idx, r in enumerate(m):
        print('{}|'.format(r_idx), end=' ')
        for e_idx, e in enumerate(r):
            if int(e) == e:
                format_str = '{:d}'
                e = int(e)
            else:
                format_str = '{:0.2f}'
            print(format_str.format(e), end=' ')
        print('')
    print('')


def mat_pretty_print_to_string(m, data_type, type_dict):

    string = io.StringIO()
    print('<code>G\P&nbsp;&#124;', end=' ', file=string)
    for r_idx, r in enumerate(m):
        print('{:4s}'.format(type_dict[r_idx]), end=' ', file=string)
    print('</code>', end='<br>', file=string)
    print('<code>' + '-'*(4+len(m)*5) + '</code>', end='<br>', file=string)

    for r_idx, r in enumerate(m):
        print('<code>{}&nbsp;&#124;</code>'.format(type_dict[r_idx]), end=' ', file=string)
        for e_idx, e in enumerate(r):
            if int(e) == e:
                format_str = '{:d}'
                e = int(e)
            else:
                if data_type == 'num':
                    format_str = '{:0.1f}'
                elif data_type == 'percent':
                    format_str = '{:0.3f}'
                else:
                    raise ValueError('Wrong data type')
            print(format_str.format(e), end=' ', file=string)
        print('', end='<br>', file=string)
    print('', end='<br>', file=string)

    return string.getvalue()


def mat_pretty_info(info):

    for i in info:
        print('{}:'.format(i), end='')
        if 'matrix' in i:
            print()
            mat_pretty_print(info[i])
        else:
            print(' {:0.3f}'.format(info[i]))


def get_optimal_threshold_roc(y1, s1s):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    for s1 in s1s:
        fpr, tpr, thresholds = roc_curve(y1, s1)
        #  Youden’s J statistic
        J = tpr - fpr
        idx = np.nanargmax(J)
        opt_thresh = thresholds[idx]

        thresh1.append(opt_thresh)

    thresh1 = np.median(thresh1)

    return thresh1


def get_optimal_threshold_pr(y1, s1s):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    for s1 in s1s:
        precision, recall, thresholds = precision_recall_curve(y1, s1)
        #  F-Measure
        fscore = (2 * precision * recall) / (precision + recall + 1e-10)
        idx = np.nanargmax(fscore)
        opt_thresh = thresholds[idx]

        thresh1.append(opt_thresh)

    thresh1 = np.median(thresh1)

    return thresh1


def get_optimal_threshold(y1, s1s, args):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    target_tpr1 = args.target_tpr
    for s1 in s1s:
        # tpr threshold for level 1
        s1_pos_sorted = np.sort(s1[y1 == 1])
        t1 = s1_pos_sorted[int(np.round(len(s1_pos_sorted) * (1 - target_tpr1)))]
        thresh1.append(t1)

    thresh1 = np.median(thresh1)

    return thresh1


def optimize_threshold(y, s, tpr, threshold_method=None):
    if threshold_method == 'roc':
        thresh = get_optimal_threshold_roc(y, [s])
    elif threshold_method == 'pr':
        thresh = get_optimal_threshold_pr(y, [s])
    elif threshold_method == 'tpr':
        s_pos_sorted = np.sort(s[y == 1])
        thresh = s_pos_sorted[int(np.round(len(s_pos_sorted) * (1 - tpr)))]
    elif threshold_method == 'default':
        thresh = 0.5
    else:
        raise ValueError('Unknown threshold method.')
    return thresh
