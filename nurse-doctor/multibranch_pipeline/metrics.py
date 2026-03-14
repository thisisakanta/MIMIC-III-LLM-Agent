import numpy as np
from sklearn import metrics


def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    # Keep label order stable so the confusion matrix is always 2x2.
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1), labels=[0, 1])
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    eps = 1e-8
    acc = (cf[0][0] + cf[1][1]) / max(np.sum(cf), eps)
    prec0 = cf[0][0] / max(cf[0][0] + cf[1][0], eps)
    prec1 = cf[1][1] / max(cf[1][1] + cf[0][1], eps)
    rec0 = cf[0][0] / max(cf[0][0] + cf[0][1], eps)
    rec1 = cf[1][1] / max(cf[1][1] + cf[1][0], eps)

    auroc = float("nan")
    auprc = float("nan")
    minpse = float("nan")
    if len(np.unique(y_true)) == 2:
        auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
        precisions, recalls, _ = metrics.precision_recall_curve(y_true, predictions[:, 1])
        auprc = metrics.auc(recalls, precisions)
        minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
    }
