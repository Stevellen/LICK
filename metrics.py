import torchmetrics as tm

METRIC_CONFIG = {
    'train': {
        'acc': (tm.Accuracy, {'threshold': 0.5}),
        'recall': (tm.Recall, {'threshold': 0.5}),
        'precision': (tm.Precision, {'threshold': 0.5}),
        'fbeta': (tm.FBeta, {'threshold': 0.5, 'beta': 1}),
    }
}


def get_metrics(threshold, beta=1):
    metrics = tm.MetricCollection({
        'accuracy': tm.Accuracy(threshold=threshold),
        'recall': tm.Recall(threshold=threshold),
        'precision': tm.Precision(threshold=threshold),
        'fbeta': tm.FBeta(threshold=threshold, beta=beta),
        }
    )
    return metrics.clone(prefix='train_'), metrics.clone(prefix='val_'), metrics.clone(prefix='test_')

def get_cmat(preds, labels, stage='test'):
    return