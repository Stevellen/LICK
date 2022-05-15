import torchmetrics as tm
import matplotlib.pyplot as plt

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

def get_cmat(matrix, stage):
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(matrix)
    ax.set_title(f"{stage} Confusion Matrix", fontsize=30)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j],
            ha='center', va='center', color='black', fontsize=40)
    fig.tight_layout()
    return fig