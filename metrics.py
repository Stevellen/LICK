import torchmetrics as tm
import matplotlib.pyplot as plt


def get_metrics(num_classes, threshold, beta=1):
    metrics = tm.MetricCollection({
        'accuracy': tm.Accuracy(
            num_classes = num_classes,
            threshold=threshold
        ),
        'recall': tm.Recall(
            num_classes = num_classes,
            threshold=threshold
        ),
        'precision': tm.Precision(
            num_classes = num_classes,
            threshold=threshold
        ),
        'fbeta': tm.FBetaScore(
            num_classes = num_classes,
            threshold=threshold,
            beta=beta
        ),
    })
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