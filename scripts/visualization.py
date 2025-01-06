import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, auc

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

def plot_confusion_matrix_each(y_true, y_pred, title):
    """
    Plot a confusion matrix for binary classification.
    """
    # No need for argmax, as the predictions are 1D for binary classification
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

    plt.title(f"{title} Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(y_true, y_pred, model_name):
    # Ensure the inputs are NumPy arrays
    y_true = y_true.to_numpy().ravel()
    y_pred = y_pred.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    return auc

def plot_model_comparison(models, accuracies, aucs):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, accuracies, alpha=0.7, label='Accuracy')
    ax.bar(models, aucs, alpha=0.7, label='AUC', width=0.5)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    plt.legend()
    plt.show()

def plot_accuracy_graphs_each(results, num_folds):
    """
        Visualize training vs. validation performance for each antibiotic.
        :param results: Dictionary containing results of each fold for antibiotics.
        :param num_folds: Number of folds used for cross-validation.
        """

    antibiotics = list(results.keys())

    # Plot for each antibiotic
    for antibiotic in antibiotics:
        train_accuracies = results[antibiotic]['train_accuracy']
        val_accuracies = results[antibiotic]['val_accuracy']

        # Create a new figure for each antibiotic
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_folds + 1), train_accuracies, label='Training Accuracy', marker='o')
        plt.plot(range(1, num_folds + 1), val_accuracies, label='Validation Accuracy', marker='o')

        plt.title(f'Training vs. Validation Accuracy: {antibiotic}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.xticks(range(1, num_folds + 1))
        plt.legend()
        plt.grid(True)
        plt.show()
def plot_roc_curve_per_antibiotic(y_train, y_train_pred, y_val, y_val_pred, antibiotic):
    """
    Plot ROC curves for training and validation datasets for a specific antibiotic.
    """
    plt.figure(figsize=(10, 6))

    # ROC curve for training data
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, label=f'Training AUC = {auc_train:.2f}', linestyle='--')

    # ROC curve for validation data
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    auc_val = auc(fpr_val, tpr_val)
    plt.plot(fpr_val, tpr_val, label=f'Validation AUC = {auc_val:.2f}')

    # Plot formatting
    plt.title(f"ROC Curve for {antibiotic}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def plot_roc_curves_for_all_antibiotics(results):
    """
    Plot all ROC curves for training and validation datasets across all antibiotics.
    """
    antibiotics = ['CIP', 'CTX', 'CTZ', 'GEN']
    for antibiotic in antibiotics:
        metrics = results[antibiotic]
        plot_roc_curve_per_antibiotic(
            metrics['y_train'], metrics['y_train_pred'],
            metrics['y_val'], metrics['y_val_pred'],
            antibiotic
        )



def plot_accuracy_graphs(results, num_folds):
    """
    Plots accuracy graphs for training and validation data across folds
    for each antibiotic (CIP, CTX, CTZ, GEN).
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    antibiotics = ['CIP', 'CTX', 'CTZ', 'GEN']
    axes = axes.ravel()

    for idx, antibiotic in enumerate(antibiotics):
        metrics = results[antibiotic]

        axes[idx].plot(
            range(1, num_folds + 1), metrics['train_acc'],
            label='Training Accuracy', marker='o'
        )
        axes[idx].plot(
            range(1, num_folds + 1), metrics['val_acc'],
            label='Validation Accuracy', marker='o'
        )

        max_train = np.max(metrics['train_acc'])
        max_val = np.max(metrics['val_acc'])
        mean_train = np.mean(metrics['train_acc'])
        mean_val = np.mean(metrics['val_acc'])

        # Add a legend with max and mean accuracy values
        legend = (
            f"Max Train Acc: {max_train:.2f}, Mean Train Acc: {mean_train:.2f}\n"
            f"Max Val Acc: {max_val:.2f}, Mean Val Acc: {mean_val:.2f}"
        )
        axes[idx].legend(title=legend)

        axes[idx].set_title(f"{antibiotic} Accuracy over {num_folds} Folds")
        axes[idx].set_xlabel("Fold")
        axes[idx].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()