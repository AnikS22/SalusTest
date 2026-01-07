"""
SALUS MVP Predictor Evaluation
Evaluate trained predictor on test data

Usage:
    python scripts/evaluate_mvp.py --checkpoint checkpoints/mvp/20260102_120000/best_f1.pth --data data/mvp_episodes/20260102_120000
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from salus.core.predictor_mvp import SALUSPredictorMVP
from salus.data.dataset_mvp import create_dataloaders


def evaluate(model, dataloader, device):
    """
    Comprehensive evaluation of the predictor

    Returns:
        Dictionary with all metrics
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward
            output = model(signals)
            probs = output['probs']  # (B, 4)

            # Store
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append((probs > 0.5).float().cpu())

    # Concatenate
    all_probs = torch.cat(all_probs, dim=0).numpy()  # (N, 4)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, 4)
    all_preds = torch.cat(all_preds, dim=0).numpy()  # (N, 4)

    # Compute metrics per class
    results = {
        'per_class': [],
        'overall': {}
    }

    failure_names = ['Collision', 'Drop', 'Miss', 'Timeout']

    print("\n" + "="*70)
    print("Per-Class Metrics")
    print("="*70)

    for i, name in enumerate(failure_names):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_probs[:, i]

        # Basic metrics
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        # AUROC (if we have positive samples)
        auroc = None
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            try:
                auroc = roc_auc_score(y_true, y_prob)
            except:
                auroc = None

        results['per_class'].append({
            'name': name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auroc': auroc,
            'support': int(y_true.sum())
        })

        print(f"\n{name}:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1:        {f1:.3f}")
        print(f"   Accuracy:  {accuracy:.3f}")
        if auroc is not None:
            print(f"   AUROC:     {auroc:.3f}")
        print(f"   Support:   {int(y_true.sum())}")

    # Overall metrics
    mean_precision = np.mean([m['precision'] for m in results['per_class']])
    mean_recall = np.mean([m['recall'] for m in results['per_class']])
    mean_f1 = np.mean([m['f1'] for m in results['per_class']])

    # Exact match accuracy (all labels correct)
    exact_match = (all_preds == all_labels).all(axis=1).mean()

    results['overall'] = {
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'exact_match_accuracy': exact_match
    }

    print("\n" + "="*70)
    print("Overall Metrics")
    print("="*70)
    print(f"Mean Precision: {mean_precision:.3f}")
    print(f"Mean Recall:    {mean_recall:.3f}")
    print(f"Mean F1:        {mean_f1:.3f}")
    print(f"Exact Match:    {exact_match:.3f}")
    print("="*70 + "\n")

    return results, all_probs, all_labels, all_preds


def plot_confusion_matrices(all_labels, all_preds, save_dir):
    """Plot confusion matrices for each class"""
    failure_names = ['Collision', 'Drop', 'Miss', 'Timeout']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, failure_names)):
        y_true = all_labels[:, i].astype(int)
        y_pred = all_preds[:, i].astype(int)

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])

    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150)
    plt.close()

    print(f"📊 Saved confusion matrices to {save_dir / 'confusion_matrices.png'}")


def plot_roc_curves(all_labels, all_probs, save_dir):
    """Plot ROC curves for each class"""
    failure_names = ['Collision', 'Drop', 'Miss', 'Timeout']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, failure_names)):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]

        # Only plot if we have both positive and negative samples
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auroc = roc_auc_score(y_true, y_prob)

                ax.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{name} ROC Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax.set_title(f'{name} ROC Curve')
        else:
            ax.text(0.5, 0.5, 'No positive samples', ha='center', va='center')
            ax.set_title(f'{name} ROC Curve')

    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=150)
    plt.close()

    print(f"📊 Saved ROC curves to {save_dir / 'roc_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SALUS MVP Predictor')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pth file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save evaluation plots')

    args = parser.parse_args()

    print("="*70)
    print("SALUS MVP Predictor Evaluation")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data: {args.data}")
    print(f"   Device: {args.device}")

    # Load model
    print(f"\n🧠 Loading Model...")
    device = torch.device(args.device)

    model = SALUSPredictorMVP(
        signal_dim=6,
        hidden_dim=64,
        num_failure_types=4
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_f1' in checkpoint:
        print(f"   Validation F1: {checkpoint['val_f1']:.3f}")

    # Load data (use validation split)
    print(f"\n💾 Loading Data...")
    _, val_loader = create_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        train_ratio=0.8,
        num_workers=2
    )

    print(f"   Val batches: {len(val_loader)}")

    # Evaluate
    print(f"\n🚀 Starting Evaluation...\n")

    results, all_probs, all_labels, all_preds = evaluate(model, val_loader, device)

    # Save plots
    if args.save_plots:
        save_dir = Path(args.checkpoint).parent / 'evaluation'
        save_dir.mkdir(exist_ok=True)

        print(f"\n📊 Generating Plots...")
        plot_confusion_matrices(all_labels, all_preds, save_dir)
        plot_roc_curves(all_labels, all_probs, save_dir)

        # Save results
        import json
        with open(save_dir / 'metrics.json', 'w') as f:
            # Convert numpy types to Python types for JSON
            results_serializable = {
                'per_class': [
                    {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in m.items()}
                    for m in results['per_class']
                ],
                'overall': {k: float(v) for k, v in results['overall'].items()}
            }
            json.dump(results_serializable, f, indent=2)

        print(f"💾 Saved results to {save_dir}")

    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
