import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from torch.utils.data import DataLoader
from network import snn
from train import dataset, transform, create_simple_split
import kagglehub
from tqdm import tqdm

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detailed_model_evaluation(model_path='model_last.pth'):
    """Generate comprehensive evaluation metrics and visualizations"""
    
    # Load model
    model = snn().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Get test data
    path = kagglehub.dataset_download("robinreni/signature-verification-dataset")
    _, test_df = create_simple_split()
    test_data = dataset(pairs_data=test_df, root_dir=f'{path}/sign_data/train/', transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Collect predictions and true labels
    all_predictions = []
    all_labels = []
    all_scores = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            scores = torch.sigmoid(output).cpu().numpy()
            predictions = (scores > 0.5).astype(int)
            
            all_scores.extend(scores.flatten())
            all_predictions.extend(predictions.flatten())
            all_labels.extend(label.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    return generate_research_plots(all_labels, all_predictions, all_scores)

def generate_research_plots(y_true, y_pred, y_scores):
    """Generate comprehensive research-level visualizations"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix with detailed metrics
    ax1 = plt.subplot(3, 4, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Forged'], 
                yticklabels=['Genuine', 'Forged'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    plt.text(2.5, 0.5, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # 2. ROC Curve
    ax2 = plt.subplot(3, 4, 2)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Analysis', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = plt.subplot(3, 4, 3)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Score Distribution
    ax4 = plt.subplot(3, 4, 4)
    genuine_scores = y_scores[y_true == 0]
    forged_scores = y_scores[y_true == 1]
    
    plt.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine', color='green', density=True)
    plt.hist(forged_scores, bins=30, alpha=0.7, label='Forged', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Score Distribution by Class', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Error Analysis
    ax5 = plt.subplot(3, 4, 5)
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    
    error_data = [true_negatives, false_positives, false_negatives, true_positives]
    error_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    colors = ['lightgreen', 'lightcoral', 'orange', 'darkgreen']
    
    plt.pie(error_data, labels=error_labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Classification Results Breakdown', fontweight='bold')
    
    # 6. Statistical Analysis
    ax6 = plt.subplot(3, 4, 6)
    stats_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
        'Score': [accuracy, precision, recall, f1, roc_auc, pr_auc]
    }
    
    bars = plt.bar(stats_data['Metric'], stats_data['Score'], color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'purple'])
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics Summary', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    
    # Add value labels on bars
    for bar, score in zip(bars, stats_data['Score']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Threshold Analysis
    ax7 = plt.subplot(3, 4, 7)
    thresholds = np.arange(0.1, 0.9, 0.05)
    threshold_metrics = []
    
    for threshold in thresholds:
        pred_threshold = (y_scores > threshold).astype(int)
        acc = accuracy_score(y_true, pred_threshold)
        prec = precision_score(y_true, pred_threshold, zero_division=0)
        rec = recall_score(y_true, pred_threshold, zero_division=0)
        f1_thresh = f1_score(y_true, pred_threshold, zero_division=0)
        threshold_metrics.append([acc, prec, rec, f1_thresh])
    
    threshold_metrics = np.array(threshold_metrics)
    
    plt.plot(thresholds, threshold_metrics[:, 0], 'o-', label='Accuracy', linewidth=2)
    plt.plot(thresholds, threshold_metrics[:, 1], 's-', label='Precision', linewidth=2)
    plt.plot(thresholds, threshold_metrics[:, 2], '^-', label='Recall', linewidth=2)
    plt.plot(thresholds, threshold_metrics[:, 3], 'd-', label='F1-Score', linewidth=2)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Current Threshold')
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Score')
    plt.title('Threshold Sensitivity Analysis', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Class-wise Performance
    ax8 = plt.subplot(3, 4, 8)
    class_metrics = pd.DataFrame({
        'Class': ['Genuine (0)', 'Forged (1)'],
        'Precision': [tn/(tn+fn) if (tn+fn)>0 else 0, tp/(tp+fp) if (tp+fp)>0 else 0],
        'Recall': [tn/(tn+fp) if (tn+fp)>0 else 0, tp/(tp+fn) if (tp+fn)>0 else 0],
        'Support': [np.sum(y_true == 0), np.sum(y_true == 1)]
    })
    
    x = np.arange(len(class_metrics['Class']))
    width = 0.35
    
    plt.bar(x - width/2, class_metrics['Precision'], width, label='Precision', color='lightblue')
    plt.bar(x + width/2, class_metrics['Recall'], width, label='Recall', color='lightcoral')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Class-wise Performance Analysis', fontweight='bold')
    plt.xticks(x, class_metrics['Class'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Feature Importance (Score ranges)
    ax9 = plt.subplot(3, 4, 9)
    score_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    range_counts = []
    range_accuracy = []
    
    for low, high in score_ranges:
        mask = (y_scores >= low) & (y_scores < high)
        count = np.sum(mask)
        if count > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
        else:
            acc = 0
        range_counts.append(count)
        range_accuracy.append(acc)
    
    range_labels = [f'{low:.1f}-{high:.1f}' for low, high in score_ranges]
    
    ax9_twin = ax9.twinx()
    bars1 = ax9.bar(range_labels, range_counts, alpha=0.7, color='skyblue', label='Sample Count')
    bars2 = ax9_twin.plot(range_labels, range_accuracy, 'ro-', linewidth=2, markersize=8, label='Accuracy')
    
    ax9.set_xlabel('Score Range')
    ax9.set_ylabel('Number of Samples', color='blue')
    ax9_twin.set_ylabel('Accuracy', color='red')
    ax9.set_title('Performance vs Score Range', fontweight='bold')
    plt.setp(ax9.get_xticklabels(), rotation=45)
    
    # 10. Detailed Statistics Table
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('tight')
    ax10.axis('off')
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    table_data = [
        ['Total Samples', f'{len(y_true)}'],
        ['Genuine Samples', f'{np.sum(y_true == 0)} ({100*np.sum(y_true == 0)/len(y_true):.1f}%)'],
        ['Forged Samples', f'{np.sum(y_true == 1)} ({100*np.sum(y_true == 1)/len(y_true):.1f}%)'],
        ['Accuracy', f'{accuracy:.4f}'],
        ['Precision', f'{precision:.4f}'],
        ['Recall (Sensitivity)', f'{sensitivity:.4f}'],
        ['Specificity', f'{specificity:.4f}'],
        ['F1-Score', f'{f1:.4f}'],
        ['AUC-ROC', f'{roc_auc:.4f}'],
        ['AUC-PR', f'{pr_auc:.4f}'],
        ['False Positive Rate', f'{fpr_rate:.4f}'],
        ['False Negative Rate', f'{fnr_rate:.4f}']
    ]
    
    table = ax10.table(cellText=table_data, 
                      colLabels=['Metric', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
    
    ax10.set_title('Detailed Performance Statistics', fontweight='bold', pad=20)
    
    # 11. Model Confidence Analysis
    ax11 = plt.subplot(3, 4, 11)
    confidence = np.abs(y_scores - 0.5) * 2  # Convert to 0-1 confidence scale
    
    plt.scatter(confidence[y_true == 0], y_scores[y_true == 0], 
                alpha=0.6, label='Genuine', color='green', s=20)
    plt.scatter(confidence[y_true == 1], y_scores[y_true == 1], 
                alpha=0.6, label='Forged', color='red', s=20)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Model Confidence')
    plt.ylabel('Similarity Score')
    plt.title('Confidence vs Score Analysis', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Error Distribution
    ax12 = plt.subplot(3, 4, 12)
    errors = np.abs(y_true - y_scores)
    
    plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(np.mean(errors), color='red', linestyle='--', 
                label=f'Mean Error: {np.mean(errors):.4f}')
    plt.axvline(np.median(errors), color='blue', linestyle='--', 
                label=f'Median Error: {np.median(errors):.4f}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Analysis', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("="*80)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    print(f"Total Test Samples: {len(y_true)}")
    print(f"Genuine Samples: {np.sum(y_true == 0)} ({100*np.sum(y_true == 0)/len(y_true):.1f}%)")
    print(f"Forged Samples: {np.sum(y_true == 1)} ({100*np.sum(y_true == 1)/len(y_true):.1f}%)")
    print("="*80)
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'auc_roc': roc_auc, 'auc_pr': pr_auc,
        'confusion_matrix': cm, 'total_samples': len(y_true)
    }

if __name__ == "__main__":
    print("Starting comprehensive model evaluation...")
    results = detailed_model_evaluation()
    print("Analysis complete! Check 'comprehensive_model_analysis.png' for detailed visualizations.")