"""
Evaluation Module for Support Ticket Classification
Comprehensive evaluation and reporting of model performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import os

class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, output_dir: str = "../outputs"):
        """
        Initialize the evaluator
        
        Args:
            output_dir: Directory to save evaluation outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = class_report
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        # Per-class metrics
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        per_class_metrics = {}
        
        for label in unique_labels:
            label_mask = (y_true == label)
            if np.sum(label_mask) > 0:
                per_class_metrics[str(label)] = {
                    'precision': precision_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)[0],
                    'recall': recall_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)[0],
                    'f1_score': f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)[0],
                    'support': np.sum(label_mask)
                }
        
        metrics['per_class_metrics'] = per_class_metrics
        
        return metrics
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, class_names: List[str],
                             title: str = "Confusion Matrix", normalize: bool = False) -> str:
        """
        Plot and save confusion matrix
        
        Args:
            conf_matrix: Confusion matrix
            class_names: List of class names
            title: Plot title
            normalize: Whether to normalize the matrix
            
        Returns:
            Path to saved plot
        """
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text for normalization
        if normalize:
            plt.text(0.5, -0.15, 'Normalized values', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=10)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str]) -> str:
        """
        Plot class distribution comparison
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        true_counts = pd.Series(y_true).value_counts().sort_index()
        ax1.bar(class_names, [true_counts.get(i, 0) for i in range(len(class_names))])
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        ax2.bar(class_names, [pred_counts.get(i, 0) for i in range(len(class_names))])
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Plot model comparison chart
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Path to saved plot
        """
        # Extract metrics for comparison
        models = list(results.keys())
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        # Create DataFrame for plotting
        comparison_data = []
        for model in models:
            for metric in metrics:
                if metric in results[model]:
                    comparison_data.append({
                        'Model': model,
                        'Metric': metric.replace('_weighted', '').title(),
                        'Score': results[model][metric]
                    })
        
        df = pd.DataFrame(comparison_data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_evaluation_report(self, results: Dict[str, Dict[str, Any]], 
                                 class_names: List[str]) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Dictionary of model results
            class_names: List of class names
            
        Returns:
            Path to saved report
        """
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("SUPPORT TICKET CLASSIFICATION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Model comparison table
            f.write("MODEL COMPARISON TABLE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10}\n")
            f.write("-" * 40 + "\n")
            
            for model_name, model_results in results.items():
                if 'accuracy' in model_results and 'f1_weighted' in model_results:
                    f.write(f"{model_name:<20} {model_results['accuracy']:<10.4f} {model_results['f1_weighted']:<10.4f}\n")
            
            f.write("\n")
            
            # Best model
            best_model = max(results.items(), key=lambda x: x[1].get('f1_weighted', 0))
            f.write(f"BEST MODEL: {best_model[0]}\n")
            f.write(f"F1 Score: {best_model[1]['f1_weighted']:.4f}\n\n")
            
            # Detailed metrics for best model
            f.write("DETAILED METRICS - BEST MODEL\n")
            f.write("-" * 40 + "\n")
            best_results = best_model[1]
            
            f.write(f"Accuracy: {best_results['accuracy']:.4f}\n")
            f.write(f"Precision (Weighted): {best_results['precision_weighted']:.4f}\n")
            f.write(f"Recall (Weighted): {best_results['recall_weighted']:.4f}\n")
            f.write(f"F1 Score (Weighted): {best_results['f1_weighted']:.4f}\n\n")
            
            # Per-class metrics
            if 'per_class_metrics' in best_results:
                f.write("PER-CLASS METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}\n")
                f.write("-" * 40 + "\n")
                
                for class_name, metrics in best_results['per_class_metrics'].items():
                    f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                           f"{metrics['f1_score']:<10.4f} {metrics['support']:<10}\n")
            
            f.write("\n")
            
            # Business insights
            f.write("BUSINESS INSIGHTS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Classification Performance:\n")
            f.write(f"   - The best model achieves {best_results['accuracy']:.1%} accuracy\n")
            f.write(f"   - This means {best_results['accuracy']:.1%} of tickets are correctly categorized\n")
            f.write(f"   - Reduces manual categorization effort by {best_results['accuracy']:.1%}\n\n")
            
            f.write("2. Priority Prediction Benefits:\n")
            f.write("   - Rule-based priority system enables automatic ticket triage\n")
            f.write("   - High-priority tickets can be routed to senior staff immediately\n")
            f.write("   - Reduces average response time for critical issues\n\n")
            
            f.write("3. Operational Impact:\n")
            f.write("   - Automated classification reduces processing time per ticket\n")
            f.write("   - Consistent categorization improves service quality\n")
            f.write("   - Enables better resource allocation based on ticket volume\n\n")
            
            f.write("4. Recommendations:\n")
            f.write("   - Deploy model in production with human review for initial period\n")
            f.write("   - Monitor model performance and retrain quarterly\n")
            f.write("   - Consider adding more categories for better granularity\n")
        
        return report_path
    
    def save_metrics_json(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Save evaluation metrics as JSON
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Path to saved JSON file
        """
        json_path = os.path.join(self.output_dir, 'metrics.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {}
            for key, value in model_results.items():
                if isinstance(value, np.ndarray):
                    json_results[model_name][key] = value.tolist()
                elif hasattr(value, '__dict__'):
                    json_results[model_name][key] = str(value)
                else:
                    json_results[model_name][key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return json_path
    
    def evaluate_complete(self, results: Dict[str, Dict[str, Any]], 
                         class_names: List[str]) -> Dict[str, str]:
        """
        Perform complete evaluation and generate all outputs
        
        Args:
            results: Dictionary of model results
            class_names: List of class names
            
        Returns:
            Dictionary with paths to all generated files
        """
        output_files = {}
        
        # Generate plots
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x].get('f1_weighted', 0))
            best_results = results[best_model_name]
            
            # Confusion matrix for best model
            if 'confusion_matrix' in best_results:
                conf_matrix = np.array(best_results['confusion_matrix'])
                output_files['confusion_matrix'] = self.plot_confusion_matrix(
                    conf_matrix, class_names, f"Confusion Matrix - {best_model_name}"
                )
            
            # Model comparison
            output_files['model_comparison'] = self.plot_model_comparison(results)
        
        # Generate report
        output_files['evaluation_report'] = self.generate_evaluation_report(results, class_names)
        
        # Save JSON metrics
        output_files['metrics_json'] = self.save_metrics_json(results)
        
        return output_files

if __name__ == "__main__":
    # Example usage
    print("Model evaluation module ready!")
    print("Use this module after training models to generate comprehensive evaluation reports.")
