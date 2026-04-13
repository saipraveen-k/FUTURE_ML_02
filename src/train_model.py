"""
Model Training Module for Support Ticket Classification
Trains and evaluates multiple classification models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Class for training and evaluating classification models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models with default parameters"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            )
        }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        model = self.models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        return {
            'model_name': model_name,
            'status': 'trained',
            'training_samples': len(X_train)
        }
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training results for all models
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.train_model(model_name, X_train, y_train)
                results[model_name] = result
                print(f"Successfully trained {model_name}")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.trained_models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist() if y_proba is not None else None
        }
        
        # Store results
        self.model_results[model_name] = results
        
        return results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results for all models
        """
        results = {}
        
        for model_name in self.trained_models.keys():
            try:
                result = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = result
                print(f"Successfully evaluated {model_name}")
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on F1 score
        
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if not self.model_results:
            raise ValueError("No models evaluated yet")
        
        best_model = max(self.model_results.items(), key=lambda x: x[1]['f1_score'])
        return best_model
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a trained model
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.trained_models[model_name] = pickle.load(f)
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to JSON file
        
        Args:
            filepath: Path to save the results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model_name, results in self.model_results.items():
            results_copy = results.copy()
            if 'confusion_matrix' in results_copy:
                results_copy['confusion_matrix'] = results_copy['confusion_matrix']
            results_serializable[model_name] = results_copy
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all models
        
        Returns:
            DataFrame with model comparison
        """
        if not self.model_results:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score']
            })
        
        return pd.DataFrame(comparison_data).sort_values('F1 Score', ascending=False)

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str], 
                         title: str = "Confusion Matrix", save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Plot model comparison
    
    Args:
        comparison_df: DataFrame with model comparison
        save_path: Path to save the plot
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.barplot(data=comparison_df, x='Model', y=metric)
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_preprocess_data
    from vectorize import prepare_data_splits, vectorize_data
    
    # Load and preprocess data
    df_processed, _ = load_and_preprocess_data("../data/tickets.csv")
    
    # Prepare data splits
    X_train, X_test, y_train, y_test = prepare_data_splits(df_processed)
    
    # Vectorize data
    X_train_vec, X_test_vec, _ = vectorize_data(X_train, X_test)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    print("Training models...")
    training_results = trainer.train_all_models(X_train_vec, y_train)
    
    # Evaluate all models
    print("\nEvaluating models...")
    evaluation_results = trainer.evaluate_all_models(X_test_vec, y_test)
    
    # Get best model
    best_model_name, best_results = trainer.get_best_model()
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {best_results['f1_score']:.4f}")
    
    # Create comparison table
    comparison_df = trainer.create_comparison_table()
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save best model and results
    trainer.save_model(best_model_name, "../outputs/best_model.pkl")
    trainer.save_results("../outputs/model_results.json")
    
    # Plot confusion matrix for best model
    class_names = list(df_processed['Ticket Type'].unique())
    plot_confusion_matrix(
        best_results['confusion_matrix'], 
        class_names,
        f"Confusion Matrix - {best_model_name}",
        "../outputs/confusion_matrix.png"
    )
    
    # Plot model comparison
    plot_model_comparison(comparison_df, "../outputs/model_comparison.png")
    
    print("\nTraining complete! Models and results saved to outputs directory.")
