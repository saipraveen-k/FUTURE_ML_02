"""
Feature Extraction Module for Support Ticket Classification
Handles text vectorization using TF-IDF and Bag of Words
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

class TextVectorizer:
    """Class for vectorizing text data using various methods"""
    
    def __init__(self, vectorizer_type: str = 'tfidf', **vectorizer_params):
        """
        Initialize the vectorizer
        
        Args:
            vectorizer_type: Type of vectorizer ('tfidf' or 'bow')
            **vectorizer_params: Additional parameters for the vectorizer
        """
        self.vectorizer_type = vectorizer_type
        self.vectorizer_params = vectorizer_params
        self.vectorizer = None
        self.feature_names = None
        
        # Default parameters
        self.default_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': 'english'
        }
        
        # Merge with user parameters
        self.params = {**self.default_params, **vectorizer_params}
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fit vectorizer and transform texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Vectorized text matrix
        """
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(**self.params)
        elif self.vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer(**self.params)
        else:
            raise ValueError("vectorizer_type must be 'tfidf' or 'bow'")
        
        # Fit and transform
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return X.toarray()
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts: List of text strings
            
        Returns:
            Vectorized text matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        X = self.vectorizer.transform(texts)
        return X.toarray()
    
    def get_feature_names(self) -> list:
        """
        Get feature names
        
        Returns:
            List of feature names
        """
        return self.feature_names.tolist() if self.feature_names is not None else []
    
    def get_top_features(self, n: int = 20) -> Dict[str, Any]:
        """
        Get top features based on TF-IDF scores
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary with top features and their scores
        """
        if self.vectorizer is None or self.vectorizer_type != 'tfidf':
            return {}
        
        # Get TF-IDF scores
        tfidf_scores = self.vectorizer.idf_
        feature_scores = list(zip(self.feature_names, tfidf_scores))
        
        # Sort by score (lower IDF = more important)
        feature_scores.sort(key=lambda x: x[1])
        
        return {
            'top_features': feature_scores[:n],
            'bottom_features': feature_scores[-n:]
        }
    
    def save_vectorizer(self, filepath: str):
        """
        Save the fitted vectorizer
        
        Args:
            filepath: Path to save the vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("No vectorizer to save. Fit the vectorizer first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_vectorizer(self, filepath: str):
        """
        Load a fitted vectorizer
        
        Args:
            filepath: Path to the saved vectorizer
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.feature_names = self.vectorizer.get_feature_names_out()

def prepare_data_splits(df: pd.DataFrame, text_column: str = 'processed_text', 
                       label_column: str = 'Ticket Type', test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare train/test splits for modeling
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        label_column: Name of the label column
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Extract features and labels
    X = df[text_column].tolist()
    y = df[label_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def vectorize_data(X_train: list, X_test: list, vectorizer_type: str = 'tfidf',
                  save_path: str = None) -> Tuple[np.ndarray, np.ndarray, TextVectorizer]:
    """
    Vectorize training and test data
    
    Args:
        X_train: Training text data
        X_test: Test text data
        vectorizer_type: Type of vectorizer to use
        save_path: Path to save the vectorizer (optional)
        
    Returns:
        Tuple of (X_train_vec, X_test_vec, vectorizer)
    """
    # Initialize and fit vectorizer
    vectorizer = TextVectorizer(vectorizer_type=vectorizer_type)
    
    # Fit on training data and transform both sets
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save vectorizer if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectorizer.save_vectorizer(save_path)
    
    print(f"Vectorization complete:")
    print(f"  - Vectorizer type: {vectorizer_type}")
    print(f"  - Feature count: {len(vectorizer.get_feature_names())}")
    print(f"  - Training shape: {X_train_vec.shape}")
    print(f"  - Test shape: {X_test_vec.shape}")
    
    return X_train_vec, X_test_vec, vectorizer

def analyze_features(vectorizer: TextVectorizer, df: pd.DataFrame, 
                    label_column: str = 'Ticket Type') -> Dict[str, Any]:
    """
    Analyze features and their importance
    
    Args:
        vectorizer: Fitted vectorizer
        df: Original DataFrame
        label_column: Label column name
        
    Returns:
        Dictionary with feature analysis
    """
    analysis = {
        'total_features': len(vectorizer.get_feature_names()),
        'feature_names': vectorizer.get_feature_names()[:20]  # First 20 features
    }
    
    # Add TF-IDF specific analysis
    if vectorizer.vectorizer_type == 'tfidf':
        top_features = vectorizer.get_top_features(20)
        analysis['top_tfidf_features'] = top_features.get('top_features', [])
        analysis['bottom_tfidf_features'] = top_features.get('bottom_features', [])
    
    # Class distribution
    analysis['class_distribution'] = df[label_column].value_counts().to_dict()
    
    return analysis

if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_preprocess_data
    
    # Load and preprocess data
    df_processed, _ = load_and_preprocess_data("../data/tickets.csv")
    
    # Prepare data splits
    X_train, X_test, y_train, y_test = prepare_data_splits(df_processed)
    
    # Vectorize data
    X_train_vec, X_test_vec, vectorizer = vectorize_data(
        X_train, X_test, vectorizer_type='tfidf', 
        save_path="../outputs/vectorizer.pkl"
    )
    
    # Analyze features
    analysis = analyze_features(vectorizer, df_processed)
    print("\nFeature Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
