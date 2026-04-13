"""
Prediction Module for Support Ticket Classification
Handles making predictions on new ticket data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
from preprocess import TicketPreprocessor
from vectorize import TextVectorizer

class TicketPredictor:
    """Class for making predictions on support tickets"""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to saved model
            vectorizer_path: Path to saved vectorizer
        """
        self.model = None
        self.vectorizer = None
        self.preprocessor = TicketPreprocessor()
        self.model_classes = None
        
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """
        Load trained model and vectorizer
        
        Args:
            model_path: Path to saved model
            vectorizer_path: Path to saved vectorizer
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        self.vectorizer = TextVectorizer()
        self.vectorizer.load_vectorizer(vectorizer_path)
        
        # Get class names
        if hasattr(self.model, 'classes_'):
            self.model_classes = self.model.classes_.tolist()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Vectorizer loaded successfully from {vectorizer_path}")
    
    def predict_single(self, ticket_text: str) -> Dict[str, Any]:
        """
        Make prediction for a single ticket
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not loaded")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(ticket_text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vector)[0]
            probability_dict = dict(zip(self.model_classes, probabilities))
            probability = probability_dict.get(prediction, 0.0)
        
        return {
            'original_text': ticket_text,
            'processed_text': processed_text,
            'predicted_category': prediction,
            'confidence': probability,
            'all_probabilities': probability_dict if 'probability_dict' in locals() else None
        }
    
    def predict_batch(self, ticket_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple tickets
        
        Args:
            ticket_texts: List of ticket texts
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text in ticket_texts:
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'original_text': text,
                    'error': str(e)
                })
        
        return results
    
    def predict_priority(self, ticket_text: str) -> str:
        """
        Predict ticket priority based on keywords (rule-based)
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Predicted priority (High/Medium/Low)
        """
        text_lower = ticket_text.lower()
        
        # High priority keywords
        high_keywords = [
            'urgent', 'emergency', 'critical', 'error', 'failed', 'not working',
            'broken', 'crash', 'down', 'unavailable', 'lost', 'stolen', 'hack',
            'security', 'breach', 'immediate', 'asap', 'as soon as possible'
        ]
        
        # Medium priority keywords
        medium_keywords = [
            'delay', 'slow', 'issue', 'problem', 'bug', 'glitch', 'freeze',
            'stuck', 'confusion', 'help', 'support', 'question', 'how to'
        ]
        
        # Low priority keywords
        low_keywords = [
            'info', 'information', 'query', 'request', 'suggestion', 'feedback',
            'improvement', 'feature', 'enhancement', 'documentation', 'tutorial'
        ]
        
        # Check for high priority keywords
        if any(keyword in text_lower for keyword in high_keywords):
            return 'High'
        
        # Check for medium priority keywords
        if any(keyword in text_lower for keyword in medium_keywords):
            return 'Medium'
        
        # Default to low priority
        return 'Low'
    
    def predict_complete(self, ticket_text: str) -> Dict[str, Any]:
        """
        Make complete prediction including category and priority
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Dictionary with complete prediction results
        """
        # Get category prediction
        category_result = self.predict_single(ticket_text)
        
        # Get priority prediction
        priority = self.predict_priority(ticket_text)
        
        # Combine results
        complete_result = {
            **category_result,
            'predicted_priority': priority
        }
        
        return complete_result
    
    def analyze_ticket_text(self, ticket_text: str) -> Dict[str, Any]:
        """
        Analyze ticket text and provide insights
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Dictionary with text analysis
        """
        processed_text = self.preprocessor.preprocess_text(ticket_text)
        
        # Basic text statistics
        word_count = len(processed_text.split())
        char_count = len(processed_text)
        
        # Sentiment analysis (basic)
        positive_words = ['good', 'great', 'excellent', 'helpful', 'thank', 'thanks', 'appreciate']
        negative_words = ['bad', 'terrible', 'awful', 'frustrated', 'angry', 'disappointed', 'worst']
        
        text_lower = processed_text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        sentiment = 'Neutral'
        if positive_count > negative_count:
            sentiment = 'Positive'
        elif negative_count > positive_count:
            sentiment = 'Negative'
        
        return {
            'original_length': len(ticket_text),
            'processed_length': len(processed_text),
            'word_count': word_count,
            'char_count': char_count,
            'sentiment': sentiment,
            'positive_words': positive_count,
            'negative_words': negative_count
        }

def create_sample_predictions() -> List[Dict[str, Any]]:
    """
    Create sample predictions for demonstration
    
    Returns:
        List of sample prediction results
    """
    sample_tickets = [
        "I'm having an issue with my product, it's not working properly and I need urgent help!",
        "Can you please provide some information about how to use the new feature?",
        "The system is running very slow today, is there any way to improve performance?",
        "Thank you for the excellent support, the issue has been resolved!",
        "I have a suggestion for improving the user interface of your application."
    ]
    
    predictor = TicketPredictor()
    results = []
    
    for ticket in sample_tickets:
        # Only analyze text since no model is loaded
        analysis = predictor.analyze_ticket_text(ticket)
        priority = predictor.predict_priority(ticket)
        
        results.append({
            'ticket_text': ticket,
            'text_analysis': analysis,
            'predicted_priority': priority
        })
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Creating sample predictions...")
    
    # Create sample predictions
    sample_results = create_sample_predictions()
    
    # Display results
    for i, result in enumerate(sample_results, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Text: {result['ticket_text']}")
        print(f"Priority: {result['predicted_priority']}")
        print(f"Word Count: {result['text_analysis']['word_count']}")
        print(f"Sentiment: {result['text_analysis']['sentiment']}")
    
    print("\nFor actual predictions, please train a model first using train_model.py")
    print("Then load the model using:")
    print("predictor = TicketPredictor('../outputs/best_model.pkl', '../outputs/vectorizer.pkl')")
