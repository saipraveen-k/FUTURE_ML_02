"""
Text Preprocessing Module for Support Ticket Classification
Handles cleaning, tokenization, and preprocessing of ticket text
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# spaCy import commented out for compatibility
# import spacy

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TicketPreprocessor:
    """Class for preprocessing support ticket text data"""
    
    def __init__(self, use_spacy: bool = False):
        """
        Initialize the preprocessor
        
        Args:
            use_spacy: Whether to use spaCy for lemmatization (True) or NLTK (False)
        """
        self.use_spacy = use_spacy
        self.stop_words = set(stopwords.words('english'))
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("spaCy model not found. Using NLTK instead.")
                self.use_spacy = False
                self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra spaces, etc.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text string
            
        Returns:
            Text without stopwords
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using spaCy or NLTK
        
        Args:
            text: Input text string
            
        Returns:
            Lemmatized text
        """
        if self.use_spacy:
            doc = self.nlp(text)
            lemmatized_words = [token.lemma_ for token in doc if not token.is_space]
        else:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(lemmatized_words)
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            
        Returns:
            Fully preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        # Lemmatize
        text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'Ticket Description') -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to preprocess
            
        Returns:
            DataFrame with preprocessed text
        """
        df_processed = df.copy()
        
        # Apply preprocessing
        df_processed['processed_text'] = df_processed[text_column].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        
        return df_processed
    
    def get_text_statistics(self, df: pd.DataFrame, text_column: str = 'processed_text') -> dict:
        """
        Get statistics about the preprocessed text
        
        Args:
            df: DataFrame with preprocessed text
            text_column: Name of the processed text column
            
        Returns:
            Dictionary with text statistics
        """
        texts = df[text_column].tolist()
        
        # Basic statistics
        total_texts = len(texts)
        total_words = sum(len(text.split()) for text in texts)
        avg_words_per_text = total_words / total_texts if total_texts > 0 else 0
        
        # Vocabulary size
        all_words = ' '.join(texts).split()
        unique_words = len(set(all_words))
        
        # Text length distribution
        text_lengths = [len(text.split()) for text in texts]
        
        return {
            'total_texts': total_texts,
            'total_words': total_words,
            'avg_words_per_text': avg_words_per_text,
            'unique_words': unique_words,
            'min_text_length': min(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'median_text_length': np.median(text_lengths) if text_lengths else 0
        }

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, TicketPreprocessor]:
    """
    Load and preprocess the support ticket dataset
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (preprocessed DataFrame, preprocessor instance)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Initialize preprocessor
    preprocessor = TicketPreprocessor()
    
    # Preprocess
    df_processed = preprocessor.preprocess_dataframe(df)
    
    print(f"Original dataset size: {len(df)}")
    print(f"Processed dataset size: {len(df_processed)}")
    print(f"Text statistics: {preprocessor.get_text_statistics(df_processed)}")
    
    return df_processed, preprocessor

if __name__ == "__main__":
    # Example usage
    file_path = "../data/tickets.csv"
    df_processed, preprocessor = load_and_preprocess_data(file_path)
    
    # Display sample processed texts
    print("\nSample processed texts:")
    for i in range(min(5, len(df_processed))):
        print(f"Original: {df_processed.iloc[i]['Ticket Description']}")
        print(f"Processed: {df_processed.iloc[i]['processed_text']}")
        print("-" * 80)
