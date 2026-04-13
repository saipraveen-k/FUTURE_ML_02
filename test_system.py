"""
Test script to demonstrate the complete Support Ticket Classification System
"""

import sys
import os
sys.path.append('src')

from preprocess import load_and_preprocess_data
from vectorize import prepare_data_splits, vectorize_data
from train_model import ModelTrainer
from predict import TicketPredictor
from evaluate import ModelEvaluator

def main():
    print("=" * 60)
    print("SUPPORT TICKET CLASSIFICATION SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    try:
        df_processed, preprocessor = load_and_preprocess_data('data/tickets.csv')
        print(f"✓ Successfully loaded {len(df_processed)} tickets")
        
        # Show sample data
        print("\nSample tickets:")
        for i in range(min(3, len(df_processed))):
            print(f"  - {df_processed.iloc[i]['Ticket Description'][:100]}...")
            print(f"    Category: {df_processed.iloc[i]['Ticket Type']}")
            print(f"    Priority: {df_processed.iloc[i]['Ticket Priority']}")
            print()
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Prepare data splits
    print("2. Preparing data splits...")
    try:
        X_train, X_test, y_train, y_test = prepare_data_splits(df_processed)
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Step 3: Vectorize data
    print("\n3. Vectorizing text data...")
    try:
        X_train_vec, X_test_vec, vectorizer = vectorize_data(
            X_train, X_test, vectorizer_type='tfidf'
        )
        print(f"✓ Feature extraction complete: {len(vectorizer.get_feature_names())} features")
    except Exception as e:
        print(f"✗ Error vectorizing data: {e}")
        return
    
    # Step 4: Train models (simplified - just Naive Bayes for demo)
    print("\n4. Training models...")
    try:
        trainer = ModelTrainer()
        
        # Train just Naive Bayes for quick demo
        result = trainer.train_model('naive_bayes', X_train_vec, y_train)
        print(f"✓ Naive Bayes trained successfully")
        
        # Evaluate
        eval_result = trainer.evaluate_model('naive_bayes', X_test_vec, y_test)
        print(f"✓ Model evaluation complete")
        print(f"  Accuracy: {eval_result['accuracy']:.4f}")
        print(f"  F1 Score: {eval_result['f1_score']:.4f}")
        
    except Exception as e:
        print(f"✗ Error training models: {e}")
        return
    
    # Step 5: Test predictions
    print("\n5. Testing predictions...")
    try:
        # Save model for testing
        os.makedirs('outputs', exist_ok=True)
        trainer.save_model('naive_bayes', 'outputs/best_model.pkl')
        vectorizer.save_vectorizer('outputs/vectorizer.pkl')
        
        # Initialize predictor
        predictor = TicketPredictor(
            model_path='outputs/best_model.pkl',
            vectorizer_path='outputs/vectorizer.pkl'
        )
        
        # Test with sample tickets
        test_tickets = [
            "I can't log into my account, the system shows an error message",
            "Could you please provide information about your pricing plans?",
            "URGENT: The payment gateway is down and we're losing sales!",
            "Thank you for the great support, everything works perfectly!",
            "I have a suggestion for improving the user interface"
        ]
        
        print("\nSample Predictions:")
        print("-" * 50)
        for i, ticket in enumerate(test_tickets, 1):
            result = predictor.predict_complete(ticket)
            priority = predictor.predict_priority(ticket)
            
            print(f"\nTicket {i}:")
            print(f"  Text: {ticket}")
            print(f"  Category: {result['predicted_category']}")
            print(f"  Priority: {priority}")
            print(f"  Confidence: {result['confidence']:.4f}")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    # Step 6: Generate business insights
    print("\n6. Business Impact Analysis...")
    try:
        total_tickets = len(df_processed)
        accuracy = eval_result['accuracy']
        time_saved_per_ticket = 4.5  # minutes (5 min manual - 0.5 min automated)
        total_time_saved = (accuracy * total_tickets * time_saved_per_ticket) / 60  # hours
        
        print(f"✓ Business Impact:")
        print(f"  - Tickets processed: {total_tickets:,}")
        print(f"  - Classification accuracy: {accuracy:.1%}")
        print(f"  - Manual effort reduction: {accuracy:.1%}")
        print(f"  - Time saved: {total_time_saved:.1f} hours")
        print(f"  - Daily capacity increase: {total_time_saved/8:.1f} staff hours")
        
    except Exception as e:
        print(f"✗ Error calculating business impact: {e}")
    
    print("\n" + "=" * 60)
    print("SYSTEM DEMONSTRATION COMPLETE ✓")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Run 'streamlit run app/dashboard.py' for interactive dashboard")
    print("2. Open 'notebook/analysis.ipynb' for detailed analysis")
    print("3. Check 'outputs/' directory for model files and results")
    print("\nTASK 2 PROJECT READY 🚀")

if __name__ == "__main__":
    main()
