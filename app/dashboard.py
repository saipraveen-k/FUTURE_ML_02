"""
Streamlit Dashboard for Support Ticket Classification
Interactive web application for ticket classification and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import sys

# Add src directory to path
sys.path.append('src')

from preprocess import TicketPreprocessor
from vectorize import TextVectorizer
from predict import TicketPredictor
from train_model import ModelTrainer

# Page configuration
st.set_page_config(
    page_title="Support Ticket Classification System",
    page_icon=":ticket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and vectorizers"""
    try:
        # Check if models exist
        model_path = 'outputs/best_model.pkl'
        vectorizer_path = 'outputs/vectorizer.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return None, None, "Models not found. Please train models first using train_model.py"
        
        # Load predictor
        predictor = TicketPredictor(model_path, vectorizer_path)
        return predictor, None, None
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"

@st.cache_data
def load_dataset():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/tickets.csv')
        return df
    except Exception as e:
        return None

def create_prediction_interface(predictor):
    """Create the prediction interface"""
    st.markdown('<h2 class="main-header">Ticket Classification</h2>', unsafe_allow_html=True)
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Enter Ticket Description")
        
        ticket_text = st.text_area(
            "Ticket Description",
            height=150,
            placeholder="Enter the support ticket description here...",
            help="Provide a detailed description of the customer issue or request."
        )
        
        submitted = st.form_submit_button("Classify Ticket", type="primary")
        
        if submitted and ticket_text.strip():
            # Make prediction
            try:
                result = predictor.predict_complete(ticket_text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Predicted Category")
                    st.markdown(f"### {result['predicted_category']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Predicted Priority")
                    priority_color = {
                        'High': 'red',
                        'Medium': 'orange', 
                        'Low': 'green'
                    }.get(result['predicted_priority'], 'blue')
                    
                    st.markdown(f"### :{priority_color}[{result['predicted_priority']}]")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence score
                if result['confidence']:
                    st.subheader("Confidence Score")
                    confidence_percentage = result['confidence'] * 100
                    st.progress(confidence_percentage / 100)
                    st.write(f"**{confidence_percentage:.1f}%** confidence")
                
                # Text analysis
                analysis = predictor.analyze_ticket_text(ticket_text)
                
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric("Word Count", analysis['word_count'])
                
                with col4:
                    st.metric("Character Count", analysis['char_count'])
                
                with col5:
                    st.metric("Sentiment", analysis['sentiment'])
                
                # Show probabilities if available
                if result['all_probabilities']:
                    st.subheader("Class Probabilities")
                    prob_df = pd.DataFrame([
                        {'Class': k, 'Probability': v * 100}
                        for k, v in result['all_probabilities'].items()
                    ]).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(prob_df, x='Class', y='Probability', 
                                title="Prediction Confidence by Class")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        elif submitted:
            st.warning("Please enter a ticket description.")

def create_batch_prediction_interface(predictor):
    """Create batch prediction interface"""
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with ticket descriptions",
        type=['csv'],
        help="CSV file should have a column named 'Ticket Description'"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            if 'Ticket Description' not in batch_df.columns:
                st.error("CSV file must have a 'Ticket Description' column")
                return
            
            st.write(f"Loaded {len(batch_df)} tickets")
            
            if st.button("Process Batch", type="primary"):
                with st.spinner("Processing tickets..."):
                    # Make predictions
                    results = predictor.predict_batch(batch_df['Ticket Description'].tolist())
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame([
                        {
                            'Ticket Description': r['original_text'],
                            'Predicted Category': r.get('predicted_category', 'Error'),
                            'Predicted Priority': predictor.predict_priority(r['original_text']),
                            'Confidence': r.get('confidence', 0),
                            'Error': r.get('error', None)
                        }
                        for r in results
                    ])
                    
                    # Display results
                    st.subheader("Batch Results")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Tickets", len(results_df))
                    
                    with col2:
                        successful = len(results_df[results_df['Error'].isna()])
                        st.metric("Successfully Processed", successful)
                    
                    with col3:
                        avg_confidence = results_df['Confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    # Results table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Category distribution
                        category_counts = results_df['Predicted Category'].value_counts()
                        fig = px.pie(values=category_counts.values, 
                                    names=category_counts.index,
                                    title="Predicted Categories")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Priority distribution
                        priority_counts = results_df['Predicted Priority'].value_counts()
                        fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                                    title="Predicted Priorities")
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def create_data_exploration_interface(df):
    """Create data exploration interface"""
    st.markdown('<h2 class="main-header">Data Exploration</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not found. Please ensure tickets.csv is in the data directory.")
        return
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(df))
    
    with col2:
        st.metric("Unique Categories", df['Ticket Type'].nunique())
    
    with col3:
        st.metric("Priority Levels", df['Ticket Priority'].nunique())
    
    with col4:
        avg_length = df['Ticket Description'].str.len().mean()
        st.metric("Avg Description Length", f"{avg_length:.0f} chars")
    
    # Class distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ticket Type Distribution")
        type_counts = df['Ticket Type'].value_counts()
        fig = px.bar(x=type_counts.index, y=type_counts.values,
                    title="Tickets by Type")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Priority Distribution")
        priority_counts = df['Ticket Priority'].value_counts()
        fig = px.pie(values=priority_counts.values, 
                    names=priority_counts.index,
                    title="Tickets by Priority")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("Sample Data")
    sample_size = st.slider("Number of samples to show", 5, 50, 10)
    
    sample_df = df[['Ticket Description', 'Ticket Type', 'Ticket Priority']].head(sample_size)
    st.dataframe(sample_df, use_container_width=True)

def create_model_insights_interface():
    """Create model insights interface"""
    st.markdown('<h2 class="main-header">Model Insights</h2>', unsafe_allow_html=True)
    
    # Try to load model results
    try:
        with open('outputs/model_results.json', 'r') as f:
            model_results = json.load(f)
        
        # Model comparison
        if model_results:
            st.subheader("Model Performance Comparison")
            
            # Create comparison table
            comparison_data = []
            for model_name, results in model_results.items():
                if 'accuracy' in results and 'f1_weighted' in results:
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': results['accuracy'],
                        'F1 Score': results['f1_weighted'],
                        'Precision': results['precision_weighted'],
                        'Recall': results['recall_weighted']
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Performance chart
                fig = go.Figure()
                
                for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=comparison_df['Model'],
                        y=comparison_df[metric],
                        yaxis='y',
                        text=comparison_df[metric].round(3),
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Model Performance Metrics",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not load model results: {str(e)}")
        st.info("Please run the training pipeline first to generate model insights.")

def main():
    """Main application"""
    # Header
    st.markdown("""
    <h1 class="main-header">:ticket: Support Ticket Classification System</h1>
    <p style="text-align: center; color: #666;">
        AI-powered ticket classification and prioritization for improved customer support
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Classification", "Batch Processing", "Data Exploration", "Model Insights"]
    )
    
    # Load models
    predictor, _, error = load_models()
    
    if error:
        st.sidebar.warning(error)
    
    # Load dataset
    df = load_dataset()
    
    # Page content
    if page == "Classification":
        if predictor:
            create_prediction_interface(predictor)
        else:
            st.error("Models not loaded. Please train the models first.")
            st.info("To train models, run: `python src/train_model.py`")
    
    elif page == "Batch Processing":
        if predictor:
            create_batch_prediction_interface(predictor)
        else:
            st.error("Models not loaded. Please train the models first.")
    
    elif page == "Data Exploration":
        create_data_exploration_interface(df)
    
    elif page == "Model Insights":
        create_model_insights_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About:**  
    This system uses NLP and machine learning to automatically classify support tickets and predict their priority level.
    
    **Models:**  
    - Logistic Regression  
    - Naive Bayes  
    - SVM  
    - Random Forest
    
    **Features:**  
    - TF-IDF Vectorization  
    - Text Preprocessing  
    - Priority Prediction  
    - Confidence Scoring
    """)

if __name__ == "__main__":
    import json
    main()
