import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import NetflixDataLoader
from src.feature_engineering import NetflixFeatureEngineer
from src.model import NetflixContentPredictor
from src.recommender import NetflixRecommender

def main():
    # Load data
    loader = NetflixDataLoader('data/netflix_dataset.csv')
    data = loader.load_data()
    data = loader.clean_data()
    data = loader.filter_recent_content()
    
    # Feature Engineering
    engineer = NetflixFeatureEngineer(data)
    data = engineer.encode_categorical()
    data = engineer.extract_genres()
    data = engineer.text_features()
    
    # Prepare ML Dataset
    X, y = engineer.prepare_ml_dataset()
    
    # Model Development
    predictor = NetflixContentPredictor(X, y)
    predictor.train_models()
    
    # Evaluate Models
    results = predictor.evaluate_models()
    print("Model Performance Results:")
    for name, metrics in results.items():
        print(f"{name} Classification Report:")
        print(metrics['Classification Report'])
    
    # Visualize Results
    predictor.visualize_results(X.columns)
    
    # Recommendation System
    recommender = NetflixRecommender(data)
    recommender.create_content_matrix()
    
    # Example Recommendations
    print("\nSample Recommendations:")
    print(recommender.get_recommendations("Stranger Things"))

if __name__ == "__main__":
    main()
