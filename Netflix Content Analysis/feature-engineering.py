import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

class NetflixFeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.encoded_features = {}
    
    def encode_categorical(self):
        """Encode categorical variables"""
        # Label encode categorical columns
        categorical_columns = ['type', 'rating']
        for col in categorical_columns:
            le = LabelEncoder()
            self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
            self.encoded_features[col] = le
        
        return self.data
    
    def extract_genres(self):
        """Extract and one-hot encode genres"""
        # Split listed_in into multiple genres
        self.data['genres'] = self.data['listed_in'].str.split(', ')
        
        # Multi-label binarization for genres
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.data['genres'])
        genre_df = pd.DataFrame(
            genre_matrix, 
            columns=[f'genre_{g}' for g in mlb.classes_]
        )
        
        # Combine with original dataframe
        self.data = pd.concat([self.data, genre_df], axis=1)
        
        return self.data
    
    def text_features(self):
        """Create text-based features using TF-IDF"""
        # TF-IDF for description
        tfidf = TfidfVectorizer(
            max_features=100, 
            stop_words='english'
        )
        description_matrix = tfidf.fit_transform(self.data['description'].fillna(''))
        
        # Convert to dataframe
        desc_features = pd.DataFrame(
            description_matrix.toarray(), 
            columns=[f'desc_feature_{i}' for i in range(100)]
        )
        
        # Combine with original dataframe
        self.data = pd.concat([self.data, desc_features], axis=1)
        
        return self.data
    
    def prepare_ml_dataset(self, target_column='type'):
        """Prepare dataset for machine learning"""
        # Select features
        feature_columns = [
            'release_year', 'type_encoded', 'rating_encoded'
        ] + [col for col in self.data.columns if col.startswith(('genre_', 'desc_feature_'))]
        
        # Prepare features and target
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        return X, y
