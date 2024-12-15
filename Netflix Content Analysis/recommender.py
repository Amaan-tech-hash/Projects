import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NetflixRecommender:
    def __init__(self, data):
        self.data = data
        self.content_matrix = None
    
    def create_content_matrix(self):
        """Create content similarity matrix"""
        # Select relevant features for recommendation
        recommendation_features = [
            'release_year', 'type_encoded', 'rating_encoded'
        ] + [col for col in self.data.columns if col.startswith(('genre_', 'desc_feature_'))]
        
        self.content_matrix = self.data[recommendation_features]
        return self.content_matrix
    
    def get_recommendations(self, title, n_recommendations=5):
        """Get content recommendations based on similarity"""
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(self.content_matrix)
        
        # Find index of input title
        try:
            idx = self.data[self.data['title'].str.lower() == title.lower()].index[0]
        except IndexError:
            return "Title not found in dataset"
        
        # Get similarity scores
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (excluding itself)
        recommended_indices = [x[0] for x in similarity_scores[1:n_recommendations+1]]
        
        return self.data.iloc[recommended_indices][['title', 'type', 'release_year']]
