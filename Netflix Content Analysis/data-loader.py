import pandas as pd
import numpy as np

class NetflixDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        """Load Netflix dataset"""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        # Handle missing values
        self.data['country'].fillna('Unknown', inplace=True)
        self.data['cast'].fillna('Unknown Cast', inplace=True)
        
        # Convert release year to numeric
        self.data['release_year'] = pd.to_numeric(self.data['release_year'], errors='coerce')
        
        # Clean text columns
        text_columns = ['title', 'description']
        for col in text_columns:
            self.data[col] = self.data[col].str.lower().str.strip()
        
        return self.data
    
    def filter_recent_content(self, years=10):
        """Filter content from recent years"""
        current_year = pd.Timestamp.now().year
        self.data = self.data[self.data['release_year'] >= (current_year - years)]
        return self.data
