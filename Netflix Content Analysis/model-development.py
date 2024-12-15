import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class NetflixContentPredictor:
    def __init__(self, X, y):
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.models = {}
        self.predictions = {}
    
    def train_models(self):
        """Train multiple classification models"""
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        self.predictions['Logistic Regression'] = lr.predict(self.X_test)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self.predictions['Random Forest'] = rf.predict(self.X_test)
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        self.predictions['Gradient Boosting'] = gb.predict(self.X_test)
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        results = {}
        for name, predictions in self.predictions.items():
            results[name] = {
                'Classification Report': classification_report(
                    self.y_test, predictions, output_dict=True
                ),
                'Confusion Matrix': confusion_matrix(self.y_test, predictions)
            }
        return results
    
    def feature_importance(self, model_name='Random Forest'):
        """Extract feature importances"""
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def visualize_results(self, feature_names):
        """Create visualization of results"""
        plt.figure(figsize=(15, 10))
        
        # Feature Importance
        plt.subplot(2, 1, 1)
        importances = self.feature_importance()
        indices = np.argsort(importances)[::-1]
        plt.title('Content Classification Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], rotation=90)
        
        # Confusion Matrix Heatmap
        plt.subplot(2, 1, 2)
        cm = self.evaluate_models()['Random Forest']['Confusion Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('netflix_content_analysis.png')
        plt.close()
