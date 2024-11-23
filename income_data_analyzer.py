import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import joblib

class IncomeDataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                                 'capital-loss', 'hours-per-week']
        self.categorical_features = ['workclass', 'education', 'marital-status', 
                                     'occupation', 'relationship', 'race', 'sex', 
                                     'native-country']
        self.preprocessor = None
        self.feature_names = None
        self.data = None

    def _ensure_features_exist(self, X):
        """Ensure all expected features exist in the dataset."""
        for feature in self.numeric_features + self.categorical_features:
            if feature not in X.columns:
                # Add missing numeric features with 0
                if feature in self.numeric_features:
                    X[feature] = 0
                # Add missing categorical features with 'missing'
                else:
                    X[feature] = 'missing'
        
        # Select and order columns exactly as they were during training
        return X[self.numeric_features + self.categorical_features]
    
    def _handle_missing_values(self, X):
        """Handle missing values in the dataset."""
        # Handle numeric features
        for feature in self.numeric_features:
            X[feature] = X[feature].fillna(X[feature].median())
            
        # Handle categorical features
        for feature in self.categorical_features:
            X[feature] = X[feature].fillna(X[feature].mode()[0])
            
        return X

    def load_and_preprocess(self, is_training=True):
        """Load and preprocess the data."""
        # Load the data
        self.data = pd.read_csv(self.data_path)
    
        # Separate features and target
        X = self.data.drop('Income', axis=1)
        y = self.data['Income']
    
        # Ensure all features exist and handle missing values
        X = self._ensure_features_exist(X)
        X = self._handle_missing_values(X)
    
        if is_training:
            # Create preprocessing steps
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
            # Combine preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )
        
            # Fit preprocessor
            X_preprocessed = self.preprocessor.fit_transform(X)
        
            # Get feature names after transformation
            numeric_features = self.numeric_features
            categorical_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
            self.feature_names = numeric_features + list(categorical_features)
        else:
            if self.preprocessor is None:
                raise ValueError("Preprocessor not fitted. Call with is_training=True first.")
            X_preprocessed = self.preprocessor.transform(X)
    
        # Convert to DataFrame with correct feature names
        self.X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=self.feature_names)
    
        # Encode the target variable
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(y)
    
        # Add encoded target to the preprocessed dataframe
        self.X_preprocessed_df['Income'] = self.y_encoded
    
        if is_training:
            # Split the data only during training
            X = self.X_preprocessed_df.drop('Income', axis=1)
            y = self.X_preprocessed_df['Income']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
    
        # Save preprocessed data to CSV
        self.X_preprocessed_df.to_csv('preprocessed.csv', index=False)
    
        return self.X_preprocessed_df

    def predict_new_data(self, new_data):
        """Preprocess new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Run load_and_preprocess with is_training=True first.")
            
        # Ensure correct features and handle missing values
        X = self._ensure_features_exist(new_data)
        X = self._handle_missing_values(X)
        
        # Transform the data
        X_preprocessed = self.preprocessor.transform(X)
        
        # Convert to DataFrame with correct feature names
        return pd.DataFrame(X_preprocessed, columns=self.feature_names)

    def save_preprocessor(self, filename='preprocessor.pkl'):
        """Save the fitted preprocessor and feature lists."""
        if self.preprocessor is not None:
            joblib.dump({
                'preprocessor': self.preprocessor,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'feature_names': self.feature_names
            }, filename)
            print(f"Preprocessor saved to {filename}")
        else:
            print("Preprocessor not fitted yet")
    
    def load_preprocessor(self, filename='preprocessor.pkl'):
        """Load a saved preprocessor and feature lists."""
        saved_data = joblib.load(filename)
        self.preprocessor = saved_data['preprocessor']
        self.numeric_features = saved_data['numeric_features']
        self.categorical_features = saved_data['categorical_features']
        self.feature_names = saved_data.get('feature_names', 
            self.numeric_features + self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features))
        print(f"Preprocessor loaded from {filename}")

    # Rest of the methods remain the same as in the original implementation