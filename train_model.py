import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(X):
    """Create advanced features engineering"""
    X = X.copy()
    
    # Age binning
    try:
        X['age_group'] = pd.qcut(X['age'], q=5, 
                                labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'VerySenior'],
                                duplicates='drop')
    except ValueError:
        age_bins = [
            X['age'].min() - 0.1,
            X['age'].min() + (X['age'].max() - X['age'].min())/5,
            X['age'].min() + 2*(X['age'].max() - X['age'].min())/5,
            X['age'].min() + 3*(X['age'].max() - X['age'].min())/5,
            X['age'].min() + 4*(X['age'].max() - X['age'].min())/5,
            X['age'].max() + 0.1
        ]
        X['age_group'] = pd.cut(X['age'], 
                               bins=age_bins,
                               labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'VerySenior'])
    
    # Hours binning
    try:
        X['hours_group'] = pd.qcut(X['hours-per-week'], q=4, 
                                 labels=['PartTime', 'Regular', 'Overtime', 'Heavy'],
                                 duplicates='drop')
    except ValueError:
        hours_bins = [
            X['hours-per-week'].min() - 0.1,
            X['hours-per-week'].min() + (X['hours-per-week'].max() - X['hours-per-week'].min())/4,
            X['hours-per-week'].min() + 2*(X['hours-per-week'].max() - X['hours-per-week'].min())/4,
            X['hours-per-week'].min() + 3*(X['hours-per-week'].max() - X['hours-per-week'].min())/4,
            X['hours-per-week'].max() + 0.1
        ]
        X['hours_group'] = pd.cut(X['hours-per-week'], 
                               bins=hours_bins,
                               labels=['PartTime', 'Regular', 'Overtime', 'Heavy'])
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['age_group', 'hours_group'])
    
    return X

def initialize_models():
    """Initialize models with optimized parameters"""
    models = {}
    
    # XGBoost
    models['xgb'] = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        scale_pos_weight=2.5,
        random_state=42
    )
    
    # LightGBM
    models['lgb'] = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
    
    # CatBoost
    models['catboost'] = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.01,
        l2_leaf_reg=3,
        border_count=128,
        bagging_temperature=1,
        random_strength=1,
        verbose=False,
        random_state=42
    )
    
    # Gradient Boosting
    models['gb'] = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=8,
        min_samples_split=200,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42
    )
    
    # SVM
    models['svm'] = SVC(
        C=10,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    # Neural Network
    models['nn'] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    
    return models

def create_ensemble(models):
    """Create stacking ensemble"""
    estimators_level1 = [(name, model) for name, model in models.items()]
    
    # Second layer estimators
    estimators_level2 = [
        ('lr', LogisticRegression(C=1.0, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=8)),
        ('xgb', XGBClassifier(n_estimators=200, max_depth=4))
    ]
    
    # Final estimator
    final_estimator = VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=200, max_depth=4)),
            ('lgb', LGBMClassifier(n_estimators=200, max_depth=4)),
            ('cb', CatBoostClassifier(iterations=200, depth=4, verbose=False))
        ],
        voting='soft'
    )
    
    # Create stacking
    stacking_model = StackingClassifier(
        estimators=estimators_level1,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_model

def main():
    print("Loading data...")
    data = pd.read_csv('preprocessed.csv')  # Make sure this file exists
    
    # Prepare data
    X = data.drop('Income', axis=1)
    y = data['Income']
    
    # Create advanced features
    print("Creating advanced features...")
    X_enhanced = create_advanced_features(X)
    
    # Initialize preprocessing
    print("Initializing preprocessing...")
    power_transformer = PowerTransformer(standardize=True)
    scaler = RobustScaler()
    
    # Transform and scale features
    X_transformed = power_transformer.fit_transform(X_enhanced)
    X_scaled = scaler.fit_transform(X_transformed)
    X_final = pd.DataFrame(X_scaled, columns=X_enhanced.columns)
    
    # Initialize models
    print("Initializing models...")
    models = initialize_models()
    
    # Create and train ensemble
    print("Training ensemble model...")
    ensemble = create_ensemble(models)
    ensemble.fit(X_final, y)
    
    # Save models and preprocessors
    print("Saving models and preprocessors...")
    with open('ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    
    with open('power_transformer.pkl', 'wb') as f:
        pickle.dump(power_transformer, f)
    
    with open('robust_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model training and saving completed!")

if __name__ == "__main__":
    main()