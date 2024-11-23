import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

class AdvancedTabularTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.embedding_dim = 256  # Increased embedding dimension
        num_heads = 8  # Increased number of heads
        num_layers = 4  # Increased number of layers
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.1)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=num_heads,
                dim_feedforward=1024,  # Increased feedforward dimension
                dropout=0.2,
                activation='gelu'  # Changed to GELU activation
            ),
            num_layers=num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        return self.classifier(x)

class AdvancedIncomePredictor:
    def __init__(self, use_gpu=False, fast_mode=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.fast_mode = fast_mode
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.initialize_models()
    
    def create_advanced_features(self, X):
        """Create advanced features engineering"""
        X = X.copy()
    
    # Previous feature engineering code remains the same...
    
    # Modified binning features code:
        try:
        # Try with duplicates='drop' first
            X['hours_group'] = pd.qcut(X['hours-per-week'], q=4, 
                                 labels=['PartTime', 'Regular', 'Overtime', 'Heavy'],
                                 duplicates='drop')
        except ValueError:
        # If that fails, fall back to pd.cut with manual bins
            hours_min = X['hours-per-week'].min()
            hours_max = X['hours-per-week'].max()
            hours_bins = [
                hours_min - 0.1,  # Add small buffer for inclusive binning
                hours_min + (hours_max - hours_min)/4,
                hours_min + 2*(hours_max - hours_min)/4,
                hours_min + 3*(hours_max - hours_min)/4,
                hours_max + 0.1   # Add small buffer for inclusive binning
            ]
            X['hours_group'] = pd.cut(X['hours-per-week'], 
                                bins=hours_bins,
                                labels=['PartTime', 'Regular', 'Overtime', 'Heavy'])
    
    # The age binning should work fine, but let's make it more robust too
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
    
    # One-hot encode the new categorical features
        X = pd.get_dummies(X, columns=['age_group', 'hours_group'])
    
        return X
    
    def initialize_models(self):
        """Initialize advanced models with optimized parameters"""
        self.models = {}
        
        # XGBoost with advanced parameters
        self.models['xgb'] = XGBClassifier(
            n_estimators=500 if not self.fast_mode else 200,
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
        
        # LightGBM with advanced parameters
        self.models['lgb'] = LGBMClassifier(
            n_estimators=500 if not self.fast_mode else 200,
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
        
        # CatBoost with advanced parameters
        self.models['catboost'] = CatBoostClassifier(
            iterations=500 if not self.fast_mode else 200,
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
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=500 if not self.fast_mode else 200,
            learning_rate=0.01,
            max_depth=8,
            min_samples_split=200,
            min_samples_leaf=50,
            subsample=0.8,
            random_state=42
        )
        
        # SVM with RBF kernel
        self.models['svm'] = SVC(
            C=10,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Advanced Neural Network
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        
        # Initialize advanced ensemble
        self.initialize_ensemble()
    
    def initialize_ensemble(self):
        """Initialize advanced stacking ensemble with multiple layers"""
        # First layer estimators
        estimators_level1 = [(name, model) for name, model in self.models.items()]
        
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
        
        # Create multi-layer stacking
        self.stacking_model = StackingClassifier(
            estimators=estimators_level1,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1 if not self.use_gpu else 1
        )
        
        # Create preprocessing pipeline
        self.pipeline = Pipeline([
            ('power_transform', PowerTransformer(standardize=True)),
            ('scaler', RobustScaler()),
            ('classifier', self.stacking_model)
        ])

def advanced_train_and_evaluate(data_path, use_gpu=False, fast_mode=False, sample_size=None):
    """Train and evaluate with advanced techniques"""
    
    print("Loading and preparing data...")
    data = pd.read_csv(data_path)
    
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)
    
    # Prepare data
    X = data.drop('Income', axis=1)
    y = data['Income']
    
    # Stratified splitting with multiple folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize predictor
    predictor = AdvancedIncomePredictor(use_gpu=use_gpu, fast_mode=fast_mode)
    
    # Store results for each fold
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining Fold {fold}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create advanced features
        X_train_enhanced = predictor.create_advanced_features(X_train)
        X_val_enhanced = predictor.create_advanced_features(X_val)
        
        # Scale features
        power_transformer = PowerTransformer(standardize=True)
        scaler = RobustScaler()
        
        X_train_transformed = power_transformer.fit_transform(X_train_enhanced)
        X_val_transformed = power_transformer.transform(X_val_enhanced)
        
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        X_val_scaled = scaler.transform(X_val_transformed)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_enhanced.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val_enhanced.columns)
        
        # Train and evaluate individual models
        fold_model_results = {}
        
        for name, model in predictor.models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
            
            fold_model_results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'report': classification_report(y_val, y_pred)
            }
            print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Train and evaluate ensemble
        print("Training Ensemble...")
        predictor.pipeline.fit(X_train_enhanced, y_train)
        y_pred_ensemble = predictor.pipeline.predict(X_val_enhanced)
        ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
        ensemble_roc_auc = roc_auc_score(y_val, predictor.pipeline.predict_proba(X_val_enhanced)[:, 1])
        
        fold_model_results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'roc_auc': ensemble_roc_auc,
            'report': classification_report(y_val, y_pred_ensemble)
        }
        
        fold_results.append(fold_model_results)
        print(f"Fold {fold} Ensemble - Accuracy: {ensemble_accuracy:.4f}, ROC-AUC: {ensemble_roc_auc:.4f}")
    
    # Calculate and print average results
    print("\nAverage Results Across Folds:")
    avg_results = {}
    
    for model_name in predictor.models.keys():
        accuracies = [fold[model_name]['accuracy'] for fold in fold_results]
        roc_aucs = [fold[model_name]['roc_auc'] for fold in fold_results]
        
        avg_results[model_name] = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'avg_roc_auc': np.mean(roc_aucs),
            'std_roc_auc': np.std(roc_aucs)
        }
        
        print(f"\n{model_name.upper()}:")
        print(f"Average Accuracy: {avg_results[model_name]['avg_accuracy']:.4f} ± {avg_results[model_name]['std_accuracy']:.4f}")
        print(f"Average ROC-AUC: {avg_results[model_name]['avg_roc_auc']:.4f} ± {avg_results[model_name]['std_roc_auc']:.4f}")
    
    # Calculate ensemble averages
    ensemble_accuracies = [fold['ensemble']['accuracy'] for fold in fold_results]
    ensemble_roc_aucs = [fold['ensemble']['roc_auc'] for fold in fold_results]
    
    print("\nENSEMBLE:")
    print(f"Average Accuracy: {np.mean(ensemble_accuracies):.4f} ± {np.std(ensemble_accuracies):.4f}")
    print(f"Average ROC-AUC: {np.mean(ensemble_roc_aucs):.4f} ± {np.std(ensemble_roc_aucs):.4f}")
    
    return predictor, avg_results

if __name__ == "__main__":
    print("Starting advanced model training and evaluation...")
    
    model, results = advanced_train_and_evaluate(
        'preprocessed.csv',
        use_gpu=False,
        fast_mode=False,
        sample_size=None
    )