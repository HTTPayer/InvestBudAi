"""
Macro Regime Classifier: Logistic Regression model to predict Risk-On/Risk-Off.
"""
import os
import pickle
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from ..data import CombinedDataPipeline


class MacroRegimeClassifier:
    """
    Logistic Regression classifier for macro regime prediction.

    Predicts Risk-On (1) vs Risk-Off (0) based on:
    - Macro indicators (M2, GDP, CPI, Fed Funds, etc.)
    - BTC momentum indicators (returns, RSI, volatility, etc.)
    """

    def __init__(self):
        """Initialize the classifier."""
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.training_date: Optional[str] = None
        self.metrics: Optional[Dict] = None
        self.cv_metrics: Optional[Dict] = None  # Cross-validation results

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'risk_regime',
        drop_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training: drop NaNs and separate features/target.

        Args:
            df: Combined dataframe with features and target
            target_col: Name of target column
            drop_features: List of feature names to exclude

        Returns:
            Tuple of (X features, y target)
        """
        # Drop rows with NaNs
        df_clean = df.dropna()
        print(f"[OK] Dropped NaN rows: {len(df)} -> {len(df_clean)} rows")

        # Separate target
        if target_col not in df_clean.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])

        # Drop specified features
        if drop_features:
            X = X.drop(columns=[col for col in drop_features if col in X.columns])

        # Store feature names
        self.feature_names = X.columns.tolist()

        print(f"[OK] Features: {len(X.columns)}")
        print(f"[OK] Target distribution: Risk-On={y.sum()} ({y.mean()*100:.1f}%), Risk-Off={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        **model_kwargs
    ) -> Dict:
        """
        Train the logistic regression classifier.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data for testing (time-aware split)
            random_state: Random seed
            **model_kwargs: Additional arguments for LogisticRegression

        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "=" * 70)
        print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
        print("=" * 70)

        # Time-aware train/test split (no shuffling for time series!)
        # Take last test_size% of data as test set
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nTrain set: {len(X_train)} rows ({X_train.index.min()} to {X_train.index.max()})")
        print(f"Test set: {len(X_test)} rows ({X_test.index.min()} to {X_test.index.max()})")

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train logistic regression
        print("\nTraining model...")
        default_kwargs = {
            'max_iter': 1000,
            'random_state': random_state,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        default_kwargs.update(model_kwargs)

        self.model = LogisticRegression(**default_kwargs)
        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Prediction probabilities
        y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate confusion matrices
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        # Evaluate
        self.metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, zero_division=0),
                'recall': recall_score(y_train, y_train_pred, zero_division=0),
                'f1': f1_score(y_train, y_train_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_train, y_train_proba),
                'confusion_matrix': cm_train
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, zero_division=0),
                'recall': recall_score(y_test, y_test_pred, zero_division=0),
                'f1': f1_score(y_test, y_test_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_test_proba),
                'confusion_matrix': cm_test
            },
            'confusion_matrix': cm_test.tolist(),  # Deprecated: kept for backward compatibility
            'classification_report': classification_report(y_test, y_test_pred, target_names=['Risk-Off', 'Risk-On'])
        }

        self.training_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Print results
        self._print_metrics()

        return self.metrics

    def _print_metrics(self):
        """Print training metrics in a nice format."""
        if not self.metrics:
            return

        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE")
        print("=" * 70)

        print("\n--- Training Set ---")
        for metric, value in self.metrics['train'].items():
            if metric != 'confusion_matrix':
                print(f"{metric.upper()}: {value:.4f}")

        print("\n--- Test Set ---")
        for metric, value in self.metrics['test'].items():
            if metric != 'confusion_matrix':
                print(f"{metric.upper()}: {value:.4f}")

        print("\n--- Confusion Matrix (Test Set) ---")
        cm = self.metrics['test']['confusion_matrix']
        print(f"                Predicted Risk-Off  Predicted Risk-On")
        print(f"Actual Risk-Off        {cm[0,0]:>4}              {cm[0,1]:>4}")
        print(f"Actual Risk-On         {cm[1,0]:>4}              {cm[1,1]:>4}")

        print("\n--- Classification Report (Test Set) ---")
        print(self.metrics['classification_report'])

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        **model_kwargs
    ) -> Dict:
        """
        Perform time-series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV folds
            **model_kwargs: Additional arguments for LogisticRegression

        Returns:
            Dictionary of CV results
        """
        from sklearn.model_selection import TimeSeriesSplit

        print("\n" + "=" * 70)
        print(f"TIME-SERIES CROSS-VALIDATION ({n_splits} folds)")
        print("=" * 70)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Store scores for each fold
        fold_results = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_auc_scores = []

        fold_num = 1
        for train_index, test_index in tscv.split(X):
            # Split data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Check if both classes present
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                print(f"\nFold {fold_num}: SKIPPED (insufficient class diversity)")
                fold_num += 1
                continue

            print(f"\nFold {fold_num}:")
            print(f"  Train: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
            print(f"  Test:  {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            default_kwargs = {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            default_kwargs.update(model_kwargs)

            model = LogisticRegression(**default_kwargs)
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Store scores
            accuracy_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
            roc_auc_scores.append(roc_auc)

            fold_results.append({
                'fold': fold_num,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc_auc
            })

            print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

            fold_num += 1

        # Calculate summary statistics
        self.cv_metrics = {
            'n_folds': len(fold_results),
            'n_splits': n_splits,
            'fold_results': fold_results,
            'mean': {
                'accuracy': float(np.mean(accuracy_scores)),
                'precision': float(np.mean(precision_scores)),
                'recall': float(np.mean(recall_scores)),
                'f1': float(np.mean(f1_scores)),
                'roc_auc': float(np.mean(roc_auc_scores))
            },
            'std': {
                'accuracy': float(np.std(accuracy_scores)),
                'precision': float(np.std(precision_scores)),
                'recall': float(np.std(recall_scores)),
                'f1': float(np.std(f1_scores)),
                'roc_auc': float(np.std(roc_auc_scores))
            },
            'min': {
                'accuracy': float(np.min(accuracy_scores)),
                'precision': float(np.min(precision_scores)),
                'recall': float(np.min(recall_scores)),
                'f1': float(np.min(f1_scores)),
                'roc_auc': float(np.min(roc_auc_scores))
            },
            'max': {
                'accuracy': float(np.max(accuracy_scores)),
                'precision': float(np.max(precision_scores)),
                'recall': float(np.max(recall_scores)),
                'f1': float(np.max(f1_scores)),
                'roc_auc': float(np.max(roc_auc_scores))
            }
        }

        # Print summary
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 70)

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            print(f"{metric.upper():<15} "
                  f"{self.cv_metrics['mean'][metric]:<10.4f} "
                  f"{self.cv_metrics['std'][metric]:<10.4f} "
                  f"{self.cv_metrics['min'][metric]:<10.4f} "
                  f"{self.cv_metrics['max'][metric]:<10.4f}")

        return self.cv_metrics

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance based on model coefficients.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with features and their coefficients
        """
        if not self.model or not self.feature_names:
            raise ValueError("Model not trained yet")

        # Get coefficients
        coefficients = self.model.coef_[0]

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df.head(top_n)

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix
            return_proba: If True, return probabilities instead of binary predictions

        Returns:
            Predictions (0/1) or probabilities
        """
        if not self.model or not self.scaler:
            raise ValueError("Model not trained yet")

        # Ensure features match training data
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        if return_proba:
            return self.model.predict_proba(X_scaled)[:, 1]  # Probability of Risk-On
        else:
            return self.model.predict(X_scaled)

    def predict_current_regime(self, verbose: bool = True) -> Dict:
        """
        Predict current market regime using latest data.

        Args:
            verbose: Print detailed output

        Returns:
            Dictionary with prediction details
        """
        # Fetch latest data from cache or API
        from ..utils.cache import get_cache
        cache = get_cache()

        def fetch_raw_data():
            pipeline = CombinedDataPipeline()
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            return pipeline.fetch_combined_data(start_date=start_date)

        # Use cached data (24h TTL) - get raw data without labels
        df = cache.get_or_set('macro_crypto_data', fetch_raw_data, ttl_seconds=86400)

        # Create risk labels on-demand (not cached, as they're cheap to compute)
        pipeline = CombinedDataPipeline()
        df = pipeline.create_risk_labels(df, method='btc_returns')

        # Get latest complete row
        df_clean = df.dropna()
        if df_clean.empty:
            raise ValueError("No complete data available")

        latest = df_clean.iloc[-1:]
        X = latest.drop(columns=['risk_regime'])

        # Predict
        prediction = self.predict(X)[0]
        probability = self.predict(X, return_proba=True)[0]

        result = {
            'date': latest.index[0],
            'regime': 'Risk-On' if prediction == 1 else 'Risk-Off',
            'regime_binary': int(prediction),
            'confidence': probability if prediction == 1 else (1 - probability),
            'risk_on_probability': probability,
            'features': X.iloc[0].to_dict()
        }

        if verbose:
            print("\n" + "=" * 70)
            print("CURRENT MARKET REGIME PREDICTION")
            print("=" * 70)
            print(f"Date: {result['date']}")
            print(f"Regime: {result['regime']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Risk-On Probability: {result['risk_on_probability']*100:.1f}%")

        return result

    def save(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.model:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_date': self.training_date,
            'metrics': self.metrics,
            'cv_metrics': self.cv_metrics
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"[OK] Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_date = model_data['training_date']
        self.metrics = model_data['metrics']
        self.cv_metrics = model_data.get('cv_metrics', None)

        print(f"[OK] Model loaded from {filepath}")
        print(f"[OK] Training date: {self.training_date}")


def main():
    """Train and evaluate the classifier."""
    # Initialize
    classifier = MacroRegimeClassifier()
    pipeline = CombinedDataPipeline()

    # Fetch data (2 years)
    print("Fetching data...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)

    # Create labels
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare data
    X, y = classifier.prepare_data(df)

    # Train
    metrics = classifier.train(X, y, test_size=0.2)

    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    importance = classifier.get_feature_importance(top_n=15)
    print(importance.to_string(index=False))

    # Predict current regime
    current = classifier.predict_current_regime()

    # Save model
    model_path = 'models/regime_classifier.pkl'
    classifier.save(model_path)


if __name__ == '__main__':
    main()
