"""
Test Logistic Regression classifier with Time-Series Cross-Validation.

Uses TimeSeriesSplit to get robust performance estimates across different time periods.
"""
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.macrocrypto.models import MacroRegimeClassifier
from src.macrocrypto.data import CombinedDataPipeline


def main():
    print("=" * 70)
    print("TIME-SERIES CROSS-VALIDATION")
    print("=" * 70)

    # Initialize
    classifier = MacroRegimeClassifier()
    pipeline = CombinedDataPipeline()

    # Fetch data (2 years)
    print("\n1. Fetching data...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)

    # Create labels
    print("\n2. Creating risk regime labels...")
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare data
    print("\n3. Preparing data...")
    X, y = classifier.prepare_data(df)

    # Time-Series Cross-Validation
    print("\n4. Running Time-Series Cross-Validation...")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    # Store scores for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    fold_num = 1
    for train_index, test_index in tscv.split(X):
        print(f"\n--- Fold {fold_num}/5 ---")

        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Train: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
        print(f"Test:  {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")

        # Check if both classes present in train and test
        train_classes = y_train.nunique()
        test_classes = y_test.nunique()

        if train_classes < 2:
            print(f"[SKIP] Only {train_classes} class in training set")
            fold_num += 1
            continue

        if test_classes < 2:
            print(f"[SKIP] Only {test_classes} class in test set")
            fold_num += 1
            continue

        print(f"Train class distribution: Risk-Off={sum(y_train==0)}, Risk-On={sum(y_train==1)}")
        print(f"Test class distribution: Risk-Off={sum(y_test==0)}, Risk-On={sum(y_test==1)}")

        # Train model for this fold (directly, not using classifier.train())
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
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

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        fold_num += 1

    # Print summary statistics
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    metrics_dict = {
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1 Score': f1_scores,
        'ROC-AUC': roc_auc_scores
    }

    for metric_name, scores in metrics_dict.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)

        print(f"{metric_name:<15} {mean_score:<10.4f} {std_score:<10.4f} {min_score:<10.4f} {max_score:<10.4f}")

    # Show individual fold scores
    print("\n" + "=" * 70)
    print("INDIVIDUAL FOLD SCORES")
    print("=" * 70)

    print(f"\n{'Fold':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 70)

    for i in range(len(accuracy_scores)):
        print(f"Fold {i+1:<3} {accuracy_scores[i]:<12.4f} {precision_scores[i]:<12.4f} "
              f"{recall_scores[i]:<12.4f} {f1_scores[i]:<12.4f} {roc_auc_scores[i]:<12.4f}")

    # Compare with single train/test split
    print("\n" + "=" * 70)
    print("COMPARISON: CV vs Single Split")
    print("=" * 70)

    # Train final model with single split for comparison
    print("\n5. Training with single 80/20 split for comparison...")
    classifier_final = MacroRegimeClassifier()
    X_final, y_final = classifier_final.prepare_data(df)
    metrics_final = classifier_final.train(X_final, y_final, test_size=0.2, random_state=42)

    print(f"\n{'Metric':<15} {'CV Mean':<15} {'Single Split':<15} {'Difference':<15}")
    print("-" * 70)

    comparison = {
        'Accuracy': (np.mean(accuracy_scores), metrics_final['test']['accuracy']),
        'Precision': (np.mean(precision_scores), metrics_final['test']['precision']),
        'Recall': (np.mean(recall_scores), metrics_final['test']['recall']),
        'F1 Score': (np.mean(f1_scores), metrics_final['test']['f1']),
        'ROC-AUC': (np.mean(roc_auc_scores), metrics_final['test']['roc_auc'])
    }

    for metric_name, (cv_score, split_score) in comparison.items():
        diff = cv_score - split_score
        diff_str = f"{diff:+.4f}"
        print(f"{metric_name:<15} {cv_score:<15.4f} {split_score:<15.4f} {diff_str:<15}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    cv_accuracy_mean = np.mean(accuracy_scores)
    cv_accuracy_std = np.std(accuracy_scores)

    print(f"\nCross-Validation Results:")
    print(f"  Mean Accuracy: {cv_accuracy_mean:.4f} +/- {cv_accuracy_std:.4f}")
    print(f"  Mean ROC-AUC:  {np.mean(roc_auc_scores):.4f} +/- {np.std(roc_auc_scores):.4f}")

    if cv_accuracy_std < 0.05:
        print("\n[OK] Model is STABLE - Low variance across folds (<5%)")
    elif cv_accuracy_std < 0.10:
        print("\n~ Model is MODERATELY STABLE - Some variance across folds (5-10%)")
    else:
        print("\n[X] Model is UNSTABLE - High variance across folds (>10%)")
        print("  Consider:")
        print("  - More data")
        print("  - Feature engineering")
        print("  - Regularization")

    single_accuracy = metrics_final['test']['accuracy']
    diff = abs(cv_accuracy_mean - single_accuracy)

    if diff < 0.05:
        print(f"\n[OK] Single split is REPRESENTATIVE - Close to CV mean (diff: {diff:.4f})")
    else:
        print(f"\n[X] Single split may be MISLEADING - Differs from CV mean (diff: {diff:.4f})")
        print("  Recommend using CV mean for performance reporting")

    print("\n" + "=" * 70)
    print("TIME-SERIES CROSS-VALIDATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
