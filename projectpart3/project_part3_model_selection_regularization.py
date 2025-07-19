#!/usr/bin/env python3
"""
Project Part III – Model selection and regularization
This script implements:
1. Model Selection with 5-fold cross-validation
2. Regularization techniques (L1, L2, Elastic Net)
3. Feature engineering and data preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load the data, remove rows with empty values and duplicated rows.
    Create new features as required by the project.
    """
    print("=" * 60)
    print("PART A: MODEL SELECTION")
    print("=" * 60)
    
    # Load the data
    print("1. Loading and preprocessing data...")
    df = pd.read_csv('down_data.csv')
    
    # Remove rows with empty values
    df = df.dropna()
    
    # Remove duplicated rows
    df = df.drop_duplicates()
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create new features as required
    # Total_toxic_replies = sum of Num_toxic_direct_replies and Num_toxic_nested_replies
    df['Total_toxic_replies'] = df['Num_toxic_direct_replies'] + df['Num_toxic_nested_replies']
    
    # Toxic_conversation = 1 if Total_toxic_replies > 0, 0 otherwise
    df['Toxic_conversation'] = (df['Total_toxic_replies'] > 0).astype(int)
    
    print(f"Toxic conversation distribution:")
    print(df['Toxic_conversation'].value_counts())
    
    return df

def part_a_model_selection(df):
    """
    Part A: Model Selection with 5-fold cross-validation
    """
    print("\n2. Splitting data (80% training, 20% testing)...")
    
    # Select features for Part A
    features_a = ['Length', 'Num_users', 'TOXICITY_x', 'Num_author_replies', 'Verified', 'Age']
    X = df[features_a]
    y = df['Toxic_conversation']
    
    # Convert boolean to int for sklearn compatibility
    X['Verified'] = X['Verified'].astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features used: {features_a}")
    
    # 3. Train and Evaluate Models with 5-Fold Cross Validation
    print("\n3. Training and evaluating models with 5-fold cross-validation...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Perform 5-fold cross-validation for accuracy
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Accuracy scores across folds: {accuracy_scores}")
        print(f"Average Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std() * 2:.4f})")
        
        # Perform 5-fold cross-validation for F1
        f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"F1 scores across folds: {f1_scores}")
        print(f"Average F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        
        results[name] = {
            'model': model,
            'avg_accuracy': accuracy_scores.mean(),
            'avg_f1': f1_scores.mean(),
            'accuracy_scores': accuracy_scores,
            'f1_scores': f1_scores
        }
    
    # 4. Retrain the Best Model on Full Training Data
    print("\n4. Retraining the best model on full training data...")
    
    # Find the best model based on F1 score
    best_model_name = max(results.keys(), key=lambda x: results[x]['avg_f1'])
    best_model = results[best_model_name]['model']
    
    print(f"Best model based on F1 score: {best_model_name}")
    print(f"Best model average F1: {results[best_model_name]['avg_f1']:.4f}")
    
    # Retrain the best model on full training data
    best_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"\nTest Set Performance (Best Model - {best_model_name}):")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Print detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return results, best_model_name, (X_train, X_test, y_train, y_test, features_a)

def part_b_regularization(df):
    """
    Part B: Regularization techniques
    """
    print("\n" + "=" * 60)
    print("PART B: REGULARIZATION")
    print("=" * 60)
    
    print("1. Preparing data for regularization with more features...")
    
    # Select more features for Part B
    features_b = ['Length', 'Num_users', 'TOXICITY_x', 'Num_author_replies', 
                  'Verified', 'Age', 'Followers', 'Friends', 'Num_tweets', 
                  'Location', 'Listed_count']
    
    X = df[features_b]
    y = df['Toxic_conversation']
    
    # Convert boolean to int and handle Location (convert to numeric)
    X['Verified'] = X['Verified'].astype(int)
    X['Location'] = X['Location'].astype('category').cat.codes
    
    # Split the data again
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features used: {features_b}")
    
    # 2. Fit Logistic Regression with 5-Fold Cross-Validation
    print("\n2. Fitting Logistic Regression with 5-fold cross-validation...")
    
    # Unregularized Logistic Regression
    logreg_unregularized = LogisticRegression(max_iter=1000, random_state=42)
    
    # Cross-validation for unregularized model
    accuracy_scores_unreg = cross_val_score(logreg_unregularized, X_train, y_train, cv=5, scoring='accuracy')
    f1_scores_unreg = cross_val_score(logreg_unregularized, X_train, y_train, cv=5, scoring='f1')
    
    print("Unregularized Logistic Regression:")
    print(f"Average Accuracy: {accuracy_scores_unreg.mean():.4f} (+/- {accuracy_scores_unreg.std() * 2:.4f})")
    print(f"Average F1 Score: {f1_scores_unreg.mean():.4f} (+/- {f1_scores_unreg.std() * 2:.4f})")
    
    # 3. Apply different regularization techniques
    print("\n3. Applying regularization techniques...")
    
    # L1 Regularization (Lasso)
    print("\nL1 Regularization (Lasso):")
    logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    accuracy_scores_l1 = cross_val_score(logreg_l1, X_train, y_train, cv=5, scoring='accuracy')
    f1_scores_l1 = cross_val_score(logreg_l1, X_train, y_train, cv=5, scoring='f1')
    
    print(f"Average Accuracy: {accuracy_scores_l1.mean():.4f} (+/- {accuracy_scores_l1.std() * 2:.4f})")
    print(f"Average F1 Score: {f1_scores_l1.mean():.4f} (+/- {f1_scores_l1.std() * 2:.4f})")
    
    # L2 Regularization (Ridge)
    print("\nL2 Regularization (Ridge):")
    logreg_l2 = LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
    accuracy_scores_l2 = cross_val_score(logreg_l2, X_train, y_train, cv=5, scoring='accuracy')
    f1_scores_l2 = cross_val_score(logreg_l2, X_train, y_train, cv=5, scoring='f1')
    
    print(f"Average Accuracy: {accuracy_scores_l2.mean():.4f} (+/- {accuracy_scores_l2.std() * 2:.4f})")
    print(f"Average F1 Score: {f1_scores_l2.mean():.4f} (+/- {f1_scores_l2.std() * 2:.4f})")
    
    # Elastic Net Regularization
    print("\nElastic Net Regularization:")
    logreg_elastic = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, 
                                       max_iter=1000, random_state=42)
    accuracy_scores_elastic = cross_val_score(logreg_elastic, X_train, y_train, cv=5, scoring='accuracy')
    f1_scores_elastic = cross_val_score(logreg_elastic, X_train, y_train, cv=5, scoring='f1')
    
    print(f"Average Accuracy: {accuracy_scores_elastic.mean():.4f} (+/- {accuracy_scores_elastic.std() * 2:.4f})")
    print(f"Average F1 Score: {f1_scores_elastic.mean():.4f} (+/- {f1_scores_elastic.std() * 2:.4f})")
    
    # 4. Compare coefficients and performance
    print("\n4. Comparing regularization techniques...")
    
    # Fit all models on full training data to examine coefficients
    logreg_unregularized.fit(X_train, y_train)
    logreg_l1.fit(X_train, y_train)
    logreg_l2.fit(X_train, y_train)
    logreg_elastic.fit(X_train, y_train)
    
    # Compare coefficients
    print("\nCoefficient Comparison:")
    print(f"{'Feature':<15} {'Unregularized':<15} {'L1 (Lasso)':<15} {'L2 (Ridge)':<15} {'Elastic Net':<15}")
    print("-" * 75)
    
    for i, feature in enumerate(features_b):
        print(f"{feature:<15} {logreg_unregularized.coef_[0][i]:<15.4f} "
              f"{logreg_l1.coef_[0][i]:<15.4f} {logreg_l2.coef_[0][i]:<15.4f} "
              f"{logreg_elastic.coef_[0][i]:<15.4f}")
    
    # Count zero coefficients in L1
    zero_coeff_l1 = np.sum(logreg_l1.coef_[0] == 0)
    zero_coeff_elastic = np.sum(logreg_elastic.coef_[0] == 0)
    
    print(f"\nFeature Selection Summary:")
    print(f"L1 (Lasso) - Features shrunk to zero: {zero_coeff_l1}/{len(features_b)}")
    print(f"Elastic Net - Features shrunk to zero: {zero_coeff_elastic}/{len(features_b)}")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"{'Model':<20} {'Avg Accuracy':<15} {'Avg F1':<15}")
    print("-" * 50)
    print(f"{'Unregularized':<20} {accuracy_scores_unreg.mean():<15.4f} {f1_scores_unreg.mean():<15.4f}")
    print(f"{'L1 (Lasso)':<20} {accuracy_scores_l1.mean():<15.4f} {f1_scores_l1.mean():<15.4f}")
    print(f"{'L2 (Ridge)':<20} {accuracy_scores_l2.mean():<15.4f} {f1_scores_l2.mean():<15.4f}")
    print(f"{'Elastic Net':<20} {accuracy_scores_elastic.mean():<15.4f} {f1_scores_elastic.mean():<15.4f}")
    
    return {
        'unregularized': (accuracy_scores_unreg.mean(), f1_scores_unreg.mean()),
        'l1': (accuracy_scores_l1.mean(), f1_scores_l1.mean()),
        'l2': (accuracy_scores_l2.mean(), f1_scores_l2.mean()),
        'elastic': (accuracy_scores_elastic.mean(), f1_scores_elastic.mean()),
        'models': {
            'unregularized': logreg_unregularized,
            'l1': logreg_l1,
            'l2': logreg_l2,
            'elastic': logreg_elastic
        }
    }

def generate_discussion():
    """
    Generate markdown discussion about regularization techniques
    """
    discussion = """
## Discussion: Regularization Techniques Comparison

### L1 Regularization (Lasso)
- **Effect**: Performs feature selection by shrinking some coefficients to exactly zero
- **Advantage**: Creates sparse models, automatically selects important features
- **Use case**: When you suspect many features are irrelevant

### L2 Regularization (Ridge)
- **Effect**: Shrinks all coefficients toward zero but doesn't eliminate any
- **Advantage**: Prevents overfitting while keeping all features
- **Use case**: When all features might be relevant but you want to prevent overfitting

### Elastic Net
- **Effect**: Combines L1 and L2 regularization (l1_ratio=0.5 means equal weight)
- **Advantage**: Benefits of both L1 and L2, handles correlated features better
- **Use case**: When you want both feature selection and regularization

### Performance Analysis
The best performing regularization technique depends on the specific dataset characteristics. In this case:
- If L1 performs best: Many irrelevant features, good feature selection needed
- If L2 performs best: All features relevant, just need overfitting prevention
- If Elastic Net performs best: Correlated features present, need balanced approach

### Feature Selection Impact
L1 regularization automatically performs feature selection by setting some coefficients to zero, which can improve model interpretability and reduce overfitting by using fewer features.
"""
    return discussion

def main():
    """
    Main function to run the complete analysis
    """
    print("Project Part III – Model selection and regularization")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Part A: Model Selection
    results_a, best_model_name, data_a = part_a_model_selection(df)
    
    # Part B: Regularization
    results_b = part_b_regularization(df)
    
    # Generate discussion
    discussion = generate_discussion()
    print("\n" + "=" * 60)
    print("DISCUSSION")
    print("=" * 60)
    print(discussion)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best model in Part A: {best_model_name}")
    
    # Find best regularization technique based on F1 score
    # Exclude 'models' key from comparison
    reg_techniques = {k: v for k, v in results_b.items() if k != 'models'}
    best_reg_technique = max(reg_techniques.keys(), key=lambda x: reg_techniques[x][1])
    print(f"Best regularization technique: {best_reg_technique}")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 