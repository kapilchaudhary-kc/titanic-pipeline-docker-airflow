import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def prepare_data(data_path, data_type='train'):
    """
    Prepare Titanic dataset with proper path handling
    Args:
        data_path: Path to directory containing the data files
        data_type: Either 'train' or 'test'
    Returns:
        Processed DataFrame
    """
    print(f"---Preparing {data_type} data---")
    
    # Load dataset
    file_path = os.path.join(data_path, f"{data_type}.csv")
    df = pd.read_csv(file_path)
    
    # Data Validation
    from func import data_validator
    validation = data_validator.Validator(df)
    report = validation.validate()
    print(report)
    
    if report['Schema Validation'] != 'No Issue with Schema':
        raise ValueError("Schema Validation Failed")
    
    # Handle duplicates
    if report['Duplicate Rows'] > 0:
        print(f"Removing {report['Duplicate Rows']} duplicate rows")
        df = df.drop_duplicates()
    
    if report['Duplicate IDs'] > 0 and 'PassengerId' in df.columns:
        print(f"Keeping first of {report['Duplicate IDs']} duplicate IDs")
        df = df.drop_duplicates(subset=['PassengerId'], keep='first')
    
    # Feature Engineering
    df['Age'] = df['Age'].clip(lower=1, upper=65)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Cabin/Deck processing
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].apply(
            lambda x: 'no cabin' if pd.isna(x) or x not in ['A','B','C','D','E','F','G'] else x
        )
    
    # Drop columns
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    if file_path == os.path.join(data_path, f"train.csv"):
        cols_to_drop.append('PassengerId')
    df = df.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'O':
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Imputed {col} (categorical) with mode: {mode_val}")
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Imputed {col} (numerical) with median: {median_val}")
    
    # One-hot encoding (consistent for train/test)
    categorical_cols = ['Sex', 'Embarked', 'Deck'] if 'Deck' in df.columns else ['Sex', 'Embarked']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True).astype(int)
    
    # Save processed data
    processed_path = os.path.join(data_path, f"processed_{data_type}.pkl")
    df.to_pickle(processed_path)
    print(f"Saved processed {data_type} data to {processed_path}")
    
    return df

def train_model(train_data_path):
    """
    Train model on prepared training data
    Args:
        train_data_path: Path to directory containing processed training data
    Returns:
        Trained model and test accuracy
    """
    print("---Training Model---")
    
    # Load processed training data
    train_path = os.path.join(train_data_path, "processed_train.pkl")
    train_df = pd.read_pickle(train_path)
    
    # Split data
    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Validate
    val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(train_data_path, "titanic_model.pkl")
    pd.to_pickle(model, model_path)
    print(f"Saved model to {model_path}")
    
    return model, accuracy

def test_model(test_data_path, model_path):
    """
    Test model on prepared test data
    Args:
        test_data_path: Path to directory containing processed test data
        model_path: Path to saved model
    Returns:
        For labeled data: Test predictions and accuracy
        For unlabeled data: Test predictions and submission DataFrame
    """
    print("---Testing Model---")
    
    # Load processed test data and model
    test_path = os.path.join(test_data_path, "processed_test.pkl")
    test_df = pd.read_pickle(test_path)
    model = pd.read_pickle(model_path)

    # Make predictions
    if 'Survived' in test_df.columns:  # If test has labels
        X_test = test_df.drop('Survived', axis=1)
        y_test = test_df['Survived']
        test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        return test_pred, accuracy
    
    else:  # For unlabeled test data
        if 'PassengerId' in test_df.columns:
            X_test = test_df.drop('PassengerId', axis=1)
            passenger_ids = test_df['PassengerId']
        else:
            X_test = test_df
            passenger_ids = pd.Series(range(1, len(test_df) + 1)) #Create dummy passengers' id

        test_pred = model.predict(X_test)
        print("Test predictions complete")

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': test_pred
        })

        return test_pred, submission_df