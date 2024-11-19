import pandas as pd
import pickle  # For saving and loading as a pickle file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


#load data
def load_data(file_path):
    """Loads and preprocesses the dataset."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  # Strip whitespace from column names
    data = data.drop_duplicates()  # Remove duplicates
    return data

#train test split
def split_data(data):
    """Splits the data into features (X) and target (y), and performs train-test split."""
    X = data.drop(['loan_id', 'loan_status'], axis=1)
    y = data['loan_status']
    return train_test_split(X, y, test_size=0.2, random_state=42)

#data pipeline
def create_pipeline():
    """Creates a preprocessing and classification pipeline."""
    # Define the numeric and categorical columns
    numeric_features = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    categorical_features = ['education', 'self_employed']

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values
        ('scaler', StandardScaler())  # Scale numerical data
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical data
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
    ])

    # Combine into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline


# Main function to train and evaluate the model
def main():
    """Main function to load data, train the model, and save the pipeline."""
    # Load the data
    data = load_data('loan_approval_dataset.csv')

    # Split the data
    X_train, X_test, Y_train, Y_test = split_data(data)

    # Create and train the pipeline
    model_pipeline = create_pipeline()
    model_pipeline.fit(X_train, Y_train)

    # Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    print('Accuracy:', accuracy_score(Y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))
    print('Classification Report:\n', classification_report(Y_test, y_pred))

    # Save the pipeline as a pickle file
    with open('loan_approval_pipeline.pkl', 'wb') as file:
        pickle.dump(model_pipeline, file)
    print("Pipeline saved as 'loan_approval_pipeline.pkl'")


# Run the script
if __name__ == "__main__":
    main()