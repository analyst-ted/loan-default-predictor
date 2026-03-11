from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


# 1. Splitting the data and scaling it to train the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def split_and_scale(df):
    """Splits data and scale features to prevent data leakage"""
    print('Splitting and scaling data...')

    # 1. Seperate features (X) and target (y)
    X = df.drop('loan_repaid', axis=1)
    y = df['loan_repaid']

    # 2. creating training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # 3. Initialize the scaler
    scaler = MinMaxScaler()

    # 4. Fit and transform on training data
    X_train = scaler.fit_transform(X_train)

    #. 5. Transform the test data
    X_test = scaler.transform(X_test)

    print(f"Training Features Shape: {X_train.shape}")
    print(f"Testing Features Shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


# 2. Training the model

def build_model(input_dim):
    """Builds a deep learning architecture for binary classification."""
    print("Initializing Neural Network Architecture...")
    
    model = Sequential()

    #Input layer
    model.add(Input(shape=(input_dim,)))
    
    # First Hidden Layer (Matching the number of features: 71)
    model.add(Dense(71, activation='relu'))
    model.add(Dropout(0.2)) # Turn off 20% of neurons randomly
    
    # Second Hidden Layer (Half the size: 35)
    model.add(Dense(35, activation='relu'))
    model.add(Dropout(0.2))
    
    # Third Hidden Layer (Half again: 17)
    model.add(Dense(17, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output Layer (Your Sigmoid answer!)
    model.add(Dense(1, activation='sigmoid'))

    # compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    print("Architecture built successfully!")
    return model

# 3. Evaluating the model

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints business metrics"""
    print('\nEvaluating Model Performance')

    # 1. Get the raw probabilities
    predictions = model.predict(X_test)

    # 2. Convert probabilities to absolute 0 and 1
    # predictions = np.round(predictions).astype(int)
    predictions = (predictions>=0.60).astype(int)
    # predictions = (predictions>=0.70).astype(int)

    # 3. print business reports
    print('\n--- Classification Report ---')
    print(classification_report(y_test,predictions))
          
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, predictions))
