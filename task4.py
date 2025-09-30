import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
   
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df.columns = ['label', 'message']
except FileNotFoundError:
    print("WARNING: 'spam.csv' not found. Creating a dummy DataFrame for demonstration.")
    data = {
        'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'ham', 'spam'],
        'message': [
            'Go until jurong point, crazy.. Available only in bugis n great world la e buffet...',
            'WINNER!! You have won a free ticket to the movies. Call now.',
            'Nah I don\'t think he goes to usf, he lives around here though',
            'Urgent! Claim your cash prize of $1000. Text YES to 8888 now.',
            'I\'ve been searching for the right words to thank you for this message.',
            'Ok lor... Joking wif u oni...',
            'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for FREE!'
        ]
    }
    df = pd.DataFrame(data)

# Show the first few rows
print("--- Data Head ---")
print(df.head())
print("\n--- Label Distribution ---")
print(df['label'].value_counts())


# --- Label Encoding ---
# Convert 'ham' to 0 and 'spam' to 1
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_encoded']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# Create a CountVectorizer object to transform text into feature vectors
vectorizer = CountVectorizer()

# Fit the vectorizer on the TRAINING data and transform it
X_train_vec = vectorizer.fit_transform(X_train)

# Transform the TEST data using the FITTED vectorizer (DO NOT re-fit)
X_test_vec = vectorizer.transform(X_test)

print(f"\nShape of Training Features (Messages x Unique Words): {X_train_vec.shape}")


# Model Initialization
model = MultinomialNB()

# Model Training
# Train the model using the vectorized training data
print("\nTraining the Multinomial Naive Bayes model...")
model.fit(X_train_vec, y_train)

print("Model training complete.")

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation Metrics
print("\n--- Model Evaluation ---")

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 2. Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)

# 3. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))


def predict_new_message(message):
    """Predicts if a new message is 'Ham' or 'Spam'."""
    # 1. Vectorize the new message using the original fitted vectorizer
    message_vec = vectorizer.transform([message])
    
    # 2. Predict the label (0 or 1)
    prediction = model.predict(message_vec)[0]
    
    # 3. Predict the probability of each class
    probability = model.predict_proba(message_vec)[0]
    
    result = "SPAM" if prediction == 1 else "HAM"
    
    print(f"\n--- Prediction for: '{message}' ---")
    print(f"Predicted Class: {result}")
    print(f"Probability (Ham): {probability[0]:.4f}")
    print(f"Probability (Spam): {probability[1]:.4f}")

# Example 1: Ham message
predict_new_message("Hey, can we meet for lunch tomorrow at 1 PM?")

# Example 2: Spam message
predict_new_message("URGENT! You've won $5000! Click here to claim your prize now.")
