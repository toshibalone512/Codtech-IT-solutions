import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import pickle
import random

# Import Keras/TensorFlow for the neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Define the Knowledge Base (Intents)
intents = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "Hello", "Hey", "Good day", "Greetings"],
         "responses": ["Hello!", "Hi there!", "How can I help you?", "Hey, glad you're here!"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "I'm leaving", "Take care"],
         "responses": ["Sad to see you go :(", "Talk to you later!", "Goodbye!", "Come back soon."],
         "context_set": ""
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Cheers"],
         "responses": ["Happy to help!", "Anytime!", "My pleasure.", "You're welcome!"],
         "context_set": ""
        },
        {"tag": "contact",
         "patterns": ["Contact info", "How to reach you", "Phone number", "Email address"],
         "responses": ["You can reach us at support@example.com or call (555) 123-4567."],
         "context_set": ""
        },
        {"tag": "about",
         "patterns": ["What is your purpose", "Who are you", "What can you do"],
         "responses": ["I am an AI assistant designed to answer your queries.", "I'm a simple chatbot powered by a neural network."],
         "context_set": ""
        }
    ]
}

# Preprocessing and Training Data Setup

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Loop through each intent and pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents (list of tuples: (word_list, tag))
        documents.append((word_list, intent['tag']))
        # Add tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase all words, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes for later use in the main chatbot loop
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Create a Bag of Words (BoW) for each pattern
for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create the bag of words array with 1 if word match is found, else 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # create output row '0' for all intents and '1' for the current intent
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Convert to numpy array and shuffle
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists: X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created.")

# Build and Train the Model

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. SGD with momentum is often used for this type of problem
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model (fit the data)
# Use verbose=0 to suppress training output
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# Save the model
model.save('chatbot_model.h5', hist)

print("Model created and saved.")

# Implement the Chatbot Function

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the user input."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """Creates a Bag-of-Words array for the user input."""
    # Clean up the user input
    sentence_words = clean_up_sentence(sentence)
    # Create the Bag of Words (BoW)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    """Predicts the intent class of the user input."""
    # Filter out predictions below a threshold (e.g., 0.25)
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Retrieves a random response based on the predicted intent."""
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        # Fallback response if no intent is matched (optional)
        result = "I'm sorry, I don't understand that. Could you try rephrasing?"
    return result

# --- Main Chat Loop ---

def chatbot_response(text):
    """Wrapper function to get the full chatbot response."""
    ints = predict_class(text, model)
    if not ints:
        # Handle case where no prediction meets the threshold
        return "I'm not sure how to respond to that. Can you simplify your question?"
    res = get_response(ints, intents)
    return res

print("Chatbot is ready! Start talking (type 'quit' to exit)")

# Run the chatbot continuously
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Bot: Talk to you later!")
        break
    
    response = chatbot_response(user_input)
    print(f"Bot: {response}")
