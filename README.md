## Demo

[Video Link](https://www.loom.com/share/d66583486bed4b34be3f4e6efdc8edf4)

## Explainability & Transparency

This app provides clear explanations for every transaction categorization:

- **ML Predictions:** When the model predicts a category, the app displays the top words (features) from your input that most influenced the decision.  
  _Example:_  
  `🛍️ Added to Shopping (confidence: 0.92) Top features: amazon, order, payment`

- **Keyword Fallback:** If the model is unsure and uses the YAML keyword fallback, the app shows the exact keyword that matched.  
  _Example:_  
  `💡 Added to Bills (via keyword: 'electricity')`

This increases user trust and makes the categorization process transparent.

## Dataset & Data Strategy

### Source & Creation

The dataset (`data/transactions.csv`) is a synthetic collection of transaction descriptions and their categories, designed to mimic real-world financial data. It includes a wide variety of merchants, payment types, and category labels (Dining, Shopping, Travel, Bills, Entertainment, Groceries, Health, Housing, Subscriptions, etc.).

### Balancing & Augmentation

To ensure robust model performance and fairness, the dataset was balanced by adding realistic, diverse examples for underrepresented categories. This helps the model achieve high accuracy across all classes and reduces bias toward frequent categories.

### Data Expansion

You can expand the dataset by:

- Adding new rows to `transactions.csv` for new merchants, payment types, or categories.
- Using the Streamlit app’s feedback loop to log user corrections (future retraining can use these logs).
- Modifying or adding categories in `config/categories.yaml` to update the taxonomy without code changes.

### Cleaning & Preprocessing

All transaction descriptions are cleaned (lowercased, special characters removed, extra spaces trimmed) before training and prediction. See `src/preprocess.py` for details.

### Reproducibility

The dataset split (train/test) is reproducible due to a fixed random seed. This ensures consistent evaluation results.

### Documentation

If you update the dataset, please document your changes and rationale in this section to keep all users up to date.

---

STEP 1
Inside the src/ folder, create this file:

src/preprocess.py

This file will contain all text cleaning logic like:

lowercase

remove special characters

remove extra spaces

tokenize

remove stopwords (optional later)

re → Python's regular expressions (used for cleaning unwanted characters)

nltk → Natural Language Toolkit (commonly used for text processing)

stopwords → A list of common English words that carry little meaning
Example stopwords: the, is, at, on, and, to

Step 2:
config/config_loader.py
use :
What this file will do

Read the YAML file.

Extract the categories list.

Make it available for ML model training & prediction.

STEP 3 :
data/transactions.csv

This is your training dataset.
Your ML model will learn from:

description → the raw text

category → the correct label

Later you can expand it, generate more automatically, or let the UI help correct predictions.

STEP 4:
src/training.py

WHY DO WE NEED training.py ?

This file is the heart of a machine learning project.

Think of it like this:

Your CSV file contains examples → “Starbucks” → “Dining”

A machine cannot understand text, so we need to train it to learn patterns.

This training produces a model file (transaction_model.joblib)

Later, your prediction script will load this model and correctly categorize new transactions.

So training.py is:

👉 a script that takes CSV → learns patterns → generates a model file

Without this file, the model does not exist, you cannot predict anything.

This is exactly what ML projects do:
dataset → train → save model → load model → use model.

Import Why needed?
pandas Reads your CSV file (dataset)
TfidfVectorizer Converts text (“Starbucks”) into numbers
LogisticRegression The AI model that learns categories
Pipeline Combines TF-IDF + model in one neat object
train_test_split Splits data into train/test for evaluation
classification_report Gives accuracy, precision, F1 score
confusion_matrix Helps show model performance for PDF
joblib Saves the trained model to a file
Path Helps create folders like /models

What is train_test_split?

You NEVER train the model using all data, otherwise:

it memorizes instead of learning

you cannot check accuracy honestly

So we split your data into two parts:

Part % Purpose
Training 80% Model learns patterns
Testing 20% Check accuracy on unseen data

This is exactly what this does:
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42
)

Why test_size = 0.2 ?

It means:

80% data → training  
20% data → testing

random_state = 42
means:

"Keep the split SAME every time I run this script."

It's like setting the same "shuffle pattern" so results are reproducible.

Why 42?
It's just a funny mathematician joke ("Answer to everything").
Could be any number.
