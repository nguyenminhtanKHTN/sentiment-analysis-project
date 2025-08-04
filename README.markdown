# Sentiment Analysis Project

This project implements a text sentiment analysis model to classify movie reviews as positive or negative using the IMDb dataset. Below is the directory structure to organize the project effectively.

## Directory Structure

```
sentiment_analysis_project/
├── data/
│   ├── raw/
│   │   └── IMDB Dataset.csv          # Raw IMDb dataset downloaded from Kaggle
│   ├── processed/
│   │   └── cleaned_reviews.csv       # Processed dataset after cleaning and preprocessing
│   └── external/
│       └── additional_data.csv        # Optional: Additional datasets (e.g., from X posts)
├── notebooks/
│   ├── data_exploration.ipynb         # Notebook for initial data exploration and visualization
│   ├── preprocessing.ipynb            # Notebook for text preprocessing steps
│   └── model_training.ipynb           # Notebook for model training and evaluation
├── src/
│   ├── __init__.py                   # Makes src a Python module
│   ├── preprocess.py                 # Python script for text preprocessing functions
│   ├── train_model.py                # Python script for training the model
│   └── predict.py                    # Python script for making predictions
├── models/
│   ├── sentiment_model.pkl           # Saved trained model
│   └── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── results/
│   ├── confusion_matrix.png          # Visualizations (e.g., confusion matrix)
│   └── metrics.txt                   # Model performance metrics
├── requirements.txt                  # List of required Python packages
└── README.md                         # Project documentation (this file)
```

## Description of Each Component

- **data/**: Stores all datasets.
  - **raw/**: Contains the original, unprocessed dataset (e.g., `IMDB Dataset.csv` from Kaggle).
  - **processed/**: Stores datasets after preprocessing (e.g., cleaned text data).
  - **external/**: Optional folder for additional data (e.g., scraped X posts or other sources).

- **notebooks/**: Contains Jupyter Notebooks for different stages of the project.
  - `data_exploration.ipynb`: For loading, exploring, and visualizing the dataset (e.g., distribution of sentiments).
  - `preprocessing.ipynb`: For implementing and testing text preprocessing steps (e.g., tokenization, removing stopwords).
  - `model_training.ipynb`: For training the model, evaluating performance, and saving results.

- **src/**: Contains reusable Python scripts for modularity.
  - `preprocess.py`: Functions for text preprocessing (e.g., cleaning, tokenization).
  - `train_model.py`: Code for training the sentiment analysis model.
  - `predict.py`: Code to load the model and make predictions on new reviews.

- **models/**: Stores trained models and vectorizers.
  - `sentiment_model.pkl`: Saved Logistic Regression model.
  - `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer for transforming new text.

- **results/**: Stores outputs like visualizations and performance metrics.
  - `confusion_matrix.png`: Plot of the confusion matrix.
  - `metrics.txt`: Text file with accuracy, precision, recall, and F1-score.

- **requirements.txt**: Lists all required Python packages for reproducibility.
  ```
  pandas==2.2.2
  scikit-learn==1.5.1
  nltk==3.8.1
  matplotlib==3.9.1
  seaborn==0.13.2
  ```

## Setup Instructions

1. **Create the directory structure**:
   - Use a terminal or file explorer to create the folders and files as shown above.
   - Example (Linux/Mac):
     ```bash
     mkdir -p sentiment_analysis_project/{data/{raw,processed,external},notebooks,src,models,results}
     touch sentiment_analysis_project/{requirements.txt,README.md}
     touch sentiment_analysis_project/src/{__init__.py,preprocess.py,train_model.py,predict.py}
     ```

2. **Download the dataset**:
   - Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25n/imdb-dataset-of-50k-movie-reviews).
   - Place it in `data/raw/`.

3. **Install dependencies**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Start working**:
   - Open Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Create and use the notebooks in the `notebooks/` folder to follow the project steps.

## Next Steps
- Follow the notebooks in order: `data_exploration.ipynb` → `preprocessing.ipynb` → `model_training.ipynb`.
- Use scripts in `src/` for reusable code.
- Save models and results in `models/` and `results/` respectively.