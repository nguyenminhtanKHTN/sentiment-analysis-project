Sentiment Analysis Project
This project implements a text sentiment analysis model to classify movie reviews as positive or negative using the IMDb dataset. Below is the directory structure and instructions to run the project.

Directory Structure
sentiment_analysis_project/
├── data/
│   ├── raw/
│   │   └── IMDB Dataset.csv
│   ├── processed/
│   │   └── cleaned_reviews.csv
│   └── external/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
├── models/
│   ├── sentiment_model_C1.pkl
│   └── tfidf_vectorizer.pkl
├── results/
│   ├── confusion_matrix_C1.png
│   ├── sentiment_distribution.png
│   ├── review_length_distribution.png
│   ├── accuracy_vs_C.png
│   └── metrics.txt
├── requirements.txt
└── README.md

Setup Instructions

Create the directory structure:

Use a terminal or file explorer to create the folders and files as shown above.
Example (Linux/Mac):mkdir -p sentiment_analysis_project/{data/{raw,processed,external},notebooks,src,models,results}
touch sentiment_analysis_project/{requirements.txt,README.md}
touch sentiment_analysis_project/src/{__init__.py,preprocess.py,train_model.py,predict.py,app.py}




Download the dataset:

Download IMDB Dataset.csv from Kaggle.
Place it in data/raw/.


Install dependencies:

Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install packages:pip install -r requirements.txt




Run the notebooks:

Open Jupyter Notebook:jupyter notebook


Follow the notebooks in order: data_exploration.ipynb → preprocessing.ipynb → model_training.ipynb.


Run the API:

Run the Flask API:python src/app.py


Send a POST request to http://localhost:5000/predict with JSON payload:curl -X POST -H "Content-Type: application/json" -d '{"review": "This movie was great!"}' http://localhost:5000/predict


Expected response:{
    "review": "This movie was great!",
    "cleaned_review": "movie great",
    "sentiment": "Positive"
}





Next Steps

Use the API to predict sentiment on new reviews.
Explore improving the model (e.g., try other algorithms like Naive Bayes).
Extend the API to handle multiple reviews or integrate with a web frontend.
