# Naive Bayes Email Spam Classifier

This project implements a Naive Bayes classifier from scratch to detect spam emails. It provides a simple pipeline for training, evaluating, and predicting whether an email is spam or not using a dataset of labeled emails.

## Features
- Preprocessing and vectorization of email text
- Training a Naive Bayes classifier
- Saving and loading trained models
- Predicting spam/ham for new emails
- Example usage in a Jupyter notebook

## Project Structure
- `main.py`: Main script for training and prediction
- `main.ipynb`: Jupyter notebook with example usage and predictions
- `emails.csv`: Dataset of emails with labels (spam/ham)

## Getting Started

### Prerequisites
- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage
1. **Train the model** (if not already trained):
   ```bash
   python main.py
   ```
2. **Run predictions**:
   - Use the provided `main.ipynb` notebook to test the model on sample emails.
   - Or, use functions in `main.py` to predict on new email text.

## Example
In `main.ipynb`, you can test the model with:
```python
predict_email("Your email text here")
```

## Dataset
- The `emails.csv` file should contain labeled emails with columns for the email text and label (spam/ham).

## License
This project is for educational purposes.
