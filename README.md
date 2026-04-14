# Bank Term Deposit Predictor

A Streamlit web app that predicts whether a bank client will subscribe to a term deposit.

The app uses a saved machine learning pipeline (`bank_pipeline.pkl`) trained on the provided `bank-full.csv` dataset.

## Project Files

- `app.py` - Streamlit application for interactive prediction.
- `bank_term_deposit_full.ipynb` - Jupyter notebook for data exploration, model training, and pipeline export.
- `bank-full.csv` - Bank marketing dataset used for training and evaluation.
- `bank_pipeline.pkl` - Saved model pipeline used by the Streamlit app.
- `requirements.txt` - Python dependencies required to run the app and notebook.

## Features

- Input customer and campaign details through an easy web interface.
- Uses feature engineering consistent with the notebook.
- Predicts whether the client will subscribe to a term deposit.
- Displays predicted probability, verdict, and input summary.

## Setup

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the App

Start the Streamlit app:

```powershell
streamlit run app.py
```

Open the browser link shown by Streamlit to use the predictor.

## Notes

- `bank_pipeline.pkl` is required by the app. If it is missing, run the notebook `bank_term_deposit_full.ipynb` to train and save the model.
- The notebook includes data preprocessing, model training, evaluation, and saving the final pipeline.

## Recommended Workflow

1. Inspect and run the notebook to retrain or adjust the model.
2. Confirm the generated `bank_pipeline.pkl` exists.
3. Run the Streamlit app with `streamlit run app.py`.
