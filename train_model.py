import pandas as pd

from credit_model import CreditScoringModel

# Get historic loan data
loans = pd.read_parquet("data/loan_table.parquet")

# Create model
model = CreditScoringModel()

# Train model (using Postgres for zipcode and credit history features)
if not model.is_model_trained():
    print('Starting model training...')
    model.train(loans)
    print('Model has been trained successfully')
else:
    print('Model was already trained in a previous run')


