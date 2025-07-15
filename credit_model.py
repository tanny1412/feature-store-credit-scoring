from pathlib import Path

import feast
import joblib
import pandas as pd
from sklearn import tree
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


class CreditScoringModel:
    # List of categorical features that require encoding before model training/prediction
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "city",
        "state",
        "location_type",
    ]

    # List of features to fetch from Feast feature store for each entity
    feast_features = [
        "zipcode_features:city",
        "zipcode_features:state",
        "zipcode_features:location_type",
        "zipcode_features:tax_returns_filed",
        "zipcode_features:population",
        "zipcode_features:total_wages",
        "credit_history:credit_card_due",
        "credit_history:mortgage_due",
        "credit_history:student_loan_due",
        "credit_history:vehicle_loan_due",
        "credit_history:hard_pulls",
        "credit_history:missed_payments_2y",
        "credit_history:missed_payments_1y",
        "credit_history:missed_payments_6m",
        "credit_history:bankruptcies",
        "total_debt_calc:total_debt_due",
    ]

    target = "loan_status"  # The target variable for training (label)
    model_filename = "model.bin"  # File to save/load the trained model
    encoder_filename = "encoder.bin"  # File to save/load the fitted encoder

    def __init__(self):
        # Load the trained model from disk if it exists, otherwise initialize a new DecisionTreeClassifier
        if Path(self.model_filename).exists():
            self.classifier = joblib.load(self.model_filename)
        else:
            self.classifier = tree.DecisionTreeClassifier()

        # Load the fitted ordinal encoder from disk if it exists, otherwise initialize a new encoder
        if Path(self.encoder_filename).exists():
            self.encoder = joblib.load(self.encoder_filename)
        else:
            self.encoder = OrdinalEncoder()

        # Initialize Feast feature store for fetching features
        self.fs = feast.FeatureStore(repo_path="feature_repo")

    def train(self, loans):
        # Prepare training features and labels from the provided loan data
        train_X, train_Y = self._get_training_features(loans)

        # Train the classifier on the sorted feature columns for consistency
        self.classifier.fit(train_X[sorted(train_X)], train_Y)
        # Save the trained model to disk for reuse
        joblib.dump(self.classifier, self.model_filename)

    def _get_training_features(self, loans):
        # Fetch historical features from Feast for the given loans
        training_df = self.fs.get_historical_features(
            entity_df=loans, features=self.feast_features
        ).to_df()

        # Fit the encoder on categorical columns and encode them
        self._fit_ordinal_encoder(training_df)
        self._apply_ordinal_encoding(training_df)

        # Select input features by dropping target, timestamps, and identifiers
        train_X = training_df[
            training_df.columns.drop(self.target)
            .drop("event_timestamp")
            .drop("created_timestamp")
            .drop("loan_id")
            .drop("zipcode")
            .drop("dob_ssn")
        ]
        # Ensure columns are sorted for consistency
        train_X = train_X.reindex(sorted(train_X.columns), axis=1)
        # Extract the target variable
        train_Y = training_df.loc[:, self.target]

        return train_X, train_Y

    def _fit_ordinal_encoder(self, requests):
        # Fit the encoder on the categorical columns of the provided DataFrame
        self.encoder.fit(requests[self.categorical_features])
        # Save the fitted encoder to disk for reuse
        joblib.dump(self.encoder, self.encoder_filename)

    def _apply_ordinal_encoding(self, requests):
        # Transform the categorical columns in place using the fitted encoder
        requests[self.categorical_features] = self.encoder.transform(
            requests[self.categorical_features]
        )

    def predict(self, request):
        # Fetch online features from Feast for the given request (real-time prediction)
        feature_vector = self._get_online_features_from_feast(request)

        # Merge user input (request) with features fetched from Feast
        features = request.copy()
        features.update(feature_vector)
        # Convert the merged dictionary to a DataFrame for model input
        features_df = pd.DataFrame.from_dict(features)
        pd.set_option('display.max_columns', 15)  # For better debug printing if needed
        # (Optional debug print statement can be added here)

        # Apply ordinal encoding to categorical columns to match training format
        self._apply_ordinal_encoding(features_df)

        # Sort columns to ensure order matches training
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        # Drop identifier columns not used for prediction
        features_df = features_df[features_df.columns.drop("zipcode").drop("dob_ssn")]

        # Make prediction using the trained classifier
        features_df["prediction"] = self.classifier.predict(features_df)

        # Return the prediction result (0 = approved, 1 = rejected)
        return features_df["prediction"].iloc[0]

    def _get_online_features_from_feast(self, request):
        # Extract entity keys from the request for feature lookup
        zipcode = request["zipcode"][0]
        dob_ssn = request["dob_ssn"][0]
        loan_amnt= request["loan_amnt"][0]

        # Fetch the latest feature values for the entity from Feast
        return self.fs.get_online_features(
            entity_rows=[{"zipcode": zipcode, "dob_ssn": dob_ssn, "loan_amnt": loan_amnt}],
            features=self.feast_features,
        ).to_dict()

    def is_model_trained(self):
        # Check if the classifier has been fitted (trained)
        try:
            check_is_fitted(self.classifier, "tree_")
        except NotFittedError:
            return False
        return True
