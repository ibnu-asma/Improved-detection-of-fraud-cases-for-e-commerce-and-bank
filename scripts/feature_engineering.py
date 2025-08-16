import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame, logging):
        self.df = df.copy()
        self.processed_df = None
        self.scaler = StandardScaler()
        self.logging = logging
        self.logging.info("FeatureEngineering class initialized with the provided DataFrame.")

    def calculate_fraud_rate(self):
        """
        Calculates the fraud rate for each country based on the transaction data.
        """
        self.logging.info("Calculating fraud rate by country...")
        try:
            # Calculate total transactions and fraudulent transactions by country
            fraud_counts = self.df[self.df['class'] == 1].groupby('country').size()
            total_counts = self.df.groupby('country').size()

            # Calculate fraud rate
            fraud_rate = fraud_counts / total_counts

            # Create a DataFrame from the fraud rate Series
            fraud_rate_df = fraud_rate.reset_index(name='fraud_rate')

            # Merge the fraud rate back into the original DataFrame
            self.df = self.df.merge(fraud_rate_df, on='country', how='left')
            self.df['fraud_rate'] = self.df['fraud_rate'].fillna(0)
            self.logging.info("Fraud rate calculated and merged successfully.")
        except Exception as e:
            self.logging.error("Error in calculating fraud rate: %s", e)
            raise

    def preprocess_datetime(self):
        self.logging.info("Preprocessing datetime features...")
        try:
            self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
            self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
            self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
            self.df['purchase_delay'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds() / 3600
            self.logging.info("Datetime features successfully created.")
        except Exception as e:
            self.logging.error("Error in preprocessing datetime features: %s", e)
            raise

    def calculate_transaction_frequency(self):
        self.logging.info("Calculating transaction frequency and velocity...")
        try:
            user_freq = self.df.groupby('user_id').size()
            self.df['user_transaction_frequency'] = self.df['user_id'].map(user_freq)

            device_freq = self.df.groupby('device_id').size()
            self.df['device_transaction_frequency'] = self.df['device_id'].map(device_freq)

            self.df['user_transaction_velocity'] = self.df['user_transaction_frequency'] / self.df['purchase_delay']
            self.logging.info("Transaction frequency and velocity calculated successfully.")
        except Exception as e:
            self.logging.error("Error in calculating transaction frequency and velocity: %s", e)
            raise

    def normalize_and_scale(self):
        self.logging.info("Normalizing and scaling numerical features...")
        try:
            numerical_features = ['purchase_value', 'user_transaction_frequency', 'device_transaction_frequency', 
                                  'user_transaction_velocity', 'hour_of_day', 'day_of_week', 'purchase_delay', 
                                  'age', 'fraud_rate']  # Include fraud_rate here
            self.df[numerical_features] = self.scaler.fit_transform(self.df[numerical_features])
            self.logging.info("Numerical features normalized and scaled successfully.")
        except Exception as e:
            self.logging.error("Error in normalizing and scaling numerical features: %s", e)
            raise

    def encode_categorical_features(self):
        self.logging.info("Encoding categorical features...")
        try:
            categorical_features = ['source', 'browser', 'sex']
            self.df = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
            boolean_cols = self.df.select_dtypes(include='bool').columns
            self.df[boolean_cols] = self.df[boolean_cols].astype(float)
            self.logging.info("Categorical features encoded successfully.")
        except Exception as e:
            self.logging.error("Error in encoding categorical features: %s", e)
            raise

    def pipeline(self):
        self.logging.info("Starting the feature engineering pipeline...")
        try:
            self.calculate_fraud_rate()  # Add this line to the pipeline
            self.preprocess_datetime()
            self.calculate_transaction_frequency()
            self.encode_categorical_features()
            self.normalize_and_scale()
            
            cols_exclude = ['signup_time', 'purchase_time', 'ip_address', 'device_id', 'ip_int', 'country']
            self.df.drop(columns=cols_exclude, inplace=True)
            self.df.set_index('user_id', inplace=True)
            
            self.processed_df = self.df
            self.logging.info("Feature engineering pipeline executed successfully.")
        except Exception as e:
            self.logging.error("Error in the feature engineering pipeline: %s", e)
            raise

    def get_processed_data(self) -> pd.DataFrame:
        self.logging.info("Retrieving processed data...")
        if self.processed_df is None:
            self.logging.error("Data has not been processed. Run the pipeline() method first.")
            raise ValueError("Data has not been processed. Run the pipeline() method first.")
        self.logging.info("Processed data retrieved successfully.")
        return self.processed_df