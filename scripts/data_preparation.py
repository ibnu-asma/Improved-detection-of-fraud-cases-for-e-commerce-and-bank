import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreparation:
    """
    A class for preprocessing datasets, including feature and target separation
    and splitting the data into training and testing sets.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset loaded as a DataFrame.
    target_column : str
        The name of the target column (e.g., 'Class' or 'class').
    X_train : pd.DataFrame, optional
        Training features.
    X_test : pd.DataFrame, optional
        Testing features.
    y_train : pd.Series, optional
        Training target.
    y_test : pd.Series, optional
        Testing target.
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Initializes the DataPreprocessor class with a DataFrame and the target column name.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the dataset.
        target_column : str
            The name of the target column ('Class' or 'class').
        """
        self.df = df
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def separate_features_and_target(self):
        """
        Separates the features and target from the DataFrame.

        Returns
        -------
        X : pd.DataFrame
            Features DataFrame.
        y : pd.Series
            Target Series.
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to include in the test split (default is 0.2).
        random_state : int, optional
            Seed used by the random number generator (default is 42).
        """
        X, y = self.separate_features_and_target()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print("Data split into training and testing sets successfully.")

    def get_train_test_data(self):
        """
        Retrieves the train and test datasets.

        Returns
        -------
        X_train : pd.DataFrame
            Training features.
        X_test : pd.DataFrame
            Testing features.
        y_train : pd.Series
            Training target.
        y_test : pd.Series
            Testing target.

        Raises
        ------
        ValueError
            If the train_test_split method has not been called.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data has not been split. Run the train_test_split() method first.")
        return self.X_train, self.X_test, self.y_train, self.y_test