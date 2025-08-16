import pandas as pd
import numpy as np

class LoadData:
    """
    A class for preprocessing a dataset, including loading, cleaning, and handling missing values.

    Attributes:
    ----------
    filepath : str
        The file path of the dataset.
    logger : logging.Logger
        The logger instance for logging actions and errors.
    data : pd.DataFrame, optional
        The dataset loaded from the file path.
    """

    def __init__(self, filepath, logger):
        """
        Initializes the DataPreprocessor with a dataset filepath and logger.

        Parameters:
        ----------
        filepath : str
            The path to the dataset file (CSV format).
        logger : logging.Logger
            A logger instance for logging information and errors.
        """
        self.filepath = filepath
        self.logger = logger
        self.data = None
    
    def load_dataset(self):
        """
        Loads the dataset from the specified filepath.

        Returns:
        -------
        pd.DataFrame
            The loaded dataset as a DataFrame.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.logger.info("Dataset loaded successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None  # Return None if there's an error loading the dataset
        