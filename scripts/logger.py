import logging
import os

class SetupLogger:
    """
    A class to set up logging for the application.

    Attributes:
    ----------
    log_file : str
        The file where logs will be saved.
    log_level : logging level
        The level of logging, default is logging.INFO.
    """

    def __init__(self, log_file='logs/app.log', log_level=logging.INFO):
        """
        Initializes the logger with a specified log file and level.

        Parameters:
        ----------
        log_file : str
            The name of the file to save logs.
        log_level : logging level, optional
            The logging level (default is logging.INFO).
        """
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create a file handler for logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Define the logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Returns the configured logger.

        Returns:
        -------
        logger : logging.Logger
            The configured logger instance.
        """
        return self.logger