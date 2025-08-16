
import shap
import joblib
from lime import lime_tabular
import matplotlib.pyplot as plt

shap.initjs()

class ModelExplainer:
    """
    A class for explaining machine learning models using SHAP and LIME.

    Attributes:
    -----------
    model : object
        The trained machine learning model.
    X_test : DataFrame
        The test dataset to explain.

    Methods:
    ----------
    __init__(self, model_path, X_test):
        Initializes the class with the model and test dataset paths.
    
    explain_with_shap(self, instance_idx=0):
        Generates SHAP Summary Plot, Force Plot, and Dependence Plot for the given model.
    
    explain_with_lime(self, instance_idx=0):
        Generates LIME Feature Importance Plot for a single instance of the dataset.
    
    explain_model(self, instance_idx=0):
        Runs both SHAP and LIME explainability functions on the model and dataset.
    """

    def __init__(self, model_path, X_test):
        """
        Initialize the ModelExplainer class with the model and test dataset.

        Parameters:
        -----------
        model_path : str
            The path to the saved model file (e.g., .pkl).
        X_test : DataFrame
            The test dataset (in pandas DataFrame format).
        """
        self.model = joblib.load(model_path)  # Load the saved model
        self.X_test = X_test  # Load the test dataset

        # If the model is part of a scikit-learn pipeline, extract the last model from the pipeline
        if hasattr(self.model, 'steps'):
            self.model = self.model.steps[-1][1]  # Extract the model from the last step of the pipeline

    def explain_with_shap(self, instance_idx=0):
        """
        Generate SHAP Summary Plot, Force Plot, and Dependence Plot for the model.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with SHAP Force Plot.
        """
        print("Generating SHAP explanations...")

        # Ensure model is correctly extracted
        model = self.model  # This should already be the final model if initialized correctly

        explainer = shap.TreeExplainer(model, self.X_test)  # General Explainer for more models
        shap_values = explainer.shap_values(self.X_test)

        # Print type and shape for debugging
        print(f"Type of SHAP values: {type(shap_values)}")
        print(f"Shape of SHAP values: {shap_values.shape}")

        # SHAP Summary Plot: Overview of important features
        plt.figure(figsize=(15, 4))
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title('SHAP Summary Plot')
        plt.show()

        # Plot SHAP force plot for the selected instance from the test data
        # Use `shap.plots.force` correctly
        shap.plots.force(explainer.expected_value, shap_values[instance_idx],feature_names=self.X_test.columns, matplotlib=True)
       
        
        # SHAP Dependence Plot: Relationship between feature and model output
        shap.dependence_plot(self.X_test.columns[0], shap_values, self.X_test, show=False)
        plt.title(f'SHAP Dependence Plot for Feature: {self.X_test.columns[0]}')
        plt.show()


    def explain_with_lime(self, instance_idx=0):
        """
        Generate LIME Feature Importance Plot for a single instance of the dataset.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with LIME.
        """
        print("Generating LIME explanations...")

        # Create LIME explainer
        explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=self.X_test.values, 
            feature_names=self.X_test.columns, 
            mode='classification'
        )

        # Select a single instance (default: first instance)
        instance = self.X_test.iloc[instance_idx].values.flatten()  # Flatten to ensure it's 1D
        print(f"Instance shape for LIME: {instance.shape}")

        explanation = explainer_lime.explain_instance(instance, self.model.predict_proba)

        # Display LIME Feature Importance Plot
        explanation.as_pyplot_figure()
        plt.title(f'LIME Feature Importance for Instance {instance_idx}')
        plt.show()


    def explain_model(self, instance_idx=0):
        """
        Run both SHAP and LIME explainability methods for the model.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with LIME and SHAP.
        """
        # Explain the model with SHAP and LIME for the specified instance
        self.explain_with_shap(instance_idx)
        self.explain_with_lime(instance_idx)
