import pandas as pd
import geopandas as gpd
import plotly.express as px
import logging
import pycountry

class GeolocationAnalyzer:
    def __init__(self, fraud_df: pd.DataFrame, world_gdf: gpd.GeoDataFrame, logger: logging.Logger):
        """
        Initializes the GeolocationAnalyzer class with DataFrames for fraud and world data and a logger.
        Automatically standardizes country names and merges data.

        Args:
            fraud_df (pd.DataFrame): DataFrame containing fraud transaction data.
            world_gdf (gpd.GeoDataFrame): GeoDataFrame containing world geometries.
            logger (logging.Logger): Logger instance for logging information.
        """
        self.fraud_df = fraud_df
        self.world_gdf = world_gdf
        self.logger = logger
        self.fraud_rate_df = None
        self.transaction_volume_df = None
        self.world_fraud_map = None

        # Standardize country names and merge data during initialization
        self.standardize_country_names()
        self.calculate_fraud_rate()
        self.calculate_transaction_volume()
        self.merge_data()

    def standardize_country_names(self):
        """
        Standardizes the country names in the world GeoDataFrame to match those in the fraud dataset.
        """
        self.logger.info("Standardizing country names in the world GeoDataFrame.")
        def map_country_name(name):
            try:
                return pycountry.countries.lookup(name).name
            except LookupError:
                return name  # Return the original name if no match is found

        self.world_gdf['standardized_name'] = self.world_gdf['NAME'].apply(map_country_name)
        self.logger.info("Country names standardized successfully.")

    def calculate_fraud_rate(self):
        """
        Calculates the fraud rate by country and stores it in the fraud_rate_df attribute.
        The fraud rate is computed as the number of fraudulent transactions divided by the total transactions for each country.

        Returns:
            pd.DataFrame: A DataFrame with fraud rates by country.
        """
        self.logger.info("Calculating fraud rates by country.")
        total_by_country = self.fraud_df['country'].value_counts()
        fraud_by_country = self.fraud_df[self.fraud_df['class'] == 1]['country'].value_counts()

        # Create a DataFrame for fraud rates
        fraud_rate_data = {
            'country': total_by_country.index,
            'fraud_rate': (fraud_by_country / total_by_country).fillna(0).values
        }
        self.fraud_rate_df = pd.DataFrame(fraud_rate_data)
        
        self.logger.info("Fraud rates calculated successfully.")
        return self.fraud_rate_df

    def calculate_transaction_volume(self):
        """
        Calculates the total transaction volume by country and stores it in the transaction_volume_df attribute.

        Returns:
            pd.DataFrame: A DataFrame with transaction volume by country.
        """
        self.logger.info("Calculating transaction volumes by country.")
        transaction_volume_by_country = self.fraud_df['country'].value_counts().reset_index()
        transaction_volume_by_country.columns = ['country', 'transaction_volume']
        self.transaction_volume_df = transaction_volume_by_country
        
        self.logger.info("Transaction volumes calculated successfully.")
        return self.transaction_volume_df

    def merge_data(self):
        """
        Merges the fraud rates and transaction volumes with the world GeoDataFrame.
        Updates the world_fraud_map attribute with the merged data.
        Missing values are filled with zeros.
        """
        self.logger.info("Merging data with world GeoDataFrame.")
        # Merge fraud rates and transaction volumes with the world GeoDataFrame
        self.world_fraud_map = self.world_gdf.merge(self.fraud_rate_df, how='left', left_on='standardized_name', right_on='country')
        self.world_fraud_map = self.world_fraud_map.merge(self.transaction_volume_df, how='left', on='country')
        
        # Fill missing values with 0
        self.world_fraud_map['fraud_rate'] = self.world_fraud_map['fraud_rate'].fillna(0)
        self.world_fraud_map['transaction_volume'] = self.world_fraud_map['transaction_volume'].fillna(0)
        
        self.logger.info("Data merged successfully.")

    def plot_fraud_rate_map(self):
        """
        Creates and displays an interactive map of fraud rates by country using Plotly.
        """
        self.logger.info("Plotting fraud rate map.")
        fig = px.choropleth(
            self.world_fraud_map,
            geojson=self.world_fraud_map.geometry,
            locations=self.world_fraud_map.index,
            color='fraud_rate',
            hover_name='standardized_name',
            hover_data=['fraud_rate', 'transaction_volume'],
            title='Fraud Rate by Country',
            color_continuous_scale='Reds',
            projection='natural earth',
        )
        fig.update_geos(fitbounds="locations", visible=False)
        # fig.show()
        self.logger.info("Fraud rate map plotted successfully.")
        return fig

    def plot_transaction_volume_map(self):
        """
        Creates and displays an interactive map of transaction volumes by country using Plotly.
        """
        self.logger.info("Plotting transaction volume map.")
        fig = px.choropleth(
            self.world_fraud_map,
            geojson=self.world_fraud_map.geometry,
            locations=self.world_fraud_map.index,
            color='transaction_volume',
            hover_name='standardized_name',
            hover_data=['transaction_volume', 'fraud_rate'],
            title='Transaction Volume by Country',
            color_continuous_scale='Blues',
            projection='natural earth',
        )
        fig.update_geos(fitbounds="locations", visible=False)
        # fig.show()
        self.logger.info("Transaction volume map plotted successfully.")
        return fig

    def analyze(self):
        """
        Executes the full geolocation analysis, including plotting the results.
        """
        self.logger.info("Starting full geolocation analysis.")
        self.plot_fraud_rate_map()
        self.plot_transaction_volume_map()
        self.logger.info("Geolocation analysis completed successfully.")