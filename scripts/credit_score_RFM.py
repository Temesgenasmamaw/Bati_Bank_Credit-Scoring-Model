import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import pandas as pd

class CreditScoreRFM:

    def __init__(self, rfm_data):

        self.rfm_data = rfm_data

    def calculate_rfm(self):

        # Convert 'TransactionStartTime' to datetime and make it timezone-aware (UTC)
        self.rfm_data['TransactionStartTime'] = pd.to_datetime(self.rfm_data['TransactionStartTime'])

        # Set the end date to the current date and make it timezone-aware (UTC)
        end_date = pd.Timestamp.utcnow()

        # Calculate Recency, Frequency, and Monetary values
        self.rfm_data['Last_Access_Date'] = self.rfm_data.groupby('CustomerId')['TransactionStartTime'].transform('max')
        self.rfm_data['Recency'] = (end_date - self.rfm_data['Last_Access_Date']).dt.days
        self.rfm_data['Frequency'] = self.rfm_data.groupby('CustomerId')['TransactionId'].transform('count')

        if 'Amount' in self.rfm_data.columns:
            self.rfm_data['Monetary'] = self.rfm_data.groupby('CustomerId')['Amount'].transform('sum')
        else:
            # Handle missing Amount column (e.g., set to 1 for each transaction)
            self.rfm_data['Monetary'] = 1

        # Remove duplicates to create a summary DataFrame for scoring
        rfm_data = self.rfm_data[['CustomerId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()

        return rfm_data
    
    def calculate_rfm_scores(self, rfm_data):

        # Quantile-based scoring
        rfm_data['r_quartile'] = pd.qcut(rfm_data['Recency'], 4, labels=['4', '3', '2', '1'])  # Lower recency is better
        rfm_data['f_quartile'] = pd.qcut(rfm_data['Frequency'], 4, labels=['1', '2', '3', '4'])  # Higher frequency is better
        rfm_data['m_quartile'] = pd.qcut(rfm_data['Monetary'], 4, labels=['1', '2', '3', '4'])  # Higher monetary is better

        # Calculate overall RFM Score
        rfm_data['RFM_Score'] = (rfm_data['r_quartile'].astype(int) * 0.1 +
                                  rfm_data['f_quartile'].astype(int) * 0.45 +
                                  rfm_data['m_quartile'].astype(int) * 0.45)

        return rfm_data
    
    def assign_label(self, rfm_data):

        high_threshold = rfm_data['RFM_Score'].quantile(0.75)  # Change to .75 to include moderate users
        low_threshold = rfm_data['RFM_Score'].quantile(0.5)  # Change to .25 to include moderate users
        rfm_data['Risk_Label'] = rfm_data['RFM_Score'].apply(lambda x: 'Good' if x >= low_threshold else 'Bad')
        return rfm_data
    
   
    def calculate_counts(self, data):
        # Group by 'RFM_bin' and calculate good and bad counts based on 'Risk_Label'
        good_count = data.groupby('RFM_bin')['Risk_Label'].apply(lambda x: (x == 'Good').sum())
        bad_count = data.groupby('RFM_bin')['Risk_Label'].apply(lambda x: (x == 'Bad').sum())
        
        return good_count, bad_count

    def calculate_woe(self, good_count, bad_count):
        total_good = good_count.sum()
        total_bad = bad_count.sum()

        # Add epsilon (small value) to avoid log(0) or division by zero
        epsilon = 1e-10
        
        good_rate = good_count / (total_good + epsilon)
        bad_rate = bad_count / (total_bad + epsilon)

        # Calculate WoE and handle cases where bad_count is zero
        woe = np.log((good_rate + epsilon) / (bad_rate + epsilon))
        iv = ((good_rate - bad_rate) * woe).sum()  # Sum over all bins to get total IV
        
        return woe, iv


        
        




    def plot_pairplot(self):
        """
        Creates a pair plot to visualize relationships between Recency, Frequency, and Monetary.
        """
        sns.pairplot(self.rfm_data[['Recency', 'Frequency', 'Monetary']])
        plt.suptitle('Pair Plot of rfm Variables', y=1.02)
        plt.show()

    def plot_heatmap(self):
        """
        Creates a heatmap to visualize correlations between rfm variables.
        """
        corr = self.rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of rfm Variables')
        plt.show()

    def plot_histograms(self):
        """
        Plots histograms for Recency, Frequency, and Monetary.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        self.rfm_data['Recency'].hist(bins=20, ax=axes[0])
        axes[0].set_title('Recency Distribution')
        
        self.rfm_data['Frequency'].hist(bins=20, ax=axes[1])
        axes[1].set_title('Frequency Distribution')
        
        self.rfm_data['Monetary'].hist(bins=20, ax=axes[2])
        axes[2].set_title('Monetary Distribution')

        plt.tight_layout()
        plt.show()
