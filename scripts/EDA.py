import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CreditScoringModelEDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA class with the DataFrame.
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.df = df
    # Load data function
    def load_data(self,df):

        try:
            df = pd.read_csv(df, index_col=0)
            print(f"Data successfully loaded from {df}")
            print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {df}")
            return pd.DataFrame()  # Return empty DataFrame in case of failure

    def data_overview(self,df):
        num_rows = df.shape[0]
        num_columns = df.shape[1]
        data_types = df.dtypes

        print(f"Number of rows:{num_rows}")
        print(f"Number of columns:{num_columns}")
        print(f"Data types of each column:\n{data_types}")
        
    def summarize_statics(self,df):
        # Select numerical columns only
        numerical_columns=['Amount','Value','CountryCode','PricingStrategy','FraudResult']
        
        # Initialize a list to hold summary statistics for each column
        summary_list = []
    
        for col in numerical_columns:
            summary_stats = {
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Mode': df[col].mode().iloc[0],  # Taking the first mode in case of multiple modes
                'Standard Deviation': df[col].std(),
                'Variance': df[col].var(),
                'Range': df[col].max() - df[col].min(),
                'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis()
            }
            
            # Append the summary statistics for the current column to the list
            summary_list.append(summary_stats)
        
        # Convert summary stats list to DataFrame with appropriate index
        summary_df = pd.DataFrame(summary_list, index=numerical_columns)
        
        return summary_df
    
    def plot_numerical_distribution(self, cols):

        # Select numeric columns
        n_cols = len(cols)
        # Automatically determine grid size (square root method)
        n_rows = math.ceil(n_cols**0.5)
        n_cols = math.ceil(n_cols / n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
        axes = axes.flatten()
        for i, col in enumerate(cols):
            sns.histplot(self.df[col], bins=15, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='dashed', linewidth=1)
            axes[i].axvline(self.df[col].median(), color='green', linestyle='dashed', linewidth=1)
            axes[i].legend({'Mean': self.df[col].mean(), 'Median': self.df[col].median()})
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()
        
    def plot_categorical_distributions(self,df):
        # Select categorical columns only
        categorical_columns=['ProviderId','ProductId','ProductCategory','ChannelId','PricingStrategy','FraudResult']
        
        # Create bar plots for each categorical feature
        for col in categorical_columns:
            plt.figure(figsize=(7, 5))
            
            # Plot a bar chart for the frequency of each category
            sns.countplot(x=df[col], order=df[col].value_counts().index, hue=df[col], palette="Set2", legend=False)

            # sns.countplot(x=df[col], order=df[col].value_counts().index, palette="Set2")
            
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

    
    def check_missing_value(self,df):
        #check missing value
        print(self.df.isnull().sum())
        if(df.isnull().sum().sum()):
            print(f"The number of missing values:{df.isnull().sum().sum()}")
            # print("There no null value in the data")
        else:
            # print(f"The number of null values:{df.isnull().sum().sum()}")
            print("There is no missing value in the data")
    def detect_outliers(self, cols):
        """
        Function to plot boxplots for numerical features to detect outliers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be analyzed.
        numerical_cols : list
            List of numerical columns to plot.
        """
        # num_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols, 1):
            plt.subplot(3, 3, i)
            sns.boxplot(y=self.df[col], color='orange')
            plt.title(f'Boxplot of {col}', fontsize=12)
        plt.tight_layout()
        plt.show()
        
    def correlation_analysis(self):
        """Generate and visualize the correlation matrix."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.show()        # Select numerical columns only
        numerical_columns = ['Amount', 'Value']
        
        # Create box plots for each numerical feature to detect outliers
        for col in numerical_columns:
            fig = plt.figure(figsize=(10, 7))
            
            # Create the box plot
            box = plt.boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='lightgreen'), 
                            medianprops=dict(color='brown'), 
                            whiskerprops=dict(color='black', linewidth=1.5), 
                            capprops=dict(color='black', linewidth=1.5))
            
            # Set title and labels
            plt.title(f'Boxplot of {col} (Outlier Detection)', fontsize=16)
            plt.xlabel(col, fontsize=14)
            
            # Set y-axis limits
            if(col=='Value'):
                plt.ylim(df[col].min(), df[col].max() * 0.002)  # Adjust as necessary to see the box clearly
            elif(col=='Amount'):
                plt.ylim(df[col].min()* 0.01,  df[col].max()* 0.002)  # Adjust as necessary to see the box clearly
            
            # Show the plot
            plt.grid()
            plt.tight_layout()
            plt.show()