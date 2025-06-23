import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the DataAnalyzer with a CSV file
        
        Parameters:
        file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the CSV file into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Successfully loaded data from: {self.file_path}")
            print(f"📊 Dataset shape: {self.df.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*60)
        print("📋 BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📝 Column Information:")
        print("-" * 40)
        for col in self.df.columns:
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            print(f"{col:<20} | {str(dtype):<10} | {null_count:>5} nulls ({null_pct:>5.1f}%)")
    
    def data_preview(self, n_rows=5):
        """Show first and last few rows of the dataset"""
        print("\n" + "="*60)
        print(f"👀 DATA PREVIEW (First {n_rows} rows)")
        print("="*60)
        print(self.df.head(n_rows))
        
        print(f"\n👀 DATA PREVIEW (Last {n_rows} rows)")
        print("="*60)
        print(self.df.tail(n_rows))
    
    def missing_data_analysis(self):
        """Analyze missing data patterns"""
        print("\n" + "="*60)
        print("🔍 MISSING DATA ANALYSIS")
        print("="*60)
        
        # Calculate missing data
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Filter columns with missing data
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print("Columns with missing data:")
            print(missing_df.to_string(index=False))
            
            # Visualize missing data
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            missing_df.plot(x='Column', y='Missing_Count', kind='bar', ax=plt.gca())
            plt.title('Missing Data Count by Column')
            plt.xticks(rotation=45)
            plt.ylabel('Missing Count')
            
            plt.subplot(1, 2, 2)
            missing_df.plot(x='Column', y='Missing_Percentage', kind='bar', ax=plt.gca())
            plt.title('Missing Data Percentage by Column')
            plt.xticks(rotation=45)
            plt.ylabel('Missing Percentage (%)')
            
            plt.tight_layout()
            plt.show()
        else:
            print("✅ No missing data found in the dataset!")
    
    def data_types_analysis(self):
        """Analyze data types and suggest improvements"""
        print("\n" + "="*60)
        print("🔧 DATA TYPES ANALYSIS")
        print("="*60)
        
        # Get data types info
        dtype_info = self.df.dtypes.value_counts()
        print("Current data types distribution:")
        print(dtype_info)
        
        # Memory usage by data type
        memory_by_dtype = self.df.memory_usage(deep=True).groupby(self.df.dtypes).sum()
        print(f"\nMemory usage by data type:")
        print(memory_by_dtype)
        
        # Suggest optimizations
        print(f"\n💡 Memory optimization suggestions:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            
            if col_min >= 0 and col_max < 255:
                print(f"  - {col}: Consider using uint8 (0-255 range)")
            elif col_min >= -128 and col_max < 127:
                print(f"  - {col}: Consider using int8 (-128 to 127 range)")
            elif col_min >= -32768 and col_max < 32767:
                print(f"  - {col}: Consider using int16 (-32768 to 32767 range)")
    
    def numerical_analysis(self):
        """Analyze numerical columns"""
        print("\n" + "="*60)
        print("📊 NUMERICAL DATA ANALYSIS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("❌ No numerical columns found in the dataset")
            return
        
        print(f"Found {len(numeric_cols)} numerical columns: {list(numeric_cols)}")
        
        # Descriptive statistics
        print("\n📈 Descriptive Statistics:")
        print("-" * 40)
        desc_stats = self.df[numeric_cols].describe()
        print(desc_stats)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            print("\n🔗 Correlation Matrix:")
            print("-" * 40)
            correlation_matrix = self.df[numeric_cols].corr()
            print(correlation_matrix.round(3))
            
            # Visualize correlation matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Variables')
            plt.tight_layout()
            plt.show()
        
        # Distribution plots for each numerical column
        n_cols = len(numeric_cols)
        if n_cols <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
        else:
            n_rows = (n_cols + 3) // 4
            fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
            axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histogram
                axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def categorical_analysis(self):
        """Analyze categorical columns"""
        print("\n" + "="*60)
        print("📝 CATEGORICAL DATA ANALYSIS")
        print("="*60)
        
        # Identify categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            print("❌ No categorical columns found in the dataset")
            return
        
        print(f"Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            print(f"\n📊 Analysis for column: {col}")
            print("-" * 40)
            
            # Basic stats
            unique_count = self.df[col].nunique()
            print(f"Unique values: {unique_count}")
            
            if unique_count <= 20:  # Only show value counts for columns with reasonable number of unique values
                value_counts = self.df[col].value_counts()
                print(f"Value counts:")
                print(value_counts)
                
                # Visualize
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar')
                plt.title(f'Value Counts for {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Too many unique values ({unique_count}) to display. Consider grouping or sampling.")
    
    def outliers_analysis(self):
        """Detect and analyze outliers in numerical columns"""
        print("\n" + "="*60)
        print("🚨 OUTLIERS ANALYSIS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("❌ No numerical columns found for outlier analysis")
            return
        
        outliers_summary = []
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            outliers_summary.append({
                'Column': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percentage': outlier_pct,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
            
            print(f"{col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        # Create summary DataFrame
        outliers_df = pd.DataFrame(outliers_summary)
        
        # Visualize outliers
        n_cols = len(numeric_cols)
        if n_cols <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
        else:
            n_rows = (n_cols + 3) // 4
            fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
            axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Box plot
                axes[i].boxplot(self.df[col].dropna())
                axes[i].set_title(f'Box Plot of {col}')
                axes[i].set_ylabel(col)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def time_series_analysis(self):
        """Analyze time series data if datetime columns exist"""
        print("\n" + "="*60)
        print("⏰ TIME SERIES ANALYSIS")
        print("="*60)
        
        # Try to identify datetime columns
        datetime_cols = []
        for col in self.df.columns:
            try:
                pd.to_datetime(self.df[col].head(100))
                datetime_cols.append(col)
            except:
                continue
        
        if len(datetime_cols) == 0:
            print("❌ No datetime columns detected")
            return
        
        print(f"Detected datetime columns: {datetime_cols}")
        
        for col in datetime_cols:
            print(f"\n📅 Analysis for datetime column: {col}")
            print("-" * 40)
            
            # Convert to datetime
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            # Basic stats
            print(f"Date range: {self.df[col].min()} to {self.df[col].max()}")
            print(f"Total days: {(self.df[col].max() - self.df[col].min()).days}")
            
            # Extract time components
            self.df[f'{col}_year'] = self.df[col].dt.year
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df[f'{col}_weekday'] = self.df[col].dt.weekday
            
            # Visualize time patterns
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Yearly distribution
            self.df[f'{col}_year'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title(f'Distribution by Year - {col}')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Monthly distribution
            self.df[f'{col}_month'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1])
            axes[0,1].set_title(f'Distribution by Month - {col}')
            axes[0,1].set_xlabel('Month')
            
            # Daily distribution
            self.df[f'{col}_day'].value_counts().sort_index().plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title(f'Distribution by Day - {col}')
            axes[1,0].set_xlabel('Day')
            
            # Weekday distribution
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = self.df[f'{col}_weekday'].value_counts().sort_index()
            weekday_counts.index = [weekday_names[i] for i in weekday_counts.index]
            weekday_counts.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title(f'Distribution by Weekday - {col}')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("📋 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Basic information
        self.basic_info()
        
        # Data preview
        self.data_preview()
        
        # Missing data analysis
        self.missing_data_analysis()
        
        # Data types analysis
        self.data_types_analysis()
        
        # Numerical analysis
        self.numerical_analysis()
        
        # Categorical analysis
        self.categorical_analysis()
        
        # Outliers analysis
        self.outliers_analysis()
        
        # Time series analysis
        self.time_series_analysis()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)

def check_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> Dict:
    """
    Comprehensive data quality check function
    
    Args:
        df: DataFrame to check
        name: Name of the dataset for reporting
    
    Returns:
        Dictionary with data quality metrics
    """
    print(f"\n{'='*50}")
    print(f"DATA QUALITY CHECK: {name}")
    print(f"{'='*50}")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes)
    
    # Missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Count', ascending=False)
    
    print(f"\nMissing Values:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Unique values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nUnique values in categorical columns:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
    
    # Summary statistics for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\nNumerical columns summary:")
        print(df[numerical_cols].describe())
    
    # Return quality metrics
    quality_metrics = {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': missing_values.to_dict(),
        'duplicates': duplicates,
        'duplicate_percent': duplicates/len(df)*100,
        'categorical_columns': list(categorical_cols),
        'numerical_columns': list(numerical_cols)
    }
    
    return quality_metrics

def merge_datasets_by_id(
    datasets: Dict[str, pd.DataFrame], 
    merge_key: str = 'client_id',
    merge_type: str = 'left'
) -> pd.DataFrame:
    """
    Merge multiple datasets by a common ID column
    
    Args:
        datasets: Dictionary of DataFrames with dataset names as keys
        merge_key: Column name to merge on
        merge_type: Type of merge ('left', 'right', 'inner', 'outer')
    
    Returns:
        Merged DataFrame
    """
    print(f"\n{'='*50}")
    print(f"MERGING DATASETS BY {merge_key.upper()}")
    print(f"{'='*50}")
    
    # Check if merge_key exists in all datasets
    missing_key_datasets = []
    for name, df in datasets.items():
        if merge_key not in df.columns:
            missing_key_datasets.append(name)
    
    if missing_key_datasets:
        raise ValueError(f"Merge key '{merge_key}' not found in datasets: {missing_key_datasets}")
    
    # Start with the first dataset
    dataset_names = list(datasets.keys())
    merged_df = datasets[dataset_names[0]].copy()
    print(f"Starting with: {dataset_names[0]} (shape: {merged_df.shape})")
    
    # Merge with remaining datasets
    for i, name in enumerate(dataset_names[1:], 1):
        df_to_merge = datasets[name]
        print(f"\nMerging with: {name} (shape: {df_to_merge.shape})")
        
        # Check for duplicate columns (excluding merge_key)
        common_cols = set(merged_df.columns) & set(df_to_merge.columns) - {merge_key}
        if common_cols:
            print(f"  Warning: Common columns found: {list(common_cols)}")
            # Rename columns in the dataset being merged to avoid conflicts
            rename_dict = {col: f"{name}_{col}" for col in common_cols}
            df_to_merge = df_to_merge.rename(columns=rename_dict)
            print(f"  Renamed columns: {rename_dict}")
        
        # Perform merge
        before_shape = merged_df.shape
        merged_df = merged_df.merge(
            df_to_merge, 
            on=merge_key, 
            how=merge_type,
            indicator=True
        )
        after_shape = merged_df.shape
        
        print(f"  Merge result: {before_shape} -> {after_shape}")
        
        # Check merge indicators
        if '_merge' in merged_df.columns:
            merge_stats = merged_df['_merge'].value_counts()
            print(f"  Merge statistics:")
            for indicator, count in merge_stats.items():
                print(f"    {indicator}: {count}")
            merged_df = merged_df.drop('_merge', axis=1)
    
    print(f"\nFinal merged dataset shape: {merged_df.shape}")
    print(f"Final columns: {list(merged_df.columns)}")
    
    return merged_df

def load_and_prepare_data(data_folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets and prepare them for merging
    
    Args:
        data_folder_path: Path to the data folder
    
    Returns:
        Dictionary of loaded DataFrames
    """
    # File names
    file_names = {
        'messages': 'apparel-messages.csv',
        'purchases': 'apparel-purchases.csv', 
        'target': 'apparel-target_binary.csv',
        'campaign': 'full_campaign_daily_event.csv',
        'campaign_channel': 'full_campaign_daily_event_channel.csv'
    }
    
    datasets = {}
    
    print("Loading datasets...")
    for name, filename in file_names.items():
        try:
            file_path = f"{data_folder_path}/{filename}"
            datasets[name] = pd.read_csv(file_path)
            print(f"✓ Loaded {name}: {datasets[name].shape}")
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
    
    return datasets

def analyze_merge_compatibility(datasets: Dict[str, pd.DataFrame], merge_key: str = 'client_id') -> Dict:
    """
    Analyze compatibility of datasets for merging
    
    Args:
        datasets: Dictionary of DataFrames
        merge_key: Column to merge on
    
    Returns:
        Dictionary with compatibility analysis
    """
    print(f"\n{'='*50}")
    print(f"MERGE COMPATIBILITY ANALYSIS")
    print(f"{'='*50}")
    
    analysis = {}
    
    # Check merge key presence
    key_presence = {}
    for name, df in datasets.items():
        key_presence[name] = merge_key in df.columns
    
    print(f"Merge key '{merge_key}' presence:")
    for name, present in key_presence.items():
        status = "✓" if present else "✗"
        print(f"  {status} {name}")
    
    # Analyze unique values in merge key
    unique_counts = {}
    for name, df in datasets.items():
        if merge_key in df.columns:
            unique_counts[name] = df[merge_key].nunique()
            print(f"  {name}: {unique_counts[name]} unique {merge_key}s")
    
    # Check for overlapping IDs
    if len([k for k, v in key_presence.items() if v]) >= 2:
        datasets_with_key = {name: df for name, df in datasets.items() if key_presence[name]}
        dataset_names = list(datasets_with_key.keys())
        
        print(f"\nOverlap analysis:")
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                set1 = set(datasets_with_key[name1][merge_key])
                set2 = set(datasets_with_key[name2][merge_key])
                overlap = len(set1 & set2)
                union = len(set1 | set2)
                overlap_percent = (overlap / union) * 100 if union > 0 else 0
                
                print(f"  {name1} ∩ {name2}: {overlap} common IDs ({overlap_percent:.1f}% of union)")
    
    analysis = {
        'key_presence': key_presence,
        'unique_counts': unique_counts
    }
    
    return analysis

# Example usage functions
def example_data_check_and_merge():
    """
    Example of how to use the data checking and merging functions
    """
    # Load data
    data_folder_path = "/home/pavel/Data/filtered_data/"
    datasets = load_and_prepare_data(data_folder_path)
    
    # Check data quality for each dataset
    quality_reports = {}
    for name, df in datasets.items():
        quality_reports[name] = check_data_quality(df, name)
    
    # Analyze merge compatibility
    compatibility = analyze_merge_compatibility(datasets, 'client_id')
    
    # Merge datasets
    merged_data = merge_datasets_by_id(datasets, 'client_id', 'left')
    
    # Check final merged dataset
    final_quality = check_data_quality(merged_data, "Merged Dataset")
    
    return merged_data, quality_reports, compatibility

def main():
    """Main function to run the analysis"""
    print("🐼 PANDAS DATA ANALYSIS TOOL")
    print("="*60)
    
    # List available CSV files
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in the current directory")
        return
    
    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    # Let user choose a file
    try:
        choice = int(input(f"\nEnter the number of the file to analyze (1-{len(csv_files)}): ")) - 1
        if 0 <= choice < len(csv_files):
            selected_file = csv_files[choice]
        else:
            print("Invalid choice. Using the first file.")
            selected_file = csv_files[0]
    except:
        print("Invalid input. Using the first file.")
        selected_file = csv_files[0]
    
    print(f"\n🎯 Analyzing: {selected_file}")
    
    # Create analyzer and run analysis
    analyzer = DataAnalyzer(selected_file)
    analyzer.generate_report()

if __name__ == "__main__":
    # Run example
    merged_data, quality_reports, compatibility = example_data_check_and_merge()
    main() 