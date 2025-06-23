import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None, keep: str = 'first', inplace: bool = False) -> pd.DataFrame:
    """
    Remove explicit duplicates from DataFrame
    
    Args:
        df: DataFrame to remove duplicates from
        subset: List of column names to consider for duplicates (None = all columns)
        keep: Which duplicates to keep ('first', 'last', False to drop all)
        inplace: Whether to modify the original DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    print(f"\n{'='*40}")
    print(f"REMOVING DUPLICATES")
    print(f"{'='*40}")
    
    # Count duplicates before removal
    duplicates_before = df.duplicated(subset=subset).sum()
    total_rows_before = len(df)
    
    print(f"Total rows before: {total_rows_before}")
    print(f"Duplicate rows found: {duplicates_before}")
    
    if duplicates_before == 0:
        print("No duplicates found!")
        return df if not inplace else None
    
    # Remove duplicates
    if inplace:
        df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        result_df = df
    else:
        result_df = df.drop_duplicates(subset=subset, keep=keep)
    
    # Count rows after removal
    total_rows_after = len(result_df)
    rows_removed = total_rows_before - total_rows_after
    
    print(f"Rows removed: {rows_removed}")
    print(f"Total rows after: {total_rows_after}")
    print(f"Reduction: {rows_removed/total_rows_before*100:.2f}%")
    
    if subset:
        print(f"Duplicates checked based on columns: {subset}")
    else:
        print("Duplicates checked based on all columns")
    
    return result_df

def quick_analysis(file_path):
    """
    Perform a quick analysis of a CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    """
    print(f"🐼 Analyzing: {file_path}")
    print("="*50)
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully!")
        print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Basic info
    print(f"\n📋 COLUMN INFORMATION:")
    print("-" * 40)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"{col:<20} | {str(dtype):<10} | {null_count:>5} nulls ({null_pct:>5.1f}%)")
    
    # Data preview
    print(f"\n👀 FIRST 5 ROWS:")
    print("-" * 40)
    print(df.head())
    
    # Missing data summary
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n🔍 MISSING DATA:")
        print("-" * 40)
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': (missing_data / len(df)) * 100
        }).sort_values('Missing_Percentage', ascending=False)
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        print(missing_df.to_string(index=False))
    else:
        print(f"\n✅ No missing data found!")
    
    # Numerical analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📊 NUMERICAL COLUMNS ({len(numeric_cols)}):")
        print("-" * 40)
        print(df[numeric_cols].describe())
        
        # Correlation matrix for multiple numerical columns
        if len(numeric_cols) > 1:
            print(f"\n🔗 CORRELATION MATRIX:")
            print("-" * 40)
            corr_matrix = df[numeric_cols].corr()
            print(corr_matrix.round(3))
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
        
        # Distribution plots
        n_cols = min(len(numeric_cols), 4)
        if n_cols > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:4]):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols[:4]), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\n📝 CATEGORICAL COLUMNS ({len(categorical_cols)}):")
        print("-" * 40)
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"\n{col}:")
            print(f"  Unique values: {unique_count}")
            
            if unique_count <= 10:
                value_counts = df[col].value_counts()
                print(f"  Value counts:")
                for val, count in value_counts.items():
                    print(f"    {val}: {count}")
                
                # Plot for categorical data with few unique values
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar')
                plt.title(f'Value Counts for {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print(f"  Too many unique values ({unique_count}) to display")
    
    # Outliers detection for numerical columns
    if len(numeric_cols) > 0:
        print(f"\n🚨 OUTLIERS ANALYSIS:")
        print("-" * 40)
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            print(f"{col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
    
    print(f"\n✅ Analysis complete!")

def main():
    """Main function"""
    print("🐼 QUICK PANDAS DATA ANALYSIS")
    print("="*50)
    
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
    
    # Run analysis
    quick_analysis(selected_file)

def check_data(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Simple data quality check function
    
    Args:
        df: DataFrame to check
        name: Name of the dataset for reporting
    """
    print(f"\n{'='*40}")
    print(f"DATA CHECK: {name}")
    print(f"{'='*40}")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes)
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print(f"\nMissing Values: None")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Unique values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nUnique values in categorical columns:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")

def merge_data_by_id(
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
    print(f"\n{'='*40}")
    print(f"MERGING DATASETS BY {merge_key.upper()}")
    print(f"{'='*40}")
    
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
    for name in dataset_names[1:]:
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

def load_datasets(data_folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets from the specified folder
    
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

def clean_and_prepare_data(data_folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load datasets, remove duplicates, and prepare for analysis
    
    Args:
        data_folder_path: Path to the data folder
    
    Returns:
        Dictionary of cleaned DataFrames
    """
    print("Loading and cleaning datasets...")
    
    # Load datasets
    datasets = load_datasets(data_folder_path)
    
    # Remove duplicates from each dataset
    cleaned_datasets = {}
    for name, df in datasets.items():
        print(f"\nProcessing {name}...")
        cleaned_df = remove_duplicates(df.copy(), keep='first')
        cleaned_datasets[name] = cleaned_df
    
    return cleaned_datasets

# Example usage
if __name__ == "__main__":
    # Example usage
    data_folder_path = "/home/pavel/Data/filtered_data/"
    
    # Load and clean datasets
    cleaned_datasets = clean_and_prepare_data(data_folder_path)
    
    # Check each cleaned dataset
    for name, df in cleaned_datasets.items():
        check_data(df, f"{name} (cleaned)")
    
    # Merge cleaned datasets
    merged_data = merge_data_by_id(cleaned_datasets, 'client_id', 'left')
    
    # Check final merged dataset
    check_data(merged_data, "Merged Dataset") 