import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def exploratory_data_analysis(df: pd.DataFrame, name: str = "DataFrame", save_plots: bool = False) -> Dict:
    """
    Comprehensive Exploratory Data Analysis
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset for reporting
        save_plots: Whether to save plots to files
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"EXPLORATORY DATA ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types overview
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    # Missing values analysis
    missing_analysis = analyze_missing_values(df)
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\nNumerical Columns Analysis:")
        numerical_analysis = analyze_numerical_columns(df[numerical_cols])
        plot_numerical_distributions(df[numerical_cols], name, save_plots)
        plot_correlation_matrix(df[numerical_cols], name, save_plots)
    else:
        numerical_analysis = {}
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical Columns Analysis:")
        categorical_analysis = analyze_categorical_columns(df[categorical_cols])
        plot_categorical_distributions(df[categorical_cols], name, save_plots)
    else:
        categorical_analysis = {}
    
    # Time series analysis (if date columns exist)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) == 0:
        # Try to convert object columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col].head(100))
                date_cols = [col]
                break
            except:
                continue
    
    if len(date_cols) > 0:
        print(f"\nTime Series Analysis:")
        time_analysis = analyze_time_series(df, date_cols[0])
        plot_time_series(df, date_cols[0], name, save_plots)
    else:
        time_analysis = {}
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(df.describe())
    
    # Return comprehensive analysis results
    analysis_results = {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_analysis': missing_analysis,
        'numerical_analysis': numerical_analysis,
        'categorical_analysis': categorical_analysis,
        'time_analysis': time_analysis,
        'data_types': df.dtypes.to_dict()
    }
    
    return analysis_results

def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Analyze missing values in the dataset"""
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Count', ascending=False)
    
    print(f"Missing Values Analysis:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Plot missing values
    if missing_values.sum() > 0:
        plt.figure(figsize=(12, 6))
        missing_df[missing_df['Missing_Count'] > 0]['Missing_Percent'].plot(kind='bar')
        plt.title('Missing Values Percentage by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return {
        'missing_counts': missing_values.to_dict(),
        'missing_percentages': missing_percent.to_dict(),
        'total_missing': missing_values.sum()
    }

def analyze_numerical_columns(df: pd.DataFrame) -> Dict:
    """Analyze numerical columns"""
    analysis = {}
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            col_analysis = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'unique_values': df[col].nunique(),
                'missing_values': df[col].isnull().sum()
            }
            analysis[col] = col_analysis
            
            print(f"\n{col}:")
            print(f"  Mean: {col_analysis['mean']:.2f}")
            print(f"  Median: {col_analysis['median']:.2f}")
            print(f"  Std: {col_analysis['std']:.2f}")
            print(f"  Range: [{col_analysis['min']:.2f}, {col_analysis['max']:.2f}]")
            print(f"  Skewness: {col_analysis['skewness']:.2f}")
            print(f"  Unique values: {col_analysis['unique_values']}")
    
    return analysis

def analyze_categorical_columns(df: pd.DataFrame) -> Dict:
    """Analyze categorical columns"""
    analysis = {}
    
    for col in df.columns:
        col_analysis = {
            'unique_values': df[col].nunique(),
            'missing_values': df[col].isnull().sum(),
            'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'most_common_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
        analysis[col] = col_analysis
        
        print(f"\n{col}:")
        print(f"  Unique values: {col_analysis['unique_values']}")
        print(f"  Missing values: {col_analysis['missing_values']}")
        print(f"  Most common: {col_analysis['most_common']} ({col_analysis['most_common_count']} times)")
        
        # Show top 5 values if not too many unique values
        if col_analysis['unique_values'] <= 20:
            print(f"  Value counts:")
            print(df[col].value_counts().head())
    
    return analysis

def analyze_time_series(df: pd.DataFrame, date_col: str) -> Dict:
    """Analyze time series data"""
    try:
        # Convert to datetime if not already
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        analysis = {
            'start_date': df[date_col].min(),
            'end_date': df[date_col].max(),
            'date_range_days': (df[date_col].max() - df[date_col].min()).days,
            'missing_dates': df[date_col].isnull().sum()
        }
        
        print(f"Time Range: {analysis['start_date']} to {analysis['end_date']}")
        print(f"Total days: {analysis['date_range_days']}")
        print(f"Missing dates: {analysis['missing_dates']}")
        
        # Daily counts
        daily_counts = df[date_col].value_counts().sort_index()
        analysis['daily_counts'] = daily_counts
        analysis['avg_daily_records'] = daily_counts.mean()
        analysis['max_daily_records'] = daily_counts.max()
        
        print(f"Average records per day: {analysis['avg_daily_records']:.2f}")
        print(f"Max records per day: {analysis['max_daily_records']}")
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing time series: {e}")
        return {}

def plot_numerical_distributions(df: pd.DataFrame, name: str, save_plots: bool = False):
    """Plot distributions of numerical columns"""
    n_cols = len(df.columns)
    if n_cols == 0:
        return
    
    # Calculate subplot layout
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(df.columns):
        if i < len(axes):
            # Histogram
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            
            # Add mean and median lines
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[i].legend()
    
    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Numerical Distributions - {name}', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{name}_numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_categorical_distributions(df: pd.DataFrame, name: str, save_plots: bool = False):
    """Plot distributions of categorical columns"""
    n_cols = len(df.columns)
    if n_cols == 0:
        return
    
    # Calculate subplot layout
    n_rows = (n_cols + 1) // 2  # 2 columns per row
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(df.columns):
        if i < len(axes):
            # Get top 10 values
            value_counts = df[col].value_counts().head(10)
            
            # Bar plot
            axes[i].bar(range(len(value_counts)), value_counts.values)
            axes[i].set_title(f'{col} - Top 10 Values')
            axes[i].set_xlabel('Values')
            axes[i].set_ylabel('Count')
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Categorical Distributions - {name}', fontsize=16)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{name}_categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, name: str, save_plots: bool = False):
    """Plot correlation matrix for numerical columns"""
    if len(df.columns) < 2:
        return
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title(f'Correlation Matrix - {name}')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{name}_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series(df: pd.DataFrame, date_col: str, name: str, save_plots: bool = False):
    """Plot time series data"""
    try:
        # Convert to datetime if not already
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Daily counts
        daily_counts = df[date_col].value_counts().sort_index()
        
        plt.figure(figsize=(15, 6))
        daily_counts.plot(kind='line', marker='o', markersize=3)
        plt.title(f'Daily Records Count - {name}')
        plt.xlabel('Date')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{name}_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting time series: {e}")

def compare_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict:
    """
    Compare multiple datasets
    
    Args:
        datasets: Dictionary of DataFrames
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*60}")
    print(f"DATASET COMPARISON")
    print(f"{'='*60}")
    
    comparison = {}
    
    # Basic comparison
    print("Dataset Sizes:")
    for name, df in datasets.items():
        size = df.shape
        memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  {name}: {size[0]} rows, {size[1]} columns, {memory:.2f} MB")
        comparison[name] = {
            'shape': size,
            'memory_mb': memory,
            'missing_total': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        }
    
    # Common columns analysis
    all_columns = [set(df.columns) for df in datasets.values()]
    common_columns = set.intersection(*all_columns)
    
    print(f"\nCommon columns across all datasets: {len(common_columns)}")
    if common_columns:
        print(f"  {list(common_columns)}")
    
    # ID overlap analysis (if client_id exists)
    if 'client_id' in common_columns:
        print(f"\nClient ID Overlap Analysis:")
        client_ids = {}
        for name, df in datasets.items():
            client_ids[name] = set(df['client_id'].dropna())
        
        # Calculate overlaps
        for i, (name1, ids1) in enumerate(client_ids.items()):
            for name2, ids2 in list(client_ids.items())[i+1:]:
                overlap = len(ids1 & ids2)
                union = len(ids1 | ids2)
                overlap_percent = (overlap / union) * 100 if union > 0 else 0
                print(f"  {name1} ∩ {name2}: {overlap} common IDs ({overlap_percent:.1f}% of union)")
    
    return comparison

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

def full_eda_pipeline(data_folder_path: str, save_plots: bool = False) -> Dict:
    """
    Complete EDA pipeline: load, clean, analyze, and visualize data
    
    Args:
        data_folder_path: Path to the data folder
        save_plots: Whether to save plots to files
    
    Returns:
        Dictionary with all analysis results
    """
    print("Starting Full EDA Pipeline...")
    
    # 1. Load and clean data
    cleaned_datasets = clean_and_prepare_data(data_folder_path)
    
    # 2. Compare datasets
    comparison_results = compare_datasets(cleaned_datasets)
    
    # 3. Individual dataset analysis
    individual_analyses = {}
    for name, df in cleaned_datasets.items():
        print(f"\nAnalyzing {name}...")
        individual_analyses[name] = exploratory_data_analysis(df, name, save_plots)
    
    # 4. Merge datasets
    merged_data = merge_data_by_id(cleaned_datasets, 'client_id', 'left')
    
    # 5. Analyze merged dataset
    merged_analysis = exploratory_data_analysis(merged_data, "Merged Dataset", save_plots)
    
    # Compile all results
    full_results = {
        'comparison': comparison_results,
        'individual_analyses': individual_analyses,
        'merged_analysis': merged_analysis,
        'merged_data': merged_data
    }
    
    print(f"\n{'='*60}")
    print(f"EDA PIPELINE COMPLETED")
    print(f"{'='*60}")
    
    return full_results

# Example usage
if __name__ == "__main__":
    # Example usage
    data_folder_path = "/home/pavel/Data/filtered_data/"
    
    # Run full EDA pipeline
    results = full_eda_pipeline(data_folder_path, save_plots=True) 