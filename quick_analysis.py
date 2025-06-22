import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

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

if __name__ == "__main__":
    main() 