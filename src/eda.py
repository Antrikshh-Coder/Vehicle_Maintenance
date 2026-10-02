"""
Exploratory Data Analysis (EDA) Module for Vehicle Maintenance Analysis.
Generates and saves publication-quality visualizations and statistical summaries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication quality plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16
})

def generate_eda_plots(df, output_dir="outputs/figures"):
    """Generate and save all key EDA charts."""
    os.makedirs(output_dir, exist_ok=True)
    features = [c for c in df.columns if c != 'Engine Condition']
    
    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    n_normal = int((df['Engine Condition'] == 0).sum())
    n_maint = int((df['Engine Condition'] == 1).sum())
    labels = ['Normal (0)', 'Maintenance Required (1)']
    values = [n_normal, n_maint]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f'{height:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    ax.set_title('Target Distribution: Engine Condition', fontweight='bold', pad=15)
    ax.set_ylabel('Number of Vehicles')
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/target_distribution.png", dpi=300)
    plt.close(fig)
    
    # 2. Feature Histograms / Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.histplot(df, x=col, hue='Engine Condition', kde=True, ax=axes[i], palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.6, element="step")
        axes[i].set_title(f'Distribution of {col}', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/feature_distributions.png", dpi=300)
    plt.close(fig)
    
    # 3. Boxplots for Outlier Visual Inspection
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(data=df, x='Engine Condition', y=col, hue='Engine Condition', ax=axes[i], palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Normal (0)', 'Maintenance (1)'])
        axes[i].set_title(f'{col} by Engine Condition', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/feature_boxplots.png", dpi=300)
    plt.close(fig)
    
    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax, linewidths=0.8, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Correlation Heatmap', fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
    plt.close(fig)
    
    # 5. Scatter Plot: Lub Oil Temp vs Coolant Temp
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Lub Oil Temp', y='Coolant Temp', hue='Engine Condition', palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.7, ax=ax)
    ax.set_title('Lub Oil Temp vs Coolant Temp by Engine Condition', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/scatter_temp.png", dpi=300)
    plt.close(fig)

    # 6. Scatter Plot: Engine RPM vs Lub Oil Pressure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Engine RPM', y='Lub Oil Pressure', hue='Engine Condition', palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.7, ax=ax)
    ax.set_title('Engine RPM vs Lub Oil Pressure by Engine Condition', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/scatter_rpm_pressure.png", dpi=300)
    plt.close(fig)
    
    print(f"All EDA visualizations successfully generated and saved to {output_dir}/")

if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data
    df = load_data()
    cleaned_df, _ = clean_data(df)
    generate_eda_plots(cleaned_df)
