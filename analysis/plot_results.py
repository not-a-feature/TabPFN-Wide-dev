import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Set style for scientific plotting
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def save_plots(fig, output_dir, filename_prefix):
    """Save figure as PDF and PNG."""
    pdf_path = os.path.join(output_dir, f"{filename_prefix}.pdf")
    png_path = os.path.join(output_dir, f"{filename_prefix}.png")
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved plots:\n  {pdf_path}\n  {png_path}")
    plt.close(fig)

def plot_metric_vs_categorical(df, x_col, hue_col, metric, output_dir, basename, xlabel=None, ylabel=None, title=None):
    """Generic boxplot for Metric vs Categorical (e.g. Dataset) grouped by Hue (e.g. Checkpoint)."""
    plt.figure(figsize=(10, 6))
    
    # Check number of categories
    unique_x = df[x_col].nunique()
    if unique_x > 20:
        # If too many items, fall back to aggregated plot grouped by hue
        print(f"Comparison has {unique_x} categories for {x_col}, switching to aggregated view.")
        sns.boxplot(data=df, x=hue_col, y=metric)
        plt.xlabel(hue_col.replace('_', ' ').title() if not xlabel else xlabel)
        plt.ylabel(ylabel if ylabel else metric.replace('_', ' ').title())
        plt.title(f"{title if title else basename} - Aggregated {metric}")
        plt.ylim(0, 1.05)
        save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_aggregated_by_{hue_col}")
        return

    # Standard detailed plot
    sns.boxplot(data=df, x=x_col, y=metric, hue=hue_col)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.xlabel(xlabel if xlabel else x_col.replace('_', ' ').title())
    plt.ylabel(ylabel if ylabel else metric.replace('_', ' ').title())
    plt.title(title if title else f"{basename} - {metric} per {x_col}")
    
    # Adjust legend
    if df[hue_col].nunique() > 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=hue_col.replace('_', ' ').title())
    
    save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_per_{x_col}")

def plot_hdlss(df, output_dir, basename):
    """Plotting logic for HDLSS benchmarks."""
    metrics = [c for c in ['accuracy', 'f1_weighted', 'roc_auc_score'] if c in df.columns]
    
    # Clean up checkpoint names if needed
    if 'checkpoint' in df.columns:
        df['checkpoint'] = df['checkpoint'].apply(lambda x: str(x).split('/')[-1].replace('.pt', ''))
    
    for metric in metrics:
        # Plot 1: Per Dataset Comparison
        plot_metric_vs_categorical(
            df, 
            x_col='dataset_name', 
            hue_col='checkpoint', 
            metric=metric, 
            output_dir=output_dir, 
            basename=basename,
            xlabel='Dataset',
            ylabel=metric.replace('_', ' ').title()
        )
        
        # Plot 2: Aggregated Summary (if more than 1 dataset)
        if df['dataset_name'].nunique() > 1:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df, x='checkpoint', y=metric, errorbar='sd', capsize=.1)
            plt.ylim(0, 1.05)
            plt.title(f"Aggregated {metric.replace('_', ' ').title()} - {basename}")
            plt.xlabel("Checkpoint")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_overall_bar")

def plot_openml(df, output_dir, basename):
    """Plotting logic for OpenML benchmarks."""
    metrics = [c for c in ['accuracy', 'f1_weighted', 'roc_auc_score'] if c in df.columns]

    if 'checkpoint' in df.columns:
        df['checkpoint'] = df['checkpoint'].apply(lambda x: str(x).split('/')[-1].replace('.pt', ''))

    # Use task_id as categorical
    df['Task'] = "Task " + df['task_id'].astype(str)
    
    for metric in metrics:
        plot_metric_vs_categorical(
            df,
            x_col='Task',
            hue_col='checkpoint',
            metric=metric,
            output_dir=output_dir,
            basename=basename
        )

        # Aggregated
        if df['task_id'].nunique() > 1:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df, x='checkpoint', y=metric, errorbar='sd', capsize=.1)
            plt.ylim(0, 1.05)
            plt.title(f"Aggregated {metric.replace('_', ' ').title()} - {basename}")
            plt.xlabel("Checkpoint")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_overall_bar")

def plot_grouping(df, output_dir, basename):
    """Plotting logic for Grouping benchmarks."""
    metrics = [c for c in ['accuracy', 'f1_weighted', 'roc_auc_score'] if c in df.columns]
    
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        # Group by features_per_group
        sns.boxplot(data=df, x='features_per_group', y=metric)
        plt.ylim(0, 1.05)
        plt.title(f"Impact of Grouping on {metric.replace('_', ' ').title()}")
        plt.xlabel("Features Per Group")
        plt.ylabel(metric.replace('_', ' ').title())
        save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_grouping")

def plot_multiomics(df, output_dir, basename):
    """Plotting logic for Multiomics Feature Reduction."""
    # Line plot: x=n_features, y=metric, hue=Checkpoint
    metrics = [c for c in ['Accuracy', 'f1_weighted'] if c in df.columns]
    
    if 'Checkpoint' in df.columns:
        df['Checkpoint'] = df['Checkpoint'].apply(lambda x: str(x).split('/')[-1].replace('.pt', ''))
        
    for metric in metrics:
        # Separate plot per Dataset if multiple
        datasets = df['Dataset'].unique()
        for ds in datasets:
            ds_df = df[df['Dataset'] == ds]
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=ds_df, x='n_features', y=metric, hue='Checkpoint', marker='o')
            plt.ylim(0, 1.05)
            plt.title(f"{ds} - {metric} vs Feature Count")
            plt.xlabel("Number of Features")
            plt.ylabel(metric)
            feats = sorted(ds_df['n_features'].unique())
            if 0 in feats:
                 # Move 0 to the end as "All"
                 pass 

            save_plots(plt.gcf(), output_dir, f"{basename}_{ds}_{metric}_feature_curve")

def main():
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'analysis_results')

    if not os.path.exists(results_dir):
        print(f"Directory '{results_dir}' not found. Skipping plotting.")
        return

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {results_dir}...")

    for csv_file in csv_files:
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"-- Processing {basename} --")
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print("   File is empty, skipping.")
                continue
                
            # Dispatch based on filename
            if 'multiomics' in basename.lower():
                 print("   Detected Multiomics Feature Reduction format.")
                 plot_multiomics(df, results_dir, basename)
            elif 'grouping' in basename.lower():
                 print("   Detected Grouping Benchmark format.")
                 plot_grouping(df, results_dir, basename)
            elif 'hdlss' in basename.lower():
                 print("   Detected HDLSS Benchmark format.")
                 plot_hdlss(df, results_dir, basename)
            elif 'openml' in basename.lower():
                 print("   Detected OpenML Benchmark format.")
                 plot_openml(df, results_dir, basename)
            else:
                 print(f"   Skipping {basename}: filename pattern not recognized.")
                 
        except Exception as e:
            print(f"   Error processing file: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
