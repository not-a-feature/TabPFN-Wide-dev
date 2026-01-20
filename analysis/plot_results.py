import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
import scipy.stats as stats

# Set style for scientific plotting
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# Hard-coded model configuration for consistent plotting
MODEL_CONFIG = {
    "v2": {"color": "#1f77b4", "order": 0, "label": "TabPFN v2"},
    "wide-v2-1.5k": {"color": "#ff7f0e", "order": 1, "label": "Wide (1.5k)"},
    "wide-v2-1.5k-nocat": {"color": "#ffbb78", "order": 2, "label": "Wide (1.5k, No-Cat)"},
    "wide-v2-5k": {"color": "#2ca02c", "order": 3, "label": "Wide (5k)"},
    "wide-v2-5k-nocat": {"color": "#98df8a", "order": 4, "label": "Wide (5k, No-Cat)"},
    "wide-v2-8k": {"color": "#d62728", "order": 5, "label": "Wide (8k)"},
    "wide-v2-8k-nocat": {"color": "#ff9896", "order": 6, "label": "Wide (8k, No-Cat)"},
    "tabicl": {"color": "#9467bd", "order": 9, "label": "TabICL"},
    "random_forest": {"color": "#5D4037", "order": 10, "label": "Random Forest"},
}


def get_model_style(df, hue_col="checkpoint"):
    """
    Returns palette and order for plotting based on MODEL_CONFIG.
    Also applies labels to the dataframe column in-place.
    """
    if hue_col not in ["checkpoint", "model"]:
        # Fallback for non-model categorical plots
        unique_vals = df[hue_col].unique()
        return None, sorted(unique_vals)

    # Apply labels to the column in-place
    label_map = {k: v["label"] for k, v in MODEL_CONFIG.items()}
    df[hue_col] = df[hue_col].apply(lambda x: label_map.get(x, x))

    models = df[hue_col].unique()
    label_to_config = {v["label"]: v for v in MODEL_CONFIG.values()}

    # Filter known models and sort by order
    known_models = sorted(
        [m for m in models if m in label_to_config],
        key=lambda m: label_to_config[m]["order"],
    )

    # Handle unknown models (append keeping alphabetical order)
    unknown_models = sorted([m for m in models if m not in label_to_config])

    # Combined order
    order = known_models + unknown_models

    # Create palette
    palette = {}
    for m in models:
        if m in label_to_config:
            palette[m] = label_to_config[m]["color"]
        else:
            palette[m] = "#333333"  # Dark Grey for unknown

    return palette, order


def clean_checkpoint_names(df, col="checkpoint"):
    """Clean checkpoint names by removing file extensions and paths."""
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).split("/")[-1].replace(".pt", ""))
    return df


def save_plots(fig, output_dir, filename_prefix):
    """Save figure as PDF and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{filename_prefix}.pdf")
    png_path = os.path.join(output_dir, f"{filename_prefix}.png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved plots:\n  {pdf_path}\n  {png_path}")
    plt.close(fig)


def format_metric(metric):
    """Format metric name for display."""
    return metric.replace("_", " ").title()


def plot_categorical_comparison(
    df,
    x_col,
    y_col,
    hue_col=None,
    output_dir=None,
    basename=None,
    xlabel=None,
    ylabel=None,
    title=None,
    plot_type="box",
    ylim=(0.4, 1.05),
    suffix="",
    agg_threshold=40,
):
    """
    Generic plotting function for categorical comparisons.
    Automatically switches to aggregated view if x_col has too many categories.
    """
    if df.empty:
        return

    df = df.copy()

    unique_x = df[x_col].nunique()

    # Clean data
    df = df.dropna(subset=[y_col])
    if df.empty:
        print(f"Skipping plot for {y_col} - No valid data.")
        return
    if unique_x > agg_threshold:
        print(f"Comparison has {unique_x} categories for {x_col}, switching to aggregated view.")

        hue_order = None
        if hue_col:
            hue_order = df.groupby(hue_col)[y_col].median().sort_values(ascending=False).index

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x=hue_col,
            y=y_col,
            order=hue_order,
            palette="tab10",
            hue=hue_col,
            legend=False,
        )

        x_label_final = (
            xlabel if xlabel else (hue_col.replace("_", " ").title() if hue_col else x_col)
        )
        y_label_final = ylabel if ylabel else format_metric(y_col)

        plt.xlabel(x_label_final)
        plt.ylabel(y_label_final)
        plt.title(f"{title if title else basename} - Aggregated {y_col}")
        plt.ylim(ylim)

        filename = f"{basename}_{y_col}_aggregated"
        if hue_col:
            filename += f"_by_{hue_col}"
        filename += suffix

        save_plots(plt.gcf(), output_dir, filename)
        return

    # Standard Plot
    plt.figure(figsize=(10, 6))

    # Seaborn 0.13.x has a bug when hue == x that causes UnboundLocalError.
    # Workaround: use hue=x with legend=False when we want colors on x.
    use_hue = hue_col and hue_col != x_col

    # Determine which column to use for colors
    color_col = hue_col if use_hue else (x_col if x_col in ["checkpoint", "model"] else None)

    # Get fixed palette and order if applicable
    # Note: get_model_style transforms the column values to labels in-place
    palette = "tab10"
    color_order = None
    if color_col in ["checkpoint", "model"]:
        palette, color_order = get_model_style(df, color_col)

    # Sort x axis by median metric value (AFTER label transformation)
    order = df.groupby(x_col)[y_col].median().sort_values(ascending=False).index

    if plot_type == "box":
        if use_hue:
            sns.boxplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                order=order,
                hue_order=color_order,
                palette=palette,
            )
        elif color_col:
            # Use hue=x for coloring, but disable legend (Seaborn 0.13 compatible)
            sns.boxplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=x_col,
                order=order,
                hue_order=color_order,
                palette=palette,
                legend=False,
            )
        else:
            sns.boxplot(
                data=df,
                x=x_col,
                y=y_col,
                order=order,
            )
    elif plot_type == "bar":
        if use_hue:
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                order=order,
                hue_order=color_order,
                palette=palette,
                errorbar="sd",
                capsize=0.1,
            )
        elif color_col:
            # Use hue=x for coloring, but disable legend (Seaborn 0.13 compatible)
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=x_col,
                order=order,
                hue_order=color_order,
                palette=palette,
                errorbar="sd",
                capsize=0.1,
                legend=False,
            )
        else:
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                order=order,
                errorbar="sd",
                capsize=0.1,
            )

    plt.xticks(rotation=45, ha="right")
    plt.ylim(ylim)
    plt.xlabel(xlabel if xlabel else x_col.replace("_", " ").title())
    plt.ylabel(ylabel if ylabel else format_metric(y_col))
    plt.title(title if title else f"{basename} - {y_col}")

    # Handle legend: remove if not needed, reposition if needed
    ax = plt.gca()
    legend = ax.get_legend()
    if legend is not None:
        if use_hue and df[hue_col].nunique() > 1:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(1.05, 1),
                title=hue_col.replace("_", " ").title(),
            )
        else:
            legend.remove()
    save_plots(plt.gcf(), output_dir, f"{basename}_{y_col}_per_{x_col}{suffix}")


def plot_line_comparison(
    df,
    x_col,
    y_col,
    hue_col=None,
    style_col=None,
    output_dir=None,
    basename=None,
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=(0.4, 1.05),
    suffix="",
    smoothing=0,
):
    df = df.copy()
    if smoothing > 0:
        # Apply smoothing per group
        if hue_col:
            # Sort is crucial for rolling
            df = df.sort_values(by=[hue_col, x_col])
            df[y_col] = df.groupby(hue_col)[y_col].transform(
                lambda x: x.rolling(window=smoothing, min_periods=1, center=True).mean()
            )
        else:
            df = df.sort_values(by=x_col)
            df[y_col] = df[y_col].rolling(window=smoothing, min_periods=1, center=True).mean()

    # Filter out the 14k outlier
    if x_col == "n_features":
        # Check if we have data around 14000
        mask = (df[x_col] > 13000) & (df[x_col] < 14800)
        if mask.any():
            df = df[~mask]

    # Get fixed palette and order if applicable
    palette = "tab10"
    hue_order = None
    if hue_col in ["checkpoint", "model"]:
        palette, hue_order = get_model_style(df, hue_col)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        style=style_col,
        style_order=hue_order if style_col == hue_col else None,
        markers=True,
        palette=palette,
        err_kws={"alpha": 0.05},
    )
    plt.ylim(ylim)
    plt.xlabel(xlabel if xlabel else x_col.replace("_", " ").title())
    plt.ylabel(ylabel if ylabel else format_metric(y_col))
    plt.title(title if title else f"{basename} - {y_col}")

    if hue_col:
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title=hue_col.replace("_", " ").title(),
        )

    save_plots(plt.gcf(), output_dir, f"{basename}_{y_col}_vs_{x_col}{suffix}")


def plot_hdlss(df, output_dir, basename):
    """Plotting logic for HDLSS benchmarks."""
    output_dir = os.path.join(output_dir, "hdlss")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    for metric in metrics:
        # Plot 1: Per Dataset Comparison
        plot_categorical_comparison(
            df,
            x_col="dataset_name",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="Dataset",
        )

        # Plot 2: Aggregated Summary (if allowed)
        if df["dataset_name"].nunique() > 1 and df["checkpoint"].nunique() > 1:
            plot_categorical_comparison(
                df,
                x_col="checkpoint",
                y_col=metric,
                output_dir=output_dir,
                basename=basename,
                plot_type="bar",
                xlabel="Checkpoint",
                title=f"Aggregated {format_metric(metric)}",
                suffix="_overall_bar",
            )

        # Plot 3: One plot per dataset
        if df["checkpoint"].nunique() > 1:
            per_dataset_dir = os.path.join(output_dir, "per_dataset_plots")
            for ds in df["dataset_name"].unique():
                ds_df = df[df["dataset_name"] == ds]
                if ds_df.empty:
                    continue
                plot_categorical_comparison(
                    ds_df,
                    x_col="checkpoint",
                    y_col=metric,
                    hue_col="checkpoint",
                    output_dir=per_dataset_dir,
                    basename=f"{basename}_{ds}",
                    plot_type="bar",
                    xlabel="Checkpoint",
                    title=f"{ds} - {format_metric(metric)}",
                    suffix="_comparison",
                )


def plot_openml(df, output_dir, basename):
    """Plotting logic for OpenML benchmarks."""
    output_dir = os.path.join(output_dir, "openml")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    # Use task_id as categorical
    df["Task"] = "Task " + df["task_id"].astype(str)

    for metric in metrics:
        # Plot 1: Task vs Metric
        plot_categorical_comparison(
            df,
            x_col="Task",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
        )

        # Plot 2: Aggregated Summary
        if df["task_id"].nunique() > 1 and df["checkpoint"].nunique() > 1:
            plot_categorical_comparison(
                df,
                x_col="checkpoint",
                y_col=metric,
                output_dir=output_dir,
                basename=basename,
                plot_type="bar",
                xlabel="Checkpoint",
                title=f"Aggregated {format_metric(metric)}",
                suffix="_overall_bar",
            )

        # Plot 3: One plot per task
        if df["checkpoint"].nunique() > 1:
            per_task_dir = os.path.join(output_dir, "per_task_plots")
            for task in df["task_id"].unique():
                task_df = df[df["task_id"] == task]
                if task_df.empty:
                    continue

                plot_categorical_comparison(
                    task_df,
                    x_col="checkpoint",
                    y_col=metric,
                    hue_col="checkpoint",
                    output_dir=per_task_dir,
                    basename=f"{basename}_task_{task}",
                    plot_type="bar",
                    xlabel="Checkpoint",
                    title=f"Task {task} - {format_metric(metric)}",
                    suffix="_comparison",
                )


def plot_grouping(df, output_dir, basename):
    """Plotting logic for Grouping benchmarks."""
    output_dir = os.path.join(output_dir, "grouping")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]

    for metric in metrics:
        plot_categorical_comparison(
            df,
            x_col="features_per_group",
            y_col=metric,
            output_dir=output_dir,
            basename=basename,
            xlabel="Features Per Group",
            title=f"Impact of Grouping on {format_metric(metric)}",
            suffix="_grouping",
        )


def plot_multiomics(df, output_dir, basename):
    """Plotting logic for Multiomics Feature Reduction."""
    output_dir = os.path.join(output_dir, "multiomics")
    metrics = [c for c in ["accuracy", "roc_auc", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    for metric in metrics:
        # Separate plot per Dataset
        for ds in df["dataset_name"].unique():
            ds_df = df[df["dataset_name"] == ds]
            plot_line_comparison(
                ds_df,
                x_col="n_features",
                y_col=metric,
                hue_col="checkpoint",
                style_col="checkpoint",
                output_dir=output_dir,
                basename=basename,
                xlabel="Number of Features",
                title=f"{ds} - {format_metric(metric)} vs Feature Count",
                suffix=f"_{ds}_{metric}_feature_curve",
            )


def plot_multiomics_overview(df, output_dir, basename):
    """Multiomics overview plots."""
    output_dir = os.path.join(output_dir, "multiomics_overview")
    metrics = [c for c in ["accuracy", "roc_auc", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    for metric in metrics:
        # 1. Bar plot: Average per checkpoint
        # Aggregate logic
        df_agg = df.groupby("checkpoint")[metric].agg(["mean", "std"]).reset_index()
        df_agg = df_agg.sort_values("mean", ascending=False)

        # Custom logic for bar colors in overview
        palette, _ = get_model_style(df_agg, "checkpoint")
        bar_colors = [palette.get(x, "#333333") for x in df_agg["checkpoint"]]

        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(df_agg))
        plt.bar(
            x_pos,
            df_agg["mean"],
            yerr=df_agg["std"],
            capsize=5,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        plt.xticks(x_pos, df_agg["checkpoint"], rotation=45, ha="right")
        plt.ylim(0.45, 1.05)
        plt.xlabel("Checkpoint")
        plt.ylabel(f"Average {format_metric(metric)}")
        plt.title(f"Multiomics Overview - Average {format_metric(metric)}")
        plt.tight_layout()
        save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_overview_bar")

        # 2. Boxplot distribution
        plot_categorical_comparison(
            df,
            x_col="checkpoint",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="Checkpoint",
            title=f"Multiomics Overview - {format_metric(metric)} Distribution",
            suffix=f"_{metric}_overview_boxplot",
        )

        # 3. Line plot: Average metric vs n_features (Absolute)
        plot_line_comparison(
            df,
            x_col="n_features",
            y_col=metric,
            hue_col="checkpoint",
            style_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="Number of Features",
            ylabel=f"Average {format_metric(metric)}",
            title=f"Multiomics Overview - {format_metric(metric)} vs Features",
            suffix=f"_{metric}_overview_features",
            smoothing=5,  # Added smoothing
        )

        # 4. Relative Performance vs Baseline (v2)
        # We need to aggregate first to compute differences
        # Group by n_features and checkpoint
        df_agg = df.groupby(["n_features", "checkpoint"])[metric].mean().reset_index()

        baseline_name = "v2"
        baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
        unique_checkpoints = df_agg["checkpoint"].unique()
        actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

        if actual_baseline in unique_checkpoints:
            df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline][["n_features", metric]]
            df_baseline = df_baseline.rename(columns={metric: "baseline_score"})

            df_merged = pd.merge(df_agg, df_baseline, on="n_features", how="left")
            df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]

            # Remove baseline from plot (relative score 0) if desired, or keep to show 0 line
            # Usually better to drop the baseline line itself if it's just 0
            df_plot = df_merged[df_merged["checkpoint"] != actual_baseline]

            if not df_plot.empty:
                plot_line_comparison(
                    df_plot,
                    x_col="n_features",
                    y_col="relative_score",
                    hue_col="checkpoint",
                    style_col="checkpoint",
                    output_dir=output_dir,
                    basename=basename,
                    xlabel="Number of Features",
                    ylabel=f"Relative {format_metric(metric)} (vs {actual_baseline})",
                    title=f"Multiomics Overview - Relative {format_metric(metric)}",
                    suffix=f"_{metric}_overview_relative",
                    ylim=(-0.05, 0.5),  # Adjust ylim for relative plots
                    smoothing=5,  # Added smoothing
                )


def plot_widening(df, output_dir, basename, comparison_mode=False):
    """Plotting logic for OpenML Widening."""
    output_dir = os.path.join(output_dir, "widening")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    has_sparsity = "sparsity" in df.columns
    sparsities = df["sparsity"].unique() if has_sparsity else [None]
    checkpoints = df["checkpoint"].unique() if "checkpoint" in df.columns else [None]
    datasets = df["dataset_id"].unique() if "dataset_id" in df.columns else [None]

    for metric in metrics:
        for ds in datasets:
            ds_df = df[df["dataset_id"] == ds] if ds is not None else df
            ds_label = f"Dataset {ds}" if ds is not None else basename
            ds_df = ds_df.sort_values("features_added")

            if comparison_mode and has_sparsity:
                # One plot per sparsity
                for sparsity in sparsities:
                    s_df = ds_df[ds_df["sparsity"] == sparsity]
                    if s_df.empty:
                        continue
                    plot_line_comparison(
                        s_df,
                        x_col="features_added",
                        y_col=metric,
                        hue_col="checkpoint",
                        style_col="checkpoint",
                        output_dir=output_dir,
                        basename=basename,
                        title=f"{ds_label} - {metric} (Sparsity={sparsity})",
                        xlabel="Features Added",
                        suffix=f"_{ds}_{metric}_sparsity_{sparsity}_comparison",
                    )

            elif has_sparsity and len(sparsities) > 1:
                # Single model, multiple sparsities
                ds_df["sparsity_label"] = ds_df["sparsity"].apply(lambda x: f"Sparsity={x}")
                plot_line_comparison(
                    ds_df,
                    x_col="features_added",
                    y_col=metric,
                    hue_col="sparsity_label",
                    style_col="sparsity_label",
                    output_dir=output_dir,
                    basename=basename,
                    title=f"{ds_label} - {metric} vs Features Added",
                    xlabel="Features Added",
                    suffix=f"_{ds}_{metric}_all_sparsities",
                )

                # Also per sparsity
                per_sparsity_dir = os.path.join(output_dir, "per_sparsity")
                for sparsity in sparsities:
                    s_df = ds_df[ds_df["sparsity"] == sparsity]
                    if s_df.empty:
                        continue
                    plot_line_comparison(
                        s_df,
                        x_col="features_added",
                        y_col=metric,
                        hue_col="checkpoint" if len(checkpoints) > 1 else None,
                        style_col="checkpoint" if len(checkpoints) > 1 else None,
                        output_dir=per_sparsity_dir,
                        basename=basename,
                        title=f"{ds_label} - {metric} (Sparsity={sparsity})",
                        xlabel="Features Added",
                        suffix=f"_{ds}_{metric}_sparsity_{sparsity}",
                    )
            else:
                # Default
                plot_line_comparison(
                    ds_df,
                    x_col="features_added",
                    y_col=metric,
                    hue_col="checkpoint" if len(checkpoints) > 1 else None,
                    style_col="checkpoint" if len(checkpoints) > 1 else None,
                    output_dir=output_dir,
                    basename=basename,
                    title=f"{ds_label} - {metric} vs Features Added",
                    xlabel="Features Added",
                    suffix=f"_{ds}_{metric}_widening_curve",
                )


def plot_widening_relative(df, output_dir, basename, baseline_name="v2"):
    """
    Plot average AUROC relative to a baseline model.
    """
    output_dir = os.path.join(output_dir, "widening")
    metric = "roc_auc_score"
    if metric not in df.columns:
        return
    if basename == baseline_name:
        return

    df = clean_checkpoint_names(df)

    if "sparsity" not in df.columns or "features" not in df.columns:
        print("Missing 'sparsity' or 'features' columns for relative plots.")
        return

    # Aggregate
    group_cols = ["checkpoint", "sparsity", "features"]
    df_agg = df.groupby(group_cols)[metric].mean().reset_index()

    # 1. Absolute Average Performance
    unique_sparsities = sorted(df_agg["sparsity"].unique())
    for sp in unique_sparsities:
        sp_df = df_agg[df_agg["sparsity"] == sp].sort_values("features")
        if sp_df.empty:
            continue

        plot_line_comparison(
            sp_df,
            x_col="features",
            y_col=metric,
            hue_col="checkpoint",
            style_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            title=f"Average Performance (All Datasets) - Sparsity {sp}",
            xlabel="Total Features",
            ylabel=f"Average {format_metric(metric)}",
            suffix=f"_average_auc_absolute_sparsity_{sp}",
        )

    # 2. Relative Performance
    baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
    unique_checkpoints = df_agg["checkpoint"].unique()
    actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

    df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline].copy()
    if df_baseline.empty:
        print(f"Baseline '{actual_baseline}' not found for relative plots.")
        return

    df_baseline = df_baseline.rename(columns={metric: "baseline_score"}).drop(
        columns=["checkpoint"]
    )
    df_merged = pd.merge(df_agg, df_baseline, on=["sparsity", "features"], how="left")
    df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]
    df_merged = df_merged.dropna(subset=["relative_score"])

    for sp in unique_sparsities:
        sp_df = df_merged[df_merged["sparsity"] == sp].sort_values("features")
        sp_df = sp_df[sp_df["checkpoint"] != actual_baseline].copy()
        if sp_df.empty:
            continue

        palette, hue_order = get_model_style(sp_df, "checkpoint")

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=sp_df,
            x="features",
            y="relative_score",
            hue="checkpoint",
            hue_order=hue_order,
            style="checkpoint",
            style_order=hue_order,
            markers=True,
            palette=palette,
        )
        plt.ylim(-0.1, 0.35)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.title(f"Relative AUROC vs {actual_baseline} (Sparsity={sp})")
        plt.xlabel("Total Features")
        plt.ylabel(f"Relative AUROC (vs {actual_baseline})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        save_plots(plt.gcf(), output_dir, f"{basename}_relative_auc_sparsity_{sp}")


def plot_forgetting(df, output_dir, basename):
    """Scatter plot comparing two models (Forgetting)."""
    output_dir = os.path.join(output_dir, "forgetting")

    required_cols = ["task_id", "checkpoint", "roc_auc_score"]
    if not all(col in df.columns for col in required_cols):
        return

    df = df.copy()

    # Filter
    # Filter: include v2 and wide models (supporting both raw and labeled names)
    def is_target(x):
        x_str = str(x)
        return (
            x_str == "v2"
            or x_str == "TabPFN v2"
            or x_str.startswith("wide-v2")
            or x_str.startswith("Wide")
        )

    df = df[df["checkpoint"].apply(is_target)]
    if df.empty:
        return

    df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
    df = df.dropna(subset=["task_id"])

    # Aggregate
    df_agg = df.groupby(["task_id", "checkpoint"])["roc_auc_score"].mean().reset_index()
    df_pivot = df_agg.pivot(index="task_id", columns="checkpoint", values="roc_auc_score")

    if df_pivot.shape[1] < 2:
        return

    cols = df_pivot.columns
    baseline = cols[0]
    others = cols[1:]

    for other in others:
        df_xy = df_pivot[[baseline, other]].dropna()
        if df_xy.empty:
            continue

        x_vals = df_xy[baseline]
        y_vals = df_xy[other]
        rho, _ = stats.spearmanr(x_vals, y_vals)

        plt.figure(figsize=(6, 6))
        plt.scatter(x_vals, y_vals, marker="x", s=100, linewidths=1.5, color="mediumblue")
        plt.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", linewidth=1.5)

        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(f"{baseline}", fontsize=12)
        plt.ylabel(f"{other}", fontsize=12)
        plt.title(f"Forgetting Comparison (Spearman rho={rho:.4f})")
        save_plots(plt.gcf(), output_dir, f"forgetting_scatter_{baseline}_vs_{other}")


def plot_snp(df, output_dir, basename):
    """Plotting logic for SNP benchmarks."""
    output_dir = os.path.join(output_dir, "snp")
    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)

    metrics = [c for c in ["roc_auc", "accuracy"] if c in df.columns]

    for metric in metrics:
        if "Polygenicity" in df.columns:
            plt.figure(figsize=(10, 6))
            palette, hue_order = get_model_style(df, hue_col)

            g = sns.FacetGrid(df, col="Polygenicity", sharey=False, height=5, aspect=1.2)
            g.map_dataframe(
                sns.lineplot,
                x="n_features",
                y=metric,
                hue=hue_col,
                hue_order=hue_order,
                style=hue_col,
                style_order=hue_order,
                markers=True,
                palette=palette,
                err_kws={"alpha": 0.05},
            )
            g.add_legend()
            g.set_axis_labels("Number of Features", format_metric(metric))
            g.set_titles(col_template="Polygenicity: {col_name}")
            g.set(ylim=(0.4, 1))
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_by_polygenicity")
        else:
            plot_line_comparison(
                df,
                x_col="n_features",
                y_col=metric,
                hue_col=hue_col,
                style_col=hue_col,
                output_dir=output_dir,
                basename=basename,
                xlabel="Number of Features",
                suffix=f"_{metric}",
            )


def clean_basename(basename):
    for suffix in ["_benchmark_results", "_results", "_benchmark"]:
        basename = basename.replace(suffix, "")
    return basename


def aggregate_files(filename, file_list):
    """Aggregate multiple CSV files into one DataFrame."""
    dfs = []
    for f in file_list:
        try:
            temp_df = pd.read_csv(f)
        except Exception:
            continue

        if temp_df.empty:
            continue

        parent_folder = os.path.dirname(f)
        folder_name = os.path.basename(parent_folder)

        # Heuristic to find checkpoint name from folder structure
        if folder_name.startswith("sparsity"):
            parent_folder = os.path.dirname(parent_folder)
            folder_name = os.path.basename(parent_folder)
        if folder_name == "openml_widening":
            folder_name = os.path.basename(os.path.dirname(parent_folder))

        temp_df["checkpoint"] = folder_name
        dfs.append(temp_df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--compare_mode", action="store_true")
    args = parser.parse_args()

    cwd = os.getcwd()
    base_results_dir = args.input_dir if args.input_dir else os.path.join(cwd, "analysis_results")

    if not os.path.exists(base_results_dir):
        print(f"Directory '{base_results_dir}' not found. Skipping plotting.")
        return

    if args.compare_mode:
        print(f"Running in COMPARISON MODE. Scanning {base_results_dir} recursively...")
        all_csvs = glob.glob(os.path.join(base_results_dir, "**/*.csv"), recursive=True)

        grouped_csvs = {}
        for csv_file in all_csvs:
            filename = os.path.basename(csv_file)
            grouped_csvs.setdefault(filename, []).append(csv_file)

        widening_filenames = [f for f in grouped_csvs if f.replace(".csv", "").isdigit()]
        other_filenames = [f for f in grouped_csvs if f not in widening_filenames]

        output_dir = os.path.join(base_results_dir, "comparison_plots")

        # Process widening
        widening_dfs = []
        for filename in widening_filenames:
            combined_df = aggregate_files(filename, grouped_csvs[filename])
            if combined_df is not None:
                widening_dfs.append(combined_df)

        if widening_dfs:
            all_widening_df = pd.concat(widening_dfs, ignore_index=True)
            plot_widening_relative(
                all_widening_df, output_dir, "openml_widening_average", baseline_name="v2"
            )

        # Process others
        for filename in other_filenames:
            combined_df = aggregate_files(filename, grouped_csvs[filename])
            if combined_df is None:
                continue

            basename = clean_basename(os.path.splitext(filename)[0])

            if "multiomics" in basename.lower():
                plot_multiomics(combined_df, output_dir, basename)
                plot_multiomics_overview(combined_df, output_dir, basename)
            elif "grouping" in basename.lower():
                plot_grouping(combined_df, output_dir, basename)
            elif "hdlss" in basename.lower():
                plot_hdlss(combined_df, output_dir, basename)
            elif "openml" in basename.lower() and "widening" not in basename.lower():
                plot_openml(combined_df, output_dir, basename)
                plot_forgetting(combined_df, output_dir, basename)
            elif "snp" in basename.lower():
                plot_snp(combined_df, output_dir, basename)

    else:
        # Single directory mode
        csv_files = glob.glob(os.path.join(base_results_dir, "*.csv"))
        widening_files = [
            f
            for f in glob.glob(os.path.join(base_results_dir, "**/*.csv"), recursive=True)
            if f not in csv_files
        ]
        all_files = csv_files + widening_files

        print(f"Found {len(all_files)} CSV files...")

        for csv_file in all_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            if df.empty:
                continue

            basename = clean_basename(os.path.splitext(os.path.basename(csv_file))[0])
            output_dir = os.path.dirname(csv_file)

            print(f"-- Processing {basename} --")

            if "multiomics" in basename.lower():
                plot_multiomics(df, output_dir, basename)
            elif "grouping" in basename.lower():
                plot_grouping(df, output_dir, basename)
            elif "hdlss" in basename.lower():
                plot_hdlss(df, output_dir, basename)
            elif "openml" in basename.lower() and "widening" not in basename.lower():
                plot_openml(df, output_dir, basename)
                plot_forgetting(df, output_dir, basename)
            elif "snp" in basename.lower():
                plot_snp(df, output_dir, basename)
            elif basename.isdigit() or "widening" in os.path.dirname(csv_file).lower():
                plot_widening(df, os.path.dirname(csv_file), basename, comparison_mode=False)


if __name__ == "__main__":
    main()
