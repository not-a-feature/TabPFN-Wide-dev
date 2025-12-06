import openml
from openml import tasks
import pandas as pd

try:
    suite_id = 337
    print(f"Fetching suite {suite_id}...")
    suite = openml.study.get_suite(suite_id=suite_id)
    print(f"Suite tasks: {len(suite.tasks)}")

    print("Listing tasks...")
    openml_df = tasks.list_tasks(output_format="dataframe", task_id=suite.tasks)
    print(f"Initial DF shape: {openml_df.shape}")
    print("Columns:", openml_df.columns.tolist())
    
    # Check filters
    min_features = 0
    max_features = 700
    max_instances = 2000
    
    if "task_type" in openml_df.columns:
        df_type = openml_df[openml_df["task_type"] == "Supervised Classification"]
        print(f"After type filter: {len(df_type)}")
    else:
        print("'task_type' column missing!")

    df_feat = openml_df[openml_df["NumberOfFeatures"] >= min_features]
    df_feat = df_feat[df_feat["NumberOfFeatures"] <= max_features]
    print(f"After feature filter (0 <= x <= {max_features}): {len(df_feat)}")

    df_inst = openml_df[openml_df["NumberOfInstances"] <= max_instances]
    print(f"After instance filter (<= {max_instances}): {len(df_inst)}")
    
    df_classes = openml_df[openml_df["NumberOfClasses"] < 10]
    df_classes = df_classes[df_classes["NumberOfClasses"] > 1]
    print(f"After class filter (1 < x < 10): {len(df_classes)}")

    # Combined
    final_df = openml_df
    if "task_type" in final_df.columns:
        final_df = final_df[final_df["task_type"] == "Supervised Classification"]
    final_df = final_df[final_df["NumberOfFeatures"] >= min_features]
    final_df = final_df[final_df["NumberOfFeatures"] <= max_features]
    final_df = final_df[final_df["NumberOfInstances"] <= max_instances]
    final_df = final_df[final_df["NumberOfClasses"] < 10]
    final_df = final_df[final_df["NumberOfClasses"] > 1]
    print(f"Final Count: {len(final_df)}")
    
    if len(final_df) == 0:
        print("\nTop 5 tasks by instances:")
        print(openml_df[["tid", "NumberOfInstances"]].sort_values("NumberOfInstances").head())

except Exception as e:
    print(f"Error: {e}")
