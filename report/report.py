import os

import altair as alt
import pandas as pd
import streamlit as st

st.title("Machine Learning Classifiers on Vehicle Silhouette Classification")


@st.cache
def load_data():
    """
    Load the data from the csv file.
    """
    return pd.read_csv("./output.csv")


# create a function that calculates the mean and standard deviation of the data
def calculate_mean_std(data):
    """
    Calculate the mean and standard deviation of the data.
    """
    mean = data.mean()
    std = data.std()
    # add the data to the mean and standard deviation
    data = data.append(
        pd.DataFrame(
            {
                "Metric": ["Mean", "Standard deviation"],
                **{col: [mean[col], std[col]] for col in data.columns},
            }
        ),
        ignore_index=True,
    )
    return data


def convert_mean_to_percentage(df):
    df_percentage = df.copy()
    for col in df.columns:
        if col != "Metric":
            df_percentage.loc[df_percentage["Metric"] == "Mean", col] *= 100
    return df_percentage


def bar_with_text_label(chart):
    # Bar chart
    bars = chart.mark_bar().properties(
        title="Mean Accuracies of Classifiers",
    )

    # Text labels
    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-2,  # Adjust this value to position the text label above the bar
    ).encode(
        text=alt.Text(
            "Accuracy:Q", format=".5f"
        )  # Format the text to display 2 decimal places
    )

    # Combine the bar chart and text labels
    return bars + text


full_data = load_data()
monolitic_classifiers = full_data.iloc[:, :5]
composable_classifiers = full_data.iloc[:, 5:]

monolitic_classifiers = calculate_mean_std(monolitic_classifiers)
composable_classifiers = calculate_mean_std(composable_classifiers)

# create a expansble section for the monolitic classifiers
st.header("Monolitic Classifiers")
st.subheader("Accuracy")

monolitic_data_percentage = convert_mean_to_percentage(monolitic_classifiers)

st.table(monolitic_data_percentage)
# Extract mean values for the bar chart
monolitic_mean_data = monolitic_classifiers[
    monolitic_classifiers["Metric"] == "Mean"
].melt(
    id_vars=["Metric"],
    value_vars=monolitic_classifiers.columns.difference(["Metric"]),
    var_name="Classifier",
    value_name="Accuracy",
)

# Create the bar chart using Altair
monolitic_mean_chart = (
    alt.Chart(monolitic_mean_data)
    .mark_bar()
    .encode(
        x="Classifier",
        y="Accuracy",
        color="Classifier",
        tooltip=["Classifier", "Accuracy"],
    )
    .properties(title="Mean Accuracies of Classifiers", width=600, height=400)
)
st.altair_chart(bar_with_text_label(monolitic_mean_chart), use_container_width=True)

# create a expansble section for the composable classifiers
st.header("Composable Classifiers")
st.subheader("Accuracy")

composable_data_percentage = convert_mean_to_percentage(composable_classifiers)

st.table(composable_data_percentage)

# Extract mean values for the bar chart
composable_mean_data = composable_classifiers[
    composable_classifiers["Metric"] == "Mean"
].melt(
    id_vars=["Metric"],
    value_vars=composable_classifiers.columns.difference(["Metric"]),
    var_name="Classifier",
    value_name="Accuracy",
)

mean_chart = (
    alt.Chart(composable_mean_data)
    .mark_bar()
    .encode(
        x="Classifier",
        y="Accuracy",
        color="Classifier",
        tooltip=["Classifier", "Accuracy"],
    )
    .properties(title="Mean Accuracies of Classifiers", width=600, height=400)
)


st.altair_chart(bar_with_text_label(mean_chart), use_container_width=True)

# Define the path to the folder containing the CSV files
folder_path = (
    "./best_params/"  # Replace 'path_to_folder' with the actual path to your folder
)

# List of CSV files and their respective columns
files_columns = {
    "dt.csv": ["criterion", "max_depth", "min_samples_split", "min_samples_leaf"],
    "knn.csv": ["K", "Distance Metric"],
    "mlp.csv": ["hidden_layer_sizes", "activation", "max_iter", "learning_rate"],
    "svm.csv": ["C", "kernel"],
}

st.title("Best Parameters")


# Function to append mode to the dataframe
def append_mode_to_df(df):
    mode_values = df.mode().iloc[0]  # Get the first mode values for all columns
    mode_values.name = "Mode"
    return df.append(mode_values)


# define tabs
tabs = st.tabs(["DT", "KNN", "MLP", "SVM"])

# Show the dataframes in each tab
for i, tab in enumerate(tabs):
    file_path = os.path.join(folder_path, list(files_columns.keys())[i])
    df = pd.read_csv(file_path)

    # Append mode to the dataframe
    df_with_mode = append_mode_to_df(df)

    with tab:  # Use the filename (without extension) as the tab name
        # let the last line in bold
        st.table(df_with_mode)
