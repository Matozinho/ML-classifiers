import altair as alt
import numpy as np
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


full_data = load_data()
monolitic_classifiers = full_data.iloc[:, :5]
composable_classifiers = full_data.iloc[:, 5:]

monolitic_classifiers = calculate_mean_std(monolitic_classifiers)
composable_classifiers = calculate_mean_std(composable_classifiers)

# create a expansble section for the monolitic classifiers
st.header("Monolitic Classifiers")
st.subheader("Data")

monolitic_data_percentage = convert_mean_to_percentage(monolitic_classifiers)

st.table(monolitic_data_percentage)
# Extract mean values for the bar chart
mean_data = monolitic_classifiers[monolitic_classifiers["Metric"] == "Mean"].melt(
    id_vars=["Metric"],
    value_vars=monolitic_classifiers.columns.difference(["Metric"]),
    var_name="Classifier",
    value_name="Accuracy",
)

# Create the bar chart using Altair
mean_chart = (
    alt.Chart(mean_data)
    .mark_bar()
    .encode(
        x="Classifier",
        y="Accuracy",
        color="Classifier",
        tooltip=["Classifier", "Accuracy"],
    )
    .properties(title="Mean Accuracies of Classifiers", width=600, height=400)
)
st.altair_chart(mean_chart, use_container_width=True)

# create a expansble section for the composable classifiers
st.header("Composable Classifiers")
st.subheader("Data")

composable_data_percentage = convert_mean_to_percentage(composable_classifiers)

st.table(composable_data_percentage)

# Extract mean values for the bar chart
mean_data = composable_classifiers[composable_classifiers["Metric"] == "Mean"].melt(
    id_vars=["Metric"],
    value_vars=composable_classifiers.columns.difference(["Metric"]),
    var_name="Classifier",
    value_name="Accuracy",
)

mean_chart = (
    alt.Chart(mean_data)
    .mark_bar()
    .encode(
        x="Classifier",
        y="Accuracy",
        color="Classifier",
        tooltip=["Classifier", "Accuracy"],
    )
    .properties(title="Mean Accuracies of Classifiers", width=600, height=400)
)


# Create the bar chart using Altair
mean_chart = (
    alt.Chart(mean_data)
    .mark_bar()
    .encode(
        x="Classifier",
        y="Accuracy",
        color="Classifier",
        tooltip=["Classifier", "Accuracy"],
    )
    .properties(title="Mean Accuracies of Classifiers", width=600, height=400)
)
st.altair_chart(mean_chart, use_container_width=True)
