import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Data Upload & Cleaning",
        "EDA",
        "Manual Visualizations",
        "Analytics",
        "Groq AI Visualization"
    ]
)


st.title("DataTalk – Conversational Data Analysis System")

st.write("""
Welcome to DataTalk 👋  
Upload your dataset and explore insights using natural language.
""")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx"]
)

from groq import Groq
import os

client = Groq(
    api_key="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"
)

def get_visualization_intent(question, columns):
    prompt = f"""
You are a data analyst assistant.

Your task is to understand the user's question and return ONLY valid JSON.
Do not include explanations, markdown, or extra text.

The system can answer these categories:
1. descriptive_statistics
2. trends_time_analysis
3. comparisons
4. statistical_insights
5. data_quality_checks
6. visualization_requests

For visualization requests, ALWAYS return JSON in this EXACT format:

{{
  "category": "visualization_requests",
  "chart_type": "bar | line | histogram | scatter | pie",
  "x_column": "<column_name_or_null>",
  "y_column": "<column_name_or_null>",
  "n": <number_or_null>
}}

Rules:
- Use only column names provided.
- If user asks for multiple charts, return ONE best chart.
- If unsure, choose a bar chart.
- Never change key names.
- Never add extra keys.

User question:
{question}

Available columns:
{columns}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df_cleaned = df.copy()

    df_cleaned.columns = (
    df_cleaned.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    )

    for col in df_cleaned.select_dtypes(include="number").columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    for col in df_cleaned.select_dtypes(include="object").columns:
        df_cleaned[col] = df_cleaned[col].fillna("Unknown")

    st.success("Dataset uploaded successfully!")

    # Preview
    if section == "Data Upload & Cleaning":

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        st.write("Shape of dataset:", df.shape)
        st.write("Column names:", df.columns.tolist())
        st.write("Data Types")
        st.write(df.dtypes)

        st.subheader("Data Cleaning")
        st.write("Missing values (Before cleaning)")
        st.write(df.isnull().sum())

        # ✅ DEFINE HERE
        df_cleaned = df.copy()

        df_cleaned.columns = (
            df_cleaned.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        for col in df_cleaned.select_dtypes(include="number").columns:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        for col in df_cleaned.select_dtypes(include="object").columns:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

        if st.checkbox("Apply Data Cleaning"):
            df = df_cleaned
            st.success("Data cleaning applied successfully!")

            st.write("Missing Values (After Cleaning)")
            st.write(df.isnull().sum())

            st.subheader("Cleaned Dataset Preview")
            st.dataframe(df.head())


    elif section == "EDA":

        st.subheader("Exploratory Data Analysis (EDA)")

        st.write("Summary Statistics")
        st.write(df.describe())

        st.write("Missing Values Report")
        st.write(df.isnull().sum())

    elif section == "Manual Visualizations":

        st.subheader("Manual Visualizations")

        numeric_df = df.select_dtypes(include="number")

        if not numeric_df.empty:
            st.write("Correlation Matrix")
            corr = numeric_df.corr()

            fig, ax = plt.subplots()
            ax.imshow(corr)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

        else:
            st.warning("No numeric columns available for correlation analysis.")

        st.subheader("Histogram (Numeric Columns)")

        num_cols = df.select_dtypes(include="number").columns.tolist()

        if num_cols:
            col = st.selectbox("Select numeric column", num_cols)

            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20)
            ax.set_title(f"Distribution of {col}")

            st.pyplot(fig)
        else:
            st.warning("No numeric columns available")
        
        st.subheader("Bar Chart (Categorical Columns)")

        if df is not None:

            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if len(cat_cols) == 0:
                st.warning("No categorical columns found for bar chart.")
            else:
                col = st.selectbox("Select categorical column", cat_cols)

                value_counts = df[col].value_counts().head(10)

                fig, ax = plt.subplots()
                ax.bar(value_counts.index.astype(str), value_counts.values)
                ax.set_title(f"Top 10 Categories in {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)

                st.pyplot(fig)
        
        st.subheader("Line Chart (Date Columns)")

        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        if date_cols:
            col = st.selectbox("Select date column", date_cols)
        
            df_sorted = df.sort_values(by=col)
        
            fig, ax = plt.subplots()
            ax.plot(df_sorted[col], range(len(df_sorted)))
            ax.set_title(f"Trend based on {col}")
        
            st.pyplot(fig)
        else:
            st.info("No datetime columns detected")

    elif section == "Analytics":

        st.subheader("Basic Analytics")

        analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Average of a column",
            "Sum of a column",
            "Maximum of a column",
            "Group by (Category → Numeric)"
        ],
        key="analysis_type"
        )

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        if analysis_type == "Average of a column" and numeric_cols:
            col = st.selectbox(
            "Select numeric column",
            numeric_cols,
            key="avg_col"
            )
            st.write("Average:", df[col].mean())

        elif analysis_type == "Sum of a column" and numeric_cols:
            col = st.selectbox(
            "Select numeric column",
            numeric_cols,
            key="sum_col"
            )
            st.write("Sum:", df[col].sum())

        elif analysis_type == "Maximum of a column" and numeric_cols:
            col = st.selectbox(
            "Select numeric column",
            numeric_cols,
            key="max_col"
            )
            st.write("Maximum:", df[col].max())

        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            st.warning("Not enough categorical or numeric columns for groupby.")
        else:
            cat_col = st.selectbox(
                "Select category column",
                categorical_cols,
                key="group_cat"
            )

            num_col = st.selectbox(
                "Select numeric column",
                numeric_cols,
                key="group_num"
            )

            result = df.groupby(cat_col)[num_col].mean().reset_index()
            st.write(result)
    

    elif section == "Groq AI Visualization":
        st.subheader("Chat with Data (Groq AI)")

        user_question = st.text_input("Ask a question about your data")

        if user_question:
            raw_response = get_visualization_intent(
                user_question,
                df.columns.tolist()
            )

            try:
                response = json.loads(raw_response)
            except json.JSONDecodeError:
                st.error("Groq returned invalid JSON")
                st.stop()

            st.write("Groq response:", response)

            category = response.get("category")

# ---------------- TEXT / NUMERIC ANSWERS ----------------
            if category != "visualization_requests":
                if "result" in response:
                    st.success(f"Answer: {response['result']}")
                else:
                    st.info(response)
                st.stop()

            # ---------------- VISUALIZATION LOGIC ----------------
            chart_type = response.get("chart_type")
            x_col = response.get("x_column")
            y_col = response.get("y_column")
            n = response.get("n")

            # HISTOGRAM
            if chart_type == "histogram" and x_col:
                fig, ax = plt.subplots()
                ax.hist(df[x_col], bins=20)
                ax.set_title(f"Distribution of {x_col}")
                st.pyplot(fig)

            # BAR
            elif chart_type == "bar" and x_col:
                vc = df[x_col].value_counts()
                if n:
                    vc = vc.head(n)

                fig, ax = plt.subplots()
                ax.bar(vc.index.astype(str), vc.values)
                ax.set_title(f"Bar Chart of {x_col}")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)

            # LINE
            elif chart_type == "line" and x_col and y_col:
                fig, ax = plt.subplots()
                ax.plot(df[x_col], df[y_col])
                ax.set_title(f"{y_col} vs {x_col}")
                st.pyplot(fig)

            # SCATTER
            elif chart_type == "scatter" and x_col and y_col:
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)

            else:
                st.warning("Unable to generate visualization from Groq response.")


