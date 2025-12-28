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
You are an expert data analyst and data visualization assistant.

You are given a pandas DataFrame with the following columns:
{columns}

The user will ask a question in natural language about the data.

User Question:
"{question}"

Your tasks are:

1. Determine whether the user's question is CLEAR or UNCLEAR.

A question is UNCLEAR if:
- It does not mention any column name directly or indirectly, AND
- It does not specify a measurable intent such as count, average, sum, distribution,
  comparison, relationship, or trend.

If the question is UNCLEAR, DO NOT guess or assume.
Return the special JSON response described below.

2. If the question is CLEAR:
   - Identify the analytical intent:
     (distribution, comparison, aggregation, trend, relationship)
   - Identify the relevant column(s)
   - Determine the required pandas operation
   - Select the most appropriate visualization

3. Visualization selection rules:
   - histogram → distribution of a single numerical column
   - bar → categorical comparison or aggregated results
   - line → trends over time or ordered data
   - scatter → relationship between two numerical columns
   - boxplot → numerical distribution with outliers
   - pie → part-to-whole comparison with few categories

4. Ensure the logic is safe for pandas execution.
   Do NOT generate raw Python code.

5. Provide a simple, student-friendly explanation.

OUTPUT FORMAT (STRICT – JSON ONLY):

If the question is UNCLEAR, return:
{{
  "intent": "unclear",
  "columns_used": [],
  "pandas_operation": "none",
  "chart_type": "none",
  "explanation": "The question is too vague. Please specify what you want to analyze, for example: count of books by author or average price per category."
}}

If the question is CLEAR, return:
{{
  "intent": "<distribution | comparison | aggregation | trend | relationship>",
  "columns_used": ["<column1>", "<column2 if applicable>"],
  "pandas_operation": "<high-level pandas logic>",
  "chart_type": "<histogram | bar | line | scatter | boxplot | pie>",
  "explanation": "<simple explanation>"
}}

Do not include any extra text outside the JSON response.
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

            chart_type = response["chart_type"].lower().strip()

            # UNCLEAR CASE
            if chart_type == "none":
                st.warning(response["explanation"])
                st.stop()

            columns_used = response["columns_used"]

            # HISTOGRAM
            if chart_type == "histogram":
                col = columns_used[0]

                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=[col])

                fig, ax = plt.subplots()
                ax.hist(df[col], bins=20)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

            # BAR CHART
            elif chart_type == "bar":
                x = columns_used[0]

                value_counts = df[x].value_counts()

                fig, ax = plt.subplots()
                ax.bar(value_counts.index.astype(str), value_counts.values)
                ax.set_title(f"Bar Chart of {x}")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)

            # SCATTER
            elif chart_type == "scatter":
                x, y = columns_used

                df[x] = pd.to_numeric(df[x], errors="coerce")
                df[y] = pd.to_numeric(df[y], errors="coerce")
                df = df.dropna(subset=[x, y])

                fig, ax = plt.subplots()
                ax.scatter(df[x], df[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"{y} vs {x}")
                st.pyplot(fig)

            # LINE
            elif chart_type == "line":
                x, y = columns_used

                fig, ax = plt.subplots()
                ax.plot(df[x], df[y])
                ax.set_title("Trend Analysis")
                st.pyplot(fig)

            else:
                st.warning("Groq could not decide a suitable chart.")

