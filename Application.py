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
        "Manual Analytics",
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

        st.subheader("Column-wise Analysis")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        if numeric_cols:
            st.write("Numeric Columns Summary")
            for col in numeric_cols:
                st.write(f"🔹 {col}")
                st.write({
                    "Mean": df[col].mean(),
                    "Median": df[col].median(),
                    "Min": df[col].min(),
                    "Max": df[col].max(),
                    "Missing Values": df[col].isnull().sum()
                })

        if categorical_cols:
            st.write("Categorical Columns Summary")
            for col in categorical_cols:
                st.write(f"🔹 {col}")
                st.write(df[col].value_counts().head(5))
    
    elif section == "Manual Analytics":
        st.subheader("Manual Data Analytics")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        analysis_type = st.selectbox(
        "Select Analysis",
        [
            "Mean",
            "Sum",
            "Min",
            "Max",
            "Group By Analysis"
        ]
        )

        if analysis_type in ["Mean", "Sum", "Min", "Max"] and numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)

            if analysis_type == "Mean":
                st.success(f"Mean of {col}: {df[col].mean()}")

            elif analysis_type == "Sum":
                st.success(f"Sum of {col}: {df[col].sum()}")

            elif analysis_type == "Min":
                st.success(f"Minimum of {col}: {df[col].min()}")

            elif analysis_type == "Max":
                st.success(f"Maximum of {col}: {df[col].max()}")

        elif analysis_type == "Group By Analysis":
            if categorical_cols and numeric_cols:
                cat_col = st.selectbox("Select category column", categorical_cols)
                num_col = st.selectbox("Select numeric column", numeric_cols)

                grouped = df.groupby(cat_col)[num_col].mean().reset_index()
                st.write("Grouped Result (Mean):")
                st.dataframe(grouped)
            else:
                st.warning("Not enough categorical or numeric columns")

            st.subheader("Trend Analysis (Time-based)")

            date_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
            for col in df.select_dtypes(include="object").columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass

            if date_cols:
                date_col = st.selectbox("Select Date Column", list(set(date_cols)))

                numeric_cols = df.select_dtypes(include="number").columns.tolist()

                if numeric_cols:
                    value_col = st.selectbox("Select Numeric Column", numeric_cols)

                    df_sorted = df.sort_values(by=date_col)

                    fig, ax = plt.subplots()
                    ax.plot(df_sorted[date_col], df_sorted[value_col])
                    ax.set_title(f"Trend of {value_col} over time")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(value_col)

                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns available")
            else:
                st.info("No date columns detected")

            st.subheader("Outlier Detection (IQR Method)")

            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

                st.write(f"{col}: {len(outliers)} potential outliers")

    elif section == "Manual Visualizations":

        st.subheader("Manual Visualizations")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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
        
        st.subheader("Advanced Histogram Analysis")

        num_cols = df.select_dtypes(include="number").columns.tolist()

        if num_cols:
            col = st.selectbox("Select numeric column", num_cols, key="adv_hist_col")
            bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)

            fig, ax = plt.subplots()
            ax.hist(df[col], bins=bins)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

            st.write("Mean:", df[col].mean())
            st.write("Median:", df[col].median())
            st.write("Standard Deviation:", df[col].std())
        else:
            st.warning("No numeric columns available.")

        st.subheader("Scatter Plot (Relationship Analysis)")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")

            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col}")

            st.pyplot(fig)

            st.write("Correlation:", df[[x_col, y_col]].corr().iloc[0,1])
        else:
            st.warning("Need at least two numeric columns")


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

        st.subheader("Box Plot (Outlier Detection)")

        num_cols = df.select_dtypes(include="number").columns.tolist()

        if num_cols:
            col = st.selectbox("Select numeric column", num_cols, key="boxplot")

            fig, ax = plt.subplots()
            ax.boxplot(df[col].dropna(), vert=True)
            ax.set_title(f"Box Plot of {col}")
            ax.set_ylabel(col)

            st.pyplot(fig)

            st.write("Q1:", df[col].quantile(0.25))
            st.write("Median:", df[col].median())
            st.write("Q3:", df[col].quantile(0.75))
        else:
            st.warning("No numeric columns available")

        
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
        st.subheader("Descriptive Analytics")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox(
                "Select a numeric column for analysis",
                numeric_cols,
                key="desc_col"
            )

            col_data = df[selected_col]

            st.write("### Key Statistics")
            st.write({
                "Mean": col_data.mean(),
                "Median": col_data.median(),
                "Minimum": col_data.min(),
                "Maximum": col_data.max(),
                "Standard Deviation": col_data.std(),
                "Variance": col_data.var(),
                "Count": col_data.count()
            })
        else:
            st.warning("No numeric columns available for descriptive analytics.")

        st.subheader("Top-N Analysis")

        n = st.slider("Select N", 1, 20, 5)
        col = st.selectbox("Select column", df.columns)

        top_n = df[col].value_counts().head(n)
        st.dataframe(top_n)

        st.subheader("Group By Analysis")

        cat_col = st.selectbox("Select category column", categorical_cols)
        num_col = st.selectbox("Select numeric column", numeric_cols)
        operation = st.selectbox("Operation", ["Mean", "Sum", "Max", "Min"])

        if operation == "Mean":
            result = df.groupby(cat_col)[num_col].mean()
        elif operation == "Sum":
            result = df.groupby(cat_col)[num_col].sum()
        elif operation == "Max":
            result = df.groupby(cat_col)[num_col].max()
        else:
            result = df.groupby(cat_col)[num_col].min()

        st.dataframe(result.reset_index())
    

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