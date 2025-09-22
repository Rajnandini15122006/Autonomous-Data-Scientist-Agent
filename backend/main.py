import os
import json
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- In-memory store for conversation state ---
shared_data_store = {}

# --- FastAPI App Initialization ---
app = FastAPI(title="Autonomous Data Scientist API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# --- Pydantic Models for API validation ---
class ChatRequest(BaseModel):
    message: str
    state: dict

# --- HELPER FUNCTIONS ---

def analyze_schema(df, dataset_path):
    """Analyzes the dataframe and returns a summary."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    schema_str = buffer.getvalue()
    categorical_info = [f"'{col}'" for col in df.columns if df[col].dtype in ['int64', 'object'] and df[col].nunique() < 15]
    shared_data_store['dataframe'] = df
    shared_data_store['dataset_path'] = dataset_path
    
    analysis_text = f"Analysis of <strong>{os.path.basename(dataset_path)}</strong> complete. I've identified <strong>{df.shape[0]}</strong> rows and <strong>{df.shape[1]}</strong> columns."
    question_text = f"I suggest one of these as a target to predict: <strong>{', '.join(categorical_info)}</strong>. <br><br>Which column would you like to analyze?"
    
    return {"analysis": analysis_text, "question": question_text}

# FIX: The dashboard is now smarter and adapts its charts to the data.
def create_dashboard():
    """Generates a beautiful and adaptive HTML dashboard."""
    try:
        df = shared_data_store['dataframe']
        target_column = shared_data_store['target_column']
        dataset_path = shared_data_store['dataset_path']
        
        # --- Create More Creative Plotly Figures ---
        kpi_total_rows = df.shape[0]
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        # --- Generate Key Insights ---
        insights = []
        insights.append(f"The dataset contains <strong>{kpi_total_rows:,}</strong> rows and <strong>{df.shape[1]}</strong> columns.")
        insights.append(f"The target variable, <strong>'{target_column}'</strong>, has <strong>{df[target_column].nunique()}</strong> unique classes.")
        if target_column in numeric_df.corr().columns and len(numeric_df.columns) > 1:
            top_corr = numeric_df.corr()[target_column].abs().sort_values(ascending=False).index[1]
            insights.append(f"The feature most correlated with '{target_column}' is <strong>'{top_corr}'</strong>.")

        # --- Create Adaptive Figures ---
        dist_fig = px.histogram(df, x=target_column, title=f'Distribution of Target: {target_column}', color=target_column, color_discrete_sequence=px.colors.qualitative.Prism)
        dist_fig.update_layout(showlegend=False, title_x=0.5)

        heatmap_html = ""
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Correlation Heatmap of Numeric Features', pad=20)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            heatmap_html = f'<img src="data:image/png;base64,{img_b64}" alt="Correlation Heatmap" class="w-full h-full object-contain">'
        else:
            heatmap_html = '<div class="text-center p-8"><h3 class="text-lg font-semibold text-gray-600">Not enough numeric data for a correlation heatmap.</h3></div>'
        
        # ADAPTIVE CHART: Choose Violin for large data, Box for small data.
        adaptive_fig = None
        if len(numeric_df.columns) > 0:
            key_numeric_1 = numeric_df.columns[0]
            chart_type = 'violin' if len(df) > 150 else 'box' # Heuristic
            
            if chart_type == 'violin':
                adaptive_fig = px.violin(df, x=target_column, y=key_numeric_1, color=target_column, 
                                   title=f'{key_numeric_1} Distribution by {target_column} (Violin Plot)',
                                   box=True, points="all",
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            else: # Box plot
                adaptive_fig = px.box(df, x=target_column, y=key_numeric_1, color=target_column, 
                               title=f'{key_numeric_1} Distribution by {target_column} (Box Plot)',
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            adaptive_fig.update_layout(showlegend=False, title_x=0.5)
        
        # --- Assemble a More Beautiful HTML Dashboard ---
        dashboard_filename = f"{os.path.basename(dataset_path).split('.')[0]}_dashboard_for_{target_column}.html"
        with open(dashboard_filename, 'w', encoding='utf-8') as f:
            f.write('<html><head><script src="https://cdn.tailwindcss.com"></script><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><title>AI Data Scientist Dashboard</title></head>')
            f.write('<body class="bg-slate-100 p-6 md:p-10 font-sans">')
            f.write('<div class="max-w-7xl mx-auto">')
            f.write(f'<h1 class="text-5xl font-bold text-slate-800 mb-2 tracking-tight">Data Science Insights</h1>')
            f.write(f'<h2 class="text-xl text-slate-600 mb-10">An Autonomous Analysis of <span class="font-semibold text-indigo-600">{os.path.basename(dataset_path)}</span> Targeting <span class="font-semibold text-indigo-600">{target_column}</span></h2>')
            
            # Key Insights Section
            f.write('<div class="bg-white p-6 rounded-xl shadow-lg mb-8 border-l-4 border-indigo-500">')
            f.write('<h3 class="text-2xl font-bold text-gray-800 mb-3">Key Insights</h3>')
            f.write('<ul class="list-disc list-inside text-gray-700 space-y-2">')
            for insight in insights:
                f.write(f'<li>{insight}</li>')
            f.write('</ul></div>')

            # Main Chart Grid
            f.write('<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">')
            if dist_fig:
                f.write(f'<div class="bg-white p-4 rounded-xl shadow-lg transition-all hover:shadow-2xl">{dist_fig.to_html(full_html=False, include_plotlyjs=False)}</div>')
            f.write(f'<div class="bg-white p-4 rounded-xl shadow-lg transition-all hover:shadow-2xl flex items-center justify-center">{heatmap_html}</div>')
            if adaptive_fig:
                f.write(f'<div class="lg:col-span-2 bg-white p-4 rounded-xl shadow-lg transition-all hover:shadow-2xl">{adaptive_fig.to_html(full_html=False, include_plotlyjs=False)}</div>')
            f.write('</div></div></body></html>')
            
        return dashboard_filename
    except Exception as e:
        print(f"Error in create_dashboard: {e}")
        return None

def create_notebook():
    """Generates the detailed Jupyter Notebook."""
    try:
        dataset_path = shared_data_store['dataset_path']
        target_column = shared_data_store['target_column']
        df = shared_data_store['dataframe']
        numerical_features = [col for col in df.columns if col != target_column and df[col].dtype in ['int64', 'float64'] and 'id' not in col.lower()]
        features_list_str = str(numerical_features)
        read_function = "pd.read_csv" if str(dataset_path).endswith('.csv') else "pd.read_excel"

        imports_code = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
print("Libraries imported successfully.")
"""
        load_and_clean_code = f"""
# Step 1: Load and Clean Data
print("Step 1: Loading and Cleaning Data...")
# Using raw string literal for path to handle backslashes correctly
df = {read_function}(r'{dataset_path}')
# General cleaning: fill numeric columns with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
# General cleaning: fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
# Convert binary categorical columns to numeric
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() == 2:
        df[col], _ = pd.factorize(df[col])
print("Data loaded and cleaned.")
df.head()
"""
        feature_engineering_code = """
# Step 2: Feature Engineering (Example for Titanic)
print("\\nStep 2: Engineering New Features...")
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    print("Created 'FamilySize' and 'IsAlone' features.")

if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {{"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)
    print("Created 'Title' feature.")
print("Feature engineering complete.")
"""
        model_evaluation_code = f"""
# Step 3: Compare Base Models
print("\\nStep 3: Comparing Base Models...")
features_list = {features_list_str}
if 'FamilySize' in df.columns: features_list.extend(['FamilySize', 'IsAlone', 'Title'])
target = '{target_column}'
X = df[[f for f in features_list if f in df.columns]]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training RandomForest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"RandomForest Base Accuracy: {{rf_accuracy:.4f}}")
print("Training LogisticRegression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_preds)
print(f"LogisticRegression Base Accuracy: {{lr_accuracy:.4f}}")
"""
        hyperparameter_tuning_code = """
# Step 4: Tune the Best Model (assuming RandomForest)
print("\\nStep 4: Tuning RandomForest Hyperparameters...")
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2, 4]}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y)
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")
print(f"Best cross-validated accuracy from tuning: {grid_search.best_score_:.4f}")
"""
        visualization_code = f"""
# Step 5: Generate Final Visualization and Key Insights
print("\\nStep 5: Generating Final Visualization and Insights...")
final_model = RandomForestClassifier(random_state=42, **best_params)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"\\n--- Key Insights ---")
print(f"The final tuned RandomForest model achieved an accuracy of {{final_accuracy:.4f}} on the test set.")
print("This is the best performing model found for the '{target_column}' target.")
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Final Tuned Model')
plt.show()
print("\\nAnalysis complete.")
"""
        notebook_filename = f"{os.path.basename(dataset_path).split('.')[0]}notebook_for{target_column}.ipynb"
        notebook_content = {
             "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": f"# Autonomous Analysis for {dataset_path}\n## Predicting '{target_column}'"},
                {"cell_type": "code", "source": imports_code, "execution_count": None, "outputs": [], "metadata": {}},
                {"cell_type": "code", "source": load_and_clean_code, "execution_count": None, "outputs": [], "metadata": {}},
                {"cell_type": "code", "source": feature_engineering_code, "execution_count": None, "outputs": [], "metadata": {}},
                {"cell_type": "code", "source": model_evaluation_code, "execution_count": None, "outputs": [], "metadata": {}},
                {"cell_type": "code", "source": hyperparameter_tuning_code, "execution_count": None, "outputs": [], "metadata": {}},
                {"cell_type": "code", "source": visualization_code, "execution_count": None, "outputs": [], "metadata": {}},
             ],
             "metadata": {"language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 4
        }
        with open(notebook_filename, 'w') as f:
            json.dump(notebook_content, f, indent=4)
        return notebook_filename
    except Exception as e:
        print(f"Error in create_notebook: {e}")
        return None

# --- API ENDPOINTS ---

@app.post("/start_analysis")
async def start_analysis(file: UploadFile = File(...)):
    """Endpoint to upload a dataset and start the analysis."""
    try:
        dataset_path = f"temp_{file.filename}"
        with open(dataset_path, "wb") as buffer:
            buffer.write(await file.read())
        df = pd.read_csv(dataset_path) if str(file.filename).endswith('.csv') else pd.read_excel(dataset_path)
        analysis_result = analyze_schema(df, dataset_path)
        return JSONResponse(content={"agent_messages": [analysis_result['analysis'], analysis_result['question']], "next_state": "AWAITING_TARGET"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/continue_conversation")
async def continue_conversation(request: ChatRequest):
    """Endpoint to handle the ongoing conversation with the agent."""
    user_message = request.message
    state = request.state
    if state.get("next_state") == "AWAITING_TARGET":
        shared_data_store['target_column'] = user_message
        return JSONResponse(content={"agent_messages": ["Excellent. I have everything I need. What would you like me to generate?"], "next_state": "AWAITING_OUTPUT_CHOICE", "choices": ["dashboard", "notebook"]})
    return JSONResponse(status_code=400, content={"error": "Unknown conversation state."})

@app.post("/generate_output")
async def generate_output(request: ChatRequest):
    """Endpoint to generate the final deliverable (notebook or dashboard)."""
    output_type = request.message
    if output_type == 'dashboard': file_path = create_dashboard()
    elif output_type == 'notebook': file_path = create_notebook()
    else: return JSONResponse(status_code=400, content={"error": "Invalid output type"})
    if file_path and os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/octet-stream', filename=os.path.basename(file_path))
    else:
        return JSONResponse(status_code=500, content={"error": "Failed to generate the output file."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)