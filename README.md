# Autonomous AI Data Scientist 🤖

This project is a full-stack web application featuring an interactive AI agent that acts as an autonomous data scientist. The agent can converse with a user, ingest any CSV or Excel dataset, perform a comprehensive analysis, and generate a choice of professional deliverables: a detailed Jupyter Notebook or a beautiful interactive dashboard.



## Features

* **Interactive Conversation:** The agent guides the user through the analysis process via a polished chat interface.
* **Accepts Any Dataset:** The user can upload any CSV or Excel file for analysis.
* **Intelligent Clarification:** The agent analyzes the data's schema and asks clarifying questions to determine the user's analytical goals.
* **Dual Outputs:** Generates a choice of two professional deliverables:
    * A detailed **Jupyter Notebook** for technical deep-dives.
    * A beautiful **Interactive Dashboard** for visual insights.
* **Full-Stack Architecture:** Built with a robust Python backend and a modern React frontend.

## Tech Stack

* **Backend:** FastAPI, Uvicorn, Pandas, Scikit-learn, Plotly, Seaborn
* **Frontend:** React, TailwindCSS, React Icons
* **AI Brain:** The backend is designed to be controlled by an AI agent (prototyped with LangChain and the Google Gemini API).

## Setup and Installation

This is a full-stack application with a separate backend and frontend.

### Backend

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Install dependencies (it's a good practice to create a `requirements.txt` file first with `pip freeze > requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the server:
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be running at `http://localhost:8000`.

### Frontend

1.  In a **new terminal**, navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the application:
    ```bash
    npm start
    ```
    The frontend will be available at `http://localhost:3000`.

## How to Use

1.  Ensure both the backend and frontend servers are running in separate terminals.
2.  Open your browser to `http://localhost:3000`.
3.  Follow the agent's prompts to upload a dataset, clarify your target, and choose your desired output.
