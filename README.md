AI-Based Fake News Detection System

Setup Instructions (Windows CMD)

Open Command Prompt (cmd) and navigate to the project folder:
cd "C:\Users\Abhishek M\Downloads\FakeNewsDetection"

Create a Virtual Environment:
python -m venv env

Activate the Virtual Environment:
env\Scripts\activate

(You should see (env) appear at the start of your command line.)

Install Dependencies:
pip install pandas scikit-learn fastapi uvicorn pydantic

Or, if you have a requirements file:
pip install -r requirements.txt
Running the Application

Train the Machine Learning Model (First Time Only):
python train.py

This command will:
Load and preprocess the Indian Fake and Real News dataset.
Train the Logistic Regression model using TF-IDF features.
Generate the following files:
model.pkl
vectorizer.pkl

Start the FastAPI Backend Server:
uvicorn app:app --reload

You will see output similar to:

INFO: Uvicorn running on http://127.0.0.1:8000
Launch the Frontend:

Open your web browser (Chrome, Edge, etc.) and go to:

http://127.0.0.1:8000

The frontend interface is automatically served by the FastAPI backend, so no separate frontend server is required.

Usage
Home Page:

Enter a news article or headline into the text area.

Check News:

Click the "Check News" button to classify the news as:

✅ REAL NEWS
❌ FAKE NEWS
Example Inputs:

Real News Example:

Delhi government held a high-level meeting to discuss measures to control pollution in the capital.

Fake News Example:

A viral WhatsApp message claims that all bank accounts in India will be frozen tomorrow.
