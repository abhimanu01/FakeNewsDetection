from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle

app = FastAPI()

# Input model
class News(BaseModel):
    text: str

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔥 HOME PAGE (Frontend UI)
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Fake News Detection</title>
        <style>
            body {
                font-family: Arial;
                text-align: center;
                background: #f4f6f8;
                margin-top: 50px;
            }
            textarea {
                width: 60%;
                height: 120px;
                padding: 10px;
                font-size: 16px;
                border-radius: 10px;
                border: 1px solid #ccc;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result {
                margin-top: 20px;
                font-size: 22px;
                font-weight: bold;
            }
        </style>
    </head>

    <body>

     <h1>📰 Fake News Detection System</h1>

        <textarea id="newsText" placeholder="Enter news here..."></textarea><br><br>

        <button onclick="predict()">Check News</button>

        <div id="result"></div>

        <script>
            async function predict() {
                let text = document.getElementById("newsText").value;

                let response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ text: text })
                });

                let data = await response.json();

                let resultDiv = document.getElementById("result");

                if (data.result === "REAL") {
                    resultDiv.innerHTML = "✅ REAL NEWS";
                    resultDiv.style.color = "green";
                } else {
                    resultDiv.innerHTML = "❌ FAKE NEWS";
                    resultDiv.style.color = "red";
                }
            }
        </script>

    </body>
    </html>
    """

# 🔥 PREDICT API
@app.post("/predict")
def predict(news: News):
    data = vectorizer.transform([news.text])
    prediction = model.predict(data)
    return {"result": prediction[0]}