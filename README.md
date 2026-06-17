# Fake News Detection API using FastAPI & Machine Learning

A REST API built with **FastAPI** and **Scikit-learn** that predicts whether a news article is **Real** or **Fake** using a trained machine learning model. The project also stores prediction history using **SQLite** and **SQLAlchemy**.

## Features

* Detects fake and real news from text input.
* FastAPI-powered REST API with interactive Swagger documentation.
* TF-IDF vectorization with a trained ML model.
* Stores prediction history in SQLite.
* CRUD operations for viewing and deleting records.
* Modular project structure for easy maintenance.

## Tech Stack

* Python
* FastAPI
* Scikit-learn
* Pandas & NumPy
* Joblib
* SQLAlchemy
* SQLite
* Uvicorn

## Project Structure

```text
fake-news-api/
│── app.py
│── train_model.py
│── model.py
│── crud.py
│── database.py
│── models.py
│── schemas.py
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
```

## Run the API

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Example Request

```json
{
  "text": "Government announces new education policy today."
}
```

## Example Response

```json
{
  "prediction": 0,
  "result": "Real News ✅"
}
```

## Author

**Mahi** – B.Tech CSE Student 
