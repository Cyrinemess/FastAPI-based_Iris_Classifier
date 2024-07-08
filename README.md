# Deployment of a FastAPI-based Iris Classifier
## Project Description

ML-based IRIS Flower Classifier that predicts the species of iris flowers based on their sepal and petal measurements.

The model is trained using the popular Iris dataset, consisting of 150 observations. 

![iris](https://github.com/Cyrinemess/iris-classifier-fastapi-docker/assets/102186127/a01c95a3-7cc9-4654-9acc-a2ff2692d982)
* Wrapped using **FastAPI**.
* Containerized using **Docker**.

## API Endpoints
### Root Endpoint
* URL: /
* Method: GET

* Description: Checks if the API is running.
* Response: `{
  "status": "Working!"
}`
### Predict Endpoint
* URL: /predict

* Method: POST

* Description: Predicts the species of an iris flower based on input measurements.

* Request Body: `{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}`
* Response:
`{
  "iris_species": "setosa"
}`
