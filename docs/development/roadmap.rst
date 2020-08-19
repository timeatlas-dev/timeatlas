Roadmap
=======

2020 - v0.1 - Get Started
-------------------------

- Provide a set of data structures specialised for time series uni and multi dimensional
- Define a standard csv export
- Provide a set of methods and plots to handle, process and explore time series dataset.
    - The tool should be generic relative to BBData. Therefore, if there is a need of some methods to make between BBData and TimeAtlas, they should be in BBData Python wrapper.
    - Time series auto labelling from Lorenz
    - Plots to display value on maps if localisation data is available
- Provide a set of models allowing for prediction, classification, clustering and other machine learning tasks on time series
    - Define standard methods for time series predictions
- Provide a set of methods and plots to validate the quality of a model
- Provide a set of models allowing for anomaly detections


2021 - v1.0 - Stabilize and Improve
-----------------------------------

- Provide a way to publish models e.g. on Kubernetes, using data from BBData to get an API giving you the prediction.
    - Allow for retraining of the model from the API