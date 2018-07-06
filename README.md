
# Forest Cover Classification
Implemented **Machine Learning** pipelines from scratch along with `gonum` (similar to NumPy in Python), to classify forest cover types.



## Installation

Install `go` and `gonum`.

    $ git clone https://github.com/mpack2018/ForestCoverPrediction.git



## Tests

    $ go main.go
    

## Feature
under `ml` directory:

- DataUtils.go				: Splitting the dataset
- Preprocessors.go		: Preprocess the dataset 
- Transformers.go			: Normalize and detect outliers
- Models.go					: Implement SGD and Logistic Regression from the scratch
- Evaluators.go				: Accuracy matrix to evaluate the performance
- RunExperiments.go	: Run my implementations along with the goml library and compare the performance.

## Future Work
Find appropriate values for the tuning parameters.
Improve KNN.