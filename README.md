
# Forest Cover Classification
Implemented **Machine Learning** pipelines from scratch along with `gonum` (similar to NumPy in Python), to classify forest cover types.



## Installation

Install `go` and `gonum`.

    $ git clone https://github.com/mpack2018/ForestCoverPrediction.git



## Usage

    $ go main.go
    

## Description of files
under `ml` directory:

- DataUtils.go				: Splitting the dataset
- Preprocessors.go		: Preprocess the dataset 
- Transformers.go			: Normalize and detect outliers
- Models.go					: Implement SGD and Logistic Regression from the scratch
- Evaluators.go				: Accuracy matrix to evaluate the performance
- RunExperiments.go	: Run my implementations along with the goml library and compare the performance.

## Result

|     | GoML Training | GoML Testing | My Implementation Training | My Implementation Testing |
|-----|---------------|--------------|----------------------------|---------------------------|
| SGD | 0.6010        | 0.5847       | 0.1642                     | 0.1555                    |
| LR  | 0.6010        | 0.5849       | 0.6018                     | 0.5863                    |
| KNN | 0.1471        | 0.1329       | -                          | -                         |


## Future Work
Find appropriate values for the tuning parameters.
Improve SGD.
Implement KNN.