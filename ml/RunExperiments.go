package ml

import (
	"fmt"
	"os"
	"gonum.org/v1/gonum/mat"
	l "github.com/cdipaolo/goml/linear"
	km "github.com/cdipaolo/goml/cluster"
	"github.com/cdipaolo/goml/base"
)

/**
This section is to provide functions for running the models from goml as well as my implementation

(1) LR represents Logistic Regression
(1.1) RunExperimentLR
(1.2) RunExperimentGoMLLR

(2) KNN: K Nearest Neighbors
(2.1) -
(2.2) RunExperimentGoMLKNN

(3) SGD: Stochastic Gradient Descent
(3.1) RunExperimentSGD
(3.2) RunExperimentGoMLSGD
 */

func RunExperimentLR(trainMatrix *mat.Dense, trainLabel *mat.Dense,
	testMatrix *mat.Dense, testLabel *mat.Dense, lRate float64,
	regCoeff float64, maxIters int) {
	fmt.Println("#################")
	fmt.Println("Experiment: Computing results of Logistic Regression on extracted features")

	// Data Setup
	_, numFeatures := trainMatrix.Dims()

	// Initialize KNN Classifier
	fmt.Println("initializing classifier...")
	logisticRegressor := CreateLogisticR(7, numFeatures, lRate, regCoeff)

	// Training
	fmt.Println("fitting the classifier...")
	logisticRegressor.Fit(*trainMatrix, *trainLabel, maxIters)


	// Training set predictions
	fmt.Println("obtaining training set predictions...")
	trainPredictions := logisticRegressor.Predict(*trainMatrix)

	// Testing set predictions
	fmt.Println("obtaining test set predictions...")
	testPredictions := logisticRegressor.Predict(*testMatrix)

	// metrics
	fmt.Println("Result Metrics:")
	// accuracy
	fmt.Println("Training Accuracy: ", GetAccuracyMatrix(*trainLabel, *trainPredictions))
	fmt.Println("Testing Accuracy: ", GetAccuracyMatrix(*testLabel, *testPredictions))
	fmt.Println("#################")
	fmt.Println()
}


func RunExperimentGoMLLR(trainMatrix *mat.Dense, trainLabel *mat.Dense,
	testMatrix *mat.Dense, testLabel *mat.Dense, lRate float64,
	regCoeff float64, maxIters int) {
	fmt.Println("#################")
	fmt.Println("Experiment: Computing results of GoML Logistic Regression on extracted features")
	// Data Setup
	trainSlice := MatrixTo2DSlice(trainMatrix)
	trainLabelSlice := MatrixToSlice(trainLabel)
	testSlice := MatrixTo2DSlice(testMatrix)
	testLabelSlice := MatrixToSlice(testLabel)
	predictionsTrain := make([]float64, len(trainLabelSlice))
	predictionsTest := make([]float64, len(testLabelSlice))

	// Initialize Logistic Regression Classifier
	fmt.Println("initializing classifier...")
	softmax := l.NewSoftmax(base.BatchGA, lRate, regCoeff, 7, maxIters, trainSlice, trainLabelSlice)

	// Training
	fmt.Println("fitting the classifier...")
	err := softmax.Learn()
	if err != nil {
		fmt.Println("Error: fail to learn with Softmax")
		os.Exit(1)
	}

	// Training set predictions
	fmt.Println("obtaining training set predictions...")
	for i, trainInstance := range trainSlice {
		prediction, err := softmax.Predict(trainInstance)
		if err != nil {
			fmt.Println("Error: fail to predict with Softmax")
			os.Exit(1)
		}
		predictedLabel := FindArgMax(prediction)
		predictionsTrain[i] = predictedLabel
	}

	// Testing set predictions
	fmt.Println("obtaining test set predictions...")
	for i, testInstance := range testSlice {
		prediction, err := softmax.Predict(testInstance)
		if err != nil {
			fmt.Println("Error: fail to predict with Softmax")
			os.Exit(1)
		}
		predictedLabel := FindArgMax(prediction)
		predictionsTest[i] = predictedLabel
	}

	// metrics
	fmt.Println("Result Metrics:")
	// accuracy
	fmt.Println("Training Accuracy: ", GetAccuracy(trainLabelSlice, predictionsTrain))
	fmt.Println("Testing Accuracy: ", GetAccuracy(testLabelSlice, predictionsTest))
	fmt.Println("#################")
	fmt.Println()
}

func RunExperimentGoMLKNN(trainMatrix *mat.Dense, trainLabel *mat.Dense, testMatrix *mat.Dense, testLabel *mat.Dense) {
	fmt.Println("#################")
	fmt.Println("Experiment: Computing results of GoML KNN on extracted features")
	// Data Setup
	trainSlice := MatrixTo2DSlice(trainMatrix)
	trainLabelSlice := MatrixToSlice(trainLabel)
	testSlice := MatrixTo2DSlice(testMatrix)
	testLabelSlice := MatrixToSlice(testLabel)
	predictionsTrain := make([]float64, len(trainLabelSlice))
	predictionsTest := make([]float64, len(testLabelSlice))

	// Initialize KNN Classifier
	fmt.Println("initializing classifier...")
	knn := km.NewKNN(16, trainSlice, trainLabelSlice, base.ManhattanDistance) // TODO: k

	// Training set predictions
	fmt.Println("obtaining training set predictions...")
	for i, trainInstance := range trainSlice {
		prediction, err := knn.Predict(trainInstance)
		if err != nil {
			fmt.Println("Error: fail to predict with Softmax")
			os.Exit(1)
		}
		predictedLabel := FindArgMax(prediction)
		predictionsTrain[i] = predictedLabel
	}

	// Testing set predictions
	fmt.Println("obtaining test set predictions...")
	for i, testInstance := range testSlice {
		prediction, err := knn.Predict(testInstance)
		if err != nil {
			fmt.Println("Error: fail to predict with Softmax")
			os.Exit(1)
		}
		predictedLabel := FindArgMax(prediction)
		predictionsTest[i] = predictedLabel
	}

	// metrics
	fmt.Println("Result Metrics:")
	// accuracy
	fmt.Println("Training Accuracy: ", GetAccuracy(trainLabelSlice, predictionsTrain))
	fmt.Println("Testing Accuracy: ", GetAccuracy(testLabelSlice, predictionsTest))
	fmt.Println("#################")
	fmt.Println()
}


// TODO: learning rate
// CreateSGD(labelNum, featureNum int, learningRate float64, regularizationCoeff float64, batchSize int)
func RunExperimentSGD(trainMatrix *mat.Dense, trainLabel *mat.Dense,
	testMatrix *mat.Dense, testLabel *mat.Dense, lRate float64,
	regCoeff float64, maxIters int, batchSize int) {
	fmt.Println("#################")
	fmt.Println("Experiment: Computing results of Stochastic Gradient Descent on extracted features")

	// Data Setup
	_, numFeatures := trainMatrix.Dims()

	// Initialize SGD  // TODO can I say SGD classifier?
	fmt.Println("initializing classifier...")
	//logisticRegressor := p.CreateLogisticR(7, numFeatures, lRate, regCoeff)
	SGD := CreateSGD(7, numFeatures, lRate, regCoeff, batchSize)

	// Training
	fmt.Println("fitting the classifier...")
	SGD.Fit(*trainMatrix, *trainLabel, maxIters)

	// Training set predictions
	fmt.Println("obtaining training set predictions...")
	trainPredictions := SGD.Predict(*trainMatrix)

	// Testing set predictions
	fmt.Println("obtaining test set predictions...")
	testPredictions := SGD.Predict(*testMatrix)

	// metrics
	fmt.Println("Result Metrics:")
	// accuracy
	fmt.Println("Training Accuracy: ", GetAccuracyMatrix(*trainLabel, *trainPredictions))
	fmt.Println("Testing Accuracy: ", GetAccuracyMatrix(*testLabel, *testPredictions))
	fmt.Println("#################")
	fmt.Println()
}



func RunExperimentGoMLSGD(trainMatrix *mat.Dense, trainLabel *mat.Dense,
	testMatrix *mat.Dense, testLabel *mat.Dense, lRate float64,
	regCoeff float64, maxIters int) {
	fmt.Println("#################")
	fmt.Println("Experiment: Computing results of GoML SGD on extracted features")
	// Data Setup
	trainSlice := MatrixTo2DSlice(trainMatrix)
	trainLabelSlice := MatrixToSlice(trainLabel)
	testSlice := MatrixTo2DSlice(testMatrix)
	testLabelSlice := MatrixToSlice(testLabel)
	predictionsTrain := make([]float64, len(trainLabelSlice))
	predictionsTest := make([]float64, len(testLabelSlice))

	// Initialize KNN Classifier
	fmt.Println("initializing classifier...")
	softmax := l.NewSoftmax(base.StochasticGA, lRate, regCoeff, 7, maxIters, trainSlice, trainLabelSlice)

	// Training
	fmt.Println("fitting the classifier...")
	err := softmax.Learn()
	if err != nil {
		fmt.Println("Error: fail to learn with Softmax")
		os.Exit(1)
	}

	// Training set predictions
	fmt.Println("obtaining training set predictions...")
	for i, trainInstance := range trainSlice {
		prediction, err := softmax.Predict(trainInstance)
		if err != nil {
			fmt.Println("Error: fail to predict with Softmax")
			os.Exit(1)
		}
		predictedLabel := FindArgMax(prediction)
		predictionsTrain[i] = predictedLabel
	}

	// Testing set predictions
	fmt.Println("obtaining test set predictions...")
	for i, testInstance := range testSlice {
		prediction, err := softmax.Predict(testInstance)
		if err != nil {
			fmt.Println("Error: fail to predict with Softmax")
			os.Exit(1)
		}
		predictedLabel := FindArgMax(prediction)
		predictionsTest[i] = predictedLabel
	}

	// metrics
	fmt.Println("Result Metrics:")
	// accuracy
	fmt.Println("Training Accuracy: ", GetAccuracy(trainLabelSlice, predictionsTrain))
	fmt.Println("Testing Accuracy: ", GetAccuracy(testLabelSlice, predictionsTest))
	fmt.Println("#################")
	fmt.Println()
}
