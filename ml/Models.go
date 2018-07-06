package ml

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

/**
In this section,

we have my implementation of SGD and Logistic Regression(LR).
There are more functions, but I only list up main parts regarding SGD and LR
(1) SGD
(1.1) SGD structure
(1.2) SGD Constructor  -- CreateSGD()
(1.3) SGD Learner	    -- SGD.Fit()
(1.4) SGD Learn batch  -- SGD.FitBatch()
(1.5) SGD Predict		-- SGD.Predict()
(1.6) SGD PredictProb  -- SGD.PredictProb()

(2) Logistic Regression
(2.1) Logistic Regressor structure
(2.2) Logistic Constructor				-- CreateLR()
(2.3) Logistic Regression Learn		-- LR.Fit()
(2.4) Logistic Regression Predict		-- LR.Predict()
(2.5) Logistic Regression PredictProb -- LR.PredictProb()

 */


// SGD ==========================================================================================================
type SGD struct{
	numLabels int
	numFeatures int
	learningRate float64
	regCoeff float64
	weights *mat.Dense
	batchSize int
}

// Constructor for SGD
func CreateSGD(labelNum, featureNum int, learningRate float64, regularizationCoeff float64, batchSize int) *SGD {
	sgd := new(SGD)
	sgd.numLabels = labelNum
	sgd.numFeatures = featureNum
	sgd.learningRate = learningRate
	sgd.regCoeff = regularizationCoeff
	sgd.weights = mat.NewDense(sgd.numFeatures, sgd.numLabels, nil)
	sgd.batchSize = batchSize
	return sgd
}

func (sgd *SGD) Fit(trainX mat.Dense, trainY mat.Dense, numIter int) {
	xMatrix := mat.DenseCopyOf(&trainX) 						// numSample x numFeatures
	yMatrix:= OneHotEncodingLabels(&trainY, sgd.numLabels) 	// numSample x numLabels

	numSamples, numFeatures := xMatrix.Dims()

	numBatches := numSamples/sgd.batchSize
	for i:=0; i<numIter; i++ {
		// Shuffle X and Y
		xMatrix, yMatrix = ShuffleXyMatrix(*xMatrix, *yMatrix)

		for batch := 0; batch < numBatches; batch++ {
			batchBegin := (numSamples*batch)/numBatches
			batchEnd := int(math.Min(float64(numSamples*(batch + 1)/numBatches), float64(numSamples)))
			batchX := xMatrix.Slice(batchBegin, batchEnd, 0, numFeatures)
			batchY := yMatrix.Slice(batchBegin, batchEnd, 0, sgd.numLabels)
			sgd.FitBatch(*mat.DenseCopyOf(batchX), *mat.DenseCopyOf(batchY))
		}
	}
}

func (sgd *SGD) FitBatch(batchX mat.Dense, batchY mat.Dense) {
	batchXMatrix := mat.DenseCopyOf(&batchX) 						// numSample x numFeatures
	batchYMatrix:= OneHotEncodingLabels(&batchY, sgd.numLabels)	// numSample x numLabels

	numSamples, numFeatures := batchX.Dims()
	score := mat.NewDense(numSamples, sgd.numLabels, nil)
	sub := mat.NewDense(numSamples, sgd.numLabels, nil)
	mul := mat.NewDense(sgd.numFeatures, sgd.numLabels, nil)
	weightsUnreg := mat.NewDense(sgd.numFeatures, sgd.numLabels, nil)

	// 1. prob = SoftMax(X, w) for now, instead of softmax just mul
	score.Mul(batchXMatrix, sgd.weights)
	prob := SoftMax(*score)

	// 2. MatrixMultiplication(X.T, (y-prob)) * (1/(n-samples))
	sub.Sub(batchYMatrix, prob)
	mul.Mul(batchXMatrix.T(), sub)

	// 3. weight = mul * scalar
	// There is no way of doing scalar * matrix, so I created scalarMatrix
	scalarMatrix := CreateScalarMatrix(sgd.learningRate/float64(numSamples), numFeatures, sgd.numLabels)
	weightsUnreg.MulElem(mul, scalarMatrix)

	// 4. Regularization Coeff
	regMatrix := CreateScalarMatrix(2*sgd.regCoeff, numFeatures, sgd.numLabels)

	sgd.weights.Add(weightsUnreg, regMatrix)
}


func (sgd *SGD) Predict(testX mat.Dense) *mat.Dense {
	prob := sgd.PredictProb(testX) // n_test_samples x n_labels

	// pick up the highest value among each row that is the predicted label
	testPredictions := FindLabelFromSoftmax(*prob)
	return testPredictions
}

func (sgd *SGD) PredictProb(testX mat.Dense) *mat.Dense {
	numTestInstances, _ :=testX.Dims()
	score := mat.NewDense(numTestInstances, 7, nil) // 7 numLabels
	score.Mul(&testX, sgd.weights)
	prob := SoftMax(*score) 									// n_test_samples x n_labels
	return prob
}

func ShuffleXyMatrix(dataMatrix mat.Dense, labelMatrix mat.Dense) (*mat.Dense, *mat.Dense) {
	numRows, numCols := dataMatrix.Dims()
	_, numLabels := labelMatrix.Dims()
	shuffledDataMatrix := mat.NewDense(numRows, numCols, nil)
	shuffledLabelMatrix := mat.NewDense(numRows, numLabels, nil)

	shuffle:=rand.Perm(numRows)
	var shufDataRow []float64
	var shufLabelRow []float64
	for r, s:= range shuffle {
		shufDataRow = make([]float64, numCols)
		shufLabelRow = make([]float64, numLabels)
		mat.Row(shufDataRow, s, &dataMatrix)
		mat.Row(shufLabelRow, s, &labelMatrix)
		shuffledDataMatrix.SetRow(r, shufDataRow)
		shuffledLabelMatrix.SetRow(r, shufLabelRow)
	}
	return shuffledDataMatrix, shuffledLabelMatrix
}

// Logistic Regression ==========================================================================================
type LogisticR struct{
	// Create new normalizer(cols) -- the columns I want to normalize
	numLabels int
	numFeatures int
	learningRate float64
	regCoeff float64
	weights *mat.Dense
}

// Constructor for Logistic Regression
func CreateLogisticR(labelNum, featureNum int, learningRate float64, regularizationCoeff float64) *LogisticR {
	logistic := new(LogisticR)
	logistic.numLabels = labelNum
	logistic.numFeatures = featureNum
	logistic.learningRate = learningRate
	logistic.regCoeff = regularizationCoeff
	logistic.weights = mat.NewDense(logistic.numFeatures, logistic.numLabels, nil)
	return logistic
}

// TODO: parameter: pass by value, return: pass by reference
func (l *LogisticR) Fit(trainX mat.Dense, trainY mat.Dense, numIter int) {
	weights := mat.NewDense(l.numFeatures, l.numLabels, nil)
	xMatrix := mat.DenseCopyOf(&trainX) 					// numSample x numFeatures
	yMatrix:= OneHotEncodingLabels(&trainY, 7) // numSample x numLabels

	numSample, numFeatures := xMatrix.Dims()

	for i:=0; i<numIter; i++ {
		score := mat.NewDense(numSample, l.numLabels, nil)
		sub := mat.NewDense(numSample, l.numLabels, nil)
		mul := mat.NewDense(l.numFeatures, l.numLabels, nil)
		weightsUnreg := mat.NewDense(l.numFeatures, l.numLabels, nil)

		// 1. score = Matrix Multiplication of xMatrix and weights
		// 2. prob = SoftMax(X, w)
		score.Mul(xMatrix, l.weights)
		prob := SoftMax(*score)

		// 3. MatrixMultiplication(X.T, (y-prob)) * (1/(n-samples))
		sub.Sub(yMatrix, prob)
		mul.Mul(xMatrix.T(), sub)

		scalarMatrix := CreateScalarMatrix(l.learningRate/float64(numSample), numFeatures, l.numLabels)

		weightsUnreg.MulElem(mul, scalarMatrix)

		// 4. Regularization Coeff
		regMatrix := CreateScalarMatrix(2*l.regCoeff, numFeatures, l.numLabels)

		weights.Add(weightsUnreg, regMatrix)

		// 5. fit !!
		l.weights = weights
	}
}

func (l *LogisticR) Predict(testX mat.Dense) *mat.Dense {
	prob := l.PredictProb(testX) // n_test_samples x n_labels
	// TODO: pick up the highest value among each row that is the predicted label
	testPredictions := FindLabelFromSoftmax(*prob)
	return testPredictions
}

func (l *LogisticR) PredictProb(testX mat.Dense) *mat.Dense {
	numTestInstances, _ :=testX.Dims()
	score := mat.NewDense(numTestInstances, 7, nil) // 7 numLabels
	score.Mul(&testX, l.weights)
	prob := SoftMax(*score) // n_test_samples x n_labels
	return prob
}


// Utility functions for Models =================================================================================

// takes labels matrix and number of labels and returns one hot encoded label matrix
func OneHotEncodingLabels(labels *mat.Dense, numLabels int) *mat.Dense {
	numRows, numCols := labels.Dims()
	encodedMatrix:=mat.NewDense(numRows, numLabels, nil)
	for r:=0; r<numRows; r++ {
		label := labels.At(r, numCols-1)
		encodedMatrix.Set(r, int(label), 1)
	}
	return encodedMatrix
}

// there is no scalar multiplication in gonum I had to create the matrix containing scalars.
func CreateScalarMatrix(scalar float64, rowNum, colNum int) *mat.Dense {
	scalars := make([]float64, rowNum*colNum)

	for i := range scalars {
		scalars[i] = scalar
	}

	scalarMatrix := mat.NewDense(rowNum, colNum, scalars)

	return scalarMatrix
}


func SoftMax(xwMatrix mat.Dense) *mat.Dense {
	// 1. dot product between X(n_sample x n_features) and w(n_features x n_labels)
	// 2. dot product result(n_sample x n_labels)

	numRows, numCols := xwMatrix.Dims()
	softMatrix := mat.NewDense(numRows, numCols, nil)

	for r:=0; r<numRows; r++ {
		denominator := SumExponentials(xwMatrix, r)
		for c:=0; c<numCols; c++ {
			temp := xwMatrix.At(r, c)
			nominator := math.Exp(temp)
			softMatrix.Set(r, c, nominator/denominator)
		}
	}
	return softMatrix
}

// in order to caculate softmax, I need denominator
func SumExponentials(xwMatrix mat.Dense, row int) float64 {
	var sumExp float64
	_, numCols := xwMatrix.Dims()

	for c:=0; c<numCols; c++ {
		temp := xwMatrix.At(row, c)
		// func Exp(x float64) float64
		sumExp += math.Exp(temp)
	}
	return sumExp
}

// from softmax probabilities, take the highest probability arguments and return a matrix, which would contain labels.
func FindLabelFromSoftmax(softMatrix mat.Dense) *mat.Dense {
	numInstances, numLabels := softMatrix.Dims()
	predictedLabels := mat.NewDense(numInstances, 1, nil) // we are taking the highest probable label, ONLY ONE
	for r:=0; r<numInstances; r++ {
		temp:=FindArgMaxMatrix(softMatrix, r, numLabels)
		predictedLabels.Set(r, 0, temp)
	}
	return predictedLabels
}

// just like FindArgMax in Evaluators.go, but this is for matrix
func FindArgMaxMatrix(softMatrix mat.Dense, row, numLabels int) float64 {
	var maxIdx, maxVal float64 // since it is from softMatrix it won't be smaller than 0, so we can set as 0.0
	for c:=0; c<numLabels; c++ {
		temp := softMatrix.At(row, c)
		if maxVal < temp {
			maxVal = temp
			maxIdx = float64(c)
		}
	}
	return maxIdx
}