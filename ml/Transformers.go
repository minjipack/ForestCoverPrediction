package ml
/* This is my own comment convention of input and output types.

(1) The name of function
(2) (type of parameter1, type of parameter2, ...) -> return type
(3) description of what I return
(4) How: The steps of my function.

In gonum, Matrix is interface and Dense is class, but I comment both as matrix for simplification

*/

import (
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/floats"
)

/**
List of Feature Transformers
1. Normalizer
2. OutlierDetector
 */

// Feature Normalization
type Normalizer struct{
	// store columns which I want to normalize
	cols []int
}

// Constructor for Normalizer
// takes slice of integers and returns normalizer object
func CreateNormalizer(columns []int) *Normalizer {
	n := new(Normalizer)
	n.cols = columns
	return n
}

/** Normalizer method(1) MeanStd
(matrix) -> ([]float64, []float64)
takes matrix and returns means and standard deviations for each column
HOW
use "gonum.org/v1/gonum/stat" library to get means and standard deviations

 */
func (n *Normalizer) MeanStd(matrix mat.Dense) ([]float64, []float64) {
	matMean :=make([]float64, len(n.cols))
	matStd := make([]float64, len(n.cols))
	numRows, _ := matrix.Dims()
	// find the mean and the standard deviation of columns aiming to normalize
	for i, colIdx := range n.cols {
		col := make([]float64, numRows)

		mat.Col(col, colIdx, &matrix)
		// use stat library to get mean and standard deviation
		matMean[i], matStd[i] = stat.MeanStdDev(col, nil)
	}
	return matMean, matStd
}

/** Normalizer method(2) Normalize
(matrix) -> (matrix)
takes matrix and returns normalized matrix

HOW
use "gonum.org/v1/gonum/stat" library to get means and standard deviations
normalization equation = (x - mean) / standard deviation
// parameter: pass by value, return: pass by reference
 */
func (n *Normalizer) Normalize(matrix mat.Dense) *mat.Dense {
	normMatrix := mat.DenseCopyOf(&matrix)
	r, _ := matrix.Dims()
	matMean, matStd := n.MeanStd(*normMatrix)
	for j, colIdx := range n.cols {
		for i:=0; i< r; i++ {
			var normVal float64
			if matStd[j] == 0 {
				normVal = 0
			} else {
				// apply the normalization equation
				normVal = (normMatrix.At(i, colIdx) - matMean[j])/matStd[j]
			}
			// update matrix with normalized values
			normMatrix.Set(i, colIdx, normVal)
		}
	}
	return normMatrix
}

// Outlier
type Outlier struct{
	// the number of outliers you want to remove
	numOutliers int
}

// Constructor for Outlier
// takes integer and returns normalizer object // TODO: check object
func CreateOutlierDetector(num int) *Outlier {
	o := new(Outlier)
	o.numOutliers = num
	return o
}


/** Outlier method(1.1) RemoveOutlierSGD
(matrix, matrix) -> (matrix, matrix)
takes train data and train label and returns train data and train label without outliers

HOW
This is for removing outliers based on SGD.
1. learn(Fit) based on SGD
2. based on SGD, get each probability of each predicted label
3. calculate residual = 1 - the probability of each predicted label
4. Find highest residuals and its corresponding indices. These indices would be outliers.
5. remove outliers
// parameter: pass by value, return: pass by reference
 */
func (o *Outlier) RemoveOutlierSGD(xTrain, yTrain mat.Dense) (*mat.Dense, *mat.Dense) {
	// 1. 2.
	sampleNum, featureNum := xTrain.Dims()
	// CreateSGD(labelNum, featureNum, learningRate, regularizationCoeff, batchSize)
	SGD := CreateSGD(7, featureNum, 0.001, 0., 100)
	SGD.Fit(xTrain, yTrain, 100)
	predictions := SGD.PredictProb(xTrain)

	// 3.
	residuals := make([]float64, sampleNum)
	for i:=0; i<sampleNum; i++ {
		residuals[i] = 1.0 - predictions.At(i, int(yTrain.At(i, 0)))
	}

	// 4.
	residualsIdx := make([]int, sampleNum)
	floats.Argsort(residuals, residualsIdx)
	rIdx := sampleNum - o.numOutliers
	outlierIdxs := residualsIdx[rIdx:]

	// 5.
	newXTrain, newYTrain:= RemoveOutliers(xTrain, yTrain, outlierIdxs)
	return newXTrain, newYTrain
}

/** Outlier method(1.2) RemoveOutlierLR
(matrix, matrix) -> (matrix, matrix)
takes train data and train label and returns train data and train label without outliers

HOW
This is for removing outliers based on LR.
1. learn(Fit) based on Logistic Regression.
2. based on SGD, get each probability of each predicted label
3. calculate residual = 1 - the probability of each predicted label
4. Find highest residuals and its corresponding indices. These indices would be outliers.
5. remove outliers
//parameter: pass by value, return: pass by reference
 */
func (o *Outlier) RemoveOutlierLR(xTrain, yTrain mat.Dense) (*mat.Dense, *mat.Dense) {
	sampleNum, featureNum := xTrain.Dims()
	// 1. 2.
	logisticRegressor := CreateLogisticR(7, featureNum, 0.001, 0.)
	logisticRegressor.Fit(xTrain, yTrain, 100)
	predictions := logisticRegressor.PredictProb(xTrain)

	// 3.
	residuals := make([]float64, sampleNum)
	for i:=0; i<sampleNum; i++ {
		residuals[i] = 1.0 - predictions.At(i, int(yTrain.At(i, 0)))
	}

	// 4.
	residualsIdx := make([]int, sampleNum)
	floats.Argsort(residuals, residualsIdx)
	rIdx := sampleNum - o.numOutliers
	outlierIdxs := residualsIdx[rIdx:]

	// 5.
	newXTrain, newYTrain:= RemoveOutliers(xTrain, yTrain, outlierIdxs)
	return newXTrain, newYTrain

}

// takes data matrix, label matrix, and outlier indices and return matrix without data matrix and label matrix without outliers
func RemoveOutliers(dataMatrix, dataLabels mat.Dense, outlierIdx []int) (*mat.Dense, *mat.Dense) {
	newData := make([]float64, 0)
	newLabel := make([]float64, 0)

	numInstances, numFeatures:=dataMatrix.Dims()

	for r:=0; r<numInstances; r++ {
		if CheckOutlierInstance(r, outlierIdx) { // check whether each instance is outlier or not and decide whether to skip or not
			continue
		}
		for c:=0; c<numFeatures; c++ {
			newData = append(newData, dataMatrix.At(r, c)) // only add data which are not outliers
		}
		newLabel = append(newLabel, dataLabels.At(r, 0)) // only add labels corresponding data are not outliers
	}
	newNumRows:=len(newLabel)
	newTestMatrix := mat.NewDense(newNumRows, numFeatures, newData)
	newTestLabels := mat.NewDense(newNumRows, 1, newLabel)

	return newTestMatrix, newTestLabels
}

// take an instance idex and indices of outliers and return bool, indicating that the instance is outlier or not.
func CheckOutlierInstance(instanceIdx int, outlierIdx []int) bool {
	for _, i:= range outlierIdx {
		if i==instanceIdx {
			return true
		}
	}
	return false
}
