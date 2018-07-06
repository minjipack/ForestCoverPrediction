package ml

import (
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

/**
In this section,
it is mainly for spliting data into Train/Test Data
we have
(1) SplitData
(2) MatrixTo2DSlice -- in order to use goml library, slice is required
(3) MatrixToSlice	 -- in order to use goml library, slice is required
 */

// Spliting ratio
var SIZE = 0.33

// take a data matrix and a label matrix and returns a train data, a train label, a test data, and a test label
func SplitData(data mat.Dense, label mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	// deep copy of matrix
	dataMatrix := mat.DenseCopyOf(&data)
	labelMatrix := mat.DenseCopyOf(&label)

	// a way to shuffle the data
	numRows, numCols := dataMatrix.Dims()
	shuffle:=rand.Perm(numRows)

	testSize := int(float64(len(shuffle))*SIZE)
	trainSize := len(shuffle)-testSize

	// these are matrices to return
	testData := mat.NewDense(testSize, numCols, nil)
	testLabel := mat.NewDense(testSize, 1, nil)
	trainData := mat.NewDense(trainSize, numCols, nil)
	trainLabel := mat.NewDense(trainSize, 1, nil)

	testIndices := shuffle[:testSize]
	trainIndices := shuffle[testSize:]

	// test row: 0 ~ testSize
	for row, shuffledIdx := range testIndices {
		// save dataMatrix.Row(shuffledIdx) to row of testData
		InitializeDataSet(row, shuffledIdx, testData, *dataMatrix)
		InitializeDataSet(row, shuffledIdx, testLabel, *labelMatrix)
	}

	// train row: 0 ~ trainSize
	for row, shuffledIdx := range trainIndices {
		// save dataMatrix.Row(shuffledIdx) to row of trainData
		InitializeDataSet(row, shuffledIdx, trainData, *dataMatrix)
		InitializeDataSet(row, shuffledIdx, trainLabel, *labelMatrix)
	}

	return trainData, trainLabel, testData, testLabel
}

// TODO: since newMatrix is based on pass by reference, therefore no need to return
// takes new index, random index, new matrix, and data matrix to give shuffled result.
func InitializeDataSet(newIdx, randomIdx int, newMatrix *mat.Dense, dataMatrix mat.Dense) {
	_, numCols := dataMatrix.Dims()
	for c:=0; c<numCols; c++ {
		newMatrix.Set(newIdx, c, dataMatrix.At(randomIdx, c))
	}
}

// converts data matrix into slice of slices of float64 type
func MatrixTo2DSlice(dataMatrix *mat.Dense) [][]float64 {
	dataMatrix = mat.DenseCopyOf(dataMatrix)
	numRows, _ := dataMatrix.Dims()
	twoDSlices := make([][]float64, numRows)

	for r:=0; r<numRows; r++ {
		twoDSlices[r] = dataMatrix.RawRowView(r)
	}
	return twoDSlices
}

// converts label matrix into float64 slice
func MatrixToSlice(dataMatrix *mat.Dense) []float64 {
	dataMatrix = mat.DenseCopyOf(dataMatrix)
	numRows, _ := dataMatrix.Dims()
	slice := make([]float64, numRows)

	for r:=0; r<numRows; r++ {
		slice[r] = dataMatrix.At(r, 0)
	}
	return slice
}