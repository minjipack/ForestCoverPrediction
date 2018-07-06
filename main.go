// This is MinJi's Final Project
// Due December 7, 2017

package main

import (
	p "project/ml"
)

func main() {
	// When running the code, you need to set up the right location of file.
	FileLocation := "/Users/minjipack/go/src/project/data/"
	initialFile := FileLocation + "train.csv"

	// 1. Preprocessor: exclude fields(first row) and id(first column) and separate data with labels
	// 0~53 (54 numbers of features)
	// 0~15119 (15120 numbers of instances)
	dataMatrix, labelMatrix := p.Preprocess(initialFile)
	_, numFeatures := dataMatrix.Dims()

	// 2. DataUtils: Split Train and Test data
	trainData, trainLabel, testData, testLabel := p.SplitData(*dataMatrix, *labelMatrix)

	// 3. Transform: Normalize data
	nCols := InitializeNormalizeIdx(numFeatures)
	normalizer := p.CreateNormalizer(nCols)

	trainMatrix := normalizer.Normalize(*trainData)
	testMatrix := normalizer.Normalize(*testData)

	// 4. Transform: Remove Outliers
	//outlier := p.CreateOutlierDetector(20)
	//trainMatrix, trainLabel = outlier.RemoveOutlierLR(*trainMatrix, *trainLabel)

	p.RunExperimentSGD(trainMatrix, trainLabel, testMatrix, testLabel, 0.1, 0., 10, 1000)
	p.RunExperimentGoMLSGD(trainMatrix, trainLabel, testMatrix, testLabel, 0.000001, 1.5, 100)
	p.RunExperimentLR(trainMatrix, trainLabel, testMatrix, testLabel, 0.000001, 1.0, 100)
	p.RunExperimentGoMLLR(trainMatrix, trainLabel, testMatrix, testLabel, 0.000001, 1.5, 100)
	p.RunExperimentGoMLKNN(trainMatrix, trainLabel, testMatrix, testLabel)

}

// Pointer, Pass by ref, Pass by value,


// 0 ~ 53 indices should be normalized
func InitializeNormalizeIdx(numFeatures int) []int {
	idx := make([]int, 0)
	for i:=0; i<numFeatures; i++ {
		idx = append(idx, i)
	}
	return idx
}
