package ml

import "gonum.org/v1/gonum/mat"

/**
In this section,
(1) FindArgMax
(2) GetAccuracyMatrix
(3) GetAccuracy
(4) ComparePredictionLabel
 */

// This is used for go ml library
// to find the index that has a maximum value
func FindArgMax(predictions []float64) float64 {
	var maxIdx, maxVal float64
	for i, val := range predictions {
		if i==0 {
			maxVal = val
		}
		if maxVal < val {
			maxVal = val
			maxIdx = float64(i)
		}
	}
	return maxIdx
}

// in order to evaluate the performance using my own implementation with gonum, we need this matrix version
func GetAccuracyMatrix(trueLabels, predictedLabels mat.Dense) float64 {
	numRows, _ := trueLabels.Dims()
	count :=0
	for r:=0; r<numRows; r++ {
		decision := ComparePredictionLabel(trueLabels.At(r, 0), predictedLabels.At(r, 0))
		if decision {
			count+=1
		}
	}
	return float64(count)/float64(numRows)
}

// in order to evaluate the performance using external libraries, we need []float64 version
func GetAccuracy(testLabelSlice, predictions []float64) float64 {
	numRows := len(testLabelSlice)
	count := 0
	for r:=0; r<numRows; r++ {
		decision := ComparePredictionLabel(testLabelSlice[r], predictions[r])
		if decision {
			count+=1
		}
	}
	return float64(count)/float64(numRows)
}

// compare prediction with label and check whether they are the same or not.
func ComparePredictionLabel(trueLabel, predictedLabel float64) bool {
	if trueLabel == predictedLabel {
		return true
	}
	return false
}
