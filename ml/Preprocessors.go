package ml

// In gonum, Matrix is interface and Dense is class, but I comment both as matrix for simplification

import(
	"fmt"
	"os"
	"encoding/csv"
	"bufio"
	"strconv"
	"gonum.org/v1/gonum/mat"
)

// Customize Reader structure
// no matter which type I put here, when reading, all the values are string,
// but still the name of fields are required to get a wanted format.
type Reader struct {
	Comma rune

	Comment rune

	// except Id, all features in first part are numerical // 11 = 1 id  + 10 features
	Id,Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,
	Horizontal_Distance_To_Roadways, Hillshade_9am,Hillshade_Noon, Hillshade_3pm,Horizontal_Distance_To_Fire_Points float64

	// boolean type features
	// Wilderness_Area: 1~4 => binary features
	// Soil_Type: 1~40 => binary features
	Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,
	Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,
	Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,
	Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,
	Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,
	Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,
	Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,
	Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40 bool

	// nominal/categorical label but instead of using string, I used int
	Cover_Type int // int
}

// Check whether converting string into float is successful and return converted float
func CheckConvertedFloat(r string) float64 {
	var temp float64
	temp, err := strconv.ParseFloat(r, 64)
	if err != nil {
		fmt.Println("Error: cannot covert string into float64")
	}
	return temp
}

var LABELFIELD = "Cover_Type"
// Exclude Field and exclude id column
func Preprocess(initialFile string) (*mat.Dense, *mat.Dense) {

	// 1. Open csv
	file, err := os.Open(initialFile)
	if err != nil {
		fmt.Println("Error: cannot open csv file")
		os.Exit(1)
	}

	// 2. Read csv
	readFile := csv.NewReader(bufio.NewReader(file))
	records, err := readFile.ReadAll()
	if err != nil {
		fmt.Println("Error: cannot read csv file")
		os.Exit(1)
	}

	numInstances := len(records)-1 // exclude fields part
	numFeatures := len(records[0])-2 // exclude id and label
	numLabel := 1

	// 3. Divide data(X) and label(Y)
	_, data, labels := DivideDataNLabel(records, numFeatures)

	preData := mat.NewDense(numInstances, numFeatures, data) // 15120 54
	preLabel := mat.NewDense(numInstances, numLabel, labels) // 15120 1

	return preData, preLabel
}

// takes records(the result of reading file contents) and number of features and return field name, data, and labels.
func DivideDataNLabel(records [][]string, numFeatures int) ([]string, []float64, []float64) {
	fields := make([]string, numFeatures+1)
	data := make([]float64, 0)
	labels := make([]float64, 0)

	for i, record := range records {
		if i == 0 {
			for fieldIdx, r := range record {
				if fieldIdx >= 1 && fieldIdx <= numFeatures { // only features
					fields[fieldIdx] = r
				}
			}
		} else {
			for col, r := range record {
				if col >= 1 && col <= numFeatures {
					temp := CheckConvertedFloat(r)
					data = append(data, temp)
				} else if col == numFeatures+1 { 				// label part
					temp := CheckConvertedFloat(r) - 1 		// change into index = type num
					labels = append(labels, temp)
				}
			}
		}
	}
	return fields, data, labels
}
