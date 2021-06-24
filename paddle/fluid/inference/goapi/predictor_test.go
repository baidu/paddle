// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package paddle

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestNewPredictor(t *testing.T) {
	t.Logf("Version:\n%+v", Version())
	config := NewConfig()
	config.SetModel("./mobilenetv1/inference.pdmodel", "./mobilenetv1/inference.pdiparams")
	config.EnableUseGpu(100, 0)
	predictor := NewPredictor(config)
	inNames := predictor.GetInputNames()
	t.Logf("InputNames:%+v", inNames)
	outNames := predictor.GetOutputNames()
	t.Logf("OutputNames:%+v", outNames)

	inHandle := predictor.GetInputHandle(inNames[0])
	inHandle.Reshape([]int32{1, 3, 224, 224})
	t.Logf("inHandle name:%+v, shape:%+v", inHandle.Name(), inHandle.Shape())

	var lod [][]uint
	lod = append(lod, []uint{0, 1, 2})
	lod = append(lod, []uint{1, 2, 3, 4})
	inHandle.SetLod(lod)
	t.Logf("inHandle Lod:%+v", inHandle.Lod())
	data := make([]float32, numElements([]int32{1, 3, 224, 224}))
	for i := 0; i < int(numElements([]int32{1, 3, 224, 224})); i++ {
		data[i] = float32(i%255) * 0.1
	}
	inHandle.CopyFromCpu(data)
	t.Logf("inHandle Type:%+v", inHandle.Type())

	predictor.Run()

	outHandle := predictor.GetOutputHandle(outNames[0])
	t.Logf("outHandle name:%+v", outHandle.Name())

	outShape := outHandle.Shape()
	t.Logf("outHandle Shape:%+v", outShape)
	outData := make([]float32, numElements(outShape))
	outHandle.CopyToCpu(outData)
	t.Log(outData)

	cloned := predictor.Clone()
	t.Logf("InputNum:%+v", cloned.GetInputNum())
	t.Logf("OutputNum:%+v", cloned.GetInputNum())
	cloned.ClearIntermediateTensor()
}

func TestFromBuffer(t *testing.T) {
	modelFile, err := os.Open("./mobilenetv1/inference.pdmodel")
	if err != nil {
		t.Fatal(err)
	}
	paramsFile, err := os.Open("./mobilenetv1/inference.pdiparams")
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()
	defer paramsFile.Close()

	model, err := ioutil.ReadAll(modelFile)
	if err != nil {
		t.Fatal(err)
	}
	params, err := ioutil.ReadAll(paramsFile)
	if err != nil {
		t.Fatal(err)
	}
	config := NewConfig()
	config.SetModelBuffer(string(model), string(params))

	predictor := NewPredictor(config)
	inNames := predictor.GetInputNames()
	outNames := predictor.GetOutputNames()
	inHandle := predictor.GetInputHandle(inNames[0])
	inHandle.Reshape([]int32{1, 3, 224, 224})
	data := make([]float32, numElements([]int32{1, 3, 224, 224}))
	for i := 0; i < int(numElements([]int32{1, 3, 224, 224})); i++ {
		data[i] = float32(i%255) * 0.1
	}
	inHandle.CopyFromCpu(data)
	predictor.Run()
	outHandle := predictor.GetOutputHandle(outNames[0])
	outShape := outHandle.Shape()
	t.Logf("outHandle Shape:%+v", outShape)
	outData := make([]float32, numElements(outShape))
	outHandle.CopyToCpu(outData)
	t.Log(outData)
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
