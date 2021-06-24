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

// #include "pd_predictor.h"
// #include "pd_tensor.h"
// #include "pd_common.h"
// #include "pd_types.h"
// #include "pd_utils.h"
// #include <stdlib.h>
// #include <string.h>
import "C"
import (
	"runtime"
	"unsafe"
)

type Predictor struct {
	c *C.PD_Predictor
}

///
/// \brief Create a new Predictor
///
/// \param[in] Config config
/// \return new predicor.
///
func NewPredictor(config *Config) *Predictor {
	cPredictor := C.PD_PredictorCreate(config.c)
	predictor := &Predictor{c: cPredictor}
	runtime.SetFinalizer(predictor, func(predictor *Predictor) {
		C.PD_PredictorDestroy(predictor.c)
	})
	return predictor
}

///
/// \brief Clone a new Predictor
///
/// \return new predictor.
///
func (p *Predictor) Clone() *Predictor {
	cPredictor := C.PD_PredictorClone(p.c)
	predictor := &Predictor{c: cPredictor}
	runtime.SetFinalizer(predictor, func(predictor *Predictor) {
		C.PD_PredictorDestroy(predictor.c)
	})
	return predictor
}

///
/// \brief Get the input number
///
/// \return input number
///
func (p *Predictor) GetInputNum() uint {
	return uint(C.PD_PredictorGetInputNum(p.c))
}

///
/// \brief Get the output number
///
/// \return output number
///
func (p *Predictor) GetOutputNum() uint {
	return uint(C.PD_PredictorGetOutputNum(p.c))
}

///
/// \brief Get the input names
///
/// \return input names
///
func (p *Predictor) GetInputNames() []string {
	cNames := C.PD_PredictorGetInputNames(p.c)
	numNames := int(cNames.size)
	names := cvtToGoSliceString(numNames, cNames.data)
	C.PD_OneDimArrayCstrDestroy(cNames)
	return names
}

///
/// \brief Get the output names
///
/// \return output names
///
func (p *Predictor) GetOutputNames() []string {
	cNames := C.PD_PredictorGetOutputNames(p.c)
	numNames := int(cNames.size)
	names := cvtToGoSliceString(numNames, cNames.data)
	C.PD_OneDimArrayCstrDestroy(cNames)
	return names
}

///
/// \brief Get the Input Tensor object
///
/// \param[in] name input name
/// \return input tensor
///
func (p *Predictor) GetInputHandle(name string) *Tensor {
	cName := C.CString(name)
	cHandle := C.PD_PredictorGetInputHandle(p.c, cName)
	C.free(unsafe.Pointer(cName))
	handle := &Tensor{c: cHandle}
	runtime.SetFinalizer(handle, func(handle *Tensor) {
		C.PD_TensorDestroy(handle.c)
	})
	return handle
}

///
/// \brief Get the Output Tensor object
///
/// \param[in] name output name
/// \return output tensor
///
func (p *Predictor) GetOutputHandle(name string) *Tensor {
	cName := C.CString(name)
	cHandle := C.PD_PredictorGetOutputHandle(p.c, cName)
	C.free(unsafe.Pointer(cName))
	handle := &Tensor{c: cHandle}
	runtime.SetFinalizer(handle, func(handle *Tensor) {
		C.PD_TensorDestroy(handle.c)
	})
	return handle
}

///
/// \brief Run the prediction engine
///
func (p *Predictor) Run() {
	C.PD_PredictorRun(p.c)
}

///
/// \brief Clear the intermediate tensors of the predictor
///
func (p *Predictor) ClearIntermediateTensor() {
	C.PD_PredictorClearIntermediateTensor(p.c)
}

///
/// \brief Release all tmp tensor to compress the size of the memory pool.
/// The memory pool is considered to be composed of a list of chunks, if
/// the chunk is not occupied, it can be released.
///
/// \return Number of bytes released. It may be smaller than the actual
/// released memory, because part of the memory is not managed by the
/// MemoryPool.
///
func (p *Predictor) TryShrinkMemory() {
	C.PD_PredictorTryShrinkMemory(p.c)
}
