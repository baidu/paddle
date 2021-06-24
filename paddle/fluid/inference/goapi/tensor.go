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

// #include "pd_tensor.h"
// #include "pd_utils.h"
// #include "pd_types.h"
// #include "pd_common.h"
// #include "stdlib.h"
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"
)

type DataType C.PD_DataType

const (
	Unk     DataType = C.PD_DATA_UNK
	Float32 DataType = C.PD_DATA_FLOAT32
	Int32   DataType = C.PD_DATA_INT32
	Int64   DataType = C.PD_DATA_INT64
	Uint8   DataType = C.PD_DATA_UINT8
	Int8    DataType = C.PD_DATA_INT8
)

type PlaceType C.PD_PlaceType

const (
	UnkPlace PlaceType = C.PD_PLACE_UNK
	CpuPlace PlaceType = C.PD_PLACE_CPU
	GpuPlace PlaceType = C.PD_PLACE_GPU
	XpuPlace PlaceType = C.PD_PLACE_XPU
)

type Tensor struct {
	c *C.PD_Tensor
}

///
/// \brief Reset the shape of the tensor.
/// Generally it's only used for the input tensor.
///
/// \param[in] shape The shape to set.
///
func (t *Tensor) Reshape(shape []int32) {
	C.PD_TensorReshape(t.c, C.size_t(len(shape)), (*C.int32_t)(unsafe.Pointer(&shape[0])))
}

///
/// \brief Get the tensor shape
///
/// \return The tensor shape.
///
func (t *Tensor) Shape() []int32 {
	cData := C.PD_TensorGetShape(t.c)
	length := int(cData.size)
	defer C.PD_OneDimArrayInt32Destroy(cData)
	return cvtToGoSliceInt32(length, cData.data)
}

///
/// \brief Set the tensor lod information
/// \param[in] pd_tensor tensor.
/// \param[in] lod lod information.
///
func (t *Tensor) SetLod(lod [][]uint) {
	cLod := (*C.struct_PD_TwoDimArraySize)(C.malloc(C.size_t(C.sizeof_struct_PD_TwoDimArraySize)))
	length := len(lod)
	cLod.size = C.size_t(uint(length))
	var lodList = make([]*C.struct_PD_OneDimArraySize, length+1)

	for i, v := range lod {
		oneDimArray := (*C.struct_PD_OneDimArraySize)(C.malloc(C.size_t(C.sizeof_struct_PD_OneDimArraySize)))
		defer C.free(unsafe.Pointer(oneDimArray))
		tmpLength := len(v)
		oneDimArray.size = C.size_t(uint(tmpLength))

		tmpC := (*C.size_t)(C.malloc(C.size_t(C.sizeof_size_t * tmpLength)))
		defer C.free(unsafe.Pointer(tmpC))
		tmpSlice := (*[1 << 27]C.size_t)(unsafe.Pointer(tmpC))[:tmpLength:tmpLength]
		for j, w := range v {
			tmpSlice[j] = C.size_t(w)
		}
		oneDimArray.data = tmpC

		lodList[i] = oneDimArray
	}
	cLod.data = (**C.struct_PD_OneDimArraySize)(unsafe.Pointer(&lodList[0]))
	C.PD_TensorSetLod(t.c, cLod)
	C.free(unsafe.Pointer(cLod))
	// C.PD_TwoDimArraySizeDestroy(cLod)
}

///
/// \brief Get the tensor lod information
///
/// \return the lod information.
///
func (t *Tensor) Lod() [][]uint {
	cLod := C.PD_TensorGetLod(t.c)
	length := int(cLod.size)
	res := make([][]uint, length)
	if length == 0 {
		return res
	}
	cLodSlice := (*[1 << 27]*C.struct_PD_OneDimArraySize)(unsafe.Pointer(cLod.data))[:length:length]

	for i := 0; i < length; i++ {
		size := uint(cLodSlice[i].size)
		lod := make([]uint, size)

		tmpSlice := (*[1 << 27]C.size_t)(unsafe.Pointer(cLodSlice[i].data))[:size:size]
		for j, v := range tmpSlice {
			lod[j] = uint(v)
		}

		res[i] = lod
	}

	C.PD_TwoDimArraySizeDestroy(cLod)
	return res
}

///
/// \brief Get the tensor data type
/// \param[in] pd_tensor tensor.
/// \return the tensor data type.
///
func (t *Tensor) Type() DataType {
	cDtype := C.PD_TensorGetDataType(t.c)
	return DataType(cDtype)
}

///
/// \brief Get the tensor name
///
/// \return the tensor name.
///
func (t *Tensor) Name() string {
	return C.GoString(C.PD_TensorGetName(t.c))
}

///
/// \brief Copy the host memory to tensor data.
/// It's usually used to set the input tensor data.
///
/// \param[in] value
///
func (t *Tensor) CopyFromCpu(value interface{}) {
	val := reflect.ValueOf(value)
	dtype, _ := dataTypeOf(val)

	switch dtype {
	case Float32:
		data := val.Interface().([]float32)
		C.PD_TensorCopyFromCpuFloat(t.c, (*C.float)(unsafe.Pointer(&data[0])))
	case Int32:
		data := val.Interface().([]int32)
		C.PD_TensorCopyFromCpuInt32(t.c, (*C.int32_t)(unsafe.Pointer(&data[0])))
	case Int64:
		data := val.Interface().([]int64)
		C.PD_TensorCopyFromCpuInt64(t.c, (*C.int64_t)(unsafe.Pointer(&data[0])))
	case Uint8:
		data := val.Interface().([]uint8)
		C.PD_TensorCopyFromCpuUint8(t.c, (*C.uint8_t)(unsafe.Pointer(&data[0])))
	case Int8:
		data := val.Interface().([]int8)
		C.PD_TensorCopyFromCpuInt8(t.c, (*C.int8_t)(unsafe.Pointer(&data[0])))
	}
}

///
/// \brief Copy the tensor data to the host memory.
/// It's usually used to get the output tensor data.
///
/// \param[value] data The tensor will copy the data to the address.
///
func (t *Tensor) CopyToCpu(value interface{}) {
	val := reflect.ValueOf(value)
	dtype, _ := dataTypeOf(val)

	switch dtype {
	case Float32:
		data := val.Interface().([]float32)
		C.PD_TensorCopyToCpuFloat(t.c, (*C.float)(unsafe.Pointer(&data[0])))
	case Int32:
		data := val.Interface().([]int32)
		C.PD_TensorCopyToCpuInt32(t.c, (*C.int32_t)(unsafe.Pointer(&data[0])))
	case Int64:
		data := val.Interface().([]int64)
		C.PD_TensorCopyToCpuInt64(t.c, (*C.int64_t)(unsafe.Pointer(&data[0])))
	case Uint8:
		data := val.Interface().([]uint8)
		C.PD_TensorCopyToCpuUint8(t.c, (*C.uint8_t)(unsafe.Pointer(&data[0])))
	case Int8:
		data := val.Interface().([]int8)
		C.PD_TensorCopyToCpuInt8(t.c, (*C.int8_t)(unsafe.Pointer(&data[0])))
	}
}

var types = []struct {
	typ      reflect.Type
	dataType C.PD_DataType
}{
	{reflect.TypeOf(float32(0)), C.PD_DATA_FLOAT32},
	{reflect.TypeOf(int32(0)), C.PD_DATA_INT32},
	{reflect.TypeOf(int64(0)), C.PD_DATA_INT64},
	{reflect.TypeOf(uint8(0)), C.PD_DATA_UINT8},
	{reflect.TypeOf(int8(0)), C.PD_DATA_INT8},
}

func dataTypeOf(val reflect.Value) (dt DataType, err error) {
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		if val.Len() > 0 {
			val = val.Index(0)
		}
		typ = typ.Elem()
	}
	for _, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return DataType(t.dataType), nil
		}
	}
	return dt, fmt.Errorf("unsupported type %v", typ)
}
