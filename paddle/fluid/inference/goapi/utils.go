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

// #include <stdint.h>
// #include <stdlib.h>
import "C"
import (
	"unsafe"
)

func cvtPDBoolToGo(b C.int8_t) bool {
	var cFalse C.int8_t
	if b != cFalse {
		return true
	}
	return false
}

func cvtGoBoolToPD(b bool) C.int8_t {
	if b == false {
		return 0
	}
	return 1
}

func cvtToGoSliceString(length int, str **C.char) []string {
	if str == nil {
		return nil
	}
	tmpSlice := (*[1 << 27]*C.char)(unsafe.Pointer(str))[:length:length]
	goStrings := make([]string, length)
	for i, s := range tmpSlice {
		goStrings[i] = C.GoString(s)
	}
	return goStrings
}

func cvtToGoSliceInt32(length int, data *C.int32_t) []int32 {
	if data == nil {
		return nil
	}
	tmpSlice := (*[1 << 27]C.int32_t)(unsafe.Pointer(data))[:length:length]
	res := make([]int32, length)
	for i, s := range tmpSlice {
		res[i] = int32(s)
	}
	return res
}
