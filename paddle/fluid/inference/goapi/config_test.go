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

import "testing"

func TestNewConfig(t *testing.T) {
	config := NewConfig()
	config.SetProgFile("model")
	config.SetParamsFile("params")

	config.SetOptimCacheDir("cache")

	config.DisableFCPadding()
	t.Logf("UseFcPadding:%+v", config.UseFcPadding())

	// It will break when we have no xpu env.
	// config.EnableXpu(100)
	// t.Logf("EnableXpu, UseXpu:%+v ", config.UseXpu())

	config.SwitchIrOptim(true)
	t.Logf("IrOptim:%+v", config.IrOptim())

	config.EnableUseGpu(100, 0)
	t.Logf("use_gpu:%+v, gpu_id:%+v", config.UseGpu(), config.GpuDeviceId())
	t.Logf("MemoryPoolInitSizeMb:%+v, FractionOfGpuMemoryForPool:%+v", config.MemoryPoolInitSizeMb(), config.FractionOfGpuMemoryForPool())

	config.EnableTensorRtEngine(1024, 16, 3, PrecisionFloat32, false, false)
	t.Logf("TensorRtEngineEnabled:%+v", config.TensorRtEngineEnabled())

	minInputShape := map[string][]int32{
		"image": []int32{-1, 3, 100, 100},
		"shape": []int32{-1, 2},
	}
	maxInputShape := map[string][]int32{
		"image": []int32{-1, 3, 608, 608},
		"shape": []int32{-1, 2},
	}
	optInputShape := map[string][]int32{
		"image": []int32{-1, 3, 406, 406},
		"shape": []int32{-1, 2},
	}
	config.SetTRTDynamicShapeInfo(minInputShape, maxInputShape, optInputShape, false)

	config.EnableTensorRtOSS()
	t.Logf("TensorrtOssEnabled:%+v", config.TensorrtOssEnabled())

	config.EnableTensorRtDLA(0)
	t.Logf("TensorrtDlaEnabled:%+v", config.TensorrtDlaEnabled())

	config.DisableTensorRtOPs([]string{"mul", "fc"})

	config.EnableGpuMultiStream()
	t.Logf("ThreadLocalStreamEnabled:%+v", config.ThreadLocalStreamEnabled())

	config.SwitchIrDebug(false)

	config.EnableMKLDNN()

	config.EnableMemoryOptim()
	t.Logf("MemoryOptimEnabled:%+v", config.MemoryOptimEnabled())

	config.EnableProfile()
	t.Logf("ProfileEnabled:%+v", config.ProfileEnabled())

	config.DisableGlogInfo()
	t.Logf("GlogInfoDisabled:%+v", config.GlogInfoDisabled())

	t.Logf("IsValid:%+v", config.IsValid())

	config.AppendPass("test_pass")
	t.Logf("After AppendPass, AllPasses:%+v", config.AllPasses())

	config.DeletePass("test_pass")
	t.Logf("After DeletePass, AllPasses:%+v", config.AllPasses())
}

func TestLite(t *testing.T) {
	config := NewConfig()
	config.SetModel("model", "params")
	t.Log(config.ProgFile())
	t.Log(config.ParamsFile())

	config.EnableLiteEngine(PrecisionFloat32, true, []string{}, []string{})
	t.Logf("LiteEngineEnabled:%+v", config.LiteEngineEnabled())
}

func TestMkldnn(t *testing.T) {
	config := NewConfig()
	config.SetModelDir("modelDir")
	t.Log(config.ModelDir())

	config.EnableMKLDNN()
	t.Logf("MkldnnEnabled:%+v", config.MkldnnEnabled())

	config.SetMkldnnCacheCapacity(4)

	config.SetCpuMathLibraryNumThreads(4)
	t.Logf("CpuMathLibraryNumThreads:%+v", config.CpuMathLibraryNumThreads())

	config.SetMKLDNNOp([]string{"fc", "conv"})

	config.EnableMkldnnQuantizer()
	t.Logf("MkldnnQuantizerEnabled:%+v", config.MkldnnQuantizerEnabled())

	config.EnableMkldnnBfloat16()
	t.Logf("MkldnnBfloat16Enabled:%+v", config.MkldnnBfloat16Enabled())

	config.SetBfloat16Op([]string{"fc", "mul"})
}
