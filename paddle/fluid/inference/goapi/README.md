# Paddle Inference golang API

Paddle Inference golang API 基于 [capi](../capi_exp) 和 cgo 实现，需要您提前准备好C预测库。

## 安装

1. 确认使用Paddle的CommitId

您可以通过`git log -1`的方式，确认您使用的Paddle版本的CommitId

2. 使用`go get`获取golang paddle api

```
# 此处使用上一步记录的CommitId，假设为76e5724
COMMITID=76e5724
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@${COMMITID}
```

3. 下载C预测库

您可以选择直接下载[paddle_inference_c](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/docs/user_guides/download_lib.md)预测库，或通过源码编译的方式安装，源码编译方式参考官网文档，注意这里cmake编译时打开`-DON_INFER=ON`,在编译目录下得到`paddle_inference_c_install_dir`。


4. 软链

go1.15新增了`GOMODCACHE`环境变量，`go get`默认会将代码下载到`GOMODCACHE`目录下，您可以通过`go env | grep GOMODCACHE`的方式，查看该路径，在官网发布的docker镜像中该路径一般默认为`/root/gopath/pkg/mod`，进入到golang api代码路径建立软连接，将c预测库命名为`paddle_inference_c`。

```bash
eval $(go env | grep GOMODCACHE)
# 按需修改最后的goapi版本号
cd ${GOMODCACHE}/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20210517084506-76e5724c16a5/
ln -s ${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c_install_dir paddle_inference_c
```

5. 运行单测，验证

```
bash test.sh
```

## 在Go中使用Paddle预测

首先创建预测配置
```go
config := paddle.NewConfig()
config.SetModel(model_file, params_file)
```

创建predictor
```go
predictor := paddle.NewPredictor(config)
```

获取输入Tensor和输出Tensor
```go
inNames := predictor.GetInputNames()
inHandle = predictor.GetInputHandle(inNames[0])

outNames := predictor.GetOutputNames()
outHandle := predictor.GetOutputHandle(outNames[0])
```

设置输入数据(假设只有一个输入)
```go
data := make([]float32, 1*3*224*224)
for i := 0; i < len(data); i++ {
    data[i] = float32(i%255) * 0.1
}
inHandle.Reshape([]int32{1, 3, 224, 224})
inHandle.CopyFromCpu(data)
```

设置Lod
```go
lod := make([][]uint, 2)
for i:=0; i < len(lod); i++ {
    lod[i] = make([]uint, 2)
    // 设置输入...
    lod[i][0] = 0
    lod[i][0] = 10
}
inHandle.SetLod(lod)
```

运行预测
```go
predictor.Run()
```

获取输入Tensor的真实值
```go
func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}

outData := make([]float32, numElements(outHandle.Shape()))
outHandle.CopyToCpu(outData)
fmt.Println(outHandle.Lod())
```

## 示例

Demo示例见[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/go)
