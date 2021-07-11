#include "paddle/fluid/operators/npu_utils.h"
#include "paddle/fluid/operators/npu_op_runner.h"

using float16 = paddle::platform::float16;

namespace paddle {
namespace operators {

void alloc_float_status(const paddle::platform::NPUDeviceContext& ctx,
        paddle::framework::Tensor* float_status){
    const auto& runner = NpuOpRunner("NPUAllocFloatStatus", {}, {*float_status});
    auto stream = ctx.stream();
    runner.Run(stream);
}


bool FoundNanOrInf(const paddle::platform::NPUDeviceContext& ctx, aclrtStream stream, 
        const paddle::framework::Tensor* float_status, Tensor* tmp){
    const auto& runner_float_status =
        NpuOpRunner("NPUGetFloatStatus", {*float_status}, {*tmp},
                    {{"message", std::string("check_nan_and_inf")}});
    runner_float_status.Run(stream);

    paddle::framework::Tensor sum;
    sum.mutable_data<float>({1}, ctx.GetPlace());
    const auto& runner_reduce_sum =
        NpuOpRunner("ReduceSumD", {*float_status}, {sum},
                    {{"axes", std::vector<int>{0}}, {"keep_dims", true}});
    runner_reduce_sum.Run(stream);

    std::vector<float> sum_vec;
    TensorToVector(sum, ctx, &sum_vec);
    bool found_inf_data = (sum_vec[0] > 1);

    VLOG(4) << "found_inf_data:" << found_inf_data;
    return found_inf_data;
}


void clear_float_status(const platform::NPUDeviceContext& ctx, 
        Tensor* float_status, Tensor* tmp){
    const auto& runner_clear_status =
        paddle::operators::NpuOpRunner("NPUClearFloatStatus", {*float_status}, {*tmp});
    runner_clear_status.Run(ctx.stream());
}


// 0/0
int hlt_hccl_aclop_compile_and_exec_test(int deviceId, aclrtStream stream){
/*
  int inputnums = 2;
  int outputnums = 1;
  aclTensorDesc *inputDesc2[2];
  aclDataBuffer *inputs[2];
  aclopAttr *attr = nullptr;
  attr = aclopCreateAttr();
  ACLCHECK(aclopSetAttrBool(attr, "keep_dims", false));
  aclTensorDesc *outputDesc[1];
  aclTensorDesc *outputDesc2[1];
  aclDataBuffer *outputs[1];
  aclrtRunMode runMode;
  ACLCHECK(aclrtGetRunMode(&runMode));

  int64_t shapecom[] = {1};
  int64_t constshapecom[] = {1};
  int64_t outShapecom[] = {1};
  inputDesc2[0] = aclCreateTensorDesc(ACL_FLOAT,1,shapecom,ACL_FORMAT_ND);
  inputDesc2[1] = aclCreateTensorDesc(ACL_FLOAT,1,constshapecom,ACL_FORMAT_ND);
  for (size_t i = 0; i < outputnums; ++i){
    outputDesc2[i] = aclCreateTensorDesc(ACL_FLOAT,1,outShapecom,ACL_FORMAT_ND);
  }
  void *constDataBuffer = nullptr;
  float k = 0.0;
  size_t lenConst = sizeof(float);

  float inputA[1] = {0.0};        
  const float K=0.0;
  const void *hostInputs[] = {inputA};
  std::vector<void *> devInputs_;
  auto *inputDescAddress = inputDesc2[0];  
  size_t tensorSize = aclGetTensorDescSize(inputDescAddress);
  LOG(LOG_DEBUG, "input tensorSize = %d\n",tensorSize);
  void *devBuffer = nullptr;
  ACLCHECK(aclrtMalloc(&devBuffer, tensorSize+padding, ACL_MEM_MALLOC_NORMAL_ONLY));
  devInputs_.emplace_back(devBuffer);

  inputs[0] = aclCreateDataBuffer(devBuffer, tensorSize);
  LOG(LOG_DEBUG, "host inputs:%f, %f, %f\n", hostInputs[0], inputA[0], K);
  assert(K==0.0);
  //LOG(LOG_DEBUG, "host inputs:%f, %f\n", hostInputs[0],  K);
  //LOG(LOG_DEBUG, "host inputs:%f\n", K);
  //ACLCHECK(aclrtMemcpy(devInputs_[0], tensorSize, &hostInputs[0], tensorSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACLCHECK(aclrtMemcpy(devInputs_[0], tensorSize, &K, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE));

  float testHostInput={3.0};
  LOG(LOG_DEBUG, "input tensorSize is %zu---------\n", tensorSize );
  ACLCHECK(aclrtMemcpy(&testHostInput, tensorSize, devInputs_[0], tensorSize, ACL_MEMCPY_DEVICE_TO_HOST));

  for (int m = 0 ; m < 1; m ++)
  {
    cout << __LINE__ << "\t" << testHostInput <<" ";
  }
  cout << endl;
  const float *devk=nullptr;
  k = 0.0;
  ACLCHECK(aclrtMalloc((void **) &devk, sizeof(float)+padding, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACLCHECK(aclrtMemcpy((void *) devk, sizeof(float), &k, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  inputs[1] = aclCreateDataBuffer((void *) devk, sizeof(float));
  //printf();
  //??
  size_t outtensorSize;
  std::vector<void *> hostOutputs_;
  std::vector<void *> devOutputs_;
  for (size_t i = 0; i < outputnums; ++i){
    auto *outputDescAddress = outputDesc2[i];
    outtensorSize = aclGetTensorDescSize(outputDescAddress);
    LOG(LOG_DEBUG, "tensorSize = %d \n",outtensorSize);
    void *devBuffer = nullptr;
    ACLCHECK(aclrtMalloc(&devBuffer, outtensorSize+padding, ACL_MEM_MALLOC_NORMAL_ONLY));
    devOutputs_.emplace_back(devBuffer);
    outputs[i] = aclCreateDataBuffer(devBuffer, outtensorSize);
    //ACL_REQUIRE_OK(aclrtMemcpy(devOutputs_[0], outtensorSize, hostInputs, outtensorSize, ACL_MEMCPY_HOST_TO_DEVICE));
  }
  char *path = getcwd(NULL,0);
  LOG(DEBUG, "path:%s\n", path);
  ACLCHECK(aclopCompileAndExecute("Div",inputnums,inputDesc2,inputs,outputnums,outputDesc2,outputs,attr,ACL_ENGINE_SYS , ACL_COMPILE_SYS, NULL,stream));
  ACLCHECK(aclrtSynchronizeStream(stream));
  float output[1];
  for (size_t i = 0; i < outputnums; ++i){
    if(runMode == 1){
      ACLCHECK(aclrtMemcpy(output, outtensorSize, devOutputs_[i], outtensorSize, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else{
      ACLCHECK(aclrtMemcpy(output, outtensorSize, devOutputs_[i], outtensorSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    for (int j = 0; j < 1; ++j) {
      cout << __LINE__ << "\t" << output[j]<<" ";
      cout << endl;
    }
  }
  //ASSERT_EQ(0, result);
  LOG(LOG_DEBUG, "To release Add op resources \n");
  for (size_t i = 0; i < inputnums; ++i) {
    aclDestroyTensorDesc(inputDesc2[i]);
    ACLCHECK(aclrtFree(aclGetDataBufferAddr(inputs[i])));
    aclDestroyDataBuffer(inputs[i]);
  }
  LOG(LOG_DEBUG, "To release outputDesc/outputs \n");
  for (size_t i = 0; i < outputnums; ++i) {
    aclDestroyTensorDesc(outputDesc2[i]);
    ACLCHECK(aclrtFree(aclGetDataBufferAddr(outputs[i])));
    aclDestroyDataBuffer(outputs[i]);
  }
  //LOG(LOG_DEBUG, "To release stream \n");
  //ACLCHECK(aclrtDestroyStream(stream));
  */
  return 0;
}

};
};


