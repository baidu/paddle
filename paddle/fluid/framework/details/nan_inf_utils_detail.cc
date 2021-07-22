// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/fluid/framework/op_proto_maker.h"
namespace paddle {
namespace framework {
namespace details {

static std::once_flag white_list_init_flag;

static int op_role_nan_inf_white_list = 0;

static constexpr int FORWARD = 0x10000;

// lazy init
static const std::unordered_map<std::string, int>& role_str2int() {
  /* In op_proto_maker.h
   * framework::OpRole::kForward      = 0x0000,
   * framework::OpRole::kBackward     = 0x0001,
   * framework::OpRole::kOptimize     = 0x0002,
   * framework::OpRole::kRPC          = 0x0004,
   * framework::OpRole::kDist         = 0x0008,
   * framework::OpRole::kLRSched      = 0x0010,
   * framework::OpRole::kLoss         = 0x0100,
   * framework::OpRole::kNotSpecified = 0x1000,
   */
  static const std::unordered_map<std::string, int> _role_str2int = {
      {"forward", FORWARD}, /* kForward=0, can't filter */
      {"backward", static_cast<int>(framework::OpRole::kBackward)},
      {"optimize", static_cast<int>(framework::OpRole::kOptimize)},
      {"rpc", static_cast<int>(framework::OpRole::kRPC)},
      {"dist", static_cast<int>(framework::OpRole::kDist)},
      {"lrsched", static_cast<int>(framework::OpRole::kLRSched)},
      {"loss", static_cast<int>(framework::OpRole::kLoss)},
      {"default", static_cast<int>(framework::OpRole::kNotSpecified)},
  };
  return _role_str2int;
}

static std::unordered_set<std::string>& op_type_nan_inf_white_list() {
  static std::unordered_set<std::string> _op_type_nan_inf_white_list = {
      "coalesce_tensor", /* This Op will alloc tensor, and may not init space */
  };
  return _op_type_nan_inf_white_list;
}

static std::unordered_map<std::string, std::vector<std::string>>&
op_var_nan_inf_white_list() {
  static std::unordered_map<std::string, std::vector<std::string>>
      _op_var_nan_inf_white_list = {
          /* encoded & gather var consist of idx&val, can't judge directly */
          {"dgc", {"__dgc_encoded__", "__dgc_gather__"}},
      };
  return _op_var_nan_inf_white_list;
}

static void InitWhiteListFormEnv() {
  // op_type_skip and op_var_skip may be NULL.
  // So need init static value in there, prevent thread competition.
  // NOTE. role_str2int needn't do this for it only used in this func.
  op_type_nan_inf_white_list();
  op_var_nan_inf_white_list();

  // export PADDLE_INF_NAN_SKIP_OP="op0,op1,op2"
  // export PADDLE_INF_NAN_SKIP_ROLE="role1,role2,role3"
  // export PADDLE_INF_NAN_SKIP_VAR="op0:var0,op0:var1,op1:var0"
  const char* op_type_skip = std::getenv("PADDLE_INF_NAN_SKIP_OP");
  const char* op_role_skip = std::getenv("PADDLE_INF_NAN_SKIP_ROLE");
  const char* op_var_skip = std::getenv("PADDLE_INF_NAN_SKIP_VAR");

  if (op_type_skip != NULL) {
    std::stringstream ss(op_type_skip);
    std::string op_type;
    while (std::getline(ss, op_type, ',')) {
      op_type_nan_inf_white_list().emplace(op_type);
    }
  }

  if (op_role_skip != NULL) {
    std::stringstream ss(op_role_skip);
    std::string op_role;
    while (std::getline(ss, op_role, ',')) {
      PADDLE_ENFORCE_EQ(role_str2int().find(op_role) != role_str2int().end(),
                        true,
                        platform::errors::InvalidArgument(
                            "Skip role must be one of "
                            "{forward,backward,optimize,rpc,dist,lrsched,loss,"
                            "default}, instead of %s",
                            op_role));
      op_role_nan_inf_white_list |= role_str2int().at(op_role);
    }
  }

  if (op_var_skip != NULL) {
    std::stringstream ss(op_var_skip);
    std::string op_var;
    while (std::getline(ss, op_var, ',')) {
      auto pos = op_var.find(":");
      PADDLE_ENFORCE_EQ(
          pos != std::string::npos, true,
          platform::errors::InvalidArgument(
              "Skip var format must be op:var, instead of %s", op_var));
      std::string op = op_var.substr(0, pos);
      std::string var = op_var.substr(pos + 1);

      op_var_nan_inf_white_list()[op].emplace_back(var);
    }
  }
}

template <typename T>
static void PrintNanInf(const T* value, const size_t numel, int print_num,
                        const std::string& op_type,
                        const std::string& var_name) {
  size_t nan_count, inf_count, num_count;
  nan_count = inf_count = num_count = 0;

  // CPU print num value
  for (size_t i = 0; i < numel; ++i) {
    size_t count = 0;
    if (std::isnan(value[i])) {
      count = nan_count++;
    } else if (std::isinf(value[i])) {
      count = inf_count++;
    } else {
      count = num_count++;
    }

    if (count < static_cast<size_t>(print_num)) {
      printf("numel:%lu index:%lu value:%f\n", static_cast<uint64_t>(numel),
             static_cast<uint64_t>(i), static_cast<float>(value[i]));
    }
  }
  printf("In cpu, there has %lu,%lu,%lu nan,inf,num\n",
         static_cast<uint64_t>(nan_count), static_cast<uint64_t>(inf_count),
         static_cast<uint64_t>(num_count));
  // PADDLE_THROW(platform::errors::PreconditionNotMet(
  //     "There are `nan` or `inf` in tensor (%s) of operator (%s).", var_name,
  //     op_type));
  VLOG(0) << "yoki: There are `nan` or `inf` in tensor (" << var_name
          << ") of operator (" << op_type << ").";
}

// openmp 4.0, reduction with fp16
#if defined _OPENMP && _OPENMP >= 201307
// more detail see: 180 page of
// https://www.openmp.org/wp-content/uploads/OpenMP4.0.0.pdf
#pragma omp declare reduction(+ : paddle::platform::float16 : omp_out += omp_in)
#pragma omp declare reduction(+ : paddle::platform::bfloat16 : omp_out += \
                              omp_in)
#pragma omp declare reduction(+ : paddle::platform::complex < \
                                  float > : omp_out += omp_in)
#pragma omp declare reduction(+ : paddle::platform::complex < \
                                  double > : omp_out += omp_in)

#endif

template <typename T>
static void CheckNanInf(const T* value, const size_t numel, int print_num,
                        const std::string& op_type,
                        const std::string& var_name) {
  T sum = static_cast<T>(0.0);
#if defined _OPENMP && _OPENMP >= 201307
#pragma omp parallel for simd reduction(+ : sum)
#elif defined _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
  for (size_t i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

#if defined _OPENMP && _OPENMP >= 201307
// openmp4.0 not need to specialization fp16
#elif defined _OPENMP
template <>
void CheckNanInf<paddle::platform::float16>(
    const paddle::platform::float16* value, const size_t numel, int print_num,
    const std::string& op_type, const std::string& var_name) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < numel; ++i) {
    sum += static_cast<float>(value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

template <>
void CheckNanInf<paddle::platform::bfloat16>(
    const paddle::platform::bfloat16* value, const size_t numel, int print_num,
    const std::string& op_type, const std::string& var_name) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < numel; ++i) {
    sum += static_cast<float>(value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

template <>
void CheckNanInf<paddle::platform::complex<float>>(
    const paddle::platform::complex<float>* value, const size_t numel,
    int print_num, const std::string& op_type, const std::string& var_name) {
  float real_sum = 0.0f;
#pragma omp parallel for reduction(+ : real_sum)
  for (size_t i = 0; i < numel; ++i) {
    real_sum += (value[i].real - value[i].real);
  }

  float imag_sum = 0.0f;
#pragma omp parallel for reduction(+ : imag_sum)
  for (size_t i = 0; i < numel; ++i) {
    imag_sum += (value[i].imag - value[i].imag);
  }

  if (std::isnan(real_sum) || std::isinf(real_sum) || std::isnan(imag_sum) ||
      std::isinf(imag_sum)) {
    // hot fix for compile failed in gcc4.8
    // here also need print detail info of nan or inf later
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are `nan` or `inf` in tensor (%s) of operator (%s).", var_name,
        op_type));
  }
}

template <>
    void CheckNanInf<paddle::platform::complex<double>>>
    (const paddle::platform::complex<double>* value, const size_t numel,
     int print_num, const std::string& op_type, const std::string& var_name) {
  double real_sum = 0.0;
#pragma omp parallel for reduction(+ : real_sum)
  for (size_t i = 0; i < numel; ++i) {
    real_sum += (value[i].real - value[i].real);
  }

  double imag_sum = 0.0;
#pragma omp parallel for reduction(+ : imag_sum)
  for (size_t i = 0; i < numel; ++i) {
    imag_sum += (value[i].imag - value[i].imag);
  }

  if (std::isnan(real_sum) || std::isinf(real_sum) || std::isnan(imag_sum) ||
      std::isinf(imag_sum)) {
    // hot fix for compile failed in gcc4.8
    // here also need print detail info of nan or inf later
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are `nan` or `inf` in tensor (%s) of operator (%s).", var_name,
        op_type));
  }
}

#endif

template <>
template <typename T>
void TensorCheckerVisitor<platform::CPUDeviceContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  // use env strategy control in future, -1=print_all.
  int print_num = 3;
  CheckNanInf(tensor_.data<T>(), tensor_.numel(), print_num, op_type_,
              var_name_);
}

template <>
void tensor_check<platform::CPUDeviceContext>(const std::string& op_type,
                                              const std::string& var_name,
                                              const framework::Tensor& tensor,
                                              const platform::Place& place) {
  TensorCheckerVisitor<platform::CPUDeviceContext> vistor(op_type, var_name,
                                                          tensor, place);
  VisitDataType(tensor.type(), vistor);
}

template <>
template <typename T>
void NPUTensorCheckerVisitor<platform::NPUDeviceContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  // use env strategy control in future, -1=print_all.
  framework::LoDTensor cpu_tensor;
  cpu_tensor.Resize(tensor_.dims());
  cpu_tensor.mutable_data(platform::CPUPlace(), tensor_.type());

  framework::TensorCopySync(tensor_, platform::CPUPlace(), &cpu_tensor);

  int print_num = 3;
  CheckNanInf(cpu_tensor.data<T>(), cpu_tensor.numel(), print_num, op_type_,
              var_name_);
}

template <>
void npu_tensor_check<platform::NPUDeviceContext>(
    const std::string& op_type, const std::string& var_name,
    const framework::Tensor& tensor, const platform::Place& place) {
  NPUTensorCheckerVisitor<platform::NPUDeviceContext> vistor(op_type, var_name,
                                                             tensor, place);
  VisitDataType(tensor.type(), vistor);
}

void CheckVarHasNanOrInf(const std::string& op_type,
                         const std::string& var_name,
                         const framework::Variable* var,
                         const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound("Cannot find var: `%s` in op `%s`.",
                                      var_name, op_type));

  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<framework::SelectedRows>()) {
    tensor = &var->Get<framework::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  if (platform::is_gpu_place(tensor->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    tensor_check<platform::CUDADeviceContext>(op_type, var_name, *tensor,
                                              place);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Tensor[%s] use gpu place. PaddlePaddle must compile with GPU.",
        var_name));
#endif
    return;
  } else if (platform::is_xpu_place(tensor->place())) {
#ifdef PADDLE_WITH_XPU
    if (tensor->type() != proto::VarType::FP32) {
      return;
    }

    float* cpu_data = new float[tensor->numel()];
    xpu_memcpy(cpu_data, tensor->data<float>(), tensor->numel() * sizeof(float),
               XPU_DEVICE_TO_HOST);
    bool flag = false;
    for (int i = 0; i < tensor->numel(); i++) {
      if (isnan(cpu_data[i]) || isinf(cpu_data[i])) {
        flag = true;
        break;
      }
    }
    delete[] cpu_data;
    PADDLE_ENFORCE_NE(
        flag, true,
        platform::errors::Fatal("Operator %s output Tensor %s contains Inf.",
                                op_type, var_name));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Tensor[%s] use xpu place. PaddlePaddle must compile with XPU.",
        var_name));
#endif
    return;
  } else if (platform::is_npu_place(tensor->place())) {
#ifdef PADDLE_WITH_ASCEND_CL
    npu_tensor_check<platform::NPUDeviceContext>(op_type, var_name, *tensor,
                                                 place);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Tensor[%s] use npu place. PaddlePaddle must compile with NPU.",
        var_name));
#endif
    return;
  }
  tensor_check<platform::CPUDeviceContext>(op_type, var_name, *tensor, place);
}

void CheckVarHasNanOrInf(const std::string& op_type,
                         const framework::Scope& scope,
                         const std::string& var_name,
                         const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  CheckVarHasNanOrInf(op_type, var_name, var, place);
}

bool IsSkipOp(const framework::OperatorBase& op) {
  if (op_type_nan_inf_white_list().count(op.Type()) != 0) return true;

  int op_role = op.template Attr<int>(
      framework::OpProtoAndCheckerMaker::OpRoleAttrName());

  // kForward=0, can't filter
  if (op_role == static_cast<int>(framework::OpRole::kForward)) {
    op_role = FORWARD;
  }
  if (op_role_nan_inf_white_list & op_role) return true;

  return false;
}

void CheckOpHasNanOrInf(const framework::OperatorBase& op,
                        const framework::Scope& exec_scope,
                        const platform::Place& place) {
  std::call_once(white_list_init_flag, InitWhiteListFormEnv);

  if (IsSkipOp(op)) return;

  if (op_var_nan_inf_white_list().count(op.Type()) == 0) {
    // NOTE. vname may destruct in the end of this func.
    for (auto& vname : op.OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  } else {
    for (auto& vname : op.OutputVars(true)) {
      bool need_check = true;
      for (auto& white_vname : op_var_nan_inf_white_list().at(op.Type())) {
        if (vname.find(white_vname) != std::string::npos) {
          need_check = false;
          break;
        }
      }
      if (!need_check) continue;
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
