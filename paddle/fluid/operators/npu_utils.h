#pragma once

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"

#include "paddle/fluid/platform/hccl_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using NPUAttribute = framework::NPUAttribute;
using NPUAttributeMap = framework::NPUAttributeMap;

void alloc_float_status(const paddle::platform::NPUDeviceContext& ctx, Tensor* float_status);

bool FoundNanOrInf(const paddle::platform::NPUDeviceContext& ctx, aclrtStream stream, 
        const Tensor* float_status, Tensor* tmp);

void clear_float_status(const paddle::platform::NPUDeviceContext& ctx,
        Tensor* float_status, Tensor* tmp);

int hlt_hccl_aclop_compile_and_exec_test(int deviceId, aclrtStream stream);
};
};

