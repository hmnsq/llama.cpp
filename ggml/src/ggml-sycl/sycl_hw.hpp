#अगर_अघोषित SYCL_HW_HPP
#घोषणा SYCL_HW_HPP

#समावेश <algorithm>
#समावेश <stdio.h>
#समावेश <vector>
#समावेश <map>

#समावेश <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
using gpu_arch = sycl::ext::oneapi::experimental::architecture;

// It's used to mark the GPU computing capacity
// The value must flow the order of performance.
क्रमागत sycl_intel_gpu_family {
  GPU_FAMILY_UKNOWN = -1,
  // iGPU without Xe core, before Meteor Lake iGPU(Xe)
  GPU_FAMILY_IGPU_NON_XE = 0,
  // iGPU with Xe core, Meteor Lake iGPU or newer.
  GPU_FAMILY_IGPU_XE = 1,
  // dGPU for gaming in client/data center (DG1/FLex 140 or newer).
  GPU_FAMILY_DGPU_CLIENT_GAME = 2,
  // dGPU for AI in cloud, PVC or newer.
  GPU_FAMILY_DGPU_CLOUD = 3
};

काष्ठा sycl_hw_info {
  syclex::architecture arch;
  स्थिर अक्षर* arch_name;
  int32_t device_id;
  std::string name;
  sycl_intel_gpu_family gpu_family;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

#पूर्ण_अगर // SYCL_HW_HPP
