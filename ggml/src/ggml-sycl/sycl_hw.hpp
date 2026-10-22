#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SYCL_HW_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_HW_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>

ਨਾਮਾਕਾਸ਼ syclex = sycl::ext::oneapi::experimental;
ਵਰਤੋਂ gpu_arch = sycl::ext::oneapi::experimental::architecture;

// It's used to mark the GPU computing capacity
// The value must flow the order of performance.
ਗਿਣਤੀ sycl_intel_gpu_family {
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

ਬਣਤਰ sycl_hw_info {
  syclex::architecture arch;
  ਅਚਲ ਅੱਖਰ* arch_name;
  int32_t device_id;
  std::string name;
  sycl_intel_gpu_family gpu_family;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

#ਜੇ_ਅੰਤ // SYCL_HW_HPP
