#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ SYCL_HW_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_HW_HPP

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <map>

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

ನಾಮಾಕಾಶ syclex = sycl::ext::oneapi::experimental;
ಬಳಸಿ gpu_arch = sycl::ext::oneapi::experimental::architecture;

// It's used to mark the GPU computing capacity
// The value must flow the order of performance.
ಎಣಿಕೆ sycl_intel_gpu_family {
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

ರಚನೆ sycl_hw_info {
  syclex::architecture arch;
  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* arch_name;
  int32_t device_id;
  std::string name;
  sycl_intel_gpu_family gpu_family;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

#ಆದರೆ_ಅಂತ್ಯ // SYCL_HW_HPP
