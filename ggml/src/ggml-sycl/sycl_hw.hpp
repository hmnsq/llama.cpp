#నిర్వచితం_కాకపోతే SYCL_HW_HPP
#నిర్వచించు SYCL_HW_HPP

#అన్తర్భావ <algorithm>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <vector>
#అన్తర్భావ <map>

#అన్తర్భావ <sycl/sycl.hpp>

నామాకాశ syclex = sycl::ext::oneapi::experimental;
ఉపయోగించి gpu_arch = sycl::ext::oneapi::experimental::architecture;

// It's used to mark the GPU computing capacity
// The value must flow the order of performance.
గణన sycl_intel_gpu_family {
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

నిర్మాణం sycl_hw_info {
  syclex::architecture arch;
  స్థిరరాశి అక్షరం* arch_name;
  int32_t device_id;
  std::string name;
  sycl_intel_gpu_family gpu_family;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

#అయితే_ముగింపు // SYCL_HW_HPP
