#هدايت once

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-et-uberkernel-common.h"

#ضمن <device-layer/IDeviceLayer.h>
#ضمن <runtime/IProfiler.h>
#ضمن <runtime/IRuntime.h>

#ضمن <cstdint>
#ضمن <fstream>
#ضمن <string>
#ضمن <unordered_map>
#ضمن <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

بنية ggml_backend_et_بفر_type_context {
    صحيح         devidx;
    std::string name;
};

بنية ggml_backend_et_بفر_context {
    صحيح          devidx;
    فراغ *       معلومات;  // Device memory pointer
    size_t       سائيز;
    rt::DeviceId rtid;
};

بنية ggml_backend_et_context {
    صحيح devidx;
};

بنية ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
بنية ggml_backend_et_uberkernel_جڳهه {
    std::ويڪٽر<ggml_et_uberkernel_inst> insts;
    std::ويڪٽر<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_صلاحيت  = 0;
    size_t      device_params_صلاحيت = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

بنية ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_نقاب = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ساكن constexpr size_t         جڳهه_ڳڻپ_2 = 4;
    ggml_backend_et_uberkernel_جڳهه slots[جڳهه_ڳڻپ_2];
    size_t                          current_جڳهه = 0;
};

بنية ggml_backend_et_device_context {
    صحيح                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     ڪل_mem;
    ggml_backend_بفر_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::وهڪرو_Id                                  default_وهڪرو;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_بفر;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

بنية ggml_backend_et_reg_ctx {
    std::ويڪٽر<ggml_backend_dev_t> devices;
};
