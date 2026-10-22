#ہدایت once

#شامل "ggml-backend-impl.h"
#شامل "ggml-et-uberkernel-common.h"

#شامل <device-layer/IDeviceLayer.h>
#شامل <runtime/IProfiler.h>
#شامل <runtime/IRuntime.h>

#شامل <cstdint>
#شامل <fstream>
#شامل <string>
#شامل <unordered_map>
#شامل <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

ساخت ggml_backend_et_بفر_type_context {
    صحیح         devidx;
    std::string name;
};

ساخت ggml_backend_et_بفر_context {
    صحیح          devidx;
    خالی *       ڈیٹا;  // Device memory pointer
    size_t       سائز;
    rt::DeviceId rtid;
};

ساخت ggml_backend_et_context {
    صحیح devidx;
};

ساخت ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
ساخت ggml_backend_et_uberkernel_خانہ {
    std::سمتیہ<ggml_et_uberkernel_inst> insts;
    std::سمتیہ<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_استعداد  = 0;
    size_t      device_params_استعداد = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

ساخت ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_نقاب = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ساکن constexpr size_t         خانہ_گنتی_2 = 4;
    ggml_backend_et_uberkernel_خانہ slots[خانہ_گنتی_2];
    size_t                          current_خانہ = 0;
};

ساخت ggml_backend_et_device_context {
    صحیح                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     کل_mem;
    ggml_backend_بفر_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::دھارا_Id                                  default_دھارا;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_بفر;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

ساخت ggml_backend_et_reg_ctx {
    std::سمتیہ<ggml_backend_dev_t> devices;
};
