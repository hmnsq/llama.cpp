#pragma once

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

ساختار ggml_backend_et_buffer_type_context {
    صحیح         devidx;
    std::string name;
};

ساختار ggml_backend_et_buffer_context {
    صحیح          devidx;
    تهی *       data;  // Device memory pointer
    size_t       size;
    rt::DeviceId rtid;
};

ساختار ggml_backend_et_context {
    صحیح devidx;
};

ساختار ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
ساختار ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

ساختار ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ایستا constexpr size_t         SLOT_شمار = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_شمار];
    size_t                          current_slot = 0;
};

ساختار ggml_backend_et_device_context {
    صحیح                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     total_mem;
    ggml_backend_buffer_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId                                  default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_buffer;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

ساختار ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
