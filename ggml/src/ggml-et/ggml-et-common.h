#pragma once

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-et-uberkernel-common.h"

#அந்தர்பாவ <device-layer/IDeviceLayer.h>
#அந்தர்பாவ <runtime/IProfiler.h>
#அந்தர்பாவ <runtime/IRuntime.h>

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <string>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

கட்டமைப்பு ggml_backend_et_buffer_type_context {
    முழு         devidx;
    std::string name;
};

கட்டமைப்பு ggml_backend_et_buffer_context {
    முழு          devidx;
    வெறுமை *       data;  // Device memory pointer
    size_t       size;
    rt::DeviceId rtid;
};

கட்டமைப்பு ggml_backend_et_context {
    முழு devidx;
};

கட்டமைப்பு ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
கட்டமைப்பு ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

கட்டமைப்பு ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    நிலையான constexpr size_t         SLOT_மொத்தம் = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_மொத்தம்];
    size_t                          current_slot = 0;
};

கட்டமைப்பு ggml_backend_et_device_context {
    முழு                        devidx;
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

கட்டமைப்பு ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
