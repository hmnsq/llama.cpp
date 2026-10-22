#pragma once

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-et-uberkernel-common.h"

#અન્તર્ભાવ <device-layer/IDeviceLayer.h>
#અન્તર્ભાવ <runtime/IProfiler.h>
#અન્તર્ભાવ <runtime/IRuntime.h>

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

સંરચના ggml_backend_et_buffer_type_context {
    પૂર્ણાંક         devidx;
    std::string name;
};

સંરચના ggml_backend_et_buffer_context {
    પૂર્ણાંક          devidx;
    શૂન્ય *       data;  // Device memory pointer
    size_t       size;
    rt::DeviceId rtid;
};

સંરચના ggml_backend_et_context {
    પૂર્ણાંક devidx;
};

સંરચના ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
સંરચના ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

સંરચના ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    સ્થિર constexpr size_t         SLOT_ગણતરી = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_ગણતરી];
    size_t                          current_slot = 0;
};

સંરચના ggml_backend_et_device_context {
    પૂર્ણાંક                        devidx;
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

સંરચના ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
