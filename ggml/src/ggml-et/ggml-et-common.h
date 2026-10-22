#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-common.h"

#ಅನ್ತರ್ಭಾವ <device-layer/IDeviceLayer.h>
#ಅನ್ತರ್ಭಾವ <runtime/IProfiler.h>
#ಅನ್ತರ್ಭಾವ <runtime/IRuntime.h>

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

ಸಂರಚನಾ ggml_backend_et_buffer_type_context {
    ಪೂರ್ಣಾಙ್ಕ         devidx;
    std::string name;
};

ಸಂರಚನಾ ggml_backend_et_buffer_context {
    ಪೂರ್ಣಾಙ್ಕ          devidx;
    ಶೂನ್ಯ *       data;  // Device memory pointer
    size_t       size;
    rt::DeviceId rtid;
};

ಸಂರಚನಾ ggml_backend_et_context {
    ಪೂರ್ಣಾಙ್ಕ devidx;
};

ಸಂರಚನಾ ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
ಸಂರಚನಾ ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

ಸಂರಚನಾ ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ಸ್ಥಿರ constexpr size_t         SLOT_ಎಣಿಕೆ = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_ಎಣಿಕೆ];
    size_t                          current_slot = 0;
};

ಸಂರಚನಾ ggml_backend_et_device_context {
    ಪೂರ್ಣಾಙ್ಕ                        devidx;
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

ಸಂರಚನಾ ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
