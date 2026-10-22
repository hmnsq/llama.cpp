#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-et-uberkernel-common.h"

#ଅନ୍ତର୍ଭାଵ <device-layer/IDeviceLayer.h>
#ଅନ୍ତର୍ଭାଵ <runtime/IProfiler.h>
#ଅନ୍ତର୍ଭାଵ <runtime/IRuntime.h>

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

ସଂରଚନା ggml_backend_et_buffer_type_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା         devidx;
    std::string name;
};

ସଂରଚନା ggml_backend_et_buffer_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା          devidx;
    ଶୂନ୍ୟ *       data;  // Device memory pointer
    size_t       size;
    rt::DeviceId rtid;
};

ସଂରଚନା ggml_backend_et_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା devidx;
};

ସଂରଚନା ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
ସଂରଚନା ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

ସଂରଚନା ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ସ୍ଥିର constexpr size_t         SLOT_ଗଣନା = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_ଗଣନା];
    size_t                          current_slot = 0;
};

ସଂରଚନା ggml_backend_et_device_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        devidx;
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

ସଂରଚନା ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
