#pragma once

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-et-uberkernel-common.h"

#অন্তর্ভাব <device-layer/IDeviceLayer.h>
#অন্তর্ভাব <runtime/IProfiler.h>
#অন্তর্ভাব <runtime/IRuntime.h>

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <fstream>
#অন্তর্ভাব <string>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

কাঠামো ggml_backend_et_বাফার_type_context {
    পূর্ণসংখ্যা         devidx;
    std::string name;
};

কাঠামো ggml_backend_et_বাফার_context {
    পূর্ণসংখ্যা          devidx;
    শূন্য *       তথ্য;  // Device memory pointer
    size_t       আকার_2;
    rt::DeviceId rtid;
};

কাঠামো ggml_backend_et_context {
    পূর্ণসংখ্যা devidx;
};

কাঠামো ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
কাঠামো ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

কাঠামো ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    স্থির constexpr size_t         SLOT_গুনতি = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_গুনতি];
    size_t                          current_slot = 0;
};

কাঠামো ggml_backend_et_device_context {
    পূর্ণসংখ্যা                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     যোগ_mem;
    ggml_backend_বাফার_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId                                  default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_বাফার;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

কাঠামো ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
