#pragma once

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-et-uberkernel-common.h"

#अन्तर्भाव <device-layer/IDeviceLayer.h>
#अन्तर्भाव <runtime/IProfiler.h>
#अन्तर्भाव <runtime/IRuntime.h>

#अन्तर्भाव <cstdint>
#अन्तर्भाव <fstream>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

संरचना ggml_backend_et_buffer_type_context {
    पूर्णाङ्क         devidx;
    std::string name;
};

संरचना ggml_backend_et_buffer_context {
    पूर्णाङ्क          devidx;
    शून्य *       data;  // Device memory pointer
    माप_प्रकार       size;
    rt::DeviceId rtid;
};

संरचना ggml_backend_et_context {
    पूर्णाङ्क devidx;
};

संरचना ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
संरचना ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    माप_प्रकार      device_insts_capacity  = 0;
    माप_प्रकार      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

संरचना ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    स्थिर constexpr माप_प्रकार         SLOT_गिनती = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_गिनती];
    माप_प्रकार                          current_slot = 0;
};

संरचना ggml_backend_et_device_context {
    पूर्णाङ्क                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    माप_प्रकार                     total_mem;
    ggml_backend_buffer_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId                                  default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_buffer;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

संरचना ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
