#आशय once

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-et-uberkernel-common.h"

#समावेश <device-layer/IDeviceLayer.h>
#समावेश <runtime/IProfiler.h>
#समावेश <runtime/IRuntime.h>

#समावेश <cstdint>
#समावेश <fstream>
#समावेश <string>
#समावेश <unordered_map>
#समावेश <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

काष्ठा ggml_backend_et_बफर_type_context {
    पूर्णांक         devidx;
    std::string name;
};

काष्ठा ggml_backend_et_बफर_context {
    पूर्णांक          devidx;
    व्योम *       तथ्य;  // Device memory pointer
    माप_प्रकार       आकार;
    rt::DeviceId rtid;
};

काष्ठा ggml_backend_et_context {
    पूर्णांक devidx;
};

काष्ठा ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
काष्ठा ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    माप_प्रकार      device_insts_capacity  = 0;
    माप_प्रकार      device_params_capacity = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

काष्ठा ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    अटल constexpr माप_प्रकार         SLOT_गिनती = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_गिनती];
    माप_प्रकार                          current_slot = 0;
};

काष्ठा ggml_backend_et_device_context {
    पूर्णांक                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    माप_प्रकार                     योग_mem;
    ggml_backend_बफर_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId                                  default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_बफर;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

काष्ठा ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
