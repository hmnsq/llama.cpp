#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-uberkernel-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <device-layer/IDeviceLayer.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <runtime/IProfiler.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <runtime/IRuntime.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

ਬਣਤਰ ggml_backend_et_ਬਫਰ_type_context {
    ਪੂਰਨ         devidx;
    std::string name;
};

ਬਣਤਰ ggml_backend_et_ਬਫਰ_context {
    ਪੂਰਨ          devidx;
    ਖਾਲੀ *       ਅੰਕੜੇ;  // Device memory pointer
    size_t       ਮਾਪ;
    rt::DeviceId rtid;
};

ਬਣਤਰ ggml_backend_et_context {
    ਪੂਰਨ devidx;
};

ਬਣਤਰ ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
ਬਣਤਰ ggml_backend_et_uberkernel_ਥਾਂ {
    std::ਸਦਿਸ਼<ggml_et_uberkernel_inst> insts;
    std::ਸਦਿਸ਼<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_ਸਮਰੱਥਾ  = 0;
    size_t      device_params_ਸਮਰੱਥਾ = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

ਬਣਤਰ ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_ਨਕਾਬ = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ਸਥਿਰ constexpr size_t         ਥਾਂ_ਗਿਣਤੀ_2 = 4;
    ggml_backend_et_uberkernel_ਥਾਂ slots[ਥਾਂ_ਗਿਣਤੀ_2];
    size_t                          current_ਥਾਂ = 0;
};

ਬਣਤਰ ggml_backend_et_device_context {
    ਪੂਰਨ                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     ਕੁੱਲ_mem;
    ggml_backend_ਬਫਰ_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::ਧਾਰਾ_Id                                  default_ਧਾਰਾ;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_ਬਫਰ;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

ਬਣਤਰ ggml_backend_et_reg_ctx {
    std::ਸਦਿਸ਼<ggml_backend_dev_t> devices;
};
