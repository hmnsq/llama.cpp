#ಸೂಚನೆ once

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

ರಚನೆ ggml_backend_et_ಬಫರ್_type_context {
    ಪೂರ್ಣಾಂಕ         devidx;
    std::string name;
};

ರಚನೆ ggml_backend_et_ಬಫರ್_context {
    ಪೂರ್ಣಾಂಕ          devidx;
    ಶೂನ್ಯ *       ದತ್ತಾಂಶ;  // Device memory pointer
    size_t       ಗಾತ್ರ_2;
    rt::DeviceId rtid;
};

ರಚನೆ ggml_backend_et_context {
    ಪೂರ್ಣಾಂಕ devidx;
};

ರಚನೆ ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
ರಚನೆ ggml_backend_et_uberkernel_ಸ್ಥಾನ {
    std::ಸದಿಶ<ggml_et_uberkernel_inst> insts;
    std::ಸದಿಶ<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_ಸಾಮರ್ಥ್ಯ  = 0;
    size_t      device_params_ಸಾಮರ್ಥ್ಯ = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

ರಚನೆ ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_ಮುಖವಾಡ = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    ಸ್ಥಿರ constexpr size_t         ಸ್ಥಾನ_ಎಣಿಕೆ_2 = 4;
    ggml_backend_et_uberkernel_ಸ್ಥಾನ slots[ಸ್ಥಾನ_ಎಣಿಕೆ_2];
    size_t                          current_ಸ್ಥಾನ = 0;
};

ರಚನೆ ggml_backend_et_device_context {
    ಪೂರ್ಣಾಂಕ                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     ಒಟ್ಟು_mem;
    ggml_backend_ಬಫರ್_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::ಹರಿವು_Id                                  default_ಹರಿವು;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_ಬಫರ್;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

ರಚನೆ ggml_backend_et_reg_ctx {
    std::ಸದಿಶ<ggml_backend_dev_t> devices;
};
