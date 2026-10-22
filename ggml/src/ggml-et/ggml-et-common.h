#సూచన once

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-et-uberkernel-common.h"

#అన్తర్భావ <device-layer/IDeviceLayer.h>
#అన్తర్భావ <runtime/IProfiler.h>
#అన్తర్భావ <runtime/IRuntime.h>

#అన్తర్భావ <cstdint>
#అన్తర్భావ <fstream>
#అన్తర్భావ <string>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

నిర్మాణం ggml_backend_et_బఫరు_type_context {
    పూర్ణాంకం         devidx;
    std::string name;
};

నిర్మాణం ggml_backend_et_బఫరు_context {
    పూర్ణాంకం          devidx;
    శూన్య *       దత్తాంశం;  // Device memory pointer
    size_t       కొలత;
    rt::DeviceId rtid;
};

నిర్మాణం ggml_backend_et_context {
    పూర్ణాంకం devidx;
};

నిర్మాణం ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
నిర్మాణం ggml_backend_et_uberkernel_స్థానం {
    std::సదిశ<ggml_et_uberkernel_inst> insts;
    std::సదిశ<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_సామర్థ్యం  = 0;
    size_t      device_params_సామర్థ్యం = 0;

    rt::EventId pending_event{};
    bool        has_pending = false;
};

నిర్మాణం ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_ముసుగు = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    స్థిర constexpr size_t         స్థానం_లెక్క_2 = 4;
    ggml_backend_et_uberkernel_స్థానం slots[స్థానం_లెక్క_2];
    size_t                          current_స్థానం = 0;
};

నిర్మాణం ggml_backend_et_device_context {
    పూర్ణాంకం                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     మొత్తం_mem;
    ggml_backend_బఫరు_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::ప్రవాహం_Id                                  default_ప్రవాహం;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_బఫరు;

    bool                               uberkernel_enabled = false;
    ggml_backend_et_uberkernel_context uberkernel;
};

నిర్మాణం ggml_backend_et_reg_ctx {
    std::సదిశ<ggml_backend_dev_t> devices;
};
