#അന്തര്ഭാവ "backend-dispatched.h"

#അന്തര്ഭാവ "backend-virgl-apir.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cstdint>

ggml_backend_reg_t reg = ശൂന്യ;
ggml_backend_dev_t dev = ശൂന്യ;
ggml_backend_t     bck = ശൂന്യ;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_എണ്ണം = 0;

uint32_t backend_dispatch_initialize(ശൂന്യം * ggml_backend_reg_fct_p) {
    എങ്കിൽ (reg != ശൂന്യ) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        തിരികെ APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(ശൂന്യം) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    എങ്കിൽ (reg == ശൂന്യ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        തിരികെ APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_എണ്ണം = reg->iface.get_device_എണ്ണം(reg);
    എങ്കിൽ (!device_എണ്ണം) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        തിരികെ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    എങ്കിൽ (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        തിരികെ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, ശൂന്യ);
    എങ്കിൽ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        തിരികെ APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    തിരികെ APIR_BACKEND_INITIALIZE_SUCCESS;
}
