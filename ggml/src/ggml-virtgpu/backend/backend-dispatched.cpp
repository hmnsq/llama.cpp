#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

ggml_backend_reg_t reg = ਨਿਰਾ;
ggml_backend_dev_t dev = ਨਿਰਾ;
ggml_backend_t     bck = ਨਿਰਾ;

uint64_t timer_ਸ਼ੁਰੂ = 0;
uint64_t timer_ਕੁੱਲ = 0;
uint64_t timer_ਗਿਣਤੀ = 0;

uint32_t backend_dispatch_initialize(ਖਾਲੀ * ggml_backend_reg_fct_p) {
    ਜੇ (reg != ਨਿਰਾ) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        ਵਾਪਸ APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(ਖਾਲੀ) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    ਜੇ (reg == ਨਿਰਾ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        ਵਾਪਸ APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_ਗਿਣਤੀ = reg->iface.get_device_ਗਿਣਤੀ(reg);
    ਜੇ (!device_ਗਿਣਤੀ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        ਵਾਪਸ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    ਜੇ (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        ਵਾਪਸ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, ਨਿਰਾ);
    ਜੇ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        ਵਾਪਸ APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    ਵਾਪਸ APIR_BACKEND_INITIALIZE_SUCCESS;
}
