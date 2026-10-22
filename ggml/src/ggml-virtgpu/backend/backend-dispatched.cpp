#அந்தர்பாவ "backend-dispatched.h"

#அந்தர்பாவ "backend-virgl-apir.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <cstdint>

ggml_backend_reg_t reg = வெற்று;
ggml_backend_dev_t dev = வெற்று;
ggml_backend_t     bck = வெற்று;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_மொத்தம் = 0;

uint32_t backend_dispatch_initialize(வெறுமை * ggml_backend_reg_fct_p) {
    எனில் (reg != வெற்று) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        திரும்பு APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(வெறுமை) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    எனில் (reg == வெற்று) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        திரும்பு APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_மொத்தம் = reg->iface.get_device_மொத்தம்(reg);
    எனில் (!device_மொத்தம்) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        திரும்பு APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    எனில் (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        திரும்பு APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, வெற்று);
    எனில் (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        திரும்பு APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    திரும்பு APIR_BACKEND_INITIALIZE_SUCCESS;
}
