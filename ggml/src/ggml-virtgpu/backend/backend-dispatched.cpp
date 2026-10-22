#شامل "backend-dispatched.h"

#شامل "backend-virgl-apir.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"

#شامل <cstdint>

ggml_backend_reg_t reg = پوچ;
ggml_backend_dev_t dev = پوچ;
ggml_backend_t     bck = پوچ;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_شمار = 0;

uint32_t backend_dispatch_initialize(تهی * ggml_backend_reg_fct_p) {
    اگر (reg != پوچ) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        بازگشت APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(تهی) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    اگر (reg == پوچ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        بازگشت APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_شمار = reg->iface.get_device_شمار(reg);
    اگر (!device_شمار) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        بازگشت APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    اگر (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        بازگشت APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, پوچ);
    اگر (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        بازگشت APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    بازگشت APIR_BACKEND_INITIALIZE_SUCCESS;
}
