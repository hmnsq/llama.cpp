#અન્તર્ભાવ "backend-dispatched.h"

#અન્તર્ભાવ "backend-virgl-apir.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <cstdint>

ggml_backend_reg_t reg = ખાલી;
ggml_backend_dev_t dev = ખાલી;
ggml_backend_t     bck = ખાલી;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_ગણતરી = 0;

uint32_t backend_dispatch_initialize(શૂન્ય * ggml_backend_reg_fct_p) {
    જો (reg != ખાલી) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        પાછા APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(શૂન્ય) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    જો (reg == ખાલી) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        પાછા APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_ગણતરી = reg->iface.get_device_ગણતરી(reg);
    જો (!device_ગણતરી) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        પાછા APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    જો (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        પાછા APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, ખાલી);
    જો (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        પાછા APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    પાછા APIR_BACKEND_INITIALIZE_SUCCESS;
}
