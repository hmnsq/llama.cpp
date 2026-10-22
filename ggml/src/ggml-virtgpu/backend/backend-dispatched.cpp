#అన్తర్భావ "backend-dispatched.h"

#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdint>

ggml_backend_reg_t reg = ఖాళీ;
ggml_backend_dev_t dev = ఖాళీ;
ggml_backend_t     bck = ఖాళీ;

uint64_t timer_ప్రారంభం = 0;
uint64_t timer_మొత్తం = 0;
uint64_t timer_లెక్క = 0;

uint32_t backend_dispatch_initialize(శూన్య * ggml_backend_reg_fct_p) {
    ఒకవేళ (reg != ఖాళీ) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        తిరిగి APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(శూన్య) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    ఒకవేళ (reg == ఖాళీ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        తిరిగి APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_లెక్క = reg->iface.get_device_లెక్క(reg);
    ఒకవేళ (!device_లెక్క) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        తిరిగి APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    ఒకవేళ (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        తిరిగి APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, ఖాళీ);
    ఒకవేళ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        తిరిగి APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    తిరిగి APIR_BACKEND_INITIALIZE_SUCCESS;
}
