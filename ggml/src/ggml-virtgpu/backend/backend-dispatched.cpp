#ضمن "backend-dispatched.h"

#ضمن "backend-virgl-apir.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"

#ضمن <cstdint>

ggml_backend_reg_t reg = عدم;
ggml_backend_dev_t dev = عدم;
ggml_backend_t     bck = عدم;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_count = 0;

uint32_t backend_dispatch_initialize(فراغ * ggml_backend_reg_fct_p) {
    אם (reg != عدم) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        החזר APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(فراغ) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    אם (reg == عدم) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        החזר APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_count = reg->iface.get_device_count(reg);
    אם (!device_count) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        החזר APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    אם (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        החזר APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, عدم);
    אם (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        החזר APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    החזר APIR_BACKEND_INITIALIZE_SUCCESS;
}
