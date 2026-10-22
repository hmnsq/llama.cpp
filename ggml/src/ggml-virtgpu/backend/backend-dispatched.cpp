#شامل "backend-dispatched.h"

#شامل "backend-virgl-apir.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"

#شامل <cstdint>

ggml_backend_reg_t reg = تہی;
ggml_backend_dev_t dev = تہی;
ggml_backend_t     bck = تہی;

uint64_t timer_آغاز = 0;
uint64_t timer_کل = 0;
uint64_t timer_گنتی = 0;

uint32_t backend_dispatch_initialize(خالی * ggml_backend_reg_fct_p) {
    اگر (reg != تہی) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        واپس APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(خالی) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    اگر (reg == تہی) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        واپس APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_گنتی = reg->iface.get_device_گنتی(reg);
    اگر (!device_گنتی) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        واپس APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    اگر (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        واپس APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, تہی);
    اگر (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        واپس APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    واپس APIR_BACKEND_INITIALIZE_SUCCESS;
}
