#समावेश "backend-dispatched.h"

#समावेश "backend-virgl-apir.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"

#समावेश <cstdint>

ggml_backend_reg_t reg = शून्य;
ggml_backend_dev_t dev = शून्य;
ggml_backend_t     bck = शून्य;

uint64_t timer_start = 0;
uint64_t timer_योग = 0;
uint64_t timer_गिनती = 0;

uint32_t backend_dispatch_initialize(व्योम * ggml_backend_reg_fct_p) {
    अगर (reg != शून्य) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        वापस APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(व्योम) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    अगर (reg == शून्य) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        वापस APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    माप_प्रकार device_गिनती = reg->iface.get_device_गिनती(reg);
    अगर (!device_गिनती) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        वापस APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    अगर (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        वापस APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, शून्य);
    अगर (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        वापस APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    वापस APIR_BACKEND_INITIALIZE_SUCCESS;
}
