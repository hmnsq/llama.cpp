#अन्तर्भाव "backend-dispatched.h"

#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdint>

ggml_backend_reg_t reg = रिक्त;
ggml_backend_dev_t dev = रिक्त;
ggml_backend_t     bck = रिक्त;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_गणनम् = 0;

uint32_t backend_dispatch_initialize(शून्य * ggml_backend_reg_fct_p) {
    जइ (reg != रिक्त) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        प्रत्यागम APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(शून्य) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    जइ (reg == रिक्त) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        प्रत्यागम APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_गणनम् = reg->iface.get_device_गणनम्(reg);
    जइ (!device_गणनम्) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        प्रत्यागम APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    जइ (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        प्रत्यागम APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, रिक्त);
    जइ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        प्रत्यागम APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    प्रत्यागम APIR_BACKEND_INITIALIZE_SUCCESS;
}
