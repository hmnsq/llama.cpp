#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"

#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdint>

ggml_backend_reg_t reg = ರಿಕ್ತ;
ggml_backend_dev_t dev = ರಿಕ್ತ;
ggml_backend_t     bck = ರಿಕ್ತ;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_ಎಣಿಕೆ = 0;

uint32_t backend_dispatch_initialize(ಶೂನ್ಯ * ggml_backend_reg_fct_p) {
    ಯದಿ (reg != ರಿಕ್ತ) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(ಶೂನ್ಯ) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    ಯದಿ (reg == ರಿಕ್ತ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_ಎಣಿಕೆ = reg->iface.get_device_ಎಣಿಕೆ(reg);
    ಯದಿ (!device_ಎಣಿಕೆ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    ಯದಿ (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, ರಿಕ್ತ);
    ಯದಿ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_INITIALIZE_SUCCESS;
}
