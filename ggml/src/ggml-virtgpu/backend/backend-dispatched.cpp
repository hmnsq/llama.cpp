#include "backend-dispatched.h"

#include "backend-virgl-apir.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstdint>

ggml_backend_reg_t reg = NULL;
ggml_backend_dev_t dev = NULL;
ggml_backend_t     bck = NULL;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_count = 0;

uint32_t backend_dispatch_initialize(void * ggml_backend_reg_fct_p) {
    အကယ်၍ (reg != NULL) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        ပြန်ပေး APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(void) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    အကယ်၍ (reg == NULL) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        ပြန်ပေး APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_count = reg->iface.get_device_count(reg);
    အကယ်၍ (!device_count) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        ပြန်ပေး APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    အကယ်၍ (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        ပြန်ပေး APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, NULL);
    အကယ်၍ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        ပြန်ပေး APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    ပြန်ပေး APIR_BACKEND_INITIALIZE_SUCCESS;
}
