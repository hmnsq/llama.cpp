#অন্তর্ভাৱ "backend-dispatched.h"

#অন্তর্ভাৱ "backend-virgl-apir.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <cstdint>

ggml_backend_reg_t reg = রিক্ত;
ggml_backend_dev_t dev = রিক্ত;
ggml_backend_t     bck = রিক্ত;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_গুনতি = 0;

uint32_t backend_dispatch_initialize(শূন্য * ggml_backend_reg_fct_p) {
    যদি (reg != রিক্ত) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        ঘূৰাই APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(শূন্য) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    যদি (reg == রিক্ত) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        ঘূৰাই APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_গুনতি = reg->iface.get_device_গুনতি(reg);
    যদি (!device_গুনতি) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        ঘূৰাই APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    যদি (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        ঘূৰাই APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, রিক্ত);
    যদি (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        ঘূৰাই APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    ঘূৰাই APIR_BACKEND_INITIALIZE_SUCCESS;
}
