#ଅନ୍ତର୍ଭାଵ "backend-dispatched.h"

#ଅନ୍ତର୍ଭାଵ "backend-virgl-apir.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>

ggml_backend_reg_t reg = ଖାଲି;
ggml_backend_dev_t dev = ଖାଲି;
ggml_backend_t     bck = ଖାଲି;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_ଗଣନା = 0;

uint32_t backend_dispatch_initialize(ଶୂନ୍ୟ * ggml_backend_reg_fct_p) {
    ଯଦି (reg != ଖାଲି) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        ଫେରନ୍ତୁ APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(ଶୂନ୍ୟ) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    ଯଦି (reg == ଖାଲି) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        ଫେରନ୍ତୁ APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    size_t device_ଗଣନା = reg->iface.get_device_ଗଣନା(reg);
    ଯଦି (!device_ଗଣନା) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        ଫେରନ୍ତୁ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    dev = reg->iface.get_device(reg, 0);

    ଯଦି (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        ଫେରନ୍ତୁ APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    bck = dev->iface.init_backend(dev, ଖାଲି);
    ଯଦି (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        ଫେରନ୍ତୁ APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    ଫେରନ୍ତୁ APIR_BACKEND_INITIALIZE_SUCCESS;
}
