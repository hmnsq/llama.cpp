#ಅನ್ತರ್ಭಾವ "../../include/ggml-virtgpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    ಬಳಸದ(backend);

    ಹಿಂತಿರುಗಿ "API Remoting backend";
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_free(ggml_backend_t backend) {
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_remoting_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(backend->device);

    ಹಿಂತಿರುಗಿ apir_backend_ನಕ್ಷೆ_compute(gpu, cgraph);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ನಕ್ಷೆ_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_ನಕ್ಷೆ_optimize_params * params) {
    ಬಳಸದ(params);
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(backend->device);
#ಆದರೆ ನಿಜ
    ಬಳಸದ(gpu);
    ಬಳಸದ(cgraph);
#ಇಲ್ಲದಿದ್ದರೆ
    // not working yet

    apir_backend_ನಕ್ಷೆ_optimize(gpu, cgraph);
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ಖಾಲಿ,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ಖಾಲಿ,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ಖಾಲಿ,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_remoting_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ಖಾಲಿ,
    /* .event_wait              = */ ಖಾಲಿ,
    /* .graph_optimize          = */ ggml_backend_remoting_ನಕ್ಷೆ_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_remoting_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    ಹಿಂತಿರುಗಿ &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ಬಳಸದ(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = ಹೊಸ ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    ಹಿಂತಿರುಗಿ remoting_backend;
}
