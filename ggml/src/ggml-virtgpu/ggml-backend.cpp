#ಅನ್ತರ್ಭಾವ "../../include/ggml-virtgpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    ಪ್ರತ್ಯಾಗಮ "API Remoting backend";
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

ಸ್ಥಿರ ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    ಪ್ರತ್ಯಾಗಮ apir_backend_graph_compute(gpu, cgraph);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#ಯದಿ ಸತ್ಯ
    UNUSED(gpu);
    UNUSED(cgraph);
#ಅನ್ಯಥಾ
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ರಿಕ್ತ,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ರಿಕ್ತ,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .get_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .cpy_tensor_async        = */ ರಿಕ್ತ,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ರಿಕ್ತ,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ರಿಕ್ತ,
    /* .graph_plan_free         = */ ರಿಕ್ತ,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ರಿಕ್ತ,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ ರಿಕ್ತ,
    /* .event_wait              = */ ರಿಕ್ತ,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_remoting_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    ಪ್ರತ್ಯಾಗಮ &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = ನೂತನ ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    ಪ್ರತ್ಯಾಗಮ remoting_backend;
}
