#അന്തര്ഭാവ "../../include/ggml-virtgpu.h"
#അന്തര്ഭാവ "ggml-remoting.h"

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    തിരികെ "API Remoting backend";
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

സ്ഥിതം ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    തിരികെ apir_backend_graph_compute(gpu, cgraph);
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#എങ്കിൽ സത്യം
    UNUSED(gpu);
    UNUSED(cgraph);
#അല്ലെങ്കിൽ
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

സ്ഥിതം ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ശൂന്യ,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ശൂന്യ,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ശൂന്യ,
    /* .get_tensor_2d_async     = */ ശൂന്യ,
    /* .cpy_tensor_async        = */ ശൂന്യ,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ശൂന്യ,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ ശൂന്യ,
    /* .event_wait              = */ ശൂന്യ,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

സ്ഥിതം ggml_guid_t ggml_backend_remoting_guid() {
    സ്ഥിതം ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    തിരികെ &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = പുതിയ ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    തിരികെ remoting_backend;
}
