#அந்தர்பாவ "../../include/ggml-virtgpu.h"
#அந்தர்பாவ "ggml-remoting.h"

நிலையான மாறிலி எழுத்து * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    திரும்பு "API Remoting backend";
}

நிலையான வெறுமை ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

நிலையான ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    திரும்பு apir_backend_graph_compute(gpu, cgraph);
}

நிலையான வெறுமை ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#எனில் உண்மை
    UNUSED(gpu);
    UNUSED(cgraph);
#இல்லையெனில்
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

நிலையான ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ வெற்று,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ வெற்று,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ வெற்று,
    /* .get_tensor_2d_async     = */ வெற்று,
    /* .cpy_tensor_async        = */ வெற்று,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ வெற்று,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ வெற்று,
    /* .graph_plan_free         = */ வெற்று,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ வெற்று,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ வெற்று,
    /* .event_wait              = */ வெற்று,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

நிலையான ggml_guid_t ggml_backend_remoting_guid() {
    நிலையான ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    திரும்பு &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = புதிய ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    திரும்பு remoting_backend;
}
