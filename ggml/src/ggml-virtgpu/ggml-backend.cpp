#अन्तर्भाव "../../include/ggml-virtgpu.h"
#अन्तर्भाव "ggml-remoting.h"

स्थिर ध्रुव वर्ण * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    प्रत्यागम "API Remoting backend";
}

स्थिर शून्य ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

स्थिर ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    प्रत्यागम apir_backend_graph_compute(gpu, cgraph);
}

स्थिर शून्य ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#यदि सत्य
    UNUSED(gpu);
    UNUSED(cgraph);
#अन्यथा
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

स्थिर ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ रिक्त,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ रिक्त,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ रिक्त,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ रिक्त,
    /* .event_wait              = */ रिक्त,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

स्थिर ggml_guid_t ggml_backend_remoting_guid() {
    स्थिर ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    प्रत्यागम &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = नूतन ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    प्रत्यागम remoting_backend;
}
