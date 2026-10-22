#समावेश "../../include/ggml-virtgpu.h"
#समावेश "ggml-remoting.h"

अटल स्थिर अक्षर * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    वापस "API Remoting backend";
}

अटल व्योम ggml_backend_remoting_free(ggml_backend_t backend) {
    मिटाओ backend;
}

अटल ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    वापस apir_backend_graph_compute(gpu, cgraph);
}

अटल व्योम ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#अगर true
    UNUSED(gpu);
    UNUSED(cgraph);
#अन्यथा
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#पूर्ण_अगर
}

अटल ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ शून्य,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ शून्य,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ शून्य,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ शून्य,
    /* .event_wait              = */ शून्य,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

अटल ggml_guid_t ggml_backend_remoting_guid() {
    अटल ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    वापस &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = नया ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    वापस remoting_backend;
}
