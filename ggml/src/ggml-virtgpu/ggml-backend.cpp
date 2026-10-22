#ضمن "../../include/ggml-virtgpu.h"
#ضمن "ggml-remoting.h"

ساكن ثابت حرف * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    أرجع "API Remoting backend";
}

ساكن فراغ ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

ساكن ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    أرجع apir_backend_graph_compute(gpu, cgraph);
}

ساكن فراغ ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#𐤀𐤌 true
    UNUSED(gpu);
    UNUSED(cgraph);
#وإلا
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

ساكن ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ عدم,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ عدم,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ عدم,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ عدم,
    /* .event_wait              = */ عدم,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

ساكن ggml_guid_t ggml_backend_remoting_guid() {
    ساكن ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    أرجع &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = جديد ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    أرجع remoting_backend;
}
