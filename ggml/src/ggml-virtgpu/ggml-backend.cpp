#شامل "../../include/ggml-virtgpu.h"
#شامل "ggml-remoting.h"

ایستا ثابت نویسه * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    بازگشت "API Remoting backend";
}

ایستا تهی ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

ایستا ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    بازگشت apir_backend_graph_compute(gpu, cgraph);
}

ایستا تهی ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#اگر درست
    UNUSED(gpu);
    UNUSED(cgraph);
#وگرنه
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

ایستا ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ پوچ,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ پوچ,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ پوچ,
    /* .get_tensor_2d_async     = */ پوچ,
    /* .cpy_tensor_async        = */ پوچ,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ پوچ,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ پوچ,
    /* .graph_plan_free         = */ پوچ,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ پوچ,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ پوچ,
    /* .event_wait              = */ پوچ,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

ایستا ggml_guid_t ggml_backend_remoting_guid() {
    ایستا ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    بازگشت &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ثابت نویسه * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = جدید ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    بازگشت remoting_backend;
}
