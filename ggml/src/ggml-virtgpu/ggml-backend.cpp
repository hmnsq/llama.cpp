#অন্তর্ভাব "../../include/ggml-virtgpu.h"
#অন্তর্ভাব "ggml-remoting.h"

স্থির ধ্রুব বর্ণ * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    প্রত্যাগম "API Remoting backend";
}

স্থির শূন্য ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

স্থির ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    প্রত্যাগম apir_backend_graph_compute(gpu, cgraph);
}

স্থির শূন্য ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#যদি সত্য
    UNUSED(gpu);
    UNUSED(cgraph);
#অন্যথা
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

স্থির ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ রিক্ত,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ রিক্ত,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ রিক্ত,
    /* .get_tensor_2d_async     = */ রিক্ত,
    /* .cpy_tensor_async        = */ রিক্ত,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ রিক্ত,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ রিক্ত,
    /* .event_wait              = */ রিক্ত,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

স্থির ggml_guid_t ggml_backend_remoting_guid() {
    স্থির ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    প্রত্যাগম &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ধ্রুব বর্ণ * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = নূতন ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    প্রত্যাগম remoting_backend;
}
