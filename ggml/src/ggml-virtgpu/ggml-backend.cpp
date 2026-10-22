#અન્તર્ભાવ "../../include/ggml-virtgpu.h"
#અન્તર્ભાવ "ggml-remoting.h"

સ્થિર અચલ અક્ષર * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    પાછા "API Remoting backend";
}

સ્થિર શૂન્ય ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

સ્થિર ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    પાછા apir_backend_graph_compute(gpu, cgraph);
}

સ્થિર શૂન્ય ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#જો સાચું
    UNUSED(gpu);
    UNUSED(cgraph);
#નહીંતર
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

સ્થિર ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ખાલી,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ખાલી,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ખાલી,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ખાલી,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ ખાલી,
    /* .event_wait              = */ ખાલી,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

સ્થિર ggml_guid_t ggml_backend_remoting_guid() {
    સ્થિર ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    પાછા &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = નવું ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    પાછા remoting_backend;
}
