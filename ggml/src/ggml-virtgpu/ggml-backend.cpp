#ଅନ୍ତର୍ଭାଵ "../../include/ggml-virtgpu.h"
#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    UNUSED(backend);

    ଫେରନ୍ତୁ "API Remoting backend";
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_free(ggml_backend_t backend) {
    delete backend;
}

ସ୍ଥିର ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_TO_GPU(backend->device);

    ଫେରନ୍ତୁ apir_backend_graph_compute(gpu, cgraph);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    UNUSED(params);
    virtgpu * gpu = DEV_TO_GPU(backend->device);
#ଯଦି ସତ୍ୟ
    UNUSED(gpu);
    UNUSED(cgraph);
#ଅନ୍ୟଥା
    // not working yet

    apir_backend_graph_optimize(gpu, cgraph);
#endif
}

ସ୍ଥିର ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ଖାଲି,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ଖାଲି,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ଖାଲି,
    /* .get_tensor_2d_async     = */ ଖାଲି,
    /* .cpy_tensor_async        = */ ଖାଲି,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ଖାଲି,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ ଖାଲି,
    /* .event_wait              = */ ଖାଲି,
    /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

ସ୍ଥିର ggml_guid_t ggml_backend_remoting_guid() {
    ସ୍ଥିର ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    ଫେରନ୍ତୁ &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    UNUSED(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = ନୂତନ ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    ଫେରନ୍ତୁ remoting_backend;
}
