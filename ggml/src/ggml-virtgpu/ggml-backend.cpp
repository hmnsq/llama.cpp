#ਅਨ੍ਤਰ੍ਭਾਵ "../../include/ggml-virtgpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    ਅਣਵਰਤਿਆ(backend);

    ਵਾਪਸ "API Remoting backend";
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_free(ggml_backend_t backend) {
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_remoting_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(backend->device);

    ਵਾਪਸ apir_backend_ਗ੍ਰਾਫ_compute(gpu, cgraph);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਗ੍ਰਾਫ_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_ਗ੍ਰਾਫ_optimize_params * params) {
    ਅਣਵਰਤਿਆ(params);
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(backend->device);
#ਜੇ ਸੱਚ
    ਅਣਵਰਤਿਆ(gpu);
    ਅਣਵਰਤਿਆ(cgraph);
#ਨਹੀਂਤਾਂ
    // not working yet

    apir_backend_ਗ੍ਰਾਫ_optimize(gpu, cgraph);
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ਨਿਰਾ,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ਨਿਰਾ,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ਨਿਰਾ,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_remoting_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ਨਿਰਾ,
    /* .event_wait              = */ ਨਿਰਾ,
    /* .graph_optimize          = */ ggml_backend_remoting_ਗ੍ਰਾਫ_optimize,
};

ਸਥਿਰ ggml_guid_t ggml_backend_remoting_guid() {
    ਸਥਿਰ ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    ਵਾਪਸ &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ਅਣਵਰਤਿਆ(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = ਨਵਾਂ ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    ਵਾਪਸ remoting_backend;
}
