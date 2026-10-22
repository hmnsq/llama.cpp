#شامل "../../include/ggml-virtgpu.h"
#شامل "ggml-remoting.h"

ساکن ثابت حرف * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    غیر_مستعمل(backend);

    واپس "API Remoting backend";
}

ساکن خالی ggml_backend_remoting_free(ggml_backend_t backend) {
    حذف backend;
}

ساکن ggml_حالت ggml_backend_remoting_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_مقصد_GPU(backend->device);

    واپس apir_backend_گراف_compute(gpu, cgraph);
}

ساکن خالی ggml_backend_remoting_گراف_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_گراف_optimize_params * params) {
    غیر_مستعمل(params);
    virtgpu * gpu = DEV_مقصد_GPU(backend->device);
#اگر سچ
    غیر_مستعمل(gpu);
    غیر_مستعمل(cgraph);
#ورنہ
    // not working yet

    apir_backend_گراف_optimize(gpu, cgraph);
#اگر_ختم
}

ساکن ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ تہی,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ تہی,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ تہی,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ تہی,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_remoting_گراف_compute,
    /* .event_record            = */ تہی,
    /* .event_wait              = */ تہی,
    /* .graph_optimize          = */ ggml_backend_remoting_گراف_optimize,
};

ساکن ggml_guid_t ggml_backend_remoting_guid() {
    ساکن ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    واپس &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    غیر_مستعمل(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = نیا ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    واپس remoting_backend;
}
