#అన్తర్భావ "../../include/ggml-virtgpu.h"
#అన్తర్భావ "ggml-remoting.h"

స్థిర స్థిరరాశి అక్షరం * ggml_backend_remoting_get_name(ggml_backend_t backend) {
    వాడని(backend);

    తిరిగి "API Remoting backend";
}

స్థిర శూన్య ggml_backend_remoting_free(ggml_backend_t backend) {
    తొలగించు backend;
}

స్థిర ggml_స్థితి ggml_backend_remoting_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    virtgpu * gpu = DEV_గమ్యం_GPU(backend->device);

    తిరిగి apir_backend_లేఖాచిత్రం_compute(gpu, cgraph);
}

స్థిర శూన్య ggml_backend_remoting_లేఖాచిత్రం_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_లేఖాచిత్రం_optimize_params * params) {
    వాడని(params);
    virtgpu * gpu = DEV_గమ్యం_GPU(backend->device);
#ఒకవేళ నిజం
    వాడని(gpu);
    వాడని(cgraph);
#లేకపోతే
    // not working yet

    apir_backend_లేఖాచిత్రం_optimize(gpu, cgraph);
#అయితే_ముగింపు
}

స్థిర ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_get_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ ఖాళీ,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ ఖాళీ,  // ggml_backend_remoting_get_tensor_async,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ ఖాళీ,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_remoting_లేఖాచిత్రం_compute,
    /* .event_record            = */ ఖాళీ,
    /* .event_wait              = */ ఖాళీ,
    /* .graph_optimize          = */ ggml_backend_remoting_లేఖాచిత్రం_optimize,
};

స్థిర ggml_guid_t ggml_backend_remoting_guid() {
    స్థిర ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02,
                              0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

    తిరిగి &guid;
}

ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    వాడని(params);

    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *) dev->context;

    ggml_backend_t remoting_backend = కొత్త ggml_backend{
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_virtgpu_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    తిరిగి remoting_backend;
}
