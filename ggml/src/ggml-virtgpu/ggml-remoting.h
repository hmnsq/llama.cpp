#సూచన once

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "virtgpu.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <string>

#నిర్వచించు GGML_VIRTGPU_NAME "ggml-virtgpu"
#నిర్వచించు GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#నిర్వచించు USE_ALWAYS_TRUE_SUPPORTS_OP 1
#నిర్వచించు USE_METAL_GUEST_SUPPORTS_OP 0

#నిర్వచించు DEV_గమ్యం_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#నిర్వచించు బఫరు_గమ్యం_GGML_CONTEXT(name) ((ggml_backend_remoting_బఫరు_context *) (name)->context)

#నిర్వచించు బఫరు_గమ్యం_APIR_CONTEXT(name) &((ggml_backend_remoting_బఫరు_context *) (name)->context)->apir_context

#నిర్వచించు బఫరు_గమ్యం_HOST_HANDLE(name) ((ggml_backend_remoting_బఫరు_context *) (name)->context)->apir_context.host_handle

#నిర్వచించు GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#నిర్వచించు BUFT_గమ్యం_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

నిర్మాణం ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::సదిశ<std::tuple<శూన్య *, size_t, virtgpu_shmem *>> shared_జ్ఞాపకం;

    virtgpu * gpu;
};

నిర్మాణం ggml_backend_remoting_బఫరు_context {
    apir_బఫరు_context_t apir_context;

    virtgpu * gpu;

    శూన్య * పునాది;

    bool is_from_ptr;
};

బాహ్య స్థిరరాశి ggml_backend_బఫరు_type_i ggml_backend_remoting_బఫరు_type_interface;
బాహ్య స్థిరరాశి ggml_backend_device_i      ggml_backend_remoting_device_interface;
బాహ్య స్థిరరాశి ggml_backend_బఫరు_i      ggml_backend_remoting_బఫరు_interface;
బాహ్య స్థిరరాశి ggml_backend_బఫరు_type_i ggml_backend_remoting_బఫరు_from_ptr_type_interface;
బాహ్య స్థిరరాశి ggml_backend_బఫరు_i      ggml_backend_remoting_బఫరు_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params);
ggml_backend_బఫరు_type_t ggml_backend_remoting_device_get_బఫరు_type(ggml_backend_dev_t dev);

స్థిర అంతఃవరుస apir_బఫరు_type_host_handle_t ggml_బఫరు_type_గమ్యం_apir_handle(ggml_backend_బఫరు_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    తిరిగి (apir_బఫరు_type_host_handle_t) buft->context;
}

స్థిర అంతఃవరుస apir_బఫరు_host_handle_t ggml_బఫరు_గమ్యం_apir_handle(ggml_backend_బఫరు_t బఫరు) {
    ఒకవేళ (!బఫరు->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    తిరిగి బఫరు_గమ్యం_HOST_HANDLE(బఫరు);
}
