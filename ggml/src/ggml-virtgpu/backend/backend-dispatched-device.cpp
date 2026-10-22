#అన్తర్భావ "backend-dispatched.h"
#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdint>

uint32_t backend_device_get_device_లెక్క(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    int32_t dev_లెక్క = reg->iface.get_device_లెక్క(reg);
    apir_encode_int32_t(enc, &dev_లెక్క);

    తిరిగి 0;
}

uint32_t backend_device_get_లెక్క(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    int32_t dev_లెక్క = reg->iface.get_device_లెక్క(reg);
    apir_encode_int32_t(enc, &dev_లెక్క);

    తిరిగి 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    స్థిరరాశి అక్షరం * string = dev->iface.get_name(dev);

    స్థిరరాశి size_t string_కొలత = అక్షరశ్రేణి_పొడవు(string) + 1;
    apir_encode_array_కొలత(enc, string_కొలత);
    apir_encode_char_array(enc, string, string_కొలత);

    తిరిగి 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    స్థిరరాశి అక్షరం * string = dev->iface.get_description(dev);

    స్థిరరాశి size_t string_కొలత = అక్షరశ్రేణి_పొడవు(string) + 1;
    apir_encode_array_కొలత(enc, string_కొలత);
    apir_encode_char_array(enc, string, string_కొలత);

    తిరిగి 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    తిరిగి 0;
}

uint32_t backend_device_get_జ్ఞాపకం(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    size_t విడుదల, మొత్తం;
    dev->iface.get_జ్ఞాపకం(dev, &విడుదల, &మొత్తం);

    apir_encode_కొలత_t(enc, &విడుదల);
    apir_encode_కొలత_t(enc, &మొత్తం);

    తిరిగి 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);

    స్థిరరాశి ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    తార్కిక supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    తిరిగి 0;
}

uint32_t backend_device_get_బఫరు_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    ggml_backend_బఫరు_type_t bufft = dev->iface.get_బఫరు_type(dev);

    apir_encode_ggml_బఫరు_type(enc, bufft);

    తిరిగి 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_బఫరు);
    apir_encode_bool_t(enc, &props.caps.బఫరు_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    తిరిగి 0;
}

uint32_t backend_device_బఫరు_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    శూన్య * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ఒకవేళ (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_సమితి_fatal(dec);
        తిరిగి 1;
    }

    size_t size;
    apir_decode_కొలత_t(dec, &size);
    size_t max_tensor_కొలత;
    apir_decode_కొలత_t(dec, &max_tensor_కొలత);

    ggml_backend_బఫరు_t బఫరు;
    బఫరు = dev->iface.బఫరు_from_host_ptr(dev, shmem_ptr, size, max_tensor_కొలత);

    apir_encode_ggml_బఫరు(enc, బఫరు);
    apir_encode_ggml_బఫరు_type(enc, బఫరు->buft);

    ఒకవేళ (బఫరు) {
        apir_track_backend_బఫరు(బఫరు);
    }

    తిరిగి 0;
}
