#समावेश "backend-dispatched.h"
#समावेश "backend-virgl-apir.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"

#समावेश <cstdint>

uint32_t backend_device_get_device_गिनती(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    int32_t dev_गिनती = reg->iface.get_device_गिनती(reg);
    apir_encode_int32_t(enc, &dev_गिनती);

    वापस 0;
}

uint32_t backend_device_get_गिनती(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    int32_t dev_गिनती = reg->iface.get_device_गिनती(reg);
    apir_encode_int32_t(enc, &dev_गिनती);

    वापस 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    स्थिर अक्षर * string = dev->iface.get_name(dev);

    स्थिर माप_प्रकार string_आकार = म_माप(string) + 1;
    apir_encode_array_आकार(enc, string_आकार);
    apir_encode_char_array(enc, string, string_आकार);

    वापस 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    स्थिर अक्षर * string = dev->iface.get_description(dev);

    स्थिर माप_प्रकार string_आकार = म_माप(string) + 1;
    apir_encode_array_आकार(enc, string_आकार);
    apir_encode_char_array(enc, string, string_आकार);

    वापस 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    वापस 0;
}

uint32_t backend_device_get_memory(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    माप_प्रकार मुक्त, योग;
    dev->iface.get_memory(dev, &मुक्त, &योग);

    apir_encode_आकार_t(enc, &मुक्त);
    apir_encode_आकार_t(enc, &योग);

    वापस 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    स्थिर ggml_tensor * op = apir_decode_ggml_tensor_inplace(डेसी);

    bool supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    वापस 0;
}

uint32_t backend_device_get_बफर_type(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    ggml_backend_बफर_type_t bufft = dev->iface.get_बफर_type(dev);

    apir_encode_ggml_बफर_type(enc, bufft);

    वापस 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_बफर);
    apir_encode_bool_t(enc, &props.caps.बफर_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    वापस 0;
}

uint32_t backend_device_बफर_from_ptr(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(डेसी, &shmem_res_id);

    व्योम * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    अगर (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(डेसी);
        वापस 1;
    }

    माप_प्रकार size;
    apir_decode_आकार_t(डेसी, &size);
    माप_प्रकार max_tensor_आकार;
    apir_decode_आकार_t(डेसी, &max_tensor_आकार);

    ggml_backend_बफर_t बफर;
    बफर = dev->iface.बफर_from_host_ptr(dev, shmem_ptr, size, max_tensor_आकार);

    apir_encode_ggml_बफर(enc, बफर);
    apir_encode_ggml_बफर_type(enc, बफर->buft);

    अगर (बफर) {
        apir_track_backend_बफर(बफर);
    }

    वापस 0;
}
