#समावेश "backend-dispatched.h"
#समावेश "backend-virgl-apir.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"

#समावेश <cstdint>

uint32_t backend_बफर_type_get_name(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(डेसी);

    स्थिर अक्षर * string = buft->iface.get_name(buft);

    स्थिर माप_प्रकार string_आकार = म_माप(string) + 1;
    apir_encode_array_आकार(enc, string_आकार);
    apir_encode_char_array(enc, string, string_आकार);

    वापस 0;
}

uint32_t backend_बफर_type_get_alignment(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(डेसी);

    माप_प्रकार value = buft->iface.get_alignment(buft);
    apir_encode_आकार_t(enc, &value);

    वापस 0;
}

uint32_t backend_बफर_type_get_max_आकार(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(डेसी);

    माप_प्रकार value = SIZE_MAX;
    अगर (buft->iface.get_max_आकार) {
        value = buft->iface.get_max_आकार(buft);
    }

    apir_encode_आकार_t(enc, &value);

    वापस 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_बफर_type_is_host(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(डेसी);
    स्थिर bool is_host = false;

    apir_encode_bool_t(enc, &is_host);

    वापस 0;
}

uint32_t backend_बफर_type_alloc_बफर(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(डेसी);

    माप_प्रकार size;
    apir_decode_आकार_t(डेसी, &size);

    ggml_backend_बफर_t बफर;

    बफर = buft->iface.alloc_बफर(buft, size);

    apir_encode_ggml_बफर(enc, बफर);

    अगर (बफर) {
        apir_track_backend_बफर(बफर);
    }

    वापस 0;
}

uint32_t backend_बफर_type_get_alloc_आकार(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(डेसी);

    स्थिर ggml_tensor * op = apir_decode_ggml_tensor_inplace(डेसी);

    // Check for decode error
    अगर (op == nullptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_set_fatal(डेसी);
        वापस 1;
    }

    माप_प्रकार value;
    अगर (buft->iface.get_alloc_आकार) {
        value = buft->iface.get_alloc_आकार(buft, op);
    } अन्यथा {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_आकार_t(enc, &value);

    वापस 0;
}
