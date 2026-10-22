#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdint>

uint32_t backend_buffer_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    ध्रुव वर्ण * string = buft->iface.get_name(buft);

    ध्रुव size_t string_size = strlen(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    प्रत्यागम 0;
}

uint32_t backend_buffer_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_size_t(enc, &value);

    प्रत्यागम 0;
}

uint32_t backend_buffer_type_get_max_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = SIZE_MAX;
    यदि (buft->iface.get_max_size) {
        value = buft->iface.get_max_size(buft);
    }

    apir_encode_size_t(enc, &value);

    प्रत्यागम 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_buffer_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);
    ध्रुव तार्किक is_host = असत्य;

    apir_encode_bool_t(enc, &is_host);

    प्रत्यागम 0;
}

uint32_t backend_buffer_type_alloc_buffer(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t size;
    apir_decode_size_t(dec, &size);

    ggml_backend_buffer_t buffer;

    buffer = buft->iface.alloc_buffer(buft, size);

    apir_encode_ggml_buffer(enc, buffer);

    यदि (buffer) {
        apir_track_backend_buffer(buffer);
    }

    प्रत्यागम 0;
}

uint32_t backend_buffer_type_get_alloc_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    ध्रुव ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    यदि (op == nullptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_set_fatal(dec);
        प्रत्यागम 1;
    }

    size_t value;
    यदि (buft->iface.get_alloc_size) {
        value = buft->iface.get_alloc_size(buft, op);
    } अन्यथा {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_size_t(enc, &value);

    प्रत्यागम 0;
}
