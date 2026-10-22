#അന്തര്ഭാവ "backend-dispatched.h"
#അന്തര്ഭാവ "backend-virgl-apir.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cstdint>

uint32_t backend_buffer_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    സ്ഥിരം അക്ഷരം * string = buft->iface.get_name(buft);

    സ്ഥിരം size_t string_size = strlen(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    തിരികെ 0;
}

uint32_t backend_buffer_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_size_t(enc, &value);

    തിരികെ 0;
}

uint32_t backend_buffer_type_get_max_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = SIZE_MAX;
    എങ്കിൽ (buft->iface.get_max_size) {
        value = buft->iface.get_max_size(buft);
    }

    apir_encode_size_t(enc, &value);

    തിരികെ 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_buffer_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);
    സ്ഥിരം തർക്കം is_host = അസത്യം;

    apir_encode_bool_t(enc, &is_host);

    തിരികെ 0;
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

    എങ്കിൽ (buffer) {
        apir_track_backend_buffer(buffer);
    }

    തിരികെ 0;
}

uint32_t backend_buffer_type_get_alloc_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    സ്ഥിരം ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    എങ്കിൽ (op == nullptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_set_fatal(dec);
        തിരികെ 1;
    }

    size_t value;
    എങ്കിൽ (buft->iface.get_alloc_size) {
        value = buft->iface.get_alloc_size(buft, op);
    } അല്ലെങ്കിൽ {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_size_t(enc, &value);

    തിരികെ 0;
}
