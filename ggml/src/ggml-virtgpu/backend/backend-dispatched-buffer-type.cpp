#அந்தர்பாவ "backend-dispatched.h"
#அந்தர்பாவ "backend-virgl-apir.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <cstdint>

uint32_t backend_buffer_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    மாறிலி எழுத்து * string = buft->iface.get_name(buft);

    மாறிலி size_t string_size = strlen(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    திரும்பு 0;
}

uint32_t backend_buffer_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_size_t(enc, &value);

    திரும்பு 0;
}

uint32_t backend_buffer_type_get_max_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = SIZE_MAX;
    எனில் (buft->iface.get_max_size) {
        value = buft->iface.get_max_size(buft);
    }

    apir_encode_size_t(enc, &value);

    திரும்பு 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_buffer_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);
    மாறிலி தருக்க is_host = பொய்;

    apir_encode_bool_t(enc, &is_host);

    திரும்பு 0;
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

    எனில் (buffer) {
        apir_track_backend_buffer(buffer);
    }

    திரும்பு 0;
}

uint32_t backend_buffer_type_get_alloc_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    மாறிலி ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    எனில் (op == nullptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_set_fatal(dec);
        திரும்பு 1;
    }

    size_t value;
    எனில் (buft->iface.get_alloc_size) {
        value = buft->iface.get_alloc_size(buft, op);
    } இல்லையெனில் {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_size_t(enc, &value);

    திரும்பு 0;
}
