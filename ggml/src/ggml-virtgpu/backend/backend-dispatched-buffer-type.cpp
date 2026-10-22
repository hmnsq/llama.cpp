#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

uint32_t backend_ਬਫਰ_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ggml_backend_ਬਫਰ_type_t buft;
    buft = apir_decode_ggml_ਬਫਰ_type(dec);

    ਅਚਲ ਅੱਖਰ * string = buft->iface.get_name(buft);

    ਅਚਲ size_t string_ਮਾਪ = ਲੜੀ_ਲੰਬਾਈ(string) + 1;
    apir_encode_array_ਮਾਪ(enc, string_ਮਾਪ);
    apir_encode_char_array(enc, string, string_ਮਾਪ);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ggml_backend_ਬਫਰ_type_t buft;
    buft = apir_decode_ggml_ਬਫਰ_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_ਮਾਪ_t(enc, &value);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_type_get_max_ਮਾਪ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ggml_backend_ਬਫਰ_type_t buft;
    buft = apir_decode_ggml_ਬਫਰ_type(dec);

    size_t value = SIZE_MAX;
    ਜੇ (buft->iface.get_max_ਮਾਪ) {
        value = buft->iface.get_max_ਮਾਪ(buft);
    }

    apir_encode_ਮਾਪ_t(enc, &value);

    ਵਾਪਸ 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_ਬਫਰ_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);
    ਅਚਲ ਤਾਰਕਿਕ is_host = ਝੂਠ;

    apir_encode_bool_t(enc, &is_host);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_type_alloc_ਬਫਰ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ggml_backend_ਬਫਰ_type_t buft;
    buft = apir_decode_ggml_ਬਫਰ_type(dec);

    size_t size;
    apir_decode_ਮਾਪ_t(dec, &size);

    ggml_backend_ਬਫਰ_t ਬਫਰ;

    ਬਫਰ = buft->iface.alloc_ਬਫਰ(buft, size);

    apir_encode_ggml_ਬਫਰ(enc, ਬਫਰ);

    ਜੇ (ਬਫਰ) {
        apir_track_backend_ਬਫਰ(ਬਫਰ);
    }

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_type_get_alloc_ਮਾਪ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ggml_backend_ਬਫਰ_type_t buft;
    buft = apir_decode_ggml_ਬਫਰ_type(dec);

    ਅਚਲ ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    ਜੇ (op == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
        ਵਾਪਸ 1;
    }

    size_t value;
    ਜੇ (buft->iface.get_alloc_ਮਾਪ) {
        value = buft->iface.get_alloc_ਮਾਪ(buft, op);
    } ਨਹੀਂਤਾਂ {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_ਮਾਪ_t(enc, &value);

    ਵਾਪਸ 0;
}
