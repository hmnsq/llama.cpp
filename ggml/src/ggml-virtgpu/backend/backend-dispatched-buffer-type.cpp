#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdint>

uint32_t backend_ಬಫರ್_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    ggml_backend_ಬಫರ್_type_t buft;
    buft = apir_decode_ggml_ಬಫರ್_type(dec);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * string = buft->iface.get_name(buft);

    ಸ್ಥಿರಾಂಕ size_t string_ಗಾತ್ರ = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(string) + 1;
    apir_encode_array_ಗಾತ್ರ(enc, string_ಗಾತ್ರ);
    apir_encode_char_array(enc, string, string_ಗಾತ್ರ);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    ggml_backend_ಬಫರ್_type_t buft;
    buft = apir_decode_ggml_ಬಫರ್_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_ಗಾತ್ರ_t(enc, &value);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_type_get_max_ಗಾತ್ರ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    ggml_backend_ಬಫರ್_type_t buft;
    buft = apir_decode_ggml_ಬಫರ್_type(dec);

    size_t value = SIZE_MAX;
    ಆದರೆ (buft->iface.get_max_ಗಾತ್ರ) {
        value = buft->iface.get_max_ಗಾತ್ರ(buft);
    }

    apir_encode_ಗಾತ್ರ_t(enc, &value);

    ಹಿಂತಿರುಗಿ 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_ಬಫರ್_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_host = ಸುಳ್ಳು;

    apir_encode_bool_t(enc, &is_host);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_type_alloc_ಬಫರ್(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    ggml_backend_ಬಫರ್_type_t buft;
    buft = apir_decode_ggml_ಬಫರ್_type(dec);

    size_t size;
    apir_decode_ಗಾತ್ರ_t(dec, &size);

    ggml_backend_ಬಫರ್_t ಬಫರ್;

    ಬಫರ್ = buft->iface.alloc_ಬಫರ್(buft, size);

    apir_encode_ggml_ಬಫರ್(enc, ಬಫರ್);

    ಆದರೆ (ಬಫರ್) {
        apir_track_backend_ಬಫರ್(ಬಫರ್);
    }

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    ggml_backend_ಬಫರ್_type_t buft;
    buft = apir_decode_ggml_ಬಫರ್_type(dec);

    ಸ್ಥಿರಾಂಕ ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    ಆದರೆ (op == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
        ಹಿಂತಿರುಗಿ 1;
    }

    size_t value;
    ಆದರೆ (buft->iface.get_alloc_ಗಾತ್ರ) {
        value = buft->iface.get_alloc_ಗಾತ್ರ(buft, op);
    } ಇಲ್ಲದಿದ್ದರೆ {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_ಗಾತ್ರ_t(enc, &value);

    ಹಿಂತಿರುಗಿ 0;
}
