#అన్తర్భావ "backend-dispatched.h"
#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdint>

uint32_t backend_బఫరు_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    ggml_backend_బఫరు_type_t buft;
    buft = apir_decode_ggml_బఫరు_type(dec);

    స్థిరరాశి అక్షరం * string = buft->iface.get_name(buft);

    స్థిరరాశి size_t string_కొలత = అక్షరశ్రేణి_పొడవు(string) + 1;
    apir_encode_array_కొలత(enc, string_కొలత);
    apir_encode_char_array(enc, string, string_కొలత);

    తిరిగి 0;
}

uint32_t backend_బఫరు_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    ggml_backend_బఫరు_type_t buft;
    buft = apir_decode_ggml_బఫరు_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_కొలత_t(enc, &value);

    తిరిగి 0;
}

uint32_t backend_బఫరు_type_get_max_కొలత(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    ggml_backend_బఫరు_type_t buft;
    buft = apir_decode_ggml_బఫరు_type(dec);

    size_t value = SIZE_MAX;
    ఒకవేళ (buft->iface.get_max_కొలత) {
        value = buft->iface.get_max_కొలత(buft);
    }

    apir_encode_కొలత_t(enc, &value);

    తిరిగి 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_బఫరు_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(dec);
    స్థిరరాశి తార్కిక is_host = అబద్ధం;

    apir_encode_bool_t(enc, &is_host);

    తిరిగి 0;
}

uint32_t backend_బఫరు_type_alloc_బఫరు(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    ggml_backend_బఫరు_type_t buft;
    buft = apir_decode_ggml_బఫరు_type(dec);

    size_t size;
    apir_decode_కొలత_t(dec, &size);

    ggml_backend_బఫరు_t బఫరు;

    బఫరు = buft->iface.alloc_బఫరు(buft, size);

    apir_encode_ggml_బఫరు(enc, బఫరు);

    ఒకవేళ (బఫరు) {
        apir_track_backend_బఫరు(బఫరు);
    }

    తిరిగి 0;
}

uint32_t backend_బఫరు_type_get_alloc_కొలత(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    ggml_backend_బఫరు_type_t buft;
    buft = apir_decode_ggml_బఫరు_type(dec);

    స్థిరరాశి ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    ఒకవేళ (op == శూన్యసూచిక) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_సమితి_fatal(dec);
        తిరిగి 1;
    }

    size_t value;
    ఒకవేళ (buft->iface.get_alloc_కొలత) {
        value = buft->iface.get_alloc_కొలత(buft, op);
    } లేకపోతే {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_కొలత_t(enc, &value);

    తిరిగి 0;
}
