#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdint>

uint32_t backend_buffer_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    ধ্রুব বর্ণ * string = buft->iface.get_name(buft);

    ধ্রুব size_t string_size = strlen(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    প্রত্যাগম 0;
}

uint32_t backend_buffer_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_size_t(enc, &value);

    প্রত্যাগম 0;
}

uint32_t backend_buffer_type_get_max_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    size_t value = SIZE_MAX;
    যদি (buft->iface.get_max_size) {
        value = buft->iface.get_max_size(buft);
    }

    apir_encode_size_t(enc, &value);

    প্রত্যাগম 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_buffer_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);
    ধ্রুব তার্কিক is_host = অসত্য;

    apir_encode_bool_t(enc, &is_host);

    প্রত্যাগম 0;
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

    যদি (buffer) {
        apir_track_backend_buffer(buffer);
    }

    প্রত্যাগম 0;
}

uint32_t backend_buffer_type_get_alloc_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_type_t buft;
    buft = apir_decode_ggml_buffer_type(dec);

    ধ্রুব ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    যদি (op == nullptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_set_fatal(dec);
        প্রত্যাগম 1;
    }

    size_t value;
    যদি (buft->iface.get_alloc_size) {
        value = buft->iface.get_alloc_size(buft, op);
    } অন্যথা {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_size_t(enc, &value);

    প্রত্যাগম 0;
}
