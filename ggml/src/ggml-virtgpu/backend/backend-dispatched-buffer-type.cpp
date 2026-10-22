#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdint>

uint32_t backend_বাফার_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_বাফার_type_t buft;
    buft = apir_decode_ggml_বাফার_type(dec);

    ধ্রুবক অক্ষর * string = buft->iface.get_name(buft);

    ধ্রুবক size_t string_আকার = strlen(string) + 1;
    apir_encode_array_আকার(enc, string_আকার);
    apir_encode_char_array(enc, string, string_আকার);

    ফেরত 0;
}

uint32_t backend_বাফার_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_বাফার_type_t buft;
    buft = apir_decode_ggml_বাফার_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_আকার_t(enc, &value);

    ফেরত 0;
}

uint32_t backend_বাফার_type_get_max_আকার(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_বাফার_type_t buft;
    buft = apir_decode_ggml_বাফার_type(dec);

    size_t value = SIZE_MAX;
    যদি (buft->iface.get_max_আকার) {
        value = buft->iface.get_max_আকার(buft);
    }

    apir_encode_আকার_t(enc, &value);

    ফেরত 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_বাফার_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);
    ধ্রুবক যৌক্তিক is_host = মিথ্যা;

    apir_encode_bool_t(enc, &is_host);

    ফেরত 0;
}

uint32_t backend_বাফার_type_alloc_বাফার(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_বাফার_type_t buft;
    buft = apir_decode_ggml_বাফার_type(dec);

    size_t size;
    apir_decode_আকার_t(dec, &size);

    ggml_backend_বাফার_t বাফার;

    বাফার = buft->iface.alloc_বাফার(buft, size);

    apir_encode_ggml_বাফার(enc, বাফার);

    যদি (বাফার) {
        apir_track_backend_বাফার(বাফার);
    }

    ফেরত 0;
}

uint32_t backend_বাফার_type_get_alloc_আকার(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_বাফার_type_t buft;
    buft = apir_decode_ggml_বাফার_type(dec);

    ধ্রুবক ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    যদি (op == nullptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_set_fatal(dec);
        ফেরত 1;
    }

    size_t value;
    যদি (buft->iface.get_alloc_আকার) {
        value = buft->iface.get_alloc_আকার(buft, op);
    } অন্যথা {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_আকার_t(enc, &value);

    ফেরত 0;
}
