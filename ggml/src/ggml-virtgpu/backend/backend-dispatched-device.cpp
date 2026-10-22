#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdint>

uint32_t backend_device_get_device_ಎಣಿಕೆ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    int32_t dev_ಎಣಿಕೆ = reg->iface.get_device_ಎಣಿಕೆ(reg);
    apir_encode_int32_t(enc, &dev_ಎಣಿಕೆ);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_ಎಣಿಕೆ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    int32_t dev_ಎಣಿಕೆ = reg->iface.get_device_ಎಣಿಕೆ(reg);
    apir_encode_int32_t(enc, &dev_ಎಣಿಕೆ);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * string = dev->iface.get_name(dev);

    ಸ್ಥಿರಾಂಕ size_t string_ಗಾತ್ರ = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(string) + 1;
    apir_encode_array_ಗಾತ್ರ(enc, string_ಗಾತ್ರ);
    apir_encode_char_array(enc, string, string_ಗಾತ್ರ);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * string = dev->iface.get_description(dev);

    ಸ್ಥಿರಾಂಕ size_t string_ಗಾತ್ರ = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(string) + 1;
    apir_encode_array_ಗಾತ್ರ(enc, string_ಗಾತ್ರ);
    apir_encode_char_array(enc, string, string_ಗಾತ್ರ);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_ಸ್ಮರಣೆ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    size_t ಬಿಡುಗಡೆ, ಒಟ್ಟು;
    dev->iface.get_ಸ್ಮರಣೆ(dev, &ಬಿಡುಗಡೆ, &ಒಟ್ಟು);

    apir_encode_ಗಾತ್ರ_t(enc, &ಬಿಡುಗಡೆ);
    apir_encode_ಗಾತ್ರ_t(enc, &ಒಟ್ಟು);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);

    ಸ್ಥಿರಾಂಕ ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    ತಾರ್ಕಿಕ supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_ಬಫರ್_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    ggml_backend_ಬಫರ್_type_t bufft = dev->iface.get_ಬಫರ್_type(dev);

    apir_encode_ggml_ಬಫರ್_type(enc, bufft);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_ಬಫರ್);
    apir_encode_bool_t(enc, &props.caps.ಬಫರ್_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_device_ಬಫರ್_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ಶೂನ್ಯ * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ಆದರೆ (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
        ಹಿಂತಿರುಗಿ 1;
    }

    size_t size;
    apir_decode_ಗಾತ್ರ_t(dec, &size);
    size_t max_tensor_ಗಾತ್ರ;
    apir_decode_ಗಾತ್ರ_t(dec, &max_tensor_ಗಾತ್ರ);

    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = dev->iface.ಬಫರ್_from_host_ptr(dev, shmem_ptr, size, max_tensor_ಗಾತ್ರ);

    apir_encode_ggml_ಬಫರ್(enc, ಬಫರ್);
    apir_encode_ggml_ಬಫರ್_type(enc, ಬಫರ್->buft);

    ಆದರೆ (ಬಫರ್) {
        apir_track_backend_ಬಫರ್(ಬಫರ್);
    }

    ಹಿಂತಿರುಗಿ 0;
}
