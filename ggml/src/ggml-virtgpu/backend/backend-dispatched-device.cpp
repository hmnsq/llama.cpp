#ضمن "backend-dispatched.h"
#ضمن "backend-virgl-apir.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"

#ضمن <cstdint>

uint32_t backend_device_get_device_شمار(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    int32_t dev_شمار = reg->iface.get_device_شمار(reg);
    apir_encode_int32_t(enc, &dev_شمار);

    أرجع 0;
}

uint32_t backend_device_get_شمار(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    int32_t dev_شمار = reg->iface.get_device_شمار(reg);
    apir_encode_int32_t(enc, &dev_شمار);

    أرجع 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    ثابت حرف * string = dev->iface.get_name(dev);

    ثابت size_t string_اندازه = طول_رشته(string) + 1;
    apir_encode_array_اندازه(enc, string_اندازه);
    apir_encode_char_array(enc, string, string_اندازه);

    أرجع 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    ثابت حرف * string = dev->iface.get_description(dev);

    ثابت size_t string_اندازه = طول_رشته(string) + 1;
    apir_encode_array_اندازه(enc, string_اندازه);
    apir_encode_char_array(enc, string, string_اندازه);

    أرجع 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    أرجع 0;
}

uint32_t backend_device_get_حافظه(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    size_t حرر, مجموع;
    dev->iface.get_حافظه(dev, &حرر, &مجموع);

    apir_encode_اندازه_t(enc, &حرر);
    apir_encode_اندازه_t(enc, &مجموع);

    أرجع 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);

    ثابت ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    منطقي supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    أرجع 0;
}

uint32_t backend_device_get_میانگیر_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    ggml_backend_میانگیر_type_t bufft = dev->iface.get_میانگیر_type(dev);

    apir_encode_ggml_میانگیر_type(enc, bufft);

    أرجع 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_میانگیر);
    apir_encode_bool_t(enc, &props.caps.میانگیر_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    أرجع 0;
}

uint32_t backend_device_میانگیر_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    فراغ * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    إذا (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_مجموعه_fatal(dec);
        أرجع 1;
    }

    size_t size;
    apir_decode_اندازه_t(dec, &size);
    size_t max_tensor_اندازه;
    apir_decode_اندازه_t(dec, &max_tensor_اندازه);

    ggml_backend_میانگیر_t میانگیر;
    میانگیر = dev->iface.میانگیر_from_host_ptr(dev, shmem_ptr, size, max_tensor_اندازه);

    apir_encode_ggml_میانگیر(enc, میانگیر);
    apir_encode_ggml_میانگیر_type(enc, میانگیر->buft);

    إذا (میانگیر) {
        apir_track_backend_میانگیر(میانگیر);
    }

    أرجع 0;
}
