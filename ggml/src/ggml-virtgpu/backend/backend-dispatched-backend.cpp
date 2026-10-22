#ضمن "backend-dispatched.h"
#ضمن "backend-virgl-apir.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "shared/apir_backend.h"

#ضمن <cstdint>

ساكن uint32_t validate_گراف_operation(size_t cgraph_سائيز, uint32_t shmem_res_id, ثابت حرف * operation) {
    إذا (cgraph_سائيز == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        أرجع 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_اڻ_استعمال_ٿيل(shmem_res_id);

    أرجع 0;  // Valid
}

uint32_t backend_backend_گراف_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);

    ساكن منطقي async_backend_initialized = خطأ;
    ساكن منطقي async_backend;

    إذا (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = سچ;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ثابت فراغ * shmem_معلومات = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    إذا (!shmem_معلومات) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_سيٽ_fatal(dec);
        أرجع 1;
    }
    size_t cgraph_سائيز;
    apir_decode_سائيز_t(dec, &cgraph_سائيز);

    إذا (validate_گراف_operation(cgraph_سائيز, shmem_res_id, __func__) != 0) {
        apir_decoder_سيٽ_fatal(dec);
        أرجع 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ثابت حرف *) shmem_معلومات, cgraph_سائيز);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_سائيز);

    إذا (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        أرجع 1;
    }

    إذا (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        أرجع 1;
    }

    ggml_حالت حالت;
#إذا APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    لكل (صحيح idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_گراف_ڳنڍ(cgraph, idx);
        إذا (dev->iface.supports_op(dev, op)) {
            استمر;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        حالت = GGML_حالت_ABORTED;
        apir_encode_ggml_حالت(enc, &حالت);

        أرجع 0;
    }
#جيڪڏهن_پورو

    // Check if backend is properly initialized
    إذا (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        أرجع 1;
    }

    حالت = bck->iface.گراف_compute(bck, cgraph);

    إذا (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_حالت(enc, &حالت);

    أرجع 0;
}
