#شامل "backend-dispatched.h"
#شامل "backend-virgl-apir.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "shared/apir_backend.h"

#شامل <cstdint>

ساکن uint32_t validate_گراف_operation(size_t cgraph_سائز, uint32_t shmem_res_id, ثابت حرف * operation) {
    اگر (cgraph_سائز == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        واپس 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_غیر_مستعمل(shmem_res_id);

    واپس 0;  // Valid
}

uint32_t backend_backend_گراف_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);

    ساکن منطقی async_backend_initialized = جھوٹ;
    ساکن منطقی async_backend;

    اگر (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = سچ;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ثابت خالی * shmem_ڈیٹا = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    اگر (!shmem_ڈیٹا) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        واپس 1;
    }
    size_t cgraph_سائز;
    apir_decode_سائز_t(dec, &cgraph_سائز);

    اگر (validate_گراف_operation(cgraph_سائز, shmem_res_id, __func__) != 0) {
        apir_decoder_سیٹ_fatal(dec);
        واپس 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ثابت حرف *) shmem_ڈیٹا, cgraph_سائز);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_سائز);

    اگر (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        واپس 1;
    }

    اگر (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        واپس 1;
    }

    ggml_حالت حالت_2;
#اگر APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    برائے (صحیح idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_گراف_گرہ(cgraph, idx);
        اگر (dev->iface.supports_op(dev, op)) {
            جاری;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        حالت_2 = GGML_حالت_ABORTED;
        apir_encode_ggml_حالت(enc, &حالت_2);

        واپس 0;
    }
#اگر_ختم

    // Check if backend is properly initialized
    اگر (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        واپس 1;
    }

    حالت_2 = bck->iface.گراف_compute(bck, cgraph);

    اگر (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_حالت(enc, &حالت_2);

    واپس 0;
}
