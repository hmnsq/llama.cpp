#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_backend.h"

#ಅನ್ತರ್ಭಾವ <cstdint>

ಸ್ಥಿರ uint32_t validate_ನಕ್ಷೆ_operation(size_t cgraph_ಗಾತ್ರ, uint32_t shmem_res_id, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * operation) {
    ಆದರೆ (cgraph_ಗಾತ್ರ == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        ಹಿಂತಿರುಗಿ 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_ಬಳಸದ(shmem_res_id);

    ಹಿಂತಿರುಗಿ 0;  // Valid
}

uint32_t backend_backend_ನಕ್ಷೆ_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);

    ಸ್ಥಿರ ತಾರ್ಕಿಕ async_backend_initialized = ಸುಳ್ಳು;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ async_backend;

    ಆದರೆ (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = ನಿಜ;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * shmem_ದತ್ತಾಂಶ = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ಆದರೆ (!shmem_ದತ್ತಾಂಶ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
        ಹಿಂತಿರುಗಿ 1;
    }
    size_t cgraph_ಗಾತ್ರ;
    apir_decode_ಗಾತ್ರ_t(dec, &cgraph_ಗಾತ್ರ);

    ಆದರೆ (validate_ನಕ್ಷೆ_operation(cgraph_ಗಾತ್ರ, shmem_res_id, __func__) != 0) {
        apir_decoder_ಗಣ_fatal(dec);
        ಹಿಂತಿರುಗಿ 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) shmem_ದತ್ತಾಂಶ, cgraph_ಗಾತ್ರ);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_ಗಾತ್ರ);

    ಆದರೆ (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಆದರೆ (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        ಹಿಂತಿರುಗಿ 1;
    }

    ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2;
#ಆದರೆ APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_ನಕ್ಷೆ_ಗಂಟು(cgraph, idx);
        ಆದರೆ (dev->iface.supports_op(dev, op)) {
            ಮುಂದುವರಿ;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        ಸ್ಥಿತಿ_2 = GGML_ಸ್ಥಿತಿ_ABORTED;
        apir_encode_ggml_ಸ್ಥಿತಿ(enc, &ಸ್ಥಿತಿ_2);

        ಹಿಂತಿರುಗಿ 0;
    }
#ಆದರೆ_ಅಂತ್ಯ

    // Check if backend is properly initialized
    ಆದರೆ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        ಹಿಂತಿರುಗಿ 1;
    }

    ಸ್ಥಿತಿ_2 = bck->iface.ನಕ್ಷೆ_compute(bck, cgraph);

    ಆದರೆ (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_ಸ್ಥಿತಿ(enc, &ಸ್ಥಿತಿ_2);

    ಹಿಂತಿರುಗಿ 0;
}
