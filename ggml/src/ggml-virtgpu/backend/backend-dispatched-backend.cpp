#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "shared/apir_backend.h"

#अन्तर्भाव <cstdint>

स्थिर uint32_t validate_graph_operation(size_t cgraph_size, uint32_t shmem_res_id, ध्रुव वर्ण * operation) {
    जइ (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        प्रत्यागम 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    प्रत्यागम 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    स्थिर तार्किक async_backend_initialized = मिच्छ;
    स्थिर तार्किक async_backend;

    जइ (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = सच्च;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ध्रुव शून्य * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    जइ (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        प्रत्यागम 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    जइ (validate_graph_operation(cgraph_size, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        प्रत्यागम 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ध्रुव वर्ण *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    जइ (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        प्रत्यागम 1;
    }

    जइ (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        प्रत्यागम 1;
    }

    ggml_status status;
#जइ APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    प्रति (पूर्णाङ्क idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_ग्रन्थिः(cgraph, idx);
        जइ (dev->iface.supports_op(dev, op)) {
            अनुवर्त;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        प्रत्यागम 0;
    }
#endif

    // Check if backend is properly initialized
    जइ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        प्रत्यागम 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    जइ (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    प्रत्यागम 0;
}
