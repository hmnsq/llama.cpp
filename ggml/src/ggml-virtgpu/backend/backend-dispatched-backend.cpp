#അന്തര്ഭാവ "backend-dispatched.h"
#അന്തര്ഭാവ "backend-virgl-apir.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "shared/apir_backend.h"

#അന്തര്ഭാവ <cstdint>

സ്ഥിതം uint32_t validate_graph_operation(size_t cgraph_size, uint32_t shmem_res_id, സ്ഥിരം അക്ഷരം * operation) {
    എങ്കിൽ (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        തിരികെ 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    തിരികെ 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    സ്ഥിതം തർക്കം async_backend_initialized = അസത്യം;
    സ്ഥിതം തർക്കം async_backend;

    എങ്കിൽ (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = സത്യം;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    സ്ഥിരം ശൂന്യം * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    എങ്കിൽ (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        തിരികെ 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    എങ്കിൽ (validate_graph_operation(cgraph_size, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        തിരികെ 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((സ്ഥിരം അക്ഷരം *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    എങ്കിൽ (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        തിരികെ 1;
    }

    എങ്കിൽ (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        തിരികെ 1;
    }

    ggml_status status;
#എങ്കിൽ APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    വേണ്ടി (പൂർണ്ണസംഖ്യ idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_കണ്ണി(cgraph, idx);
        എങ്കിൽ (dev->iface.supports_op(dev, op)) {
            തുടരുക;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        തിരികെ 0;
    }
#endif

    // Check if backend is properly initialized
    എങ്കിൽ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        തിരികെ 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    എങ്കിൽ (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    തിരികെ 0;
}
