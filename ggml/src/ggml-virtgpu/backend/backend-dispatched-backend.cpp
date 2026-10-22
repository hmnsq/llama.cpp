#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "shared/apir_backend.h"

#অন্তর্ভাব <cstdint>

স্থির uint32_t validate_graph_operation(size_t cgraph_size, uint32_t shmem_res_id, ধ্রুব বর্ণ * operation) {
    যদি (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        প্রত্যাগম 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    প্রত্যাগম 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    স্থির তার্কিক async_backend_initialized = অসত্য;
    স্থির তার্কিক async_backend;

    যদি (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = সত্য;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ধ্রুব শূন্য * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    যদি (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        প্রত্যাগম 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    যদি (validate_graph_operation(cgraph_size, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        প্রত্যাগম 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ধ্রুব বর্ণ *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    যদি (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        প্রত্যাগম 1;
    }

    যদি (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        প্রত্যাগম 1;
    }

    ggml_status status;
#যদি APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    প্রতি (পূর্ণাঙ্ক idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_গ্রন্থি(cgraph, idx);
        যদি (dev->iface.supports_op(dev, op)) {
            অনুবর্ত;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        প্রত্যাগম 0;
    }
#endif

    // Check if backend is properly initialized
    যদি (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        প্রত্যাগম 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    যদি (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    প্রত্যাগম 0;
}
