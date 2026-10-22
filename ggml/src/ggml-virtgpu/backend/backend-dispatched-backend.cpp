#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "shared/apir_backend.h"

#অন্তর্ভাব <cstdint>

স্থির uint32_t validate_graph_operation(size_t cgraph_আকার, uint32_t shmem_res_id, ধ্রুবক অক্ষর * operation) {
    যদি (cgraph_আকার == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        ফেরত 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    ফেরত 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    স্থির যৌক্তিক async_backend_initialized = মিথ্যা;
    স্থির যৌক্তিক async_backend;

    যদি (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = সত্য;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ধ্রুবক শূন্য * shmem_তথ্য = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    যদি (!shmem_তথ্য) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        ফেরত 1;
    }
    size_t cgraph_আকার;
    apir_decode_আকার_t(dec, &cgraph_আকার);

    যদি (validate_graph_operation(cgraph_আকার, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        ফেরত 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ধ্রুবক অক্ষর *) shmem_তথ্য, cgraph_আকার);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_আকার);

    যদি (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        ফেরত 1;
    }

    যদি (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        ফেরত 1;
    }

    ggml_status status;
#যদি APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    জন্য (পূর্ণসংখ্যা idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_গ্রন্থি(cgraph, idx);
        যদি (dev->iface.supports_op(dev, op)) {
            চালু;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        ফেরত 0;
    }
#endif

    // Check if backend is properly initialized
    যদি (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        ফেরত 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    যদি (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    ফেরত 0;
}
