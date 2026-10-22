#समावेश "backend-dispatched.h"
#समावेश "backend-virgl-apir.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "shared/apir_backend.h"

#समावेश <cstdint>

अटल uint32_t validate_graph_operation(माप_प्रकार cgraph_आकार, uint32_t shmem_res_id, स्थिर अक्षर * operation) {
    अगर (cgraph_आकार == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        वापस 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    वापस 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    अटल bool async_backend_initialized = false;
    अटल bool async_backend;

    अगर (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = true;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(डेसी, &shmem_res_id);

    स्थिर व्योम * shmem_तथ्य = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    अगर (!shmem_तथ्य) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(डेसी);
        वापस 1;
    }
    माप_प्रकार cgraph_आकार;
    apir_decode_आकार_t(डेसी, &cgraph_आकार);

    अगर (validate_graph_operation(cgraph_आकार, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(डेसी);
        वापस 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((स्थिर अक्षर *) shmem_तथ्य, cgraph_आकार);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_आकार);

    अगर (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        वापस 1;
    }

    अगर (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        वापस 1;
    }

    ggml_status status;
#अगर APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    क्रम (पूर्णांक idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_गाँठ(cgraph, idx);
        अगर (dev->iface.supports_op(dev, op)) {
            जारी;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        वापस 0;
    }
#पूर्ण_अगर

    // Check if backend is properly initialized
    अगर (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        वापस 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    अगर (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    वापस 0;
}
