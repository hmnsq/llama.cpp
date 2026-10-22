#అన్తర్భావ "apir_cs.h"
#అన్తర్భావ "apir_cs_rpc.h"
#అన్తర్భావ "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_బఫరు_host_handle(apir_encoder * enc, స్థిరరాశి apir_బఫరు_host_handle_t * handle);

స్థిర అంతఃవరుస ggml_backend_బఫరు_t apir_decode_ggml_బఫరు(apir_decoder * dec);

/* apir_rpc_tensor */

స్థిర అంతఃవరుస శూన్య apir_encode_rcp_tensor(apir_encoder * enc, స్థిరరాశి apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_కొలత = పరిమాణం(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_కొలత, apir_rpc_tensor, apir_rpc_tensor_కొలత);
}

స్థిర అంతఃవరుస apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_కొలత = పరిమాణం(apir_rpc_tensor);

    తిరిగి (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_కొలత);
}

స్థిర అంతఃవరుస apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_కొలత = పరిమాణం(apir_rpc_tensor) * n_tensors;

    తిరిగి (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_కొలత);
}

/* ggml_tensor */

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_tensor(apir_encoder * enc, స్థిరరాశి ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

స్థిర అంతఃవరుస స్థిరరాశి ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    స్థిరరాశి apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    ఒకవేళ (!apir_rpc_tensor) {
        తిరిగి ఖాళీ;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/ఖాళీ,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    స్థిరరాశి ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    తిరిగి tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_బఫరు_type(apir_encoder * enc, ggml_backend_బఫరు_type_t buft) {
    apir_బఫరు_type_host_handle_t handle = ggml_బఫరు_type_గమ్యం_apir_handle(buft);
    apir_encoder_write(enc, పరిమాణం(handle), &handle, పరిమాణం(handle));
}

స్థిర అంతఃవరుస ggml_backend_బఫరు_type_t apir_decode_ggml_బఫరు_type(apir_decoder * dec) {
    apir_బఫరు_type_host_handle_t handle;

    apir_decoder_read(dec, పరిమాణం(handle), &handle, పరిమాణం(handle));

    తిరిగి (ggml_backend_బఫరు_type_t) handle;
}

స్థిర అంతఃవరుస శూన్య apir_encode_apir_బఫరు_type_host_handle(apir_encoder * enc, apir_బఫరు_type_host_handle_t handle) {
    apir_encoder_write(enc, పరిమాణం(handle), &handle, పరిమాణం(handle));
}

స్థిర అంతఃవరుస apir_బఫరు_type_host_handle_t apir_decode_apir_బఫరు_type_host_handle(apir_decoder * dec) {
    apir_బఫరు_type_host_handle_t handle;

    apir_decoder_read(dec, పరిమాణం(handle), &handle, పరిమాణం(handle));

    తిరిగి handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_బఫరు(apir_encoder * enc, స్థిరరాశి ggml_backend_బఫరు_t బఫరు) {
    apir_బఫరు_host_handle_t handle = బఫరు_గమ్యం_HOST_HANDLE(బఫరు);
    apir_encoder_write(enc, పరిమాణం(handle), &handle, పరిమాణం(handle));
}

స్థిర అంతఃవరుస ggml_backend_బఫరు_t apir_decode_ggml_బఫరు(apir_decoder * dec) {
    ggml_backend_బఫరు_t బఫరు;
    size_t                బఫరు_ptr_కొలత = పరిమాణం(బఫరు);

    apir_decoder_read(dec, బఫరు_ptr_కొలత, &బఫరు, బఫరు_ptr_కొలత);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    ఒకవేళ (బఫరు) {
        బాహ్య std::unordered_సమితి<ggml_backend_బఫరు_t> backend_buffers;
        ఒకవేళ (backend_buffers.కనుగొను(బఫరు) == backend_buffers.చివర()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (శూన్య *) బఫరు);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_సమితి_fatal(dec);
            తిరిగి ఖాళీ;
        }
    }

    తిరిగి బఫరు;
}

/* enum ggml_status */

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_స్థితి(apir_encoder * enc, స్థిరరాశి ggml_స్థితి * స్థితి_2) {
    apir_encoder_write(enc, పరిమాణం(*స్థితి_2), స్థితి_2, పరిమాణం(*స్థితి_2));
}

స్థిర అంతఃవరుస శూన్య apir_decode_ggml_స్థితి(apir_decoder * dec, ggml_స్థితి * స్థితి_2) {
    apir_decoder_read(dec, పరిమాణం(*స్థితి_2), స్థితి_2, పరిమాణం(*స్థితి_2));
}

/* virtgpu_shmem */

స్థిర అంతఃవరుస శూన్య apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

స్థిర అంతఃవరుస శూన్య apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

స్థిర అంతఃవరుస size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::సదిశ<uint8_t> & cgraph_దత్తాంశం) {
    apir_serialize_లేఖాచిత్రం(cgraph, cgraph_దత్తాంశం);

    తిరిగి cgraph_దత్తాంశం.కొలత();
}

స్థిర అంతఃవరుస శూన్య apir_encode_cgraph_దత్తాంశం(apir_encoder * enc, std::సదిశ<uint8_t> & cgraph_దత్తాంశం) {
    size_t cgraph_కొలత = cgraph_దత్తాంశం.కొలత();

    apir_encode(enc, cgraph_కొలత, cgraph_దత్తాంశం.దత్తాంశం(), cgraph_కొలత);
}

స్థిర అంతఃవరుస ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_కొలత) {
    GGML_వాడని(cgraph_కొలత);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    స్థిరరాశి uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    స్థిరరాశి apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    తిరిగి apir_deserialize_లేఖాచిత్రం(n_nodes, n_tensors, tensors, nodes);
}

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_బఫరు_handle(apir_encoder * enc, స్థిరరాశి apir_బఫరు_host_handle_t * handle) {
    apir_encoder_write(enc, పరిమాణం(*handle), &handle, పరిమాణం(*handle));
}

స్థిర అంతఃవరుస శూన్య apir_encode_ggml_tensor_inline(apir_encoder * enc, స్థిరరాశి ggml_tensor * tensor) {
    size_t tensor_కొలత = పరిమాణం(*tensor);

    ఒకవేళ (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    ఒకవేళ (tensor->src[0] && tensor->బఫరు) {
        స్థిర పూర్ణాంకం మొదటి = 1;
        ఒకవేళ (మొదటి) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            మొదటి = 0;
        }
    }

    apir_encoder_write(enc, tensor_కొలత, tensor, tensor_కొలత);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    ఒకవేళ (tensor->బఫరు) {
        apir_బఫరు_host_handle_t బఫరు_handle = ggml_బఫరు_గమ్యం_apir_handle(tensor->బఫరు);
        apir_encode_ggml_బఫరు_handle(enc, &బఫరు_handle);
    }

    ఒకవేళ (tensor->view_src) {
        apir_encoder_write(enc, tensor_కొలత, tensor->view_src, tensor_కొలత);
    }

    కొరకు (పూర్ణాంకం i = 0; tensor->src[i]; i++) {
        స్థిరరాశి ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_కొలత, tensor_src, tensor_కొలత);
    }
}

స్థిర అంతఃవరుస స్థిరరాశి ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, పరిమాణం(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    ఒకవేళ (tensor->బఫరు) {
        tensor->బఫరు = apir_decode_ggml_బఫరు(dec);
    }

    ఒకవేళ (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, పరిమాణం(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    కొరకు (పూర్ణాంకం i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, పరిమాణం(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    తిరిగి tensor;
}
