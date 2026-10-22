#ਅਨ੍ਤਰ੍ਭਾਵ "apir_cs.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "apir_cs_rpc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_ਬਫਰ_host_handle(apir_encoder * enc, ਅਚਲ apir_ਬਫਰ_host_handle_t * handle);

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_backend_ਬਫਰ_t apir_decode_ggml_ਬਫਰ(apir_decoder * dec);

/* apir_rpc_tensor */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_rcp_tensor(apir_encoder * enc, ਅਚਲ apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_ਮਾਪ = ਆਕਾਰ(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_ਮਾਪ, apir_rpc_tensor, apir_rpc_tensor_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_ਮਾਪ = ਆਕਾਰ(apir_rpc_tensor);

    ਵਾਪਸ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_ਮਾਪ = ਆਕਾਰ(apir_rpc_tensor) * n_tensors;

    ਵਾਪਸ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_ਮਾਪ);
}

/* ggml_tensor */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_tensor(apir_encoder * enc, ਅਚਲ ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ਅਚਲ apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    ਜੇ (!apir_rpc_tensor) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/ਨਿਰਾ,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ਅਚਲ ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    ਵਾਪਸ tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_ਬਫਰ_type(apir_encoder * enc, ggml_backend_ਬਫਰ_type_t buft) {
    apir_ਬਫਰ_type_host_handle_t handle = ggml_ਬਫਰ_type_ਮੰਜ਼ਿਲ_apir_handle(buft);
    apir_encoder_write(enc, ਆਕਾਰ(handle), &handle, ਆਕਾਰ(handle));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_backend_ਬਫਰ_type_t apir_decode_ggml_ਬਫਰ_type(apir_decoder * dec) {
    apir_ਬਫਰ_type_host_handle_t handle;

    apir_decoder_read(dec, ਆਕਾਰ(handle), &handle, ਆਕਾਰ(handle));

    ਵਾਪਸ (ggml_backend_ਬਫਰ_type_t) handle;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_apir_ਬਫਰ_type_host_handle(apir_encoder * enc, apir_ਬਫਰ_type_host_handle_t handle) {
    apir_encoder_write(enc, ਆਕਾਰ(handle), &handle, ਆਕਾਰ(handle));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_ਬਫਰ_type_host_handle_t apir_decode_apir_ਬਫਰ_type_host_handle(apir_decoder * dec) {
    apir_ਬਫਰ_type_host_handle_t handle;

    apir_decoder_read(dec, ਆਕਾਰ(handle), &handle, ਆਕਾਰ(handle));

    ਵਾਪਸ handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_ਬਫਰ(apir_encoder * enc, ਅਚਲ ggml_backend_ਬਫਰ_t ਬਫਰ) {
    apir_ਬਫਰ_host_handle_t handle = ਬਫਰ_ਮੰਜ਼ਿਲ_HOST_HANDLE(ਬਫਰ);
    apir_encoder_write(enc, ਆਕਾਰ(handle), &handle, ਆਕਾਰ(handle));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_backend_ਬਫਰ_t apir_decode_ggml_ਬਫਰ(apir_decoder * dec) {
    ggml_backend_ਬਫਰ_t ਬਫਰ;
    size_t                ਬਫਰ_ptr_ਮਾਪ = ਆਕਾਰ(ਬਫਰ);

    apir_decoder_read(dec, ਬਫਰ_ptr_ਮਾਪ, &ਬਫਰ, ਬਫਰ_ptr_ਮਾਪ);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    ਜੇ (ਬਫਰ) {
        ਬਾਹਰੀ std::unordered_ਸੈੱਟ<ggml_backend_ਬਫਰ_t> backend_buffers;
        ਜੇ (backend_buffers.ਲੱਭੋ(ਬਫਰ) == backend_buffers.ਸਿਰਾ()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (ਖਾਲੀ *) ਬਫਰ);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_ਸੈੱਟ_fatal(dec);
            ਵਾਪਸ ਨਿਰਾ;
        }
    }

    ਵਾਪਸ ਬਫਰ;
}

/* enum ggml_status */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_ਸਥਿਤੀ(apir_encoder * enc, ਅਚਲ ggml_ਸਥਿਤੀ * ਸਥਿਤੀ) {
    apir_encoder_write(enc, ਆਕਾਰ(*ਸਥਿਤੀ), ਸਥਿਤੀ, ਆਕਾਰ(*ਸਥਿਤੀ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_ggml_ਸਥਿਤੀ(apir_decoder * dec, ggml_ਸਥਿਤੀ * ਸਥਿਤੀ) {
    apir_decoder_read(dec, ਆਕਾਰ(*ਸਥਿਤੀ), ਸਥਿਤੀ, ਆਕਾਰ(*ਸਥਿਤੀ));
}

/* virtgpu_shmem */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::ਸਦਿਸ਼<uint8_t> & cgraph_ਅੰਕੜੇ) {
    apir_serialize_ਗ੍ਰਾਫ(cgraph, cgraph_ਅੰਕੜੇ);

    ਵਾਪਸ cgraph_ਅੰਕੜੇ.ਮਾਪ();
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_cgraph_ਅੰਕੜੇ(apir_encoder * enc, std::ਸਦਿਸ਼<uint8_t> & cgraph_ਅੰਕੜੇ) {
    size_t cgraph_ਮਾਪ = cgraph_ਅੰਕੜੇ.ਮਾਪ();

    apir_encode(enc, cgraph_ਮਾਪ, cgraph_ਅੰਕੜੇ.ਅੰਕੜੇ(), cgraph_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_ਮਾਪ) {
    GGML_ਅਣਵਰਤਿਆ(cgraph_ਮਾਪ);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ਅਚਲ uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ਅਚਲ apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    ਵਾਪਸ apir_deserialize_ਗ੍ਰਾਫ(n_nodes, n_tensors, tensors, nodes);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_ਬਫਰ_handle(apir_encoder * enc, ਅਚਲ apir_ਬਫਰ_host_handle_t * handle) {
    apir_encoder_write(enc, ਆਕਾਰ(*handle), &handle, ਆਕਾਰ(*handle));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ggml_tensor_inline(apir_encoder * enc, ਅਚਲ ggml_tensor * tensor) {
    size_t tensor_ਮਾਪ = ਆਕਾਰ(*tensor);

    ਜੇ (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    ਜੇ (tensor->src[0] && tensor->ਬਫਰ) {
        ਸਥਿਰ ਪੂਰਨ ਪਹਿਲਾ = 1;
        ਜੇ (ਪਹਿਲਾ) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            ਪਹਿਲਾ = 0;
        }
    }

    apir_encoder_write(enc, tensor_ਮਾਪ, tensor, tensor_ਮਾਪ);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    ਜੇ (tensor->ਬਫਰ) {
        apir_ਬਫਰ_host_handle_t ਬਫਰ_handle = ggml_ਬਫਰ_ਮੰਜ਼ਿਲ_apir_handle(tensor->ਬਫਰ);
        apir_encode_ggml_ਬਫਰ_handle(enc, &ਬਫਰ_handle);
    }

    ਜੇ (tensor->view_src) {
        apir_encoder_write(enc, tensor_ਮਾਪ, tensor->view_src, tensor_ਮਾਪ);
    }

    ਲਈ (ਪੂਰਨ i = 0; tensor->src[i]; i++) {
        ਅਚਲ ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_ਮਾਪ, tensor_src, tensor_ਮਾਪ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ਆਕਾਰ(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    ਜੇ (tensor->ਬਫਰ) {
        tensor->ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);
    }

    ਜੇ (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ਆਕਾਰ(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    ਲਈ (ਪੂਰਨ i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ਆਕਾਰ(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    ਵਾਪਸ tensor;
}
