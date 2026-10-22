#ಅನ್ತರ್ಭಾವ "apir_cs.h"
#ಅನ್ತರ್ಭಾವ "apir_cs_rpc.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_ಬಫರ್_host_handle(apir_encoder * enc, ಸ್ಥಿರಾಂಕ apir_ಬಫರ್_host_handle_t * handle);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_backend_ಬಫರ್_t apir_decode_ggml_ಬಫರ್(apir_decoder * dec);

/* apir_rpc_tensor */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_rcp_tensor(apir_encoder * enc, ಸ್ಥಿರಾಂಕ apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_ಗಾತ್ರ = ಗಾತ್ರ(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_ಗಾತ್ರ, apir_rpc_tensor, apir_rpc_tensor_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_ಗಾತ್ರ = ಗಾತ್ರ(apir_rpc_tensor);

    ಹಿಂತಿರುಗಿ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_ಗಾತ್ರ = ಗಾತ್ರ(apir_rpc_tensor) * n_tensors;

    ಹಿಂತಿರುಗಿ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_ಗಾತ್ರ);
}

/* ggml_tensor */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_tensor(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ಸ್ಥಿರಾಂಕ apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    ಆದರೆ (!apir_rpc_tensor) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/ಖಾಲಿ,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ಸ್ಥಿರಾಂಕ ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    ಹಿಂತಿರುಗಿ tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_ಬಫರ್_type(apir_encoder * enc, ggml_backend_ಬಫರ್_type_t buft) {
    apir_ಬಫರ್_type_host_handle_t handle = ggml_ಬಫರ್_type_ಗಮ್ಯ_apir_handle(buft);
    apir_encoder_write(enc, ಗಾತ್ರ(handle), &handle, ಗಾತ್ರ(handle));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_backend_ಬಫರ್_type_t apir_decode_ggml_ಬಫರ್_type(apir_decoder * dec) {
    apir_ಬಫರ್_type_host_handle_t handle;

    apir_decoder_read(dec, ಗಾತ್ರ(handle), &handle, ಗಾತ್ರ(handle));

    ಹಿಂತಿರುಗಿ (ggml_backend_ಬಫರ್_type_t) handle;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_apir_ಬಫರ್_type_host_handle(apir_encoder * enc, apir_ಬಫರ್_type_host_handle_t handle) {
    apir_encoder_write(enc, ಗಾತ್ರ(handle), &handle, ಗಾತ್ರ(handle));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_ಬಫರ್_type_host_handle_t apir_decode_apir_ಬಫರ್_type_host_handle(apir_decoder * dec) {
    apir_ಬಫರ್_type_host_handle_t handle;

    apir_decoder_read(dec, ಗಾತ್ರ(handle), &handle, ಗಾತ್ರ(handle));

    ಹಿಂತಿರುಗಿ handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_ಬಫರ್(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_t ಬಫರ್) {
    apir_ಬಫರ್_host_handle_t handle = ಬಫರ್_ಗಮ್ಯ_HOST_HANDLE(ಬಫರ್);
    apir_encoder_write(enc, ಗಾತ್ರ(handle), &handle, ಗಾತ್ರ(handle));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_backend_ಬಫರ್_t apir_decode_ggml_ಬಫರ್(apir_decoder * dec) {
    ggml_backend_ಬಫರ್_t ಬಫರ್;
    size_t                ಬಫರ್_ptr_ಗಾತ್ರ = ಗಾತ್ರ(ಬಫರ್);

    apir_decoder_read(dec, ಬಫರ್_ptr_ಗಾತ್ರ, &ಬಫರ್, ಬಫರ್_ptr_ಗಾತ್ರ);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    ಆದರೆ (ಬಫರ್) {
        ಬಾಹ್ಯ std::unordered_ಗಣ<ggml_backend_ಬಫರ್_t> backend_buffers;
        ಆದರೆ (backend_buffers.ಕಂಡುಹಿಡಿ(ಬಫರ್) == backend_buffers.ಕೊನೆ()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (ಶೂನ್ಯ *) ಬಫರ್);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_ಗಣ_fatal(dec);
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }
    }

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

/* enum ggml_status */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_ಸ್ಥಿತಿ(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ggml_ಸ್ಥಿತಿ * ಸ್ಥಿತಿ_2) {
    apir_encoder_write(enc, ಗಾತ್ರ(*ಸ್ಥಿತಿ_2), ಸ್ಥಿತಿ_2, ಗಾತ್ರ(*ಸ್ಥಿತಿ_2));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_ggml_ಸ್ಥಿತಿ(apir_decoder * dec, ggml_ಸ್ಥಿತಿ * ಸ್ಥಿತಿ_2) {
    apir_decoder_read(dec, ಗಾತ್ರ(*ಸ್ಥಿತಿ_2), ಸ್ಥಿತಿ_2, ಗಾತ್ರ(*ಸ್ಥಿತಿ_2));
}

/* virtgpu_shmem */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::ಸದಿಶ<uint8_t> & cgraph_ದತ್ತಾಂಶ) {
    apir_serialize_ನಕ್ಷೆ(cgraph, cgraph_ದತ್ತಾಂಶ);

    ಹಿಂತಿರುಗಿ cgraph_ದತ್ತಾಂಶ.ಗಾತ್ರ_2();
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_cgraph_ದತ್ತಾಂಶ(apir_encoder * enc, std::ಸದಿಶ<uint8_t> & cgraph_ದತ್ತಾಂಶ) {
    size_t cgraph_ಗಾತ್ರ = cgraph_ದತ್ತಾಂಶ.ಗಾತ್ರ_2();

    apir_encode(enc, cgraph_ಗಾತ್ರ, cgraph_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), cgraph_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_ಗಾತ್ರ) {
    GGML_ಬಳಸದ(cgraph_ಗಾತ್ರ);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ಸ್ಥಿರಾಂಕ uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ಸ್ಥಿರಾಂಕ apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    ಹಿಂತಿರುಗಿ apir_deserialize_ನಕ್ಷೆ(n_nodes, n_tensors, tensors, nodes);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_ಬಫರ್_handle(apir_encoder * enc, ಸ್ಥಿರಾಂಕ apir_ಬಫರ್_host_handle_t * handle) {
    apir_encoder_write(enc, ಗಾತ್ರ(*handle), &handle, ಗಾತ್ರ(*handle));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ggml_tensor_inline(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    size_t tensor_ಗಾತ್ರ = ಗಾತ್ರ(*tensor);

    ಆದರೆ (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    ಆದರೆ (tensor->src[0] && tensor->ಬಫರ್) {
        ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ಮೊದಲ = 1;
        ಆದರೆ (ಮೊದಲ) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            ಮೊದಲ = 0;
        }
    }

    apir_encoder_write(enc, tensor_ಗಾತ್ರ, tensor, tensor_ಗಾತ್ರ);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    ಆದರೆ (tensor->ಬಫರ್) {
        apir_ಬಫರ್_host_handle_t ಬಫರ್_handle = ggml_ಬಫರ್_ಗಮ್ಯ_apir_handle(tensor->ಬಫರ್);
        apir_encode_ggml_ಬಫರ್_handle(enc, &ಬಫರ್_handle);
    }

    ಆದರೆ (tensor->view_src) {
        apir_encoder_write(enc, tensor_ಗಾತ್ರ, tensor->view_src, tensor_ಗಾತ್ರ);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; tensor->src[i]; i++) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_ಗಾತ್ರ, tensor_src, tensor_ಗಾತ್ರ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ಗಾತ್ರ(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    ಆದರೆ (tensor->ಬಫರ್) {
        tensor->ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);
    }

    ಆದರೆ (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ಗಾತ್ರ(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ಗಾತ್ರ(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    ಹಿಂತಿರುಗಿ tensor;
}
