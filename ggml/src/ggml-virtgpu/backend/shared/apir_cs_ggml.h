#അന്തര്ഭാവ "apir_cs.h"
#അന്തര്ഭാവ "apir_cs_rpc.h"
#അന്തര്ഭാവ "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_buffer_host_handle(apir_encoder * enc, സ്ഥിരം apir_buffer_host_handle_t * handle);

സ്ഥിതം വരിയിൽ ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec);

/* apir_rpc_tensor */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_rcp_tensor(apir_encoder * enc, സ്ഥിരം apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_size = വലുപ്പം(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_size, apir_rpc_tensor, apir_rpc_tensor_size);
}

സ്ഥിതം വരിയിൽ apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_size = വലുപ്പം(apir_rpc_tensor);

    തിരികെ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

സ്ഥിതം വരിയിൽ apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_size = വലുപ്പം(apir_rpc_tensor) * n_tensors;

    തിരികെ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

/* ggml_tensor */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_tensor(apir_encoder * enc, സ്ഥിരം ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

സ്ഥിതം വരിയിൽ സ്ഥിരം ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    സ്ഥിരം apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    എങ്കിൽ (!apir_rpc_tensor) {
        തിരികെ ശൂന്യ;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/ശൂന്യ,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    സ്ഥിരം ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    തിരികെ tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_buffer_type(apir_encoder * enc, ggml_backend_buffer_type_t buft) {
    apir_buffer_type_host_handle_t handle = ggml_buffer_type_to_apir_handle(buft);
    apir_encoder_write(enc, വലുപ്പം(handle), &handle, വലുപ്പം(handle));
}

സ്ഥിതം വരിയിൽ ggml_backend_buffer_type_t apir_decode_ggml_buffer_type(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, വലുപ്പം(handle), &handle, വലുപ്പം(handle));

    തിരികെ (ggml_backend_buffer_type_t) handle;
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_apir_buffer_type_host_handle(apir_encoder * enc, apir_buffer_type_host_handle_t handle) {
    apir_encoder_write(enc, വലുപ്പം(handle), &handle, വലുപ്പം(handle));
}

സ്ഥിതം വരിയിൽ apir_buffer_type_host_handle_t apir_decode_apir_buffer_type_host_handle(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, വലുപ്പം(handle), &handle, വലുപ്പം(handle));

    തിരികെ handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_buffer(apir_encoder * enc, സ്ഥിരം ggml_backend_buffer_t buffer) {
    apir_buffer_host_handle_t handle = BUFFER_TO_HOST_HANDLE(buffer);
    apir_encoder_write(enc, വലുപ്പം(handle), &handle, വലുപ്പം(handle));
}

സ്ഥിതം വരിയിൽ ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec) {
    ggml_backend_buffer_t buffer;
    size_t                buffer_ptr_size = വലുപ്പം(buffer);

    apir_decoder_read(dec, buffer_ptr_size, &buffer, buffer_ptr_size);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    എങ്കിൽ (buffer) {
        ബാഹ്യ std::unordered_set<ggml_backend_buffer_t> backend_buffers;
        എങ്കിൽ (backend_buffers.find(buffer) == backend_buffers.end()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (ശൂന്യം *) buffer);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_set_fatal(dec);
            തിരികെ ശൂന്യ;
        }
    }

    തിരികെ buffer;
}

/* enum ggml_status */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_status(apir_encoder * enc, സ്ഥിരം ggml_status * status) {
    apir_encoder_write(enc, വലുപ്പം(*status), status, വലുപ്പം(*status));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_ggml_status(apir_decoder * dec, ggml_status * status) {
    apir_decoder_read(dec, വലുപ്പം(*status), status, വലുപ്പം(*status));
}

/* virtgpu_shmem */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

സ്ഥിതം വരിയിൽ size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::vector<uint8_t> & cgraph_data) {
    apir_serialize_graph(cgraph, cgraph_data);

    തിരികെ cgraph_data.size();
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_cgraph_data(apir_encoder * enc, std::vector<uint8_t> & cgraph_data) {
    size_t cgraph_size = cgraph_data.size();

    apir_encode(enc, cgraph_size, cgraph_data.data(), cgraph_size);
}

സ്ഥിതം വരിയിൽ ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_size) {
    GGML_UNUSED(cgraph_size);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    സ്ഥിരം uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    സ്ഥിരം apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    തിരികെ apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_buffer_handle(apir_encoder * enc, സ്ഥിരം apir_buffer_host_handle_t * handle) {
    apir_encoder_write(enc, വലുപ്പം(*handle), &handle, വലുപ്പം(*handle));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_ggml_tensor_inline(apir_encoder * enc, സ്ഥിരം ggml_tensor * tensor) {
    size_t tensor_size = വലുപ്പം(*tensor);

    എങ്കിൽ (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    എങ്കിൽ (tensor->src[0] && tensor->buffer) {
        സ്ഥിതം പൂർണ്ണസംഖ്യ first = 1;
        എങ്കിൽ (first) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_size, tensor, tensor_size);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    എങ്കിൽ (tensor->buffer) {
        apir_buffer_host_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
        apir_encode_ggml_buffer_handle(enc, &buffer_handle);
    }

    എങ്കിൽ (tensor->view_src) {
        apir_encoder_write(enc, tensor_size, tensor->view_src, tensor_size);
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; tensor->src[i]; i++) {
        സ്ഥിരം ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_size, tensor_src, tensor_size);
    }
}

സ്ഥിതം വരിയിൽ സ്ഥിരം ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, വലുപ്പം(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    എങ്കിൽ (tensor->buffer) {
        tensor->buffer = apir_decode_ggml_buffer(dec);
    }

    എങ്കിൽ (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, വലുപ്പം(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, വലുപ്പം(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    തിരികെ tensor;
}
