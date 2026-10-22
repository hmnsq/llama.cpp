#समावेश "apir_cs.h"
#समावेश "apir_cs_rpc.h"
#समावेश "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

अटल अंतरभूत व्योम apir_encode_ggml_बफर_host_handle(apir_encoder * enc, स्थिर apir_बफर_host_handle_t * handle);

अटल अंतरभूत ggml_backend_बफर_t apir_decode_ggml_बफर(apir_decoder * dec);

/* apir_rpc_tensor */

अटल अंतरभूत व्योम apir_encode_rcp_tensor(apir_encoder * enc, स्थिर apir_rpc_tensor * apir_rpc_tensor) {
    माप_प्रकार apir_rpc_tensor_आकार = माप(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_आकार, apir_rpc_tensor, apir_rpc_tensor_आकार);
}

अटल अंतरभूत apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    माप_प्रकार apir_rpc_tensor_आकार = माप(apir_rpc_tensor);

    वापस (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_आकार);
}

अटल अंतरभूत apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    माप_प्रकार apir_rpc_tensor_आकार = माप(apir_rpc_tensor) * n_tensors;

    वापस (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_आकार);
}

/* ggml_tensor */

अटल अंतरभूत व्योम apir_encode_ggml_tensor(apir_encoder * enc, स्थिर ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

अटल अंतरभूत स्थिर ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    स्थिर apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    अगर (!apir_rpc_tensor) {
        वापस शून्य;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/शून्य,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    स्थिर ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    वापस tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

अटल अंतरभूत व्योम apir_encode_ggml_बफर_type(apir_encoder * enc, ggml_backend_बफर_type_t buft) {
    apir_बफर_type_host_handle_t handle = ggml_बफर_type_to_apir_handle(buft);
    apir_encoder_write(enc, माप(handle), &handle, माप(handle));
}

अटल अंतरभूत ggml_backend_बफर_type_t apir_decode_ggml_बफर_type(apir_decoder * dec) {
    apir_बफर_type_host_handle_t handle;

    apir_decoder_read(dec, माप(handle), &handle, माप(handle));

    वापस (ggml_backend_बफर_type_t) handle;
}

अटल अंतरभूत व्योम apir_encode_apir_बफर_type_host_handle(apir_encoder * enc, apir_बफर_type_host_handle_t handle) {
    apir_encoder_write(enc, माप(handle), &handle, माप(handle));
}

अटल अंतरभूत apir_बफर_type_host_handle_t apir_decode_apir_बफर_type_host_handle(apir_decoder * dec) {
    apir_बफर_type_host_handle_t handle;

    apir_decoder_read(dec, माप(handle), &handle, माप(handle));

    वापस handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

अटल अंतरभूत व्योम apir_encode_ggml_बफर(apir_encoder * enc, स्थिर ggml_backend_बफर_t बफर) {
    apir_बफर_host_handle_t handle = बफर_TO_HOST_HANDLE(बफर);
    apir_encoder_write(enc, माप(handle), &handle, माप(handle));
}

अटल अंतरभूत ggml_backend_बफर_t apir_decode_ggml_बफर(apir_decoder * dec) {
    ggml_backend_बफर_t बफर;
    माप_प्रकार                बफर_ptr_आकार = माप(बफर);

    apir_decoder_read(dec, बफर_ptr_आकार, &बफर, बफर_ptr_आकार);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    अगर (बफर) {
        बाह्य std::unordered_set<ggml_backend_बफर_t> backend_buffers;
        अगर (backend_buffers.find(बफर) == backend_buffers.end()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (व्योम *) बफर);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_set_fatal(dec);
            वापस शून्य;
        }
    }

    वापस बफर;
}

/* enum ggml_status */

अटल अंतरभूत व्योम apir_encode_ggml_status(apir_encoder * enc, स्थिर ggml_status * status) {
    apir_encoder_write(enc, माप(*status), status, माप(*status));
}

अटल अंतरभूत व्योम apir_decode_ggml_status(apir_decoder * dec, ggml_status * status) {
    apir_decoder_read(dec, माप(*status), status, माप(*status));
}

/* virtgpu_shmem */

अटल अंतरभूत व्योम apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

अटल अंतरभूत व्योम apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

अटल अंतरभूत माप_प्रकार apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::vector<uint8_t> & cgraph_तथ्य) {
    apir_serialize_graph(cgraph, cgraph_तथ्य);

    वापस cgraph_तथ्य.आकार();
}

अटल अंतरभूत व्योम apir_encode_cgraph_तथ्य(apir_encoder * enc, std::vector<uint8_t> & cgraph_तथ्य) {
    माप_प्रकार cgraph_आकार = cgraph_तथ्य.आकार();

    apir_encode(enc, cgraph_आकार, cgraph_तथ्य.तथ्य(), cgraph_आकार);
}

अटल अंतरभूत ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, माप_प्रकार cgraph_आकार) {
    GGML_UNUSED(cgraph_आकार);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    स्थिर uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    स्थिर apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    वापस apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

अटल अंतरभूत व्योम apir_encode_ggml_बफर_handle(apir_encoder * enc, स्थिर apir_बफर_host_handle_t * handle) {
    apir_encoder_write(enc, माप(*handle), &handle, माप(*handle));
}

अटल अंतरभूत व्योम apir_encode_ggml_tensor_inline(apir_encoder * enc, स्थिर ggml_tensor * tensor) {
    माप_प्रकार tensor_आकार = माप(*tensor);

    अगर (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    अगर (tensor->src[0] && tensor->बफर) {
        अटल पूर्णांक first = 1;
        अगर (first) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_आकार, tensor, tensor_आकार);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    अगर (tensor->बफर) {
        apir_बफर_host_handle_t बफर_handle = ggml_बफर_to_apir_handle(tensor->बफर);
        apir_encode_ggml_बफर_handle(enc, &बफर_handle);
    }

    अगर (tensor->view_src) {
        apir_encoder_write(enc, tensor_आकार, tensor->view_src, tensor_आकार);
    }

    क्रम (पूर्णांक i = 0; tensor->src[i]; i++) {
        स्थिर ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_आकार, tensor_src, tensor_आकार);
    }
}

अटल अंतरभूत स्थिर ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, माप(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    अगर (tensor->बफर) {
        tensor->बफर = apir_decode_ggml_बफर(dec);
    }

    अगर (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, माप(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    क्रम (पूर्णांक i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, माप(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    वापस tensor;
}
