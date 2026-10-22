#अन्तर्भाव "apir_cs.h"
#अन्तर्भाव "apir_cs_rpc.h"
#अन्तर्भाव "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

स्थिर अन्तःस्थ शून्य apir_encode_ggml_बफर_host_handle(apir_encoder * enc, ध्रुव apir_बफर_host_handle_t * handle);

स्थिर अन्तःस्थ ggml_backend_बफर_t apir_decode_ggml_बफर(apir_decoder * dec);

/* apir_rpc_tensor */

स्थिर अन्तःस्थ शून्य apir_encode_rcp_tensor(apir_encoder * enc, ध्रुव apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_आकार = परिमाण(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_आकार, apir_rpc_tensor, apir_rpc_tensor_आकार);
}

स्थिर अन्तःस्थ apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_आकार = परिमाण(apir_rpc_tensor);

    प्रत्यागम (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_आकार);
}

स्थिर अन्तःस्थ apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_आकार = परिमाण(apir_rpc_tensor) * n_tensors;

    प्रत्यागम (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_आकार);
}

/* ggml_tensor */

स्थिर अन्तःस्थ शून्य apir_encode_ggml_tensor(apir_encoder * enc, ध्रुव ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

स्थिर अन्तःस्थ ध्रुव ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ध्रुव apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    यदि (!apir_rpc_tensor) {
        प्रत्यागम रिक्त;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/रिक्त,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ध्रुव ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    प्रत्यागम tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

स्थिर अन्तःस्थ शून्य apir_encode_ggml_बफर_type(apir_encoder * enc, ggml_backend_बफर_type_t buft) {
    apir_बफर_type_host_handle_t handle = ggml_बफर_type_गन्तव्यम्_apir_handle(buft);
    apir_encoder_write(enc, परिमाण(handle), &handle, परिमाण(handle));
}

स्थिर अन्तःस्थ ggml_backend_बफर_type_t apir_decode_ggml_बफर_type(apir_decoder * dec) {
    apir_बफर_type_host_handle_t handle;

    apir_decoder_read(dec, परिमाण(handle), &handle, परिमाण(handle));

    प्रत्यागम (ggml_backend_बफर_type_t) handle;
}

स्थिर अन्तःस्थ शून्य apir_encode_apir_बफर_type_host_handle(apir_encoder * enc, apir_बफर_type_host_handle_t handle) {
    apir_encoder_write(enc, परिमाण(handle), &handle, परिमाण(handle));
}

स्थिर अन्तःस्थ apir_बफर_type_host_handle_t apir_decode_apir_बफर_type_host_handle(apir_decoder * dec) {
    apir_बफर_type_host_handle_t handle;

    apir_decoder_read(dec, परिमाण(handle), &handle, परिमाण(handle));

    प्रत्यागम handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

स्थिर अन्तःस्थ शून्य apir_encode_ggml_बफर(apir_encoder * enc, ध्रुव ggml_backend_बफर_t बफर) {
    apir_बफर_host_handle_t handle = बफर_गन्तव्यम्_HOST_HANDLE(बफर);
    apir_encoder_write(enc, परिमाण(handle), &handle, परिमाण(handle));
}

स्थिर अन्तःस्थ ggml_backend_बफर_t apir_decode_ggml_बफर(apir_decoder * dec) {
    ggml_backend_बफर_t बफर;
    size_t                बफर_ptr_आकार = परिमाण(बफर);

    apir_decoder_read(dec, बफर_ptr_आकार, &बफर, बफर_ptr_आकार);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    यदि (बफर) {
        बाह्य std::unordered_समुच्चयः<ggml_backend_बफर_t> backend_buffers;
        यदि (backend_buffers.find(बफर) == backend_buffers.अन्तः_2()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (शून्य *) बफर);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_समुच्चयः_fatal(dec);
            प्रत्यागम रिक्त;
        }
    }

    प्रत्यागम बफर;
}

/* enum ggml_status */

स्थिर अन्तःस्थ शून्य apir_encode_ggml_स्थितिः(apir_encoder * enc, ध्रुव ggml_स्थितिः * स्थितिः) {
    apir_encoder_write(enc, परिमाण(*स्थितिः), स्थितिः, परिमाण(*स्थितिः));
}

स्थिर अन्तःस्थ शून्य apir_decode_ggml_स्थितिः(apir_decoder * dec, ggml_स्थितिः * स्थितिः) {
    apir_decoder_read(dec, परिमाण(*स्थितिः), स्थितिः, परिमाण(*स्थितिः));
}

/* virtgpu_shmem */

स्थिर अन्तःस्थ शून्य apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

स्थिर अन्तःस्थ शून्य apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

स्थिर अन्तःस्थ size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::सदिशम्<uint8_t> & cgraph_दत्त) {
    apir_serialize_graph(cgraph, cgraph_दत्त);

    प्रत्यागम cgraph_दत्त.आकार();
}

स्थिर अन्तःस्थ शून्य apir_encode_cgraph_दत्त(apir_encoder * enc, std::सदिशम्<uint8_t> & cgraph_दत्त) {
    size_t cgraph_आकार = cgraph_दत्त.आकार();

    apir_encode(enc, cgraph_आकार, cgraph_दत्त.दत्त(), cgraph_आकार);
}

स्थिर अन्तःस्थ ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_आकार) {
    GGML_अप्रयुक्तम्(cgraph_आकार);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ध्रुव uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ध्रुव apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    प्रत्यागम apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

स्थिर अन्तःस्थ शून्य apir_encode_ggml_बफर_handle(apir_encoder * enc, ध्रुव apir_बफर_host_handle_t * handle) {
    apir_encoder_write(enc, परिमाण(*handle), &handle, परिमाण(*handle));
}

स्थिर अन्तःस्थ शून्य apir_encode_ggml_tensor_inline(apir_encoder * enc, ध्रुव ggml_tensor * tensor) {
    size_t tensor_आकार = परिमाण(*tensor);

    यदि (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    यदि (tensor->src[0] && tensor->बफर) {
        स्थिर पूर्णाङ्क first = 1;
        यदि (first) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_आकार, tensor, tensor_आकार);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    यदि (tensor->बफर) {
        apir_बफर_host_handle_t बफर_handle = ggml_बफर_गन्तव्यम्_apir_handle(tensor->बफर);
        apir_encode_ggml_बफर_handle(enc, &बफर_handle);
    }

    यदि (tensor->view_src) {
        apir_encoder_write(enc, tensor_आकार, tensor->view_src, tensor_आकार);
    }

    प्रति (पूर्णाङ्क i = 0; tensor->src[i]; i++) {
        ध्रुव ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_आकार, tensor_src, tensor_आकार);
    }
}

स्थिर अन्तःस्थ ध्रुव ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, परिमाण(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    यदि (tensor->बफर) {
        tensor->बफर = apir_decode_ggml_बफर(dec);
    }

    यदि (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, परिमाण(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    प्रति (पूर्णाङ्क i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, परिमाण(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    प्रत्यागम tensor;
}
