#شامل "apir_cs.h"
#شامل "apir_cs_rpc.h"
#شامل "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

ایستا درونخطی تهی apir_encode_ggml_buffer_host_handle(apir_encoder * enc, ثابت apir_buffer_host_handle_t * handle);

ایستا درونخطی ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec);

/* apir_rpc_tensor */

ایستا درونخطی تهی apir_encode_rcp_tensor(apir_encoder * enc, ثابت apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_size = اندازه(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_size, apir_rpc_tensor, apir_rpc_tensor_size);
}

ایستا درونخطی apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_size = اندازه(apir_rpc_tensor);

    بازگشت (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

ایستا درونخطی apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_size = اندازه(apir_rpc_tensor) * n_tensors;

    بازگشت (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

/* ggml_tensor */

ایستا درونخطی تهی apir_encode_ggml_tensor(apir_encoder * enc, ثابت ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

ایستا درونخطی ثابت ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ثابت apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    اگر (!apir_rpc_tensor) {
        بازگشت پوچ;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/پوچ,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ثابت ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    بازگشت tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

ایستا درونخطی تهی apir_encode_ggml_buffer_type(apir_encoder * enc, ggml_backend_buffer_type_t buft) {
    apir_buffer_type_host_handle_t handle = ggml_buffer_type_to_apir_handle(buft);
    apir_encoder_write(enc, اندازه(handle), &handle, اندازه(handle));
}

ایستا درونخطی ggml_backend_buffer_type_t apir_decode_ggml_buffer_type(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, اندازه(handle), &handle, اندازه(handle));

    بازگشت (ggml_backend_buffer_type_t) handle;
}

ایستا درونخطی تهی apir_encode_apir_buffer_type_host_handle(apir_encoder * enc, apir_buffer_type_host_handle_t handle) {
    apir_encoder_write(enc, اندازه(handle), &handle, اندازه(handle));
}

ایستا درونخطی apir_buffer_type_host_handle_t apir_decode_apir_buffer_type_host_handle(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, اندازه(handle), &handle, اندازه(handle));

    بازگشت handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

ایستا درونخطی تهی apir_encode_ggml_buffer(apir_encoder * enc, ثابت ggml_backend_buffer_t buffer) {
    apir_buffer_host_handle_t handle = BUFFER_TO_HOST_HANDLE(buffer);
    apir_encoder_write(enc, اندازه(handle), &handle, اندازه(handle));
}

ایستا درونخطی ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec) {
    ggml_backend_buffer_t buffer;
    size_t                buffer_ptr_size = اندازه(buffer);

    apir_decoder_read(dec, buffer_ptr_size, &buffer, buffer_ptr_size);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    اگر (buffer) {
        بیرونی std::unordered_set<ggml_backend_buffer_t> backend_buffers;
        اگر (backend_buffers.find(buffer) == backend_buffers.end()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (تهی *) buffer);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_set_fatal(dec);
            بازگشت پوچ;
        }
    }

    بازگشت buffer;
}

/* enum ggml_status */

ایستا درونخطی تهی apir_encode_ggml_status(apir_encoder * enc, ثابت ggml_status * status) {
    apir_encoder_write(enc, اندازه(*status), status, اندازه(*status));
}

ایستا درونخطی تهی apir_decode_ggml_status(apir_decoder * dec, ggml_status * status) {
    apir_decoder_read(dec, اندازه(*status), status, اندازه(*status));
}

/* virtgpu_shmem */

ایستا درونخطی تهی apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

ایستا درونخطی تهی apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

ایستا درونخطی size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::vector<uint8_t> & cgraph_data) {
    apir_serialize_graph(cgraph, cgraph_data);

    بازگشت cgraph_data.size();
}

ایستا درونخطی تهی apir_encode_cgraph_data(apir_encoder * enc, std::vector<uint8_t> & cgraph_data) {
    size_t cgraph_size = cgraph_data.size();

    apir_encode(enc, cgraph_size, cgraph_data.data(), cgraph_size);
}

ایستا درونخطی ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_size) {
    GGML_UNUSED(cgraph_size);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ثابت uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ثابت apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    بازگشت apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

ایستا درونخطی تهی apir_encode_ggml_buffer_handle(apir_encoder * enc, ثابت apir_buffer_host_handle_t * handle) {
    apir_encoder_write(enc, اندازه(*handle), &handle, اندازه(*handle));
}

ایستا درونخطی تهی apir_encode_ggml_tensor_inline(apir_encoder * enc, ثابت ggml_tensor * tensor) {
    size_t tensor_size = اندازه(*tensor);

    اگر (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    اگر (tensor->src[0] && tensor->buffer) {
        ایستا صحیح first = 1;
        اگر (first) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_size, tensor, tensor_size);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    اگر (tensor->buffer) {
        apir_buffer_host_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
        apir_encode_ggml_buffer_handle(enc, &buffer_handle);
    }

    اگر (tensor->view_src) {
        apir_encoder_write(enc, tensor_size, tensor->view_src, tensor_size);
    }

    برای (صحیح i = 0; tensor->src[i]; i++) {
        ثابت ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_size, tensor_src, tensor_size);
    }
}

ایستا درونخطی ثابت ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, اندازه(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    اگر (tensor->buffer) {
        tensor->buffer = apir_decode_ggml_buffer(dec);
    }

    اگر (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, اندازه(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    برای (صحیح i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, اندازه(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    بازگشت tensor;
}
