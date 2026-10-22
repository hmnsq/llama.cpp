#ଅନ୍ତର୍ଭାଵ "apir_cs.h"
#ଅନ୍ତର୍ଭାଵ "apir_cs_rpc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_buffer_host_handle(apir_encoder * enc, ଅଚଳ apir_buffer_host_handle_t * handle);

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec);

/* apir_rpc_tensor */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_rcp_tensor(apir_encoder * enc, ଅଚଳ apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_size = ଆକାର(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_size, apir_rpc_tensor, apir_rpc_tensor_size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_size = ଆକାର(apir_rpc_tensor);

    ଫେରନ୍ତୁ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_size = ଆକାର(apir_rpc_tensor) * n_tensors;

    ଫେରନ୍ତୁ (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

/* ggml_tensor */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_tensor(apir_encoder * enc, ଅଚଳ ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ଅଚଳ apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    ଯଦି (!apir_rpc_tensor) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/ଖାଲି,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ଅଚଳ ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    ଫେରନ୍ତୁ tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_buffer_type(apir_encoder * enc, ggml_backend_buffer_type_t buft) {
    apir_buffer_type_host_handle_t handle = ggml_buffer_type_to_apir_handle(buft);
    apir_encoder_write(enc, ଆକାର(handle), &handle, ଆକାର(handle));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_backend_buffer_type_t apir_decode_ggml_buffer_type(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, ଆକାର(handle), &handle, ଆକାର(handle));

    ଫେରନ୍ତୁ (ggml_backend_buffer_type_t) handle;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_apir_buffer_type_host_handle(apir_encoder * enc, apir_buffer_type_host_handle_t handle) {
    apir_encoder_write(enc, ଆକାର(handle), &handle, ଆକାର(handle));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ apir_buffer_type_host_handle_t apir_decode_apir_buffer_type_host_handle(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, ଆକାର(handle), &handle, ଆକାର(handle));

    ଫେରନ୍ତୁ handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_buffer(apir_encoder * enc, ଅଚଳ ggml_backend_buffer_t buffer) {
    apir_buffer_host_handle_t handle = BUFFER_TO_HOST_HANDLE(buffer);
    apir_encoder_write(enc, ଆକାର(handle), &handle, ଆକାର(handle));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec) {
    ggml_backend_buffer_t buffer;
    size_t                buffer_ptr_size = ଆକାର(buffer);

    apir_decoder_read(dec, buffer_ptr_size, &buffer, buffer_ptr_size);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    ଯଦି (buffer) {
        ବାହ୍ୟ std::unordered_set<ggml_backend_buffer_t> backend_buffers;
        ଯଦି (backend_buffers.find(buffer) == backend_buffers.end()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (ଶୂନ୍ୟ *) buffer);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_set_fatal(dec);
            ଫେରନ୍ତୁ ଖାଲି;
        }
    }

    ଫେରନ୍ତୁ buffer;
}

/* enum ggml_status */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_status(apir_encoder * enc, ଅଚଳ ggml_status * status) {
    apir_encoder_write(enc, ଆକାର(*status), status, ଆକାର(*status));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_ggml_status(apir_decoder * dec, ggml_status * status) {
    apir_decoder_read(dec, ଆକାର(*status), status, ଆକାର(*status));
}

/* virtgpu_shmem */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::vector<uint8_t> & cgraph_data) {
    apir_serialize_graph(cgraph, cgraph_data);

    ଫେରନ୍ତୁ cgraph_data.size();
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_cgraph_data(apir_encoder * enc, std::vector<uint8_t> & cgraph_data) {
    size_t cgraph_size = cgraph_data.size();

    apir_encode(enc, cgraph_size, cgraph_data.data(), cgraph_size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_size) {
    GGML_UNUSED(cgraph_size);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ଅଚଳ uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ଅଚଳ apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    ଫେରନ୍ତୁ apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_buffer_handle(apir_encoder * enc, ଅଚଳ apir_buffer_host_handle_t * handle) {
    apir_encoder_write(enc, ଆକାର(*handle), &handle, ଆକାର(*handle));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_ggml_tensor_inline(apir_encoder * enc, ଅଚଳ ggml_tensor * tensor) {
    size_t tensor_size = ଆକାର(*tensor);

    ଯଦି (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    ଯଦି (tensor->src[0] && tensor->buffer) {
        ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା first = 1;
        ଯଦି (first) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_size, tensor, tensor_size);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    ଯଦି (tensor->buffer) {
        apir_buffer_host_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
        apir_encode_ggml_buffer_handle(enc, &buffer_handle);
    }

    ଯଦି (tensor->view_src) {
        apir_encoder_write(enc, tensor_size, tensor->view_src, tensor_size);
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; tensor->src[i]; i++) {
        ଅଚଳ ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_size, tensor_src, tensor_size);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ଆକାର(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    ଯଦି (tensor->buffer) {
        tensor->buffer = apir_decode_ggml_buffer(dec);
    }

    ଯଦି (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ଆକାର(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, ଆକାର(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    ଫେରନ୍ତୁ tensor;
}
