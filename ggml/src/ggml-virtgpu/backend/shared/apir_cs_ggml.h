#অন্তর্ভাব "apir_cs.h"
#অন্তর্ভাব "apir_cs_rpc.h"
#অন্তর্ভাব "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_বাফার_host_handle(apir_encoder * enc, ধ্রুবক apir_বাফার_host_handle_t * handle);

স্থির অন্তঃরেখ ggml_backend_বাফার_t apir_decode_ggml_বাফার(apir_decoder * dec);

/* apir_rpc_tensor */

স্থির অন্তঃরেখ শূন্য apir_encode_rcp_tensor(apir_encoder * enc, ধ্রুবক apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_আকার = আকার(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_আকার, apir_rpc_tensor, apir_rpc_tensor_আকার);
}

স্থির অন্তঃরেখ apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_আকার = আকার(apir_rpc_tensor);

    ফেরত (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_আকার);
}

স্থির অন্তঃরেখ apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_আকার = আকার(apir_rpc_tensor) * n_tensors;

    ফেরত (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_আকার);
}

/* ggml_tensor */

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_tensor(apir_encoder * enc, ধ্রুবক ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

স্থির অন্তঃরেখ ধ্রুবক ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ধ্রুবক apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    যদি (!apir_rpc_tensor) {
        ফেরত ফাঁকা;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/ফাঁকা,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ধ্রুবক ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    ফেরত tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_বাফার_type(apir_encoder * enc, ggml_backend_বাফার_type_t buft) {
    apir_বাফার_type_host_handle_t handle = ggml_বাফার_type_to_apir_handle(buft);
    apir_encoder_write(enc, আকার(handle), &handle, আকার(handle));
}

স্থির অন্তঃরেখ ggml_backend_বাফার_type_t apir_decode_ggml_বাফার_type(apir_decoder * dec) {
    apir_বাফার_type_host_handle_t handle;

    apir_decoder_read(dec, আকার(handle), &handle, আকার(handle));

    ফেরত (ggml_backend_বাফার_type_t) handle;
}

স্থির অন্তঃরেখ শূন্য apir_encode_apir_বাফার_type_host_handle(apir_encoder * enc, apir_বাফার_type_host_handle_t handle) {
    apir_encoder_write(enc, আকার(handle), &handle, আকার(handle));
}

স্থির অন্তঃরেখ apir_বাফার_type_host_handle_t apir_decode_apir_বাফার_type_host_handle(apir_decoder * dec) {
    apir_বাফার_type_host_handle_t handle;

    apir_decoder_read(dec, আকার(handle), &handle, আকার(handle));

    ফেরত handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_বাফার(apir_encoder * enc, ধ্রুবক ggml_backend_বাফার_t বাফার) {
    apir_বাফার_host_handle_t handle = বাফার_TO_HOST_HANDLE(বাফার);
    apir_encoder_write(enc, আকার(handle), &handle, আকার(handle));
}

স্থির অন্তঃরেখ ggml_backend_বাফার_t apir_decode_ggml_বাফার(apir_decoder * dec) {
    ggml_backend_বাফার_t বাফার;
    size_t                বাফার_ptr_আকার = আকার(বাফার);

    apir_decoder_read(dec, বাফার_ptr_আকার, &বাফার, বাফার_ptr_আকার);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    যদি (বাফার) {
        বাহ্যিক std::unordered_set<ggml_backend_বাফার_t> backend_buffers;
        যদি (backend_buffers.find(বাফার) == backend_buffers.end()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (শূন্য *) বাফার);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_set_fatal(dec);
            ফেরত ফাঁকা;
        }
    }

    ফেরত বাফার;
}

/* enum ggml_status */

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_status(apir_encoder * enc, ধ্রুবক ggml_status * status) {
    apir_encoder_write(enc, আকার(*status), status, আকার(*status));
}

স্থির অন্তঃরেখ শূন্য apir_decode_ggml_status(apir_decoder * dec, ggml_status * status) {
    apir_decoder_read(dec, আকার(*status), status, আকার(*status));
}

/* virtgpu_shmem */

স্থির অন্তঃরেখ শূন্য apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

স্থির অন্তঃরেখ শূন্য apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

স্থির অন্তঃরেখ size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::vector<uint8_t> & cgraph_তথ্য) {
    apir_serialize_graph(cgraph, cgraph_তথ্য);

    ফেরত cgraph_তথ্য.আকার_2();
}

স্থির অন্তঃরেখ শূন্য apir_encode_cgraph_তথ্য(apir_encoder * enc, std::vector<uint8_t> & cgraph_তথ্য) {
    size_t cgraph_আকার = cgraph_তথ্য.আকার_2();

    apir_encode(enc, cgraph_আকার, cgraph_তথ্য.তথ্য(), cgraph_আকার);
}

স্থির অন্তঃরেখ ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_আকার) {
    GGML_UNUSED(cgraph_আকার);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ধ্রুবক uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ধ্রুবক apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    ফেরত apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_বাফার_handle(apir_encoder * enc, ধ্রুবক apir_বাফার_host_handle_t * handle) {
    apir_encoder_write(enc, আকার(*handle), &handle, আকার(*handle));
}

স্থির অন্তঃরেখ শূন্য apir_encode_ggml_tensor_inline(apir_encoder * enc, ধ্রুবক ggml_tensor * tensor) {
    size_t tensor_আকার = আকার(*tensor);

    যদি (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    যদি (tensor->src[0] && tensor->বাফার) {
        স্থির পূর্ণসংখ্যা first = 1;
        যদি (first) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_আকার, tensor, tensor_আকার);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    যদি (tensor->বাফার) {
        apir_বাফার_host_handle_t বাফার_handle = ggml_বাফার_to_apir_handle(tensor->বাফার);
        apir_encode_ggml_বাফার_handle(enc, &বাফার_handle);
    }

    যদি (tensor->view_src) {
        apir_encoder_write(enc, tensor_আকার, tensor->view_src, tensor_আকার);
    }

    জন্য (পূর্ণসংখ্যা i = 0; tensor->src[i]; i++) {
        ধ্রুবক ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_আকার, tensor_src, tensor_আকার);
    }
}

স্থির অন্তঃরেখ ধ্রুবক ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, আকার(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    যদি (tensor->বাফার) {
        tensor->বাফার = apir_decode_ggml_বাফার(dec);
    }

    যদি (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, আকার(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    জন্য (পূর্ণসংখ্যা i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, আকার(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    ফেরত tensor;
}
