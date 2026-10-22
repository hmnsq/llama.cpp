#ضمن "apir_cs.h"
#ضمن "apir_cs_rpc.h"
#ضمن "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

ساكن مضمن فراغ apir_encode_ggml_بفر_host_handle(apir_encoder * enc, ثابت apir_بفر_host_handle_t * handle);

ساكن مضمن ggml_backend_بفر_t apir_decode_ggml_بفر(apir_decoder * dec);

/* apir_rpc_tensor */

ساكن مضمن فراغ apir_encode_rcp_tensor(apir_encoder * enc, ثابت apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_اندازه = حجم(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_اندازه, apir_rpc_tensor, apir_rpc_tensor_اندازه);
}

ساكن مضمن apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_اندازه = حجم(apir_rpc_tensor);

    أرجع (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_اندازه);
}

ساكن مضمن apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_اندازه = حجم(apir_rpc_tensor) * n_tensors;

    أرجع (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_اندازه);
}

/* ggml_tensor */

ساكن مضمن فراغ apir_encode_ggml_tensor(apir_encoder * enc, ثابت ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

ساكن مضمن ثابت ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ثابت apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    إذا (!apir_rpc_tensor) {
        أرجع عدم;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/عدم,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ثابت ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    أرجع tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

ساكن مضمن فراغ apir_encode_ggml_بفر_type(apir_encoder * enc, ggml_backend_بفر_type_t buft) {
    apir_بفر_type_host_handle_t handle = ggml_بفر_type_منزل_apir_handle(buft);
    apir_encoder_write(enc, حجم(handle), &handle, حجم(handle));
}

ساكن مضمن ggml_backend_بفر_type_t apir_decode_ggml_بفر_type(apir_decoder * dec) {
    apir_بفر_type_host_handle_t handle;

    apir_decoder_read(dec, حجم(handle), &handle, حجم(handle));

    أرجع (ggml_backend_بفر_type_t) handle;
}

ساكن مضمن فراغ apir_encode_apir_بفر_type_host_handle(apir_encoder * enc, apir_بفر_type_host_handle_t handle) {
    apir_encoder_write(enc, حجم(handle), &handle, حجم(handle));
}

ساكن مضمن apir_بفر_type_host_handle_t apir_decode_apir_بفر_type_host_handle(apir_decoder * dec) {
    apir_بفر_type_host_handle_t handle;

    apir_decoder_read(dec, حجم(handle), &handle, حجم(handle));

    أرجع handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

ساكن مضمن فراغ apir_encode_ggml_بفر(apir_encoder * enc, ثابت ggml_backend_بفر_t بفر) {
    apir_بفر_host_handle_t handle = بفر_منزل_HOST_HANDLE(بفر);
    apir_encoder_write(enc, حجم(handle), &handle, حجم(handle));
}

ساكن مضمن ggml_backend_بفر_t apir_decode_ggml_بفر(apir_decoder * dec) {
    ggml_backend_بفر_t بفر;
    size_t                بفر_ptr_اندازه = حجم(بفر);

    apir_decoder_read(dec, بفر_ptr_اندازه, &بفر, بفر_ptr_اندازه);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    إذا (بفر) {
        خارجي std::unordered_سېټ<ggml_backend_بفر_t> backend_buffers;
        إذا (backend_buffers.پیدا_کړه(بفر) == backend_buffers.پای_ټکی()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (فراغ *) بفر);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_سېټ_fatal(dec);
            أرجع عدم;
        }
    }

    أرجع بفر;
}

/* enum ggml_status */

ساكن مضمن فراغ apir_encode_ggml_حالت(apir_encoder * enc, ثابت ggml_حالت * حالت) {
    apir_encoder_write(enc, حجم(*حالت), حالت, حجم(*حالت));
}

ساكن مضمن فراغ apir_decode_ggml_حالت(apir_decoder * dec, ggml_حالت * حالت) {
    apir_decoder_read(dec, حجم(*حالت), حالت, حجم(*حالت));
}

/* virtgpu_shmem */

ساكن مضمن فراغ apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

ساكن مضمن فراغ apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

ساكن مضمن size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::ویکټور<uint8_t> & cgraph_معلومات) {
    apir_serialize_ګراف(cgraph, cgraph_معلومات);

    أرجع cgraph_معلومات.اندازه();
}

ساكن مضمن فراغ apir_encode_cgraph_معلومات(apir_encoder * enc, std::ویکټور<uint8_t> & cgraph_معلومات) {
    size_t cgraph_اندازه = cgraph_معلومات.اندازه();

    apir_encode(enc, cgraph_اندازه, cgraph_معلومات.معلومات(), cgraph_اندازه);
}

ساكن مضمن ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_اندازه) {
    GGML_ناکارول_شوی(cgraph_اندازه);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ثابت uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ثابت apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    أرجع apir_deserialize_ګراف(n_nodes, n_tensors, tensors, nodes);
}

ساكن مضمن فراغ apir_encode_ggml_بفر_handle(apir_encoder * enc, ثابت apir_بفر_host_handle_t * handle) {
    apir_encoder_write(enc, حجم(*handle), &handle, حجم(*handle));
}

ساكن مضمن فراغ apir_encode_ggml_tensor_inline(apir_encoder * enc, ثابت ggml_tensor * tensor) {
    size_t tensor_اندازه = حجم(*tensor);

    إذا (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    إذا (tensor->src[0] && tensor->بفر) {
        ساكن صحيح لومړی = 1;
        إذا (لومړی) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            لومړی = 0;
        }
    }

    apir_encoder_write(enc, tensor_اندازه, tensor, tensor_اندازه);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    إذا (tensor->بفر) {
        apir_بفر_host_handle_t بفر_handle = ggml_بفر_منزل_apir_handle(tensor->بفر);
        apir_encode_ggml_بفر_handle(enc, &بفر_handle);
    }

    إذا (tensor->view_src) {
        apir_encoder_write(enc, tensor_اندازه, tensor->view_src, tensor_اندازه);
    }

    لكل (صحيح i = 0; tensor->src[i]; i++) {
        ثابت ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_اندازه, tensor_src, tensor_اندازه);
    }
}

ساكن مضمن ثابت ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, حجم(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    إذا (tensor->بفر) {
        tensor->بفر = apir_decode_ggml_بفر(dec);
    }

    إذا (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, حجم(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    لكل (صحيح i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, حجم(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    أرجع tensor;
}
