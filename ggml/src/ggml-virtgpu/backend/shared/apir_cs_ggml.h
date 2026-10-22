#شامل "apir_cs.h"
#شامل "apir_cs_rpc.h"
#شامل "ggml-impl.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

ساکن اندرونی خالی apir_encode_ggml_بفر_host_handle(apir_encoder * enc, ثابت apir_بفر_host_handle_t * handle);

ساکن اندرونی ggml_backend_بفر_t apir_decode_ggml_بفر(apir_decoder * dec);

/* apir_rpc_tensor */

ساکن اندرونی خالی apir_encode_rcp_tensor(apir_encoder * enc, ثابت apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_سائز = حجم(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_سائز, apir_rpc_tensor, apir_rpc_tensor_سائز);
}

ساکن اندرونی apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_سائز = حجم(apir_rpc_tensor);

    واپس (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_سائز);
}

ساکن اندرونی apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec, uint32_t n_tensors) {
    size_t apir_rpc_tensor_سائز = حجم(apir_rpc_tensor) * n_tensors;

    واپس (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_سائز);
}

/* ggml_tensor */

ساکن اندرونی خالی apir_encode_ggml_tensor(apir_encoder * enc, ثابت ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

ساکن اندرونی ثابت ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    ثابت apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);

    اگر (!apir_rpc_tensor) {
        واپس تہی;
    }

    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/تہی,
        /*.no_alloc   =*/true,
    };

    ggml_context * ctx = ggml_init(params);

    ثابت ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    واپس tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

ساکن اندرونی خالی apir_encode_ggml_بفر_type(apir_encoder * enc, ggml_backend_بفر_type_t buft) {
    apir_بفر_type_host_handle_t handle = ggml_بفر_type_مقصد_apir_handle(buft);
    apir_encoder_write(enc, حجم(handle), &handle, حجم(handle));
}

ساکن اندرونی ggml_backend_بفر_type_t apir_decode_ggml_بفر_type(apir_decoder * dec) {
    apir_بفر_type_host_handle_t handle;

    apir_decoder_read(dec, حجم(handle), &handle, حجم(handle));

    واپس (ggml_backend_بفر_type_t) handle;
}

ساکن اندرونی خالی apir_encode_apir_بفر_type_host_handle(apir_encoder * enc, apir_بفر_type_host_handle_t handle) {
    apir_encoder_write(enc, حجم(handle), &handle, حجم(handle));
}

ساکن اندرونی apir_بفر_type_host_handle_t apir_decode_apir_بفر_type_host_handle(apir_decoder * dec) {
    apir_بفر_type_host_handle_t handle;

    apir_decoder_read(dec, حجم(handle), &handle, حجم(handle));

    واپس handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

ساکن اندرونی خالی apir_encode_ggml_بفر(apir_encoder * enc, ثابت ggml_backend_بفر_t بفر) {
    apir_بفر_host_handle_t handle = بفر_مقصد_HOST_HANDLE(بفر);
    apir_encoder_write(enc, حجم(handle), &handle, حجم(handle));
}

ساکن اندرونی ggml_backend_بفر_t apir_decode_ggml_بفر(apir_decoder * dec) {
    ggml_backend_بفر_t بفر;
    size_t                بفر_ptr_سائز = حجم(بفر);

    apir_decoder_read(dec, بفر_ptr_سائز, &بفر, بفر_ptr_سائز);

    // SECURITY: Validate buffer handle against tracked buffers to prevent
    // guest VM from providing arbitrary host memory addresses
    اگر (بفر) {
        خارجی std::unordered_سیٹ<ggml_backend_بفر_t> backend_buffers;
        اگر (backend_buffers.ڈھونڈو(بفر) == backend_buffers.انتہا()) {
            GGML_LOG_WARN("ggml-virtgpu-backend: %s: Invalid buffer handle from guest: %p\n", __func__,
                          (خالی *) بفر);
            // Set fatal flag to prevent further processing with invalid handle
            apir_decoder_سیٹ_fatal(dec);
            واپس تہی;
        }
    }

    واپس بفر;
}

/* enum ggml_status */

ساکن اندرونی خالی apir_encode_ggml_حالت(apir_encoder * enc, ثابت ggml_حالت * حالت_2) {
    apir_encoder_write(enc, حجم(*حالت_2), حالت_2, حجم(*حالت_2));
}

ساکن اندرونی خالی apir_decode_ggml_حالت(apir_decoder * dec, ggml_حالت * حالت_2) {
    apir_decoder_read(dec, حجم(*حالت_2), حالت_2, حجم(*حالت_2));
}

/* virtgpu_shmem */

ساکن اندرونی خالی apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

ساکن اندرونی خالی apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

ساکن اندرونی size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::سمتیہ<uint8_t> & cgraph_ڈیٹا) {
    apir_serialize_گراف(cgraph, cgraph_ڈیٹا);

    واپس cgraph_ڈیٹا.سائز();
}

ساکن اندرونی خالی apir_encode_cgraph_ڈیٹا(apir_encoder * enc, std::سمتیہ<uint8_t> & cgraph_ڈیٹا) {
    size_t cgraph_سائز = cgraph_ڈیٹا.سائز();

    apir_encode(enc, cgraph_سائز, cgraph_ڈیٹا.ڈیٹا(), cgraph_سائز);
}

ساکن اندرونی ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_سائز) {
    GGML_غیر_مستعمل(cgraph_سائز);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    ثابت uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    ثابت apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    واپس apir_deserialize_گراف(n_nodes, n_tensors, tensors, nodes);
}

ساکن اندرونی خالی apir_encode_ggml_بفر_handle(apir_encoder * enc, ثابت apir_بفر_host_handle_t * handle) {
    apir_encoder_write(enc, حجم(*handle), &handle, حجم(*handle));
}

ساکن اندرونی خالی apir_encode_ggml_tensor_inline(apir_encoder * enc, ثابت ggml_tensor * tensor) {
    size_t tensor_سائز = حجم(*tensor);

    اگر (tensor->extra) {
        GGML_ABORT("%s: Cannot pass tensors with extra", __func__);
    }

    اگر (tensor->src[0] && tensor->بفر) {
        ساکن صحیح پہلا = 1;
        اگر (پہلا) {
            GGML_LOG_WARN("%s: Cannot pass tensors with src and buffer\n", __func__);
            پہلا = 0;
        }
    }

    apir_encoder_write(enc, tensor_سائز, tensor, tensor_سائز);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    اگر (tensor->بفر) {
        apir_بفر_host_handle_t بفر_handle = ggml_بفر_مقصد_apir_handle(tensor->بفر);
        apir_encode_ggml_بفر_handle(enc, &بفر_handle);
    }

    اگر (tensor->view_src) {
        apir_encoder_write(enc, tensor_سائز, tensor->view_src, tensor_سائز);
    }

    برائے (صحیح i = 0; tensor->src[i]; i++) {
        ثابت ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_سائز, tensor_src, tensor_سائز);
    }
}

ساکن اندرونی ثابت ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, حجم(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    اگر (tensor->بفر) {
        tensor->بفر = apir_decode_ggml_بفر(dec);
    }

    اگر (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, حجم(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    برائے (صحیح i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, حجم(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    واپس tensor;
}
