#அந்தர்பாவ "ggml-metal.h"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-backend-impl.h"

#அந்தர்பாவ "ggml-metal-device.h"
#அந்தர்பாவ "ggml-metal-fusion.h"
#அந்தர்பாவ "ggml-metal-context.h"
#அந்தர்பாவ "ggml-metal-ops.h"
#அந்தர்பாவ "ggml-metal-tuning.h"

#அந்தர்பாவ <mutex>
#அந்தர்பாவ <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
நிலையான முழு g_devices = 1;

// forward declaration
நிலையான தருக்க ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

நிலையான வெறுமை ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

நிலையான வெறுமை * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    திரும்பு ggml_metal_buffer_get_base(ctx);
}

நிலையான வெறுமை ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

நிலையான வெறுமை ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

நிலையான வெறுமை ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

நிலையான தருக்க ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    எனில் (!ggml_backend_buffer_is_metal(src->buffer)) {
        திரும்பு பொய்;
    }

    திரும்பு ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

நிலையான வெறுமை ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

நிலையான ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ வெற்று,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ வெற்று,
    /* .get_tensor_2d = */ வெற்று,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ வெற்று,
};

// private buffer

நிலையான வெறுமை ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

நிலையான வெறுமை * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    திரும்பு ggml_metal_buffer_get_base(ctx);
}

நிலையான வெறுமை ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

நிலையான வெறுமை ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

நிலையான வெறுமை ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

நிலையான தருக்க ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    எனில் (!ggml_backend_buffer_is_metal(src->buffer)) {
        திரும்பு பொய்;
    }

    திரும்பு ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

நிலையான வெறுமை ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

நிலையான ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ வெற்று,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ வெற்று,
    /* .get_tensor_2d = */ வெற்று,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ வெற்று,
};

நிலையான தருக்க ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    திரும்பு buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

கட்டமைப்பு ggml_backend_metal_buffer_type {
    முழு device;
    std::string name;
};

கட்டமைப்பு ggml_backend_metal_buffer_type_deleter {
    வெறுமை operator()(ggml_backend_metal_buffer_type * ctx) மாறிலி {
        delete ctx;
    }
};

வகைவரை std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
நிலையான ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, தருக்க shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    எனில் (res == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        திரும்பு வெற்று;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    திரும்பு ggml_backend_buffer_init(buft, buf_i, res, size);
}

நிலையான size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    தேர்வு (tensor->op) {
        நிலை GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } நிறுத்து;
        நிலை GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } நிறுத்து;
        நிலை GGML_OP_CUMSUM:
        நிலை GGML_OP_ARGSORT:
            {
                res *= 2;
            } நிறுத்து;
        நிலை GGML_OP_உச்சி_K:
            {
                res = 2*அளவு(int32_t)*ggml_nelements(tensor->src[0]);
            } நிறுத்து;
        நிலை GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } நிறுத்து;
        இயல்பு:
            நிறுத்து;
    }

    திரும்பு res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

நிலையான மாறிலி எழுத்து * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    திரும்பு ctx->name.c_str();
}

நிலையான ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    திரும்பு ggml_backend_metal_buffer_type_alloc_buffer(buft, size, உண்மை);
}

நிலையான size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு 32;

    GGML_UNUSED(buft);
}

நிலையான size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    திரும்பு ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

நிலையான size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி ggml_tensor * tensor) {
    திரும்பு ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

நிலையான தருக்க ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    திரும்பு பொய்;

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(முழு device) {
    நிலையான std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    நிலையான std::vector<ggml_backend_buffer_type> bufts;
    நிலையான std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    நிலையான தருக்க initialized = பொய்;
    எனில் (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        க்கு (முழு i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                புதிய ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ வெற்று,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ வெற்று,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = உண்மை;
    }

    திரும்பு &bufts[device];
}

// default (private) buffer type

நிலையான மாறிலி எழுத்து * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    திரும்பு ctx->name.c_str();
}

நிலையான ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    திரும்பு ggml_backend_metal_buffer_type_alloc_buffer(buft, size, பொய்);
}

நிலையான size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு 32;

    GGML_UNUSED(buft);
}

நிலையான size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    திரும்பு ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

நிலையான size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி ggml_tensor * tensor) {
    திரும்பு ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

நிலையான தருக்க ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    திரும்பு பொய்;

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(முழு device) {
    நிலையான std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    நிலையான std::vector<ggml_backend_buffer_type> bufts;
    நிலையான std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    நிலையான தருக்க initialized = பொய்;
    எனில் (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        க்கு (முழு i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = புதிய ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ வெற்று,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ வெற்று,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = உண்மை;
    }

    திரும்பு &bufts[device];
}

// mapped buffer type

நிலையான மாறிலி எழுத்து * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    திரும்பு ctx->name.c_str();
}

நிலையான ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    திரும்பு ggml_backend_metal_buffer_type_alloc_buffer(buft, size, உண்மை);
}

நிலையான size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு 32;

    GGML_UNUSED(buft);
}

நிலையான size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    திரும்பு ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

நிலையான size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி ggml_tensor * tensor) {
    திரும்பு ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

நிலையான தருக்க ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    திரும்பு பொய்;

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(முழு device) {
    நிலையான std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    நிலையான std::vector<ggml_backend_buffer_type> bufts;
    நிலையான std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    நிலையான தருக்க initialized = பொய்;
    எனில் (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        க்கு (முழு i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = புதிய ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .alloc_buffer_n      = */ வெற்று,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ வெற்று,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = உண்மை;
    }

    திரும்பு &bufts[device];
}

// backend

நிலையான மாறிலி எழுத்து * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    திரும்பு ggml_metal_get_name(ctx);
}

நிலையான வெறுமை ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    விடுவி(backend);
}

நிலையான வெறுமை ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

நிலையான வெறுமை ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

நிலையான வெறுமை ggml_backend_metal_get_tensor_async(ggml_backend_t backend, மாறிலி ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

நிலையான தருக்க ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    எனில் (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        திரும்பு பொய்;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    திரும்பு ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    திரும்பு ggml_metal_graph_compute(ctx, cgraph);
}

நிலையான வெறுமை ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

நிலையான வெறுமை ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

நிலையான வெறுமை ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

நிலையான வெறுமை ggml_backend_metal_set_n_cb(ggml_backend_t backend, முழு n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

நிலையான ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ வெற்று,
    /* .get_tensor_2d_async     = */ வெற்று,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ வெற்று,
    /* .graph_plan_free         = */ வெற்று,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ வெற்று,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

நிலையான ggml_guid_t ggml_backend_metal_guid(வெறுமை) {
    நிலையான ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    திரும்பு &guid;
}

ggml_backend_t ggml_backend_metal_init(வெறுமை) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    எனில் (ctx == வெற்று) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        திரும்பு வெற்று;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(அளவு(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    திரும்பு backend;
}

தருக்க ggml_backend_is_metal(ggml_backend_t backend) {
    திரும்பு backend != வெற்று && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

வெறுமை ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, வெறுமை * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

தருக்க ggml_backend_metal_supports_family(ggml_backend_t backend, முழு family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    திரும்பு ggml_metal_supports_family(ctx, family);
}

வெறுமை ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

நிலையான மாறிலி எழுத்து * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    மாறிலி ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    திரும்பு props_dev->name;
}

நிலையான மாறிலி எழுத்து * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    திரும்பு ggml_metal_device_get_props(ctx_dev)->desc;
}

நிலையான வெறுமை ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, விடுவி, total);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    திரும்பு GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ உண்மை,
        /* .host_buffer          = */ பொய்,
        /* .buffer_from_host_ptr = */ உண்மை,
        /* .events               = */ உண்மை,
        /* .mmap_support         = */ உண்மை,
    };
}

நிலையான ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    எனில் (ctx == வெற்று) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        திரும்பு வெற்று;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(அளவு(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    திரும்பு backend;

    GGML_UNUSED(params);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    மாறிலி ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    திரும்பு props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

நிலையான ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, வெறுமை * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    மாறிலி ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    திரும்பு ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

நிலையான தருக்க ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, மாறிலி ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    திரும்பு ggml_metal_device_supports_op(ctx_dev, op);
}

நிலையான தருக்க ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    திரும்பு
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

நிலையான int64_t get_op_batch_size(மாறிலி ggml_tensor * op) {
    தேர்வு (op->op) {
        நிலை GGML_OP_MUL_MAT:
            திரும்பு op->ne[1];
        நிலை GGML_OP_MUL_MAT_ID:
            திரும்பு op->ne[2];
        இயல்பு:
            திரும்பு ggml_nrows(op);
    }
}

நிலையான தருக்க ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, மாறிலி ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    திரும்பு (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

நிலையான ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = புதிய ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    திரும்பு ev;
}

நிலையான வெறுமை ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

நிலையான வெறுமை ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

நிலையான ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ வெற்று,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

கட்டமைப்பு ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

வகைவரை கட்டமைப்பு ggml_backend_metal_reg * ggml_backend_metal_reg_t;

நிலையான ggml_backend_metal_reg_t ggml_backend_metal_reg_init(வெறுமை) {
    ggml_backend_metal_reg_t ctx = புதிய கட்டமைப்பு ggml_backend_metal_reg;

    திரும்பு ctx;
}

நிலையான வெறுமை ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

கட்டமைப்பு ggml_backend_metal_reg_deleter {
    வெறுமை operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

வகைவரை std::unique_ptr<கட்டமைப்பு ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

நிலையான மாறிலி எழுத்து * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    திரும்பு GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

நிலையான size_t ggml_backend_metal_reg_device_மொத்தம்(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    திரும்பு ctx->devices.size();
}

நிலையான ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    திரும்பு ctx->devices[index];
}

நிலையான ggml_backend_feature g_ggml_backend_metal_features[] = {
#எனில் defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { வெற்று, வெற்று },
};

நிலையான ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    திரும்பு g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
நிலையான வெறுமை ggml_backend_metal_tuning_set_fa_vec_override(முழு Q, முழு NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

நிலையான வெறுமை ggml_backend_metal_tuning_clear_fa_vec_override(வெறுமை) {
    ggml_metal_tuning::fa_vec_clear_override();
}

நிலையான முழு ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    திரும்பு ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

நிலையான முழு ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    திரும்பு ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

நிலையான முழு ggml_backend_metal_tuning_fa_vec_baseline_ne(முழு dk, முழு dv) {
    திரும்பு ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

நிலையான மாறிலி எழுத்து * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    திரும்பு ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
வகைவரை வெறுமை * ggml_backend_fusion_t;

நிலையான ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    திரும்பு ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

நிலையான வெறுமை ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((கட்டமைப்பு ggml_metal_fusion_info *) finfo);
}

நிலையான வெறுமை ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((கட்டமைப்பு ggml_metal_fusion_info *) finfo);
}

நிலையான முழு ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, மாறிலி எழுத்து ** labels, uint64_t * counts, முழு n) {
    திரும்பு ggml_metal_fusion_info_stats_get((கட்டமைப்பு ggml_metal_fusion_info *) finfo, labels, counts, n);
}

நிலையான வெறுமை ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, தருக்க enabled) {
    ggml_metal_fusion_info_set_enabled((கட்டமைப்பு ggml_metal_fusion_info *) finfo, enabled);
}

நிலையான வெறுமை * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    எனில் (strcmp(name, "ggml_backend_get_features") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_get_features;
    }
    எனில் (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    எனில் (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    எனில் (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    எனில் (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    எனில் (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    எனில் (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    எனில் (strcmp(name, "ggml_backend_fusion_get") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_fusion_get;
    }
    எனில் (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_fusion_stats_init;
    }
    எனில் (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_fusion_stats_reset;
    }
    எனில் (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_fusion_stats_get;
    }
    எனில் (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_metal_fusion_set_enabled;
    }

    திரும்பு வெற்று;

    GGML_UNUSED(reg);
}

நிலையான ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

நிலையான ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, முழு device) {
    திரும்பு புதிய ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

நிலையான வெறுமை ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

கட்டமைப்பு ggml_backend_device_deleter {
    வெறுமை operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

வகைவரை std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(வெறுமை) {
    நிலையான ggml_backend_reg reg;
    நிலையான தருக்க initialized = பொய்;

    {
        நிலையான std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        மாறிலி எழுத்து * env = getenv("GGML_METAL_DEVICES");
        எனில் (env) {
            g_devices = atoi(env);
        }

        நிலையான std::vector<ggml_backend_device_ptr> devs;

        எனில் (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", உண்மை);

            நிலையான ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            க்கு (முழு i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = உண்மை;
    }

    திரும்பு &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
