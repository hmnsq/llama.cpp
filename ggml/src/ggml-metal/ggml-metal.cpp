#شامل "ggml-metal.h"

#شامل "ggml-impl.h"
#شامل "ggml-backend-impl.h"

#شامل "ggml-metal-device.h"
#شامل "ggml-metal-fusion.h"
#شامل "ggml-metal-context.h"
#شامل "ggml-metal-ops.h"
#شامل "ggml-metal-tuning.h"

#شامل <mutex>
#شامل <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ایستا صحیح g_devices = 1;

// forward declaration
ایستا منطقی ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ایستا تهی ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

ایستا تهی * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    بازگشت ggml_metal_buffer_get_base(ctx);
}

ایستا تهی ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

ایستا تهی ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

ایستا تهی ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

ایستا منطقی ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    اگر (!ggml_backend_buffer_is_metal(src->buffer)) {
        بازگشت نادرست;
    }

    بازگشت ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

ایستا تهی ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

ایستا ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ پوچ,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ پوچ,
    /* .get_tensor_2d = */ پوچ,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ پوچ,
};

// private buffer

ایستا تهی ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

ایستا تهی * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    بازگشت ggml_metal_buffer_get_base(ctx);
}

ایستا تهی ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

ایستا تهی ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

ایستا تهی ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

ایستا منطقی ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    اگر (!ggml_backend_buffer_is_metal(src->buffer)) {
        بازگشت نادرست;
    }

    بازگشت ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

ایستا تهی ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

ایستا ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ پوچ,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ پوچ,
    /* .get_tensor_2d = */ پوچ,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ پوچ,
};

ایستا منطقی ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    بازگشت buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

ساختار ggml_backend_metal_buffer_type {
    صحیح device;
    std::string name;
};

ساختار ggml_backend_metal_buffer_type_deleter {
    تهی operator()(ggml_backend_metal_buffer_type * ctx) ثابت {
        delete ctx;
    }
};

تعریفنوع std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
ایستا ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, منطقی shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    اگر (res == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        بازگشت پوچ;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    بازگشت ggml_backend_buffer_init(buft, buf_i, res, size);
}

ایستا size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    انتخاب (tensor->op) {
        حالت GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } بشکن;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } بشکن;
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_ARGSORT:
            {
                res *= 2;
            } بشکن;
        حالت GGML_OP_بالاترین_K:
            {
                res = 2*اندازه(int32_t)*ggml_nelements(tensor->src[0]);
            } بشکن;
        حالت GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } بشکن;
        پیشفرض:
            بشکن;
    }

    بازگشت res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

ایستا ثابت نویسه * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    بازگشت ctx->name.c_str();
}

ایستا ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    بازگشت ggml_backend_metal_buffer_type_alloc_buffer(buft, size, درست);
}

ایستا size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    بازگشت 32;

    GGML_UNUSED(buft);
}

ایستا size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    بازگشت ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ایستا size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    بازگشت ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ایستا منطقی ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    بازگشت نادرست;

    GGML_UNUSED(buft);
}

ایستا ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(صحیح device) {
    ایستا std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ایستا std::vector<ggml_backend_buffer_type> bufts;
    ایستا std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ایستا منطقی initialized = نادرست;
    اگر (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        برای (صحیح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                جدید ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ پوچ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ پوچ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = درست;
    }

    بازگشت &bufts[device];
}

// default (private) buffer type

ایستا ثابت نویسه * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    بازگشت ctx->name.c_str();
}

ایستا ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    بازگشت ggml_backend_metal_buffer_type_alloc_buffer(buft, size, نادرست);
}

ایستا size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    بازگشت 32;

    GGML_UNUSED(buft);
}

ایستا size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    بازگشت ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ایستا size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    بازگشت ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ایستا منطقی ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    بازگشت نادرست;

    GGML_UNUSED(buft);
}

ایستا ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(صحیح device) {
    ایستا std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ایستا std::vector<ggml_backend_buffer_type> bufts;
    ایستا std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ایستا منطقی initialized = نادرست;
    اگر (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        برای (صحیح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = جدید ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ پوچ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ پوچ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = درست;
    }

    بازگشت &bufts[device];
}

// mapped buffer type

ایستا ثابت نویسه * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    بازگشت ctx->name.c_str();
}

ایستا ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    بازگشت ggml_backend_metal_buffer_type_alloc_buffer(buft, size, درست);
}

ایستا size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    بازگشت 32;

    GGML_UNUSED(buft);
}

ایستا size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    بازگشت ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ایستا size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    بازگشت ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ایستا منطقی ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    بازگشت نادرست;

    GGML_UNUSED(buft);
}

ایستا ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(صحیح device) {
    ایستا std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ایستا std::vector<ggml_backend_buffer_type> bufts;
    ایستا std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ایستا منطقی initialized = نادرست;
    اگر (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        برای (صحیح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = جدید ggml_backend_metal_buffer_type{
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
                    /* .alloc_buffer_n      = */ پوچ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ پوچ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = درست;
    }

    بازگشت &bufts[device];
}

// backend

ایستا ثابت نویسه * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    بازگشت ggml_metal_get_name(ctx);
}

ایستا تهی ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    آزاد(backend);
}

ایستا تهی ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ایستا تهی ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

ایستا تهی ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

ایستا منطقی ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        بازگشت نادرست;
    }

    اگر (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        بازگشت نادرست;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    بازگشت ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ایستا شمارش ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    بازگشت ggml_metal_graph_compute(ctx, cgraph);
}

ایستا تهی ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

ایستا تهی ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ایستا تهی ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

ایستا تهی ggml_backend_metal_set_n_cb(ggml_backend_t backend, صحیح n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

ایستا ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ پوچ,
    /* .get_tensor_2d_async     = */ پوچ,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ پوچ,
    /* .graph_plan_free         = */ پوچ,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ پوچ,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

ایستا ggml_guid_t ggml_backend_metal_guid(تهی) {
    ایستا ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    بازگشت &guid;
}

ggml_backend_t ggml_backend_metal_init(تهی) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    اگر (ctx == پوچ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        بازگشت پوچ;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(اندازه(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    بازگشت backend;
}

منطقی ggml_backend_is_metal(ggml_backend_t backend) {
    بازگشت backend != پوچ && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

تهی ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, تهی * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

منطقی ggml_backend_metal_supports_family(ggml_backend_t backend, صحیح family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    بازگشت ggml_metal_supports_family(ctx, family);
}

تهی ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

ایستا ثابت نویسه * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    بازگشت props_dev->name;
}

ایستا ثابت نویسه * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    بازگشت ggml_metal_device_get_props(ctx_dev)->desc;
}

ایستا تهی ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * آزاد, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, آزاد, total);
}

ایستا شمارش ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    بازگشت GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ درست,
        /* .host_buffer          = */ نادرست,
        /* .buffer_from_host_ptr = */ درست,
        /* .events               = */ درست,
        /* .mmap_support         = */ درست,
    };
}

ایستا ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ثابت نویسه * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    اگر (ctx == پوچ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        بازگشت پوچ;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(اندازه(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    بازگشت backend;

    GGML_UNUSED(params);
}

ایستا ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    بازگشت props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

ایستا ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, تهی * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    بازگشت ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

ایستا منطقی ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    بازگشت ggml_metal_device_supports_op(ctx_dev, op);
}

ایستا منطقی ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    بازگشت
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

ایستا int64_t get_op_batch_size(ثابت ggml_tensor * op) {
    انتخاب (op->op) {
        حالت GGML_OP_MUL_MAT:
            بازگشت op->ne[1];
        حالت GGML_OP_MUL_MAT_ID:
            بازگشت op->ne[2];
        پیشفرض:
            بازگشت ggml_nrows(op);
    }
}

ایستا منطقی ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    بازگشت (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

ایستا ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = جدید ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    بازگشت ev;
}

ایستا تهی ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

ایستا تهی ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ایستا ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ پوچ,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ساختار ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

تعریفنوع ساختار ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ایستا ggml_backend_metal_reg_t ggml_backend_metal_reg_init(تهی) {
    ggml_backend_metal_reg_t ctx = جدید ساختار ggml_backend_metal_reg;

    بازگشت ctx;
}

ایستا تهی ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

ساختار ggml_backend_metal_reg_deleter {
    تهی operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

تعریفنوع std::unique_ptr<ساختار ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ایستا ثابت نویسه * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    بازگشت GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

ایستا size_t ggml_backend_metal_reg_device_شمار(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    بازگشت ctx->devices.size();
}

ایستا ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    بازگشت ctx->devices[index];
}

ایستا ggml_backend_feature g_ggml_backend_metal_features[] = {
#اگر defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { پوچ, پوچ },
};

ایستا ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    بازگشت g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ایستا تهی ggml_backend_metal_tuning_set_fa_vec_override(صحیح Q, صحیح NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

ایستا تهی ggml_backend_metal_tuning_clear_fa_vec_override(تهی) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ایستا صحیح ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    بازگشت ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

ایستا صحیح ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    بازگشت ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

ایستا صحیح ggml_backend_metal_tuning_fa_vec_baseline_ne(صحیح dk, صحیح dv) {
    بازگشت ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ایستا ثابت نویسه * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    بازگشت ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
تعریفنوع تهی * ggml_backend_fusion_t;

ایستا ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    بازگشت ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ایستا تهی ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ساختار ggml_metal_fusion_info *) finfo);
}

ایستا تهی ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ساختار ggml_metal_fusion_info *) finfo);
}

ایستا صحیح ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ثابت نویسه ** labels, uint64_t * counts, صحیح n) {
    بازگشت ggml_metal_fusion_info_stats_get((ساختار ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ایستا تهی ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, منطقی enabled) {
    ggml_metal_fusion_info_set_enabled((ساختار ggml_metal_fusion_info *) finfo, enabled);
}

ایستا تهی * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    اگر (strcmp(name, "ggml_backend_get_features") == 0) {
        بازگشت (تهی *)ggml_backend_metal_get_features;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        بازگشت (تهی *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        بازگشت (تهی *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        بازگشت (تهی *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        بازگشت (تهی *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        بازگشت (تهی *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        بازگشت (تهی *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    اگر (strcmp(name, "ggml_backend_fusion_get") == 0) {
        بازگشت (تهی *)ggml_backend_metal_fusion_get;
    }
    اگر (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        بازگشت (تهی *)ggml_backend_metal_fusion_stats_init;
    }
    اگر (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        بازگشت (تهی *)ggml_backend_metal_fusion_stats_reset;
    }
    اگر (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        بازگشت (تهی *)ggml_backend_metal_fusion_stats_get;
    }
    اگر (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        بازگشت (تهی *)ggml_backend_metal_fusion_set_enabled;
    }

    بازگشت پوچ;

    GGML_UNUSED(reg);
}

ایستا ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_شمار,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

ایستا ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, صحیح device) {
    بازگشت جدید ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ایستا تهی ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

ساختار ggml_backend_device_deleter {
    تهی operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

تعریفنوع std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(تهی) {
    ایستا ggml_backend_reg reg;
    ایستا منطقی initialized = نادرست;

    {
        ایستا std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ثابت نویسه * env = getenv("GGML_METAL_DEVICES");
        اگر (env) {
            g_devices = atoi(env);
        }

        ایستا std::vector<ggml_backend_device_ptr> devs;

        اگر (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", درست);

            ایستا ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            برای (صحیح i = 0; i < g_devices; ++i) {
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

        initialized = درست;
    }

    بازگشت &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
