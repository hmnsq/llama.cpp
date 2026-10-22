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

#تعریف_کرو GGML_METAL_NAME "MTL"
#تعریف_کرو GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ساکن صحیح g_devices = 1;

// forward declaration
ساکن منطقی ggml_backend_بفر_is_metal(ggml_backend_بفر_t بفر);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ساکن خالی ggml_backend_metal_بفر_shared_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_free(ctx);
}

ساکن خالی * ggml_backend_metal_بفر_shared_get_بنیاد(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    واپس ggml_metal_بفر_get_بنیاد(ctx);
}

ساکن خالی ggml_backend_metal_بفر_shared_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t انحراف, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_memset_tensor(ctx, tensor, value, انحراف, size);
}

ساکن خالی ggml_backend_metal_بفر_shared_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_سیٹ_tensor(ctx, tensor, ڈیٹا, انحراف, size);
}

ساکن خالی ggml_backend_metal_بفر_shared_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_get_tensor(ctx, tensor, ڈیٹا, انحراف, size);
}

ساکن منطقی ggml_backend_metal_بفر_shared_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    اگر (!ggml_backend_بفر_is_metal(src->بفر)) {
        واپس جھوٹ;
    }

    واپس ggml_metal_بفر_cpy_tensor(ctx, src, dst);
}

ساکن خالی ggml_backend_metal_بفر_shared_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_clear(ctx, value);
}

ساکن ggml_backend_بفر_i ggml_backend_metal_بفر_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_بفر_shared_free_بفر,
    /* .get_base      = */ ggml_backend_metal_بفر_shared_get_بنیاد,
    /* .init_tensor   = */ تہی,
    /* .memset_tensor = */ ggml_backend_metal_بفر_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_بفر_shared_سیٹ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_بفر_shared_get_tensor,
    /* .set_tensor_2d = */ تہی,
    /* .get_tensor_2d = */ تہی,
    /* .cpy_tensor    = */ ggml_backend_metal_بفر_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_بفر_shared_clear,
    /* .reset         = */ تہی,
};

// private buffer

ساکن خالی ggml_backend_metal_بفر_نجی_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_free(ctx);
}

ساکن خالی * ggml_backend_metal_بفر_نجی_get_بنیاد(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    واپس ggml_metal_بفر_get_بنیاد(ctx);
}

ساکن خالی ggml_backend_metal_بفر_نجی_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t انحراف, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_memset_tensor(ctx, tensor, value, انحراف, size);
}

ساکن خالی ggml_backend_metal_بفر_نجی_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_سیٹ_tensor(ctx, tensor, ڈیٹا, انحراف, size);
}

ساکن خالی ggml_backend_metal_بفر_نجی_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_get_tensor(ctx, tensor, ڈیٹا, انحراف, size);
}

ساکن منطقی ggml_backend_metal_بفر_نجی_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    اگر (!ggml_backend_بفر_is_metal(src->بفر)) {
        واپس جھوٹ;
    }

    واپس ggml_metal_بفر_cpy_tensor(ctx, src, dst);
}

ساکن خالی ggml_backend_metal_بفر_نجی_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_clear(ctx, value);
}

ساکن ggml_backend_بفر_i ggml_backend_metal_بفر_نجی_i = {
    /* .free_buffer   = */ ggml_backend_metal_بفر_نجی_free_بفر,
    /* .get_base      = */ ggml_backend_metal_بفر_نجی_get_بنیاد,
    /* .init_tensor   = */ تہی,
    /* .memset_tensor = */ ggml_backend_metal_بفر_نجی_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_بفر_نجی_سیٹ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_بفر_نجی_get_tensor,
    /* .set_tensor_2d = */ تہی,
    /* .get_tensor_2d = */ تہی,
    /* .cpy_tensor    = */ ggml_backend_metal_بفر_نجی_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_بفر_نجی_clear,
    /* .reset         = */ تہی,
};

ساکن منطقی ggml_backend_بفر_is_metal(ggml_backend_بفر_t بفر) {
    واپس بفر->iface.free_بفر == ggml_backend_metal_بفر_shared_free_بفر ||
           بفر->iface.free_بفر == ggml_backend_metal_بفر_نجی_free_بفر;
}

//
// buffer types
//

ساخت ggml_backend_metal_بفر_type {
    صحیح device;
    std::string name;
};

ساخت ggml_backend_metal_بفر_type_deleter {
    خالی عامل()(ggml_backend_metal_بفر_type * ctx) ثابت {
        حذف ctx;
    }
};

قسمتعریف std::unique_ptr<ggml_backend_metal_بفر_type, ggml_backend_metal_بفر_type_deleter> ggml_backend_metal_بفر_type_ptr;

// common method for allocating shread or private Metal buffers
ساکن ggml_backend_بفر_t ggml_backend_metal_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size, منطقی shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_بفر_t res = ggml_metal_بفر_init(ctx_dev, size, shared);

    اگر (res == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        واپس تہی;
    }

    ggml_backend_بفر_i buf_i = ggml_metal_بفر_is_shared(res)
        ? ggml_backend_metal_بفر_shared_i
        : ggml_backend_metal_بفر_نجی_i;

    واپس ggml_backend_بفر_init(buft, buf_i, res, size);
}

ساکن size_t ggml_backend_metal_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    انتخاب (tensor->op) {
        حالت GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } توڑو;
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_ARGSORT:
            {
                res *= 2;
            } توڑو;
        حالت GGML_OP_چوٹی_K:
            {
                res = 2*حجم(int32_t)*ggml_nelements(tensor->src[0]);
            } توڑو;
        حالت GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } توڑو;
        طےشدہ:
            توڑو;
    }

    واپس res;

    GGML_غیر_مستعمل(buft);
}

// default (shared) buffer type

ساکن ثابت حرف * ggml_backend_metal_بفر_type_shared_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_metal_بفر_type * ctx = (ggml_backend_metal_بفر_type *)buft->context;

    واپس ctx->name.c_str();
}

ساکن ggml_backend_بفر_t ggml_backend_metal_بفر_type_shared_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    واپس ggml_backend_metal_بفر_type_alloc_بفر(buft, size, سچ);
}

ساکن size_t ggml_backend_metal_بفر_type_shared_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس 32;

    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_metal_بفر_type_shared_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    واپس ggml_metal_device_get_props(ctx_dev)->max_بفر_سائز;
}

ساکن size_t ggml_backend_metal_بفر_type_shared_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    واپس ggml_backend_metal_بفر_type_get_alloc_سائز(buft, tensor);
}

ساکن منطقی ggml_backend_metal_بفر_type_shared_is_host(ggml_backend_بفر_type_t buft) {
    واپس جھوٹ;

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_type_t ggml_backend_metal_بفر_type_shared(صحیح device) {
    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);

    ساکن std::vector<ggml_backend_بفر_type> bufts;
    ساکن std::vector<ggml_backend_metal_بفر_type_ptr> ctxs;

    ساکن منطقی initialized = جھوٹ;
    اگر (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        برائے (صحیح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_بفر_type * raw_ctx =
                نیا ggml_backend_metal_بفر_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_بفر_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_بفر_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_بفر_type_shared_alloc_بفر,
                    /* .alloc_buffer_n      = */ تہی,
                    /* .get_alignment       = */ ggml_backend_metal_بفر_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_بفر_type_shared_get_max_سائز,
                    /* .get_alloc_size      = */ ggml_backend_metal_بفر_type_shared_get_alloc_سائز,
                    /* .get_alloc_size_n    = */ تہی,
                    /* .is_host             = */ ggml_backend_metal_بفر_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = سچ;
    }

    واپس &bufts[device];
}

// default (private) buffer type

ساکن ثابت حرف * ggml_backend_metal_بفر_type_نجی_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_metal_بفر_type * ctx = (ggml_backend_metal_بفر_type *)buft->context;

    واپس ctx->name.c_str();
}

ساکن ggml_backend_بفر_t ggml_backend_metal_بفر_type_نجی_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    واپس ggml_backend_metal_بفر_type_alloc_بفر(buft, size, جھوٹ);
}

ساکن size_t ggml_backend_metal_بفر_type_نجی_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس 32;

    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_metal_بفر_type_نجی_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    واپس ggml_metal_device_get_props(ctx_dev)->max_بفر_سائز;
}

ساکن size_t ggml_backend_metal_بفر_type_نجی_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    واپس ggml_backend_metal_بفر_type_get_alloc_سائز(buft, tensor);
}

ساکن منطقی ggml_backend_metal_بفر_type_نجی_is_host(ggml_backend_بفر_type_t buft) {
    واپس جھوٹ;

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_type_t ggml_backend_metal_بفر_type_نجی(صحیح device) {
    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);

    ساکن std::vector<ggml_backend_بفر_type> bufts;
    ساکن std::vector<ggml_backend_metal_بفر_type_ptr> ctxs;

    ساکن منطقی initialized = جھوٹ;
    اگر (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        برائے (صحیح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_بفر_type * raw_ctx = نیا ggml_backend_metal_بفر_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_بفر_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_بفر_type_نجی_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_بفر_type_نجی_alloc_بفر,
                    /* .alloc_buffer_n      = */ تہی,
                    /* .get_alignment       = */ ggml_backend_metal_بفر_type_نجی_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_بفر_type_نجی_get_max_سائز,
                    /* .get_alloc_size      = */ ggml_backend_metal_بفر_type_نجی_get_alloc_سائز,
                    /* .get_alloc_size_n    = */ تہی,
                    /* .is_host             = */ ggml_backend_metal_بفر_type_نجی_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = سچ;
    }

    واپس &bufts[device];
}

// mapped buffer type

ساکن ثابت حرف * ggml_backend_metal_بفر_type_mapped_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_metal_بفر_type * ctx = (ggml_backend_metal_بفر_type *)buft->context;

    واپس ctx->name.c_str();
}

ساکن ggml_backend_بفر_t ggml_backend_metal_بفر_type_mapped_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    واپس ggml_backend_metal_بفر_type_alloc_بفر(buft, size, سچ);
}

ساکن size_t ggml_backend_metal_بفر_type_mapped_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس 32;

    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_metal_بفر_type_mapped_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    واپس ggml_metal_device_get_props(ctx_dev)->max_بفر_سائز;
}

ساکن size_t ggml_backend_metal_بفر_type_mapped_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    واپس ggml_backend_metal_بفر_type_get_alloc_سائز(buft, tensor);
}

ساکن منطقی ggml_backend_metal_بفر_type_mapped_is_host(ggml_backend_بفر_type_t buft) {
    واپس جھوٹ;

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_type_t ggml_backend_metal_بفر_type_mapped(صحیح device) {
    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);

    ساکن std::vector<ggml_backend_بفر_type> bufts;
    ساکن std::vector<ggml_backend_metal_بفر_type_ptr> ctxs;

    ساکن منطقی initialized = جھوٹ;
    اگر (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        برائے (صحیح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_بفر_type * raw_ctx = نیا ggml_backend_metal_بفر_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_بفر_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_بفر_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_بفر_type_mapped_alloc_بفر,
                    /* .alloc_buffer_n      = */ تہی,
                    /* .get_alignment       = */ ggml_backend_metal_بفر_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_بفر_type_mapped_get_max_سائز,
                    /* .get_alloc_size      = */ ggml_backend_metal_بفر_type_mapped_get_alloc_سائز,
                    /* .get_alloc_size_n    = */ تہی,
                    /* .is_host             = */ ggml_backend_metal_بفر_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = سچ;
    }

    واپس &bufts[device];
}

// backend

ساکن ثابت حرف * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    واپس ggml_metal_get_name(ctx);
}

ساکن خالی ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    آزاد(backend);
}

ساکن خالی ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ساکن خالی ggml_backend_metal_سیٹ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_سیٹ_tensor_async(ctx, tensor, ڈیٹا, انحراف, size);
}

ساکن خالی ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, ڈیٹا, انحراف, size);
}

ساکن منطقی ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        واپس جھوٹ;
    }

    اگر (!ggml_backend_بفر_is_metal(src->بفر) || !ggml_backend_بفر_is_metal(dst->بفر)) {
        واپس جھوٹ;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    واپس ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ساکن شمار ggml_حالت ggml_backend_metal_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    واپس ggml_metal_گراف_compute(ctx, cgraph);
}

ساکن خالی ggml_backend_metal_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_ریکارڈ(ctx, ev);
}

ساکن خالی ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ساکن خالی ggml_backend_metal_گراف_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_گراف_optimize_params * params) {
    GGML_ASSERT(params && params->جوڑو_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_جوڑو_alloc_deps(params->user_ڈیٹا, params->جوڑو_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_گراف_optimize(ctx, cgraph);
}

ساکن خالی ggml_backend_metal_سیٹ_n_cb(ggml_backend_t backend, صحیح n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_سیٹ_n_cb(ctx, n_cb);
}

ساکن ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_metal_گراف_compute,
    /* .event_record            = */ ggml_backend_metal_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_گراف_optimize,
};

ساکن ggml_guid_t ggml_backend_metal_guid(خالی) {
    ساکن ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    واپس &guid;
}

ggml_backend_t ggml_backend_metal_init(خالی) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    اگر (ctx == تہی) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        واپس تہی;
    }

    ggml_backend_t backend = (ggml_backend_t) حافظہ_دو(حجم(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_سیٹ_n_cb(backend, 1);

    واپس backend;
}

منطقی ggml_backend_is_metal(ggml_backend_t backend) {
    واپس backend != تہی && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

خالی ggml_backend_metal_سیٹ_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, خالی * user_ڈیٹا) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_سیٹ_abort_callback(ctx, abort_callback, user_ڈیٹا);
}

منطقی ggml_backend_metal_supports_family(ggml_backend_t backend, صحیح family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    واپس ggml_metal_supports_family(ctx, family);
}

خالی ggml_backend_metal_capture_اگلا_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_اگلا_compute(ctx);
}

// backend device

ساکن ثابت حرف * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    واپس props_dev->name;
}

ساکن ثابت حرف * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    واپس ggml_metal_device_get_props(ctx_dev)->desc;
}

ساکن خالی ggml_backend_metal_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_حافظہ(ctx_dev, آزاد, کل);
}

ساکن شمار ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    واپس GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    props->caps = {
        /* .async                = */ سچ,
        /* .host_buffer          = */ جھوٹ,
        /* .buffer_from_host_ptr = */ سچ,
        /* .events               = */ سچ,
        /* .mmap_support         = */ سچ,
    };
}

ساکن ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    اگر (ctx == تہی) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        واپس تہی;
    }

    ggml_backend_t backend = (ggml_backend_t) حافظہ_دو(حجم(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_سیٹ_n_cb(backend, 1);

    واپس backend;

    GGML_غیر_مستعمل(params);
}

ساکن ggml_backend_بفر_type_t ggml_backend_metal_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    واپس props_dev->use_shared_buffers ? ggml_backend_metal_بفر_type_shared(props_dev->device) : ggml_backend_metal_بفر_type_نجی(props_dev->device);
}

ساکن ggml_backend_بفر_t ggml_backend_metal_device_بفر_mapped(ggml_backend_dev_t dev, خالی * ptr, size_t size, size_t max_tensor_سائز) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_بفر_t res = ggml_metal_بفر_map(ctx_dev, ptr, size, max_tensor_سائز);

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    واپس ggml_backend_بفر_init(ggml_backend_metal_بفر_type_mapped(props_dev->device), ggml_backend_metal_بفر_shared_i, res, size);
}

ساکن منطقی ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    واپس ggml_metal_device_supports_op(ctx_dev, op);
}

ساکن منطقی ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    واپس
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_بفر_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_بفر_type_نجی_get_name ||
        buft->iface.get_name == ggml_backend_metal_بفر_type_mapped_get_name);

    GGML_غیر_مستعمل(dev);
}

ساکن int64_t get_op_batch_سائز(ثابت ggml_tensor * op) {
    انتخاب (op->op) {
        حالت GGML_OP_MUL_MAT:
            واپس op->ne[1];
        حالت GGML_OP_MUL_MAT_ID:
            واپس op->ne[2];
        طےشدہ:
            واپس ggml_nrows(op);
    }
}

ساکن منطقی ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    واپس (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_سائز(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_سائز;
}

ساکن ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = نیا ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    واپس ev;
}

ساکن خالی ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    حذف event;
}

ساکن خالی ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ساکن ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_بفر_type,
    /* .get_host_buffer_type = */ تہی,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_بفر_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ساخت ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

قسمتعریف ساخت ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ساکن ggml_backend_metal_reg_t ggml_backend_metal_reg_init(خالی) {
    ggml_backend_metal_reg_t ctx = نیا ساخت ggml_backend_metal_reg;

    واپس ctx;
}

ساکن خالی ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    حذف ctx;
}

ساخت ggml_backend_metal_reg_deleter {
    خالی عامل()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

قسمتعریف std::unique_ptr<ساخت ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ساکن ثابت حرف * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    واپس GGML_METAL_NAME;

    GGML_غیر_مستعمل(reg);
}

ساکن size_t ggml_backend_metal_reg_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    واپس ctx->devices.size();
}

ساکن ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t اشاریہ) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(اشاریہ < ctx->devices.size());
    واپس ctx->devices[اشاریہ];
}

ساکن ggml_backend_feature g_ggml_backend_metal_features[] = {
#اگر defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#اگر_ختم
    { تہی, تہی },
};

ساکن ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    واپس g_ggml_backend_metal_features;

    GGML_غیر_مستعمل(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ساکن خالی ggml_backend_metal_tuning_سیٹ_fa_vec_override(صحیح Q, صحیح NE) {
    ggml_metal_tuning::fa_vec_سیٹ_override({ (int8_t) Q, (int8_t) NE });
}

ساکن خالی ggml_backend_metal_tuning_clear_fa_vec_override(خالی) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ساکن صحیح ggml_backend_metal_tuning_fa_vec_ne11_بالٹی(int64_t ne11) {
    واپس ggml_metal_tuning::fa_vec_ne11_بالٹی(ne11);
}

ساکن صحیح ggml_backend_metal_tuning_fa_vec_ne01_بالٹی(int64_t ne01) {
    واپس ggml_metal_tuning::fa_vec_ne01_بالٹی(ne01);
}

ساکن صحیح ggml_backend_metal_tuning_fa_vec_baseline_ne(صحیح dk, صحیح dv) {
    واپس ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ساکن ثابت حرف * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    واپس ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
قسمتعریف خالی * ggml_backend_fusion_t;

ساکن ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    واپس ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ساکن خالی ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ساخت ggml_metal_fusion_info *) finfo);
}

ساکن خالی ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ساخت ggml_metal_fusion_info *) finfo);
}

ساکن صحیح ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ثابت حرف ** labels, uint64_t * counts, صحیح n) {
    واپس ggml_metal_fusion_info_stats_get((ساخت ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ساکن خالی ggml_backend_metal_fusion_سیٹ_enabled(ggml_backend_fusion_t finfo, منطقی enabled) {
    ggml_metal_fusion_info_سیٹ_enabled((ساخت ggml_metal_fusion_info *) finfo, enabled);
}

ساکن خالی * ggml_backend_metal_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    اگر (strcmp(name, "ggml_backend_get_features") == 0) {
        واپس (خالی *)ggml_backend_metal_get_features;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        واپس (خالی *)ggml_backend_metal_tuning_سیٹ_fa_vec_override;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        واپس (خالی *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        واپس (خالی *)ggml_backend_metal_tuning_fa_vec_ne11_بالٹی;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        واپس (خالی *)ggml_backend_metal_tuning_fa_vec_ne01_بالٹی;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        واپس (خالی *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    اگر (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        واپس (خالی *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    اگر (strcmp(name, "ggml_backend_fusion_get") == 0) {
        واپس (خالی *)ggml_backend_metal_fusion_get;
    }
    اگر (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        واپس (خالی *)ggml_backend_metal_fusion_stats_init;
    }
    اگر (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        واپس (خالی *)ggml_backend_metal_fusion_stats_reset;
    }
    اگر (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        واپس (خالی *)ggml_backend_metal_fusion_stats_get;
    }
    اگر (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        واپس (خالی *)ggml_backend_metal_fusion_سیٹ_enabled;
    }

    واپس تہی;

    GGML_غیر_مستعمل(reg);
}

ساکن ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_گنتی,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_پتہ,
};

ساکن ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, صحیح device) {
    واپس نیا ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ساکن خالی ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    حذف dev;
}

ساخت ggml_backend_device_deleter {
    خالی عامل()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

قسمتعریف std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(خالی) {
    ساکن ggml_backend_reg reg;
    ساکن منطقی initialized = جھوٹ;

    {
        ساکن std::mutex mutex;
        std::lock_guard<std::mutex> تالا(mutex);

        ثابت حرف * env = getenv("GGML_METAL_DEVICES");
        اگر (env) {
            g_devices = atoi(env);
        }

        ساکن std::vector<ggml_backend_device_ptr> devs;

        اگر (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", سچ);

            ساکن ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            برائے (صحیح i = 0; i < g_devices; ++i) {
                خودکار * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_نسخہ,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = سچ;
    }

    واپس &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
