#ضمن "ggml-metal.h"

#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"

#ضمن "ggml-metal-device.h"
#ضمن "ggml-metal-fusion.h"
#ضمن "ggml-metal-context.h"
#ضمن "ggml-metal-ops.h"
#ضمن "ggml-metal-tuning.h"

#ضمن <mutex>
#ضمن <string>

#تعریف_کرو GGML_METAL_NAME "MTL"
#تعریف_کرو GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ساكن صحيح g_devices = 1;

// forward declaration
ساكن منطقي ggml_backend_بفر_is_metal(ggml_backend_بفر_t بفر);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ساكن فراغ ggml_backend_metal_بفر_shared_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_free(ctx);
}

ساكن فراغ * ggml_backend_metal_بفر_shared_get_ادھار(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    أرجع ggml_metal_بفر_get_ادھار(ctx);
}

ساكن فراغ ggml_backend_metal_بفر_shared_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_memset_tensor(ctx, tensor, value, وتھ, size);
}

ساكن فراغ ggml_backend_metal_بفر_shared_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_سیٹ_tensor(ctx, tensor, انکڑے, وتھ, size);
}

ساكن فراغ ggml_backend_metal_بفر_shared_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_get_tensor(ctx, tensor, انکڑے, وتھ, size);
}

ساكن منطقي ggml_backend_metal_بفر_shared_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    إذا (!ggml_backend_بفر_is_metal(src->بفر)) {
        أرجع خطأ;
    }

    أرجع ggml_metal_بفر_cpy_tensor(ctx, src, dst);
}

ساكن فراغ ggml_backend_metal_بفر_shared_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_clear(ctx, value);
}

ساكن ggml_backend_بفر_i ggml_backend_metal_بفر_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_بفر_shared_free_بفر,
    /* .get_base      = */ ggml_backend_metal_بفر_shared_get_ادھار,
    /* .init_tensor   = */ عدم,
    /* .memset_tensor = */ ggml_backend_metal_بفر_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_بفر_shared_سیٹ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_بفر_shared_get_tensor,
    /* .set_tensor_2d = */ عدم,
    /* .get_tensor_2d = */ عدم,
    /* .cpy_tensor    = */ ggml_backend_metal_بفر_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_بفر_shared_clear,
    /* .reset         = */ عدم,
};

// private buffer

ساكن فراغ ggml_backend_metal_بفر_نجی_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_free(ctx);
}

ساكن فراغ * ggml_backend_metal_بفر_نجی_get_ادھار(ggml_backend_بفر_t بفر) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    أرجع ggml_metal_بفر_get_ادھار(ctx);
}

ساكن فراغ ggml_backend_metal_بفر_نجی_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_memset_tensor(ctx, tensor, value, وتھ, size);
}

ساكن فراغ ggml_backend_metal_بفر_نجی_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_سیٹ_tensor(ctx, tensor, انکڑے, وتھ, size);
}

ساكن فراغ ggml_backend_metal_بفر_نجی_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_get_tensor(ctx, tensor, انکڑے, وتھ, size);
}

ساكن منطقي ggml_backend_metal_بفر_نجی_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    إذا (!ggml_backend_بفر_is_metal(src->بفر)) {
        أرجع خطأ;
    }

    أرجع ggml_metal_بفر_cpy_tensor(ctx, src, dst);
}

ساكن فراغ ggml_backend_metal_بفر_نجی_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_metal_بفر_t ctx = (ggml_metal_بفر_t)بفر->context;

    GGML_ASSERT(!ggml_metal_بفر_is_shared(ctx));

    ggml_metal_بفر_clear(ctx, value);
}

ساكن ggml_backend_بفر_i ggml_backend_metal_بفر_نجی_i = {
    /* .free_buffer   = */ ggml_backend_metal_بفر_نجی_free_بفر,
    /* .get_base      = */ ggml_backend_metal_بفر_نجی_get_ادھار,
    /* .init_tensor   = */ عدم,
    /* .memset_tensor = */ ggml_backend_metal_بفر_نجی_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_بفر_نجی_سیٹ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_بفر_نجی_get_tensor,
    /* .set_tensor_2d = */ عدم,
    /* .get_tensor_2d = */ عدم,
    /* .cpy_tensor    = */ ggml_backend_metal_بفر_نجی_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_بفر_نجی_clear,
    /* .reset         = */ عدم,
};

ساكن منطقي ggml_backend_بفر_is_metal(ggml_backend_بفر_t بفر) {
    أرجع بفر->iface.free_بفر == ggml_backend_metal_بفر_shared_free_بفر ||
           بفر->iface.free_بفر == ggml_backend_metal_بفر_نجی_free_بفر;
}

//
// buffer types
//

بنية ggml_backend_metal_بفر_type {
    صحيح device;
    std::string name;
};

بنية ggml_backend_metal_بفر_type_deleter {
    فراغ عامل()(ggml_backend_metal_بفر_type * ctx) ثابت {
        مٹاؤ ctx;
    }
};

تعريفنوع std::unique_ptr<ggml_backend_metal_بفر_type, ggml_backend_metal_بفر_type_deleter> ggml_backend_metal_بفر_type_ptr;

// common method for allocating shread or private Metal buffers
ساكن ggml_backend_بفر_t ggml_backend_metal_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size, منطقي shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_بفر_t res = ggml_metal_بفر_init(ctx_dev, size, shared);

    إذا (res == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        أرجع عدم;
    }

    ggml_backend_بفر_i buf_i = ggml_metal_بفر_is_shared(res)
        ? ggml_backend_metal_بفر_shared_i
        : ggml_backend_metal_بفر_نجی_i;

    أرجع ggml_backend_بفر_init(buft, buf_i, res, size);
}

ساكن size_t ggml_backend_metal_بفر_type_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    بدل (tensor->op) {
        حالة GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } اكسر;
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_ARGSORT:
            {
                res *= 2;
            } اكسر;
        حالة GGML_OP_چوٹی_K:
            {
                res = 2*حجم(int32_t)*ggml_nelements(tensor->src[0]);
            } اكسر;
        حالة GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } اكسر;
        افتراضي:
            اكسر;
    }

    أرجع res;

    GGML_بن_ورتیا(buft);
}

// default (shared) buffer type

ساكن ثابت حرف * ggml_backend_metal_بفر_type_shared_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_metal_بفر_type * ctx = (ggml_backend_metal_بفر_type *)buft->context;

    أرجع ctx->name.c_str();
}

ساكن ggml_backend_بفر_t ggml_backend_metal_بفر_type_shared_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    أرجع ggml_backend_metal_بفر_type_alloc_بفر(buft, size, سچ);
}

ساكن size_t ggml_backend_metal_بفر_type_shared_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 32;

    GGML_بن_ورتیا(buft);
}

ساكن size_t ggml_backend_metal_بفر_type_shared_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    أرجع ggml_metal_device_get_props(ctx_dev)->max_بفر_ناپ;
}

ساكن size_t ggml_backend_metal_بفر_type_shared_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_backend_metal_بفر_type_get_alloc_ناپ(buft, tensor);
}

ساكن منطقي ggml_backend_metal_بفر_type_shared_is_host(ggml_backend_بفر_type_t buft) {
    أرجع خطأ;

    GGML_بن_ورتیا(buft);
}

ساكن ggml_backend_بفر_type_t ggml_backend_metal_بفر_type_shared(صحيح device) {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> جندرا(mutex);

    ساكن std::vector<ggml_backend_بفر_type> bufts;
    ساكن std::vector<ggml_backend_metal_بفر_type_ptr> ctxs;

    ساكن منطقي initialized = خطأ;
    إذا (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        لكل (صحيح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_بفر_type * raw_ctx =
                جديد ggml_backend_metal_بفر_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_بفر_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_بفر_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_بفر_type_shared_alloc_بفر,
                    /* .alloc_buffer_n      = */ عدم,
                    /* .get_alignment       = */ ggml_backend_metal_بفر_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_بفر_type_shared_get_max_ناپ,
                    /* .get_alloc_size      = */ ggml_backend_metal_بفر_type_shared_get_alloc_ناپ,
                    /* .get_alloc_size_n    = */ عدم,
                    /* .is_host             = */ ggml_backend_metal_بفر_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = سچ;
    }

    أرجع &bufts[device];
}

// default (private) buffer type

ساكن ثابت حرف * ggml_backend_metal_بفر_type_نجی_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_metal_بفر_type * ctx = (ggml_backend_metal_بفر_type *)buft->context;

    أرجع ctx->name.c_str();
}

ساكن ggml_backend_بفر_t ggml_backend_metal_بفر_type_نجی_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    أرجع ggml_backend_metal_بفر_type_alloc_بفر(buft, size, خطأ);
}

ساكن size_t ggml_backend_metal_بفر_type_نجی_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 32;

    GGML_بن_ورتیا(buft);
}

ساكن size_t ggml_backend_metal_بفر_type_نجی_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    أرجع ggml_metal_device_get_props(ctx_dev)->max_بفر_ناپ;
}

ساكن size_t ggml_backend_metal_بفر_type_نجی_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_backend_metal_بفر_type_get_alloc_ناپ(buft, tensor);
}

ساكن منطقي ggml_backend_metal_بفر_type_نجی_is_host(ggml_backend_بفر_type_t buft) {
    أرجع خطأ;

    GGML_بن_ورتیا(buft);
}

ساكن ggml_backend_بفر_type_t ggml_backend_metal_بفر_type_نجی(صحيح device) {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> جندرا(mutex);

    ساكن std::vector<ggml_backend_بفر_type> bufts;
    ساكن std::vector<ggml_backend_metal_بفر_type_ptr> ctxs;

    ساكن منطقي initialized = خطأ;
    إذا (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        لكل (صحيح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_بفر_type * raw_ctx = جديد ggml_backend_metal_بفر_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_بفر_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_بفر_type_نجی_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_بفر_type_نجی_alloc_بفر,
                    /* .alloc_buffer_n      = */ عدم,
                    /* .get_alignment       = */ ggml_backend_metal_بفر_type_نجی_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_بفر_type_نجی_get_max_ناپ,
                    /* .get_alloc_size      = */ ggml_backend_metal_بفر_type_نجی_get_alloc_ناپ,
                    /* .get_alloc_size_n    = */ عدم,
                    /* .is_host             = */ ggml_backend_metal_بفر_type_نجی_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = سچ;
    }

    أرجع &bufts[device];
}

// mapped buffer type

ساكن ثابت حرف * ggml_backend_metal_بفر_type_mapped_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_metal_بفر_type * ctx = (ggml_backend_metal_بفر_type *)buft->context;

    أرجع ctx->name.c_str();
}

ساكن ggml_backend_بفر_t ggml_backend_metal_بفر_type_mapped_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    أرجع ggml_backend_metal_بفر_type_alloc_بفر(buft, size, سچ);
}

ساكن size_t ggml_backend_metal_بفر_type_mapped_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 32;

    GGML_بن_ورتیا(buft);
}

ساكن size_t ggml_backend_metal_بفر_type_mapped_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    أرجع ggml_metal_device_get_props(ctx_dev)->max_بفر_ناپ;
}

ساكن size_t ggml_backend_metal_بفر_type_mapped_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_backend_metal_بفر_type_get_alloc_ناپ(buft, tensor);
}

ساكن منطقي ggml_backend_metal_بفر_type_mapped_is_host(ggml_backend_بفر_type_t buft) {
    أرجع خطأ;

    GGML_بن_ورتیا(buft);
}

ساكن ggml_backend_بفر_type_t ggml_backend_metal_بفر_type_mapped(صحيح device) {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> جندرا(mutex);

    ساكن std::vector<ggml_backend_بفر_type> bufts;
    ساكن std::vector<ggml_backend_metal_بفر_type_ptr> ctxs;

    ساكن منطقي initialized = خطأ;
    إذا (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        لكل (صحيح i = 0; i < g_devices; ++i) {
            ggml_backend_metal_بفر_type * raw_ctx = جديد ggml_backend_metal_بفر_type{
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
                    /* .alloc_buffer_n      = */ عدم,
                    /* .get_alignment       = */ ggml_backend_metal_بفر_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_بفر_type_mapped_get_max_ناپ,
                    /* .get_alloc_size      = */ ggml_backend_metal_بفر_type_mapped_get_alloc_ناپ,
                    /* .get_alloc_size_n    = */ عدم,
                    /* .is_host             = */ ggml_backend_metal_بفر_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = سچ;
    }

    أرجع &bufts[device];
}

// backend

ساكن ثابت حرف * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    أرجع ggml_metal_get_name(ctx);
}

ساكن فراغ ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    حرر(backend);
}

ساكن فراغ ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ساكن فراغ ggml_backend_metal_سیٹ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_سیٹ_tensor_async(ctx, tensor, انکڑے, وتھ, size);
}

ساكن فراغ ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, انکڑے, وتھ, size);
}

ساكن منطقي ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    إذا (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        أرجع خطأ;
    }

    إذا (!ggml_backend_بفر_is_metal(src->بفر) || !ggml_backend_بفر_is_metal(dst->بفر)) {
        أرجع خطأ;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    أرجع ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ساكن تعداد ggml_حالت ggml_backend_metal_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    أرجع ggml_metal_گراف_compute(ctx, cgraph);
}

ساكن فراغ ggml_backend_metal_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_ریکارڈ(ctx, ev);
}

ساكن فراغ ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ساكن فراغ ggml_backend_metal_گراف_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_گراف_optimize_params * params) {
    GGML_ASSERT(params && params->جوڑو_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_جوڑو_alloc_deps(params->user_انکڑے, params->جوڑو_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_گراف_optimize(ctx, cgraph);
}

ساكن فراغ ggml_backend_metal_سیٹ_n_cb(ggml_backend_t backend, صحيح n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_سیٹ_n_cb(ctx, n_cb);
}

ساكن ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_metal_گراف_compute,
    /* .event_record            = */ ggml_backend_metal_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_گراف_optimize,
};

ساكن ggml_guid_t ggml_backend_metal_guid(فراغ) {
    ساكن ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    أرجع &guid;
}

ggml_backend_t ggml_backend_metal_init(فراغ) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    إذا (ctx == عدم) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        أرجع عدم;
    }

    ggml_backend_t backend = (ggml_backend_t) حافظہ_دیو(حجم(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_سیٹ_n_cb(backend, 1);

    أرجع backend;
}

منطقي ggml_backend_is_metal(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

فراغ ggml_backend_metal_سیٹ_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, فراغ * user_انکڑے) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_سیٹ_abort_callback(ctx, abort_callback, user_انکڑے);
}

منطقي ggml_backend_metal_supports_family(ggml_backend_t backend, صحيح family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    أرجع ggml_metal_supports_family(ctx, family);
}

فراغ ggml_backend_metal_capture_اگلا_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_اگلا_compute(ctx);
}

// backend device

ساكن ثابت حرف * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    أرجع props_dev->name;
}

ساكن ثابت حرف * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    أرجع ggml_metal_device_get_props(ctx_dev)->desc;
}

ساكن فراغ ggml_backend_metal_device_get_حافظہ(ggml_backend_dev_t dev, size_t * حرر, size_t * کل) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_حافظہ(ctx_dev, حرر, کل);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_بن_ورتیا(dev);
}

ساكن فراغ ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    props->caps = {
        /* .async                = */ سچ,
        /* .host_buffer          = */ خطأ,
        /* .buffer_from_host_ptr = */ سچ,
        /* .events               = */ سچ,
        /* .mmap_support         = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    إذا (ctx == عدم) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        أرجع عدم;
    }

    ggml_backend_t backend = (ggml_backend_t) حافظہ_دیو(حجم(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_سیٹ_n_cb(backend, 1);

    أرجع backend;

    GGML_بن_ورتیا(params);
}

ساكن ggml_backend_بفر_type_t ggml_backend_metal_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    أرجع props_dev->use_shared_buffers ? ggml_backend_metal_بفر_type_shared(props_dev->device) : ggml_backend_metal_بفر_type_نجی(props_dev->device);
}

ساكن ggml_backend_بفر_t ggml_backend_metal_device_بفر_mapped(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_ناپ) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_بفر_t res = ggml_metal_بفر_map(ctx_dev, ptr, size, max_tensor_ناپ);

    ثابت ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    أرجع ggml_backend_بفر_init(ggml_backend_metal_بفر_type_mapped(props_dev->device), ggml_backend_metal_بفر_shared_i, res, size);
}

ساكن منطقي ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    أرجع ggml_metal_device_supports_op(ctx_dev, op);
}

ساكن منطقي ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    أرجع
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_بفر_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_بفر_type_نجی_get_name ||
        buft->iface.get_name == ggml_backend_metal_بفر_type_mapped_get_name);

    GGML_بن_ورتیا(dev);
}

ساكن int64_t get_op_batch_ناپ(ثابت ggml_tensor * op) {
    بدل (op->op) {
        حالة GGML_OP_MUL_MAT:
            أرجع op->ne[1];
        حالة GGML_OP_MUL_MAT_ID:
            أرجع op->ne[2];
        افتراضي:
            أرجع ggml_nrows(op);
    }
}

ساكن منطقي ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    أرجع (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_ناپ(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_ناپ;
}

ساكن ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = جديد ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    أرجع ev;
}

ساكن فراغ ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    مٹاؤ event;
}

ساكن فراغ ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ساكن ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_بفر_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_بفر_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

بنية ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

تعريفنوع بنية ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ساكن ggml_backend_metal_reg_t ggml_backend_metal_reg_init(فراغ) {
    ggml_backend_metal_reg_t ctx = جديد بنية ggml_backend_metal_reg;

    أرجع ctx;
}

ساكن فراغ ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    مٹاؤ ctx;
}

بنية ggml_backend_metal_reg_deleter {
    فراغ عامل()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

تعريفنوع std::unique_ptr<بنية ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ساكن ثابت حرف * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    أرجع GGML_METAL_NAME;

    GGML_بن_ورتیا(reg);
}

ساكن size_t ggml_backend_metal_reg_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    أرجع ctx->devices.size();
}

ساكن ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t اشاریہ) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(اشاریہ < ctx->devices.size());
    أرجع ctx->devices[اشاریہ];
}

ساكن ggml_backend_feature g_ggml_backend_metal_features[] = {
#إذا defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#جے_انت
    { عدم, عدم },
};

ساكن ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    أرجع g_ggml_backend_metal_features;

    GGML_بن_ورتیا(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ساكن فراغ ggml_backend_metal_tuning_سیٹ_fa_vec_override(صحيح Q, صحيح NE) {
    ggml_metal_tuning::fa_vec_سیٹ_override({ (int8_t) Q, (int8_t) NE });
}

ساكن فراغ ggml_backend_metal_tuning_clear_fa_vec_override(فراغ) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ساكن صحيح ggml_backend_metal_tuning_fa_vec_ne11_بالٹی(int64_t ne11) {
    أرجع ggml_metal_tuning::fa_vec_ne11_بالٹی(ne11);
}

ساكن صحيح ggml_backend_metal_tuning_fa_vec_ne01_بالٹی(int64_t ne01) {
    أرجع ggml_metal_tuning::fa_vec_ne01_بالٹی(ne01);
}

ساكن صحيح ggml_backend_metal_tuning_fa_vec_baseline_ne(صحيح dk, صحيح dv) {
    أرجع ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ساكن ثابت حرف * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    أرجع ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
تعريفنوع فراغ * ggml_backend_fusion_t;

ساكن ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    أرجع ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ساكن فراغ ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((بنية ggml_metal_fusion_info *) finfo);
}

ساكن فراغ ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((بنية ggml_metal_fusion_info *) finfo);
}

ساكن صحيح ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ثابت حرف ** labels, uint64_t * counts, صحيح n) {
    أرجع ggml_metal_fusion_info_stats_get((بنية ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ساكن فراغ ggml_backend_metal_fusion_سیٹ_enabled(ggml_backend_fusion_t finfo, منطقي enabled) {
    ggml_metal_fusion_info_سیٹ_enabled((بنية ggml_metal_fusion_info *) finfo, enabled);
}

ساكن فراغ * ggml_backend_metal_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    إذا (strcmp(name, "ggml_backend_get_features") == 0) {
        أرجع (فراغ *)ggml_backend_metal_get_features;
    }
    إذا (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        أرجع (فراغ *)ggml_backend_metal_tuning_سیٹ_fa_vec_override;
    }
    إذا (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        أرجع (فراغ *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    إذا (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        أرجع (فراغ *)ggml_backend_metal_tuning_fa_vec_ne11_بالٹی;
    }
    إذا (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        أرجع (فراغ *)ggml_backend_metal_tuning_fa_vec_ne01_بالٹی;
    }
    إذا (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        أرجع (فراغ *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    إذا (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        أرجع (فراغ *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    إذا (strcmp(name, "ggml_backend_fusion_get") == 0) {
        أرجع (فراغ *)ggml_backend_metal_fusion_get;
    }
    إذا (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        أرجع (فراغ *)ggml_backend_metal_fusion_stats_init;
    }
    إذا (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        أرجع (فراغ *)ggml_backend_metal_fusion_stats_reset;
    }
    إذا (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        أرجع (فراغ *)ggml_backend_metal_fusion_stats_get;
    }
    إذا (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        أرجع (فراغ *)ggml_backend_metal_fusion_سیٹ_enabled;
    }

    أرجع عدم;

    GGML_بن_ورتیا(reg);
}

ساكن ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_گنتی,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_پتہ,
};

ساكن ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, صحيح device) {
    أرجع جديد ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ساكن فراغ ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    مٹاؤ dev;
}

بنية ggml_backend_device_deleter {
    فراغ عامل()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

تعريفنوع std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(فراغ) {
    ساكن ggml_backend_reg reg;
    ساكن منطقي initialized = خطأ;

    {
        ساكن std::mutex mutex;
        std::lock_guard<std::mutex> جندرا(mutex);

        ثابت حرف * env = getenv("GGML_METAL_DEVICES");
        إذا (env) {
            g_devices = atoi(env);
        }

        ساكن std::vector<ggml_backend_device_ptr> devs;

        إذا (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", سچ);

            ساكن ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            لكل (صحيح i = 0; i < g_devices; ++i) {
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

    أرجع &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
