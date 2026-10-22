#অন্তর্ভাব "ggml-metal.h"

#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব "ggml-metal-device.h"
#অন্তর্ভাব "ggml-metal-fusion.h"
#অন্তর্ভাব "ggml-metal-context.h"
#অন্তর্ভাব "ggml-metal-ops.h"
#অন্তর্ভাব "ggml-metal-tuning.h"

#অন্তর্ভাব <mutex>
#অন্তর্ভাব <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
স্থির পূর্ণসংখ্যা g_devices = 1;

// forward declaration
স্থির যৌক্তিক ggml_backend_বাফার_is_metal(ggml_backend_বাফার_t বাফার);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

স্থির শূন্য ggml_backend_metal_বাফার_shared_free_বাফার(ggml_backend_বাফার_t বাফার) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_free(ctx);
}

স্থির শূন্য * ggml_backend_metal_বাফার_shared_get_base(ggml_backend_বাফার_t বাফার) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    ফেরত ggml_metal_বাফার_get_base(ctx);
}

স্থির শূন্য ggml_backend_metal_বাফার_shared_memset_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_memset_tensor(ctx, tensor, value, offset, size);
}

স্থির শূন্য ggml_backend_metal_বাফার_shared_set_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_set_tensor(ctx, tensor, তথ্য, offset, size);
}

স্থির শূন্য ggml_backend_metal_বাফার_shared_get_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_get_tensor(ctx, tensor, তথ্য, offset, size);
}

স্থির যৌক্তিক ggml_backend_metal_বাফার_shared_cpy_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    যদি (!ggml_backend_বাফার_is_metal(src->বাফার)) {
        ফেরত মিথ্যা;
    }

    ফেরত ggml_metal_বাফার_cpy_tensor(ctx, src, dst);
}

স্থির শূন্য ggml_backend_metal_বাফার_shared_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_clear(ctx, value);
}

স্থির ggml_backend_বাফার_i ggml_backend_metal_বাফার_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_বাফার_shared_free_বাফার,
    /* .get_base      = */ ggml_backend_metal_বাফার_shared_get_base,
    /* .init_tensor   = */ ফাঁকা,
    /* .memset_tensor = */ ggml_backend_metal_বাফার_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_বাফার_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_বাফার_shared_get_tensor,
    /* .set_tensor_2d = */ ফাঁকা,
    /* .get_tensor_2d = */ ফাঁকা,
    /* .cpy_tensor    = */ ggml_backend_metal_বাফার_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_বাফার_shared_clear,
    /* .reset         = */ ফাঁকা,
};

// private buffer

স্থির শূন্য ggml_backend_metal_বাফার_private_free_বাফার(ggml_backend_বাফার_t বাফার) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_free(ctx);
}

স্থির শূন্য * ggml_backend_metal_বাফার_private_get_base(ggml_backend_বাফার_t বাফার) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    ফেরত ggml_metal_বাফার_get_base(ctx);
}

স্থির শূন্য ggml_backend_metal_বাফার_private_memset_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_memset_tensor(ctx, tensor, value, offset, size);
}

স্থির শূন্য ggml_backend_metal_বাফার_private_set_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_set_tensor(ctx, tensor, তথ্য, offset, size);
}

স্থির শূন্য ggml_backend_metal_বাফার_private_get_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_get_tensor(ctx, tensor, তথ্য, offset, size);
}

স্থির যৌক্তিক ggml_backend_metal_বাফার_private_cpy_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    যদি (!ggml_backend_বাফার_is_metal(src->বাফার)) {
        ফেরত মিথ্যা;
    }

    ফেরত ggml_metal_বাফার_cpy_tensor(ctx, src, dst);
}

স্থির শূন্য ggml_backend_metal_বাফার_private_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    ggml_metal_বাফার_t ctx = (ggml_metal_বাফার_t)বাফার->context;

    GGML_ASSERT(!ggml_metal_বাফার_is_shared(ctx));

    ggml_metal_বাফার_clear(ctx, value);
}

স্থির ggml_backend_বাফার_i ggml_backend_metal_বাফার_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_বাফার_private_free_বাফার,
    /* .get_base      = */ ggml_backend_metal_বাফার_private_get_base,
    /* .init_tensor   = */ ফাঁকা,
    /* .memset_tensor = */ ggml_backend_metal_বাফার_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_বাফার_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_বাফার_private_get_tensor,
    /* .set_tensor_2d = */ ফাঁকা,
    /* .get_tensor_2d = */ ফাঁকা,
    /* .cpy_tensor    = */ ggml_backend_metal_বাফার_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_বাফার_private_clear,
    /* .reset         = */ ফাঁকা,
};

স্থির যৌক্তিক ggml_backend_বাফার_is_metal(ggml_backend_বাফার_t বাফার) {
    ফেরত বাফার->iface.free_বাফার == ggml_backend_metal_বাফার_shared_free_বাফার ||
           বাফার->iface.free_বাফার == ggml_backend_metal_বাফার_private_free_বাফার;
}

//
// buffer types
//

কাঠামো ggml_backend_metal_বাফার_type {
    পূর্ণসংখ্যা device;
    std::string name;
};

কাঠামো ggml_backend_metal_বাফার_type_deleter {
    শূন্য operator()(ggml_backend_metal_বাফার_type * ctx) ধ্রুবক {
        delete ctx;
    }
};

প্রকারলেখ std::unique_ptr<ggml_backend_metal_বাফার_type, ggml_backend_metal_বাফার_type_deleter> ggml_backend_metal_বাফার_type_ptr;

// common method for allocating shread or private Metal buffers
স্থির ggml_backend_বাফার_t ggml_backend_metal_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size, যৌক্তিক shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_বাফার_t res = ggml_metal_বাফার_init(ctx_dev, size, shared);

    যদি (res == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        ফেরত ফাঁকা;
    }

    ggml_backend_বাফার_i buf_i = ggml_metal_বাফার_is_shared(res)
        ? ggml_backend_metal_বাফার_shared_i
        : ggml_backend_metal_বাফার_private_i;

    ফেরত ggml_backend_বাফার_init(buft, buf_i, res, size);
}

স্থির size_t ggml_backend_metal_বাফার_type_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    নির্বাচন (tensor->op) {
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } বিরতি;
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } বিরতি;
        ক্ষেত্র GGML_OP_CUMSUM:
        ক্ষেত্র GGML_OP_ARGSORT:
            {
                res *= 2;
            } বিরতি;
        ক্ষেত্র GGML_OP_শীর্ষ_K:
            {
                res = 2*আকার(int32_t)*ggml_nelements(tensor->src[0]);
            } বিরতি;
        ক্ষেত্র GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } বিরতি;
        পূর্বনির্ধারিত:
            বিরতি;
    }

    ফেরত res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_বাফার_type_shared_get_name(ggml_backend_বাফার_type_t buft) {
    ggml_backend_metal_বাফার_type * ctx = (ggml_backend_metal_বাফার_type *)buft->context;

    ফেরত ctx->name.c_str();
}

স্থির ggml_backend_বাফার_t ggml_backend_metal_বাফার_type_shared_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    ফেরত ggml_backend_metal_বাফার_type_alloc_বাফার(buft, size, সত্য);
}

স্থির size_t ggml_backend_metal_বাফার_type_shared_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত 32;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_metal_বাফার_type_shared_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ফেরত ggml_metal_device_get_props(ctx_dev)->max_বাফার_আকার;
}

স্থির size_t ggml_backend_metal_বাফার_type_shared_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    ফেরত ggml_backend_metal_বাফার_type_get_alloc_আকার(buft, tensor);
}

স্থির যৌক্তিক ggml_backend_metal_বাফার_type_shared_is_host(ggml_backend_বাফার_type_t buft) {
    ফেরত মিথ্যা;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_metal_বাফার_type_shared(পূর্ণসংখ্যা device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    স্থির std::vector<ggml_backend_বাফার_type> bufts;
    স্থির std::vector<ggml_backend_metal_বাফার_type_ptr> ctxs;

    স্থির যৌক্তিক initialized = মিথ্যা;
    যদি (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        জন্য (পূর্ণসংখ্যা i = 0; i < g_devices; ++i) {
            ggml_backend_metal_বাফার_type * raw_ctx =
                নতুন ggml_backend_metal_বাফার_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_বাফার_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_বাফার_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_বাফার_type_shared_alloc_বাফার,
                    /* .alloc_buffer_n      = */ ফাঁকা,
                    /* .get_alignment       = */ ggml_backend_metal_বাফার_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_বাফার_type_shared_get_max_আকার,
                    /* .get_alloc_size      = */ ggml_backend_metal_বাফার_type_shared_get_alloc_আকার,
                    /* .get_alloc_size_n    = */ ফাঁকা,
                    /* .is_host             = */ ggml_backend_metal_বাফার_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = সত্য;
    }

    ফেরত &bufts[device];
}

// default (private) buffer type

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_বাফার_type_private_get_name(ggml_backend_বাফার_type_t buft) {
    ggml_backend_metal_বাফার_type * ctx = (ggml_backend_metal_বাফার_type *)buft->context;

    ফেরত ctx->name.c_str();
}

স্থির ggml_backend_বাফার_t ggml_backend_metal_বাফার_type_private_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    ফেরত ggml_backend_metal_বাফার_type_alloc_বাফার(buft, size, মিথ্যা);
}

স্থির size_t ggml_backend_metal_বাফার_type_private_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত 32;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_metal_বাফার_type_private_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ফেরত ggml_metal_device_get_props(ctx_dev)->max_বাফার_আকার;
}

স্থির size_t ggml_backend_metal_বাফার_type_private_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    ফেরত ggml_backend_metal_বাফার_type_get_alloc_আকার(buft, tensor);
}

স্থির যৌক্তিক ggml_backend_metal_বাফার_type_private_is_host(ggml_backend_বাফার_type_t buft) {
    ফেরত মিথ্যা;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_metal_বাফার_type_private(পূর্ণসংখ্যা device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    স্থির std::vector<ggml_backend_বাফার_type> bufts;
    স্থির std::vector<ggml_backend_metal_বাফার_type_ptr> ctxs;

    স্থির যৌক্তিক initialized = মিথ্যা;
    যদি (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        জন্য (পূর্ণসংখ্যা i = 0; i < g_devices; ++i) {
            ggml_backend_metal_বাফার_type * raw_ctx = নতুন ggml_backend_metal_বাফার_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_বাফার_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_বাফার_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_বাফার_type_private_alloc_বাফার,
                    /* .alloc_buffer_n      = */ ফাঁকা,
                    /* .get_alignment       = */ ggml_backend_metal_বাফার_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_বাফার_type_private_get_max_আকার,
                    /* .get_alloc_size      = */ ggml_backend_metal_বাফার_type_private_get_alloc_আকার,
                    /* .get_alloc_size_n    = */ ফাঁকা,
                    /* .is_host             = */ ggml_backend_metal_বাফার_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = সত্য;
    }

    ফেরত &bufts[device];
}

// mapped buffer type

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_বাফার_type_mapped_get_name(ggml_backend_বাফার_type_t buft) {
    ggml_backend_metal_বাফার_type * ctx = (ggml_backend_metal_বাফার_type *)buft->context;

    ফেরত ctx->name.c_str();
}

স্থির ggml_backend_বাফার_t ggml_backend_metal_বাফার_type_mapped_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    ফেরত ggml_backend_metal_বাফার_type_alloc_বাফার(buft, size, সত্য);
}

স্থির size_t ggml_backend_metal_বাফার_type_mapped_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত 32;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_metal_বাফার_type_mapped_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ফেরত ggml_metal_device_get_props(ctx_dev)->max_বাফার_আকার;
}

স্থির size_t ggml_backend_metal_বাফার_type_mapped_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    ফেরত ggml_backend_metal_বাফার_type_get_alloc_আকার(buft, tensor);
}

স্থির যৌক্তিক ggml_backend_metal_বাফার_type_mapped_is_host(ggml_backend_বাফার_type_t buft) {
    ফেরত মিথ্যা;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_metal_বাফার_type_mapped(পূর্ণসংখ্যা device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    স্থির std::vector<ggml_backend_বাফার_type> bufts;
    স্থির std::vector<ggml_backend_metal_বাফার_type_ptr> ctxs;

    স্থির যৌক্তিক initialized = মিথ্যা;
    যদি (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        জন্য (পূর্ণসংখ্যা i = 0; i < g_devices; ++i) {
            ggml_backend_metal_বাফার_type * raw_ctx = নতুন ggml_backend_metal_বাফার_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_বাফার_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_বাফার_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_বাফার_type_mapped_alloc_বাফার,
                    /* .alloc_buffer_n      = */ ফাঁকা,
                    /* .get_alignment       = */ ggml_backend_metal_বাফার_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_বাফার_type_mapped_get_max_আকার,
                    /* .get_alloc_size      = */ ggml_backend_metal_বাফার_type_mapped_get_alloc_আকার,
                    /* .get_alloc_size_n    = */ ফাঁকা,
                    /* .is_host             = */ ggml_backend_metal_বাফার_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = সত্য;
    }

    ফেরত &bufts[device];
}

// backend

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ফেরত ggml_metal_get_name(ctx);
}

স্থির শূন্য ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    মুক্ত(backend);
}

স্থির শূন্য ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

স্থির শূন্য ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, তথ্য, offset, size);
}

স্থির শূন্য ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, তথ্য, offset, size);
}

স্থির যৌক্তিক ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুবক ggml_tensor * src, ggml_tensor * dst) {
    যদি (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        ফেরত মিথ্যা;
    }

    যদি (!ggml_backend_বাফার_is_metal(src->বাফার) || !ggml_backend_বাফার_is_metal(dst->বাফার)) {
        ফেরত মিথ্যা;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    ফেরত ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

স্থির গণনা ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ফেরত ggml_metal_graph_compute(ctx, cgraph);
}

স্থির শূন্য ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

স্থির শূন্য ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

স্থির শূন্য ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_তথ্য, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

স্থির শূন্য ggml_backend_metal_set_n_cb(ggml_backend_t backend, পূর্ণসংখ্যা n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

স্থির ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ফাঁকা,
    /* .get_tensor_2d_async     = */ ফাঁকা,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ফাঁকা,
    /* .graph_plan_free         = */ ফাঁকা,
    /* .graph_plan_update       = */ ফাঁকা,
    /* .graph_plan_compute      = */ ফাঁকা,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

স্থির ggml_guid_t ggml_backend_metal_guid(শূন্য) {
    স্থির ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    ফেরত &guid;
}

ggml_backend_t ggml_backend_metal_init(শূন্য) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    যদি (ctx == ফাঁকা) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ফেরত ফাঁকা;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(আকার(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    ফেরত backend;
}

যৌক্তিক ggml_backend_is_metal(ggml_backend_t backend) {
    ফেরত backend != ফাঁকা && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

শূন্য ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, শূন্য * user_তথ্য) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_তথ্য);
}

যৌক্তিক ggml_backend_metal_supports_family(ggml_backend_t backend, পূর্ণসংখ্যা family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ফেরত ggml_metal_supports_family(ctx, family);
}

শূন্য ggml_backend_metal_capture_পরবর্তী_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_পরবর্তী_compute(ctx);
}

// backend device

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ধ্রুবক ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ফেরত props_dev->name;
}

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ফেরত ggml_metal_device_get_props(ctx_dev)->desc;
}

স্থির শূন্য ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, মুক্ত, যোগ);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    ফেরত GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);

    props->caps = {
        /* .async                = */ সত্য,
        /* .host_buffer          = */ মিথ্যা,
        /* .buffer_from_host_ptr = */ সত্য,
        /* .events               = */ সত্য,
        /* .mmap_support         = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    যদি (ctx == ফাঁকা) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ফেরত ফাঁকা;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(আকার(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    ফেরত backend;

    GGML_UNUSED(params);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_metal_device_get_বাফার_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ধ্রুবক ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ফেরত props_dev->use_shared_buffers ? ggml_backend_metal_বাফার_type_shared(props_dev->device) : ggml_backend_metal_বাফার_type_private(props_dev->device);
}

স্থির ggml_backend_বাফার_t ggml_backend_metal_device_বাফার_mapped(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_আকার) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_বাফার_t res = ggml_metal_বাফার_map(ctx_dev, ptr, size, max_tensor_আকার);

    ধ্রুবক ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ফেরত ggml_backend_বাফার_init(ggml_backend_metal_বাফার_type_mapped(props_dev->device), ggml_backend_metal_বাফার_shared_i, res, size);
}

স্থির যৌক্তিক ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ফেরত ggml_metal_device_supports_op(ctx_dev, op);
}

স্থির যৌক্তিক ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    ফেরত
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_বাফার_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_বাফার_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_বাফার_type_mapped_get_name);

    GGML_UNUSED(dev);
}

স্থির int64_t get_op_batch_আকার(ধ্রুবক ggml_tensor * op) {
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_MUL_MAT:
            ফেরত op->ne[1];
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            ফেরত op->ne[2];
        পূর্বনির্ধারিত:
            ফেরত ggml_nrows(op);
    }
}

স্থির যৌক্তিক ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ফেরত (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_আকার(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_আকার;
}

স্থির ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = নতুন ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    ফেরত ev;
}

স্থির শূন্য ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

স্থির শূন্য ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

স্থির ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ফাঁকা,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_বাফার_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

কাঠামো ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

প্রকারলেখ কাঠামো ggml_backend_metal_reg * ggml_backend_metal_reg_t;

স্থির ggml_backend_metal_reg_t ggml_backend_metal_reg_init(শূন্য) {
    ggml_backend_metal_reg_t ctx = নতুন কাঠামো ggml_backend_metal_reg;

    ফেরত ctx;
}

স্থির শূন্য ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

কাঠামো ggml_backend_metal_reg_deleter {
    শূন্য operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

প্রকারলেখ std::unique_ptr<কাঠামো ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    ফেরত GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_metal_reg_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    ফেরত ctx->devices.size();
}

স্থির ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t সূচক) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(সূচক < ctx->devices.size());
    ফেরত ctx->devices[সূচক];
}

স্থির ggml_backend_feature g_ggml_backend_metal_features[] = {
#যদি defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { ফাঁকা, ফাঁকা },
};

স্থির ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    ফেরত g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
স্থির শূন্য ggml_backend_metal_tuning_set_fa_vec_override(পূর্ণসংখ্যা Q, পূর্ণসংখ্যা NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

স্থির শূন্য ggml_backend_metal_tuning_clear_fa_vec_override(শূন্য) {
    ggml_metal_tuning::fa_vec_clear_override();
}

স্থির পূর্ণসংখ্যা ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    ফেরত ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

স্থির পূর্ণসংখ্যা ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    ফেরত ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

স্থির পূর্ণসংখ্যা ggml_backend_metal_tuning_fa_vec_baseline_ne(পূর্ণসংখ্যা dk, পূর্ণসংখ্যা dv) {
    ফেরত ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

স্থির ধ্রুবক অক্ষর * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ফেরত ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
প্রকারলেখ শূন্য * ggml_backend_fusion_t;

স্থির ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    ফেরত ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

স্থির শূন্য ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((কাঠামো ggml_metal_fusion_info *) finfo);
}

স্থির শূন্য ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((কাঠামো ggml_metal_fusion_info *) finfo);
}

স্থির পূর্ণসংখ্যা ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ধ্রুবক অক্ষর ** labels, uint64_t * counts, পূর্ণসংখ্যা n) {
    ফেরত ggml_metal_fusion_info_stats_get((কাঠামো ggml_metal_fusion_info *) finfo, labels, counts, n);
}

স্থির শূন্য ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, যৌক্তিক enabled) {
    ggml_metal_fusion_info_set_enabled((কাঠামো ggml_metal_fusion_info *) finfo, enabled);
}

স্থির শূন্য * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    যদি (strcmp(name, "ggml_backend_get_features") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_get_features;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    যদি (strcmp(name, "ggml_backend_fusion_get") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_fusion_get;
    }
    যদি (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_fusion_stats_init;
    }
    যদি (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_fusion_stats_reset;
    }
    যদি (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_fusion_stats_get;
    }
    যদি (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        ফেরত (শূন্য *)ggml_backend_metal_fusion_set_enabled;
    }

    ফেরত ফাঁকা;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_গুনতি,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

স্থির ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, পূর্ণসংখ্যা device) {
    ফেরত নতুন ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

স্থির শূন্য ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

কাঠামো ggml_backend_device_deleter {
    শূন্য operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

প্রকারলেখ std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(শূন্য) {
    স্থির ggml_backend_reg reg;
    স্থির যৌক্তিক initialized = মিথ্যা;

    {
        স্থির std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ধ্রুবক অক্ষর * env = getenv("GGML_METAL_DEVICES");
        যদি (env) {
            g_devices = atoi(env);
        }

        স্থির std::vector<ggml_backend_device_ptr> devs;

        যদি (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", সত্য);

            স্থির ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            জন্য (পূর্ণসংখ্যা i = 0; i < g_devices; ++i) {
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

        initialized = সত্য;
    }

    ফেরত &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
