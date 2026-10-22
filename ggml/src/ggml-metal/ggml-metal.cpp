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
স্থির পূর্ণাঙ্ক g_devices = 1;

// forward declaration
স্থির তার্কিক ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

স্থির শূন্য ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

স্থির শূন্য * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    প্রত্যাগম ggml_metal_buffer_get_base(ctx);
}

স্থির শূন্য ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

স্থির শূন্য ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুব শূন্য * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

স্থির শূন্য ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, ধ্রুব ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

স্থির তার্কিক ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, ধ্রুব ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    যদি (!ggml_backend_buffer_is_metal(src->buffer)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

স্থির শূন্য ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

স্থির ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ রিক্ত,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ রিক্ত,
    /* .get_tensor_2d = */ রিক্ত,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ রিক্ত,
};

// private buffer

স্থির শূন্য ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

স্থির শূন্য * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    প্রত্যাগম ggml_metal_buffer_get_base(ctx);
}

স্থির শূন্য ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

স্থির শূন্য ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুব শূন্য * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

স্থির শূন্য ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, ধ্রুব ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

স্থির তার্কিক ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, ধ্রুব ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    যদি (!ggml_backend_buffer_is_metal(src->buffer)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

স্থির শূন্য ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

স্থির ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ রিক্ত,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ রিক্ত,
    /* .get_tensor_2d = */ রিক্ত,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ রিক্ত,
};

স্থির তার্কিক ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    প্রত্যাগম buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

সংরচনা ggml_backend_metal_buffer_type {
    পূর্ণাঙ্ক device;
    std::string name;
};

সংরচনা ggml_backend_metal_buffer_type_deleter {
    শূন্য operator()(ggml_backend_metal_buffer_type * ctx) ধ্রুব {
        delete ctx;
    }
};

প্রকারনির্দেশ std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
স্থির ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, তার্কিক shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    যদি (res == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        প্রত্যাগম রিক্ত;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    প্রত্যাগম ggml_backend_buffer_init(buft, buf_i, res, size);
}

স্থির size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুব ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    বিকল্প (tensor->op) {
        পক্ষ GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CUMSUM:
        পক্ষ GGML_OP_ARGSORT:
            {
                res *= 2;
            } ভঙ্গ;
        পক্ষ GGML_OP_শীর্ষ_K:
            {
                res = 2*পরিমাণ(int32_t)*ggml_nelements(tensor->src[0]);
            } ভঙ্গ;
        পক্ষ GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } ভঙ্গ;
        স্বভাব:
            ভঙ্গ;
    }

    প্রত্যাগম res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

স্থির ধ্রুব বর্ণ * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    প্রত্যাগম ctx->name.c_str();
}

স্থির ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    প্রত্যাগম ggml_backend_metal_buffer_type_alloc_buffer(buft, size, সত্য);
}

স্থির size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম 32;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    প্রত্যাগম ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

স্থির size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুব ggml_tensor * tensor) {
    প্রত্যাগম ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

স্থির তার্কিক ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম অসত্য;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(পূর্ণাঙ্ক device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    স্থির std::vector<ggml_backend_buffer_type> bufts;
    স্থির std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    স্থির তার্কিক initialized = অসত্য;
    যদি (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        প্রতি (পূর্ণাঙ্ক i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                নূতন ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ রিক্ত,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ রিক্ত,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = সত্য;
    }

    প্রত্যাগম &bufts[device];
}

// default (private) buffer type

স্থির ধ্রুব বর্ণ * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    প্রত্যাগম ctx->name.c_str();
}

স্থির ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    প্রত্যাগম ggml_backend_metal_buffer_type_alloc_buffer(buft, size, অসত্য);
}

স্থির size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম 32;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    প্রত্যাগম ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

স্থির size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুব ggml_tensor * tensor) {
    প্রত্যাগম ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

স্থির তার্কিক ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম অসত্য;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(পূর্ণাঙ্ক device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    স্থির std::vector<ggml_backend_buffer_type> bufts;
    স্থির std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    স্থির তার্কিক initialized = অসত্য;
    যদি (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        প্রতি (পূর্ণাঙ্ক i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = নূতন ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ রিক্ত,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ রিক্ত,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = সত্য;
    }

    প্রত্যাগম &bufts[device];
}

// mapped buffer type

স্থির ধ্রুব বর্ণ * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    প্রত্যাগম ctx->name.c_str();
}

স্থির ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    প্রত্যাগম ggml_backend_metal_buffer_type_alloc_buffer(buft, size, সত্য);
}

স্থির size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম 32;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    প্রত্যাগম ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

স্থির size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুব ggml_tensor * tensor) {
    প্রত্যাগম ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

স্থির তার্কিক ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম অসত্য;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(পূর্ণাঙ্ক device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    স্থির std::vector<ggml_backend_buffer_type> bufts;
    স্থির std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    স্থির তার্কিক initialized = অসত্য;
    যদি (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        প্রতি (পূর্ণাঙ্ক i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = নূতন ggml_backend_metal_buffer_type{
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
                    /* .alloc_buffer_n      = */ রিক্ত,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ রিক্ত,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = সত্য;
    }

    প্রত্যাগম &bufts[device];
}

// backend

স্থির ধ্রুব বর্ণ * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    প্রত্যাগম ggml_metal_get_name(ctx);
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

স্থির শূন্য ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ধ্রুব শূন্য * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

স্থির শূন্য ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ধ্রুব ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

স্থির তার্কিক ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুব ggml_tensor * src, ggml_tensor * dst) {
    যদি (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        প্রত্যাগম অসত্য;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    প্রত্যাগম ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

স্থির গণনা ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    প্রত্যাগম ggml_metal_graph_compute(ctx, cgraph);
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
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

স্থির শূন্য ggml_backend_metal_set_n_cb(ggml_backend_t backend, পূর্ণাঙ্ক n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

স্থির ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ রিক্ত,
    /* .get_tensor_2d_async     = */ রিক্ত,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

স্থির ggml_guid_t ggml_backend_metal_guid(শূন্য) {
    স্থির ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    প্রত্যাগম &guid;
}

ggml_backend_t ggml_backend_metal_init(শূন্য) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    যদি (ctx == রিক্ত) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        প্রত্যাগম রিক্ত;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(পরিমাণ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    প্রত্যাগম backend;
}

তার্কিক ggml_backend_is_metal(ggml_backend_t backend) {
    প্রত্যাগম backend != রিক্ত && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

শূন্য ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, শূন্য * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

তার্কিক ggml_backend_metal_supports_family(ggml_backend_t backend, পূর্ণাঙ্ক family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    প্রত্যাগম ggml_metal_supports_family(ctx, family);
}

শূন্য ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

স্থির ধ্রুব বর্ণ * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ধ্রুব ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    প্রত্যাগম props_dev->name;
}

স্থির ধ্রুব বর্ণ * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    প্রত্যাগম ggml_metal_device_get_props(ctx_dev)->desc;
}

স্থির শূন্য ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, মুক্ত, total);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    প্রত্যাগম GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ সত্য,
        /* .host_buffer          = */ অসত্য,
        /* .buffer_from_host_ptr = */ সত্য,
        /* .events               = */ সত্য,
        /* .mmap_support         = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ধ্রুব বর্ণ * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    যদি (ctx == রিক্ত) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        প্রত্যাগম রিক্ত;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(পরিমাণ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    প্রত্যাগম backend;

    GGML_UNUSED(params);
}

স্থির ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ধ্রুব ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    প্রত্যাগম props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

স্থির ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    ধ্রুব ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    প্রত্যাগম ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

স্থির তার্কিক ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ধ্রুব ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    প্রত্যাগম ggml_metal_device_supports_op(ctx_dev, op);
}

স্থির তার্কিক ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    প্রত্যাগম
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

স্থির int64_t get_op_batch_size(ধ্রুব ggml_tensor * op) {
    বিকল্প (op->op) {
        পক্ষ GGML_OP_MUL_MAT:
            প্রত্যাগম op->ne[1];
        পক্ষ GGML_OP_MUL_MAT_ID:
            প্রত্যাগম op->ne[2];
        স্বভাব:
            প্রত্যাগম ggml_nrows(op);
    }
}

স্থির তার্কিক ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ধ্রুব ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    প্রত্যাগম (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

স্থির ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = নূতন ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    প্রত্যাগম ev;
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
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ রিক্ত,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

সংরচনা ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

প্রকারনির্দেশ সংরচনা ggml_backend_metal_reg * ggml_backend_metal_reg_t;

স্থির ggml_backend_metal_reg_t ggml_backend_metal_reg_init(শূন্য) {
    ggml_backend_metal_reg_t ctx = নূতন সংরচনা ggml_backend_metal_reg;

    প্রত্যাগম ctx;
}

স্থির শূন্য ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

সংরচনা ggml_backend_metal_reg_deleter {
    শূন্য operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

প্রকারনির্দেশ std::unique_ptr<সংরচনা ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

স্থির ধ্রুব বর্ণ * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    প্রত্যাগম GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_metal_reg_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    প্রত্যাগম ctx->devices.size();
}

স্থির ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    প্রত্যাগম ctx->devices[index];
}

স্থির ggml_backend_feature g_ggml_backend_metal_features[] = {
#যদি defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { রিক্ত, রিক্ত },
};

স্থির ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    প্রত্যাগম g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
স্থির শূন্য ggml_backend_metal_tuning_set_fa_vec_override(পূর্ণাঙ্ক Q, পূর্ণাঙ্ক NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

স্থির শূন্য ggml_backend_metal_tuning_clear_fa_vec_override(শূন্য) {
    ggml_metal_tuning::fa_vec_clear_override();
}

স্থির পূর্ণাঙ্ক ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    প্রত্যাগম ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

স্থির পূর্ণাঙ্ক ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    প্রত্যাগম ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

স্থির পূর্ণাঙ্ক ggml_backend_metal_tuning_fa_vec_baseline_ne(পূর্ণাঙ্ক dk, পূর্ণাঙ্ক dv) {
    প্রত্যাগম ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

স্থির ধ্রুব বর্ণ * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    প্রত্যাগম ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
প্রকারনির্দেশ শূন্য * ggml_backend_fusion_t;

স্থির ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    প্রত্যাগম ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

স্থির শূন্য ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((সংরচনা ggml_metal_fusion_info *) finfo);
}

স্থির শূন্য ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((সংরচনা ggml_metal_fusion_info *) finfo);
}

স্থির পূর্ণাঙ্ক ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ধ্রুব বর্ণ ** labels, uint64_t * counts, পূর্ণাঙ্ক n) {
    প্রত্যাগম ggml_metal_fusion_info_stats_get((সংরচনা ggml_metal_fusion_info *) finfo, labels, counts, n);
}

স্থির শূন্য ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, তার্কিক enabled) {
    ggml_metal_fusion_info_set_enabled((সংরচনা ggml_metal_fusion_info *) finfo, enabled);
}

স্থির শূন্য * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, ধ্রুব বর্ণ * name) {
    যদি (strcmp(name, "ggml_backend_get_features") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_get_features;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    যদি (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    যদি (strcmp(name, "ggml_backend_fusion_get") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_fusion_get;
    }
    যদি (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_fusion_stats_init;
    }
    যদি (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_fusion_stats_reset;
    }
    যদি (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_fusion_stats_get;
    }
    যদি (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_metal_fusion_set_enabled;
    }

    প্রত্যাগম রিক্ত;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_গুনতি,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

স্থির ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, পূর্ণাঙ্ক device) {
    প্রত্যাগম নূতন ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

স্থির শূন্য ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

সংরচনা ggml_backend_device_deleter {
    শূন্য operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

প্রকারনির্দেশ std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(শূন্য) {
    স্থির ggml_backend_reg reg;
    স্থির তার্কিক initialized = অসত্য;

    {
        স্থির std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ধ্রুব বর্ণ * env = getenv("GGML_METAL_DEVICES");
        যদি (env) {
            g_devices = atoi(env);
        }

        স্থির std::vector<ggml_backend_device_ptr> devs;

        যদি (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", সত্য);

            স্থির ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            প্রতি (পূর্ণাঙ্ক i = 0; i < g_devices; ++i) {
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

    প্রত্যাগম &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
