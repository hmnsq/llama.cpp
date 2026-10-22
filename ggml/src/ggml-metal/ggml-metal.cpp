#അന്തര്ഭാവ "ggml-metal.h"

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ "ggml-metal-device.h"
#അന്തര്ഭാവ "ggml-metal-fusion.h"
#അന്തര്ഭാവ "ggml-metal-context.h"
#അന്തര്ഭാവ "ggml-metal-ops.h"
#അന്തര്ഭാവ "ggml-metal-tuning.h"

#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
സ്ഥിതം പൂർണ്ണസംഖ്യ g_devices = 1;

// forward declaration
സ്ഥിതം തർക്കം ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

സ്ഥിതം ശൂന്യം * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    തിരികെ ggml_metal_buffer_get_base(ctx);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

സ്ഥിതം തർക്കം ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    എങ്കിൽ (!ggml_backend_buffer_is_metal(src->buffer)) {
        തിരികെ അസത്യം;
    }

    തിരികെ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

സ്ഥിതം ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ ശൂന്യ,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ ശൂന്യ,
    /* .get_tensor_2d = */ ശൂന്യ,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ ശൂന്യ,
};

// private buffer

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

സ്ഥിതം ശൂന്യം * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    തിരികെ ggml_metal_buffer_get_base(ctx);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

സ്ഥിതം തർക്കം ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    എങ്കിൽ (!ggml_backend_buffer_is_metal(src->buffer)) {
        തിരികെ അസത്യം;
    }

    തിരികെ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

സ്ഥിതം ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ ശൂന്യ,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ ശൂന്യ,
    /* .get_tensor_2d = */ ശൂന്യ,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ ശൂന്യ,
};

സ്ഥിതം തർക്കം ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    തിരികെ buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

ഘടന ggml_backend_metal_buffer_type {
    പൂർണ്ണസംഖ്യ device;
    std::string name;
};

ഘടന ggml_backend_metal_buffer_type_deleter {
    ശൂന്യം operator()(ggml_backend_metal_buffer_type * ctx) സ്ഥിരം {
        delete ctx;
    }
};

തരനിർവചനം std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
സ്ഥിതം ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, തർക്കം shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    എങ്കിൽ (res == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        തിരികെ ശൂന്യ;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    തിരികെ ggml_backend_buffer_init(buft, buf_i, res, size);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    തിരഞ്ഞെടുപ്പ് (tensor->op) {
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CUMSUM:
        അവസ്ഥ GGML_OP_ARGSORT:
            {
                res *= 2;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_മുകൾ_K:
            {
                res = 2*വലുപ്പം(int32_t)*ggml_nelements(tensor->src[0]);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }

    തിരികെ res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    തിരികെ ctx->name.c_str();
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    തിരികെ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, സത്യം);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ 32;

    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    തിരികെ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

സ്ഥിതം തർക്കം ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ അസത്യം;

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(പൂർണ്ണസംഖ്യ device) {
    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    സ്ഥിതം std::vector<ggml_backend_buffer_type> bufts;
    സ്ഥിതം std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    സ്ഥിതം തർക്കം initialized = അസത്യം;
    എങ്കിൽ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                പുതിയ ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ ശൂന്യ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ ശൂന്യ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = സത്യം;
    }

    തിരികെ &bufts[device];
}

// default (private) buffer type

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    തിരികെ ctx->name.c_str();
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    തിരികെ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, അസത്യം);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ 32;

    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    തിരികെ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

സ്ഥിതം തർക്കം ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ അസത്യം;

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(പൂർണ്ണസംഖ്യ device) {
    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    സ്ഥിതം std::vector<ggml_backend_buffer_type> bufts;
    സ്ഥിതം std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    സ്ഥിതം തർക്കം initialized = അസത്യം;
    എങ്കിൽ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = പുതിയ ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ ശൂന്യ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ ശൂന്യ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = സത്യം;
    }

    തിരികെ &bufts[device];
}

// mapped buffer type

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    തിരികെ ctx->name.c_str();
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    തിരികെ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, സത്യം);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ 32;

    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    തിരികെ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

സ്ഥിതം size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

സ്ഥിതം തർക്കം ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ അസത്യം;

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(പൂർണ്ണസംഖ്യ device) {
    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    സ്ഥിതം std::vector<ggml_backend_buffer_type> bufts;
    സ്ഥിതം std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    സ്ഥിതം തർക്കം initialized = അസത്യം;
    എങ്കിൽ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = പുതിയ ggml_backend_metal_buffer_type{
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
                    /* .alloc_buffer_n      = */ ശൂന്യ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ ശൂന്യ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = സത്യം;
    }

    തിരികെ &bufts[device];
}

// backend

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    തിരികെ ggml_metal_get_name(ctx);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    സ്വതന്ത്രം(backend);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_get_tensor_async(ggml_backend_t backend, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

സ്ഥിതം തർക്കം ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    എങ്കിൽ (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        തിരികെ അസത്യം;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    തിരികെ ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    തിരികെ ggml_metal_graph_compute(ctx, cgraph);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_set_n_cb(ggml_backend_t backend, പൂർണ്ണസംഖ്യ n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

സ്ഥിതം ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ശൂന്യ,
    /* .get_tensor_2d_async     = */ ശൂന്യ,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

സ്ഥിതം ggml_guid_t ggml_backend_metal_guid(ശൂന്യം) {
    സ്ഥിതം ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    തിരികെ &guid;
}

ggml_backend_t ggml_backend_metal_init(ശൂന്യം) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    എങ്കിൽ (ctx == ശൂന്യ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        തിരികെ ശൂന്യ;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(വലുപ്പം(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    തിരികെ backend;
}

തർക്കം ggml_backend_is_metal(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

ശൂന്യം ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, ശൂന്യം * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

തർക്കം ggml_backend_metal_supports_family(ggml_backend_t backend, പൂർണ്ണസംഖ്യ family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    തിരികെ ggml_metal_supports_family(ctx, family);
}

ശൂന്യം ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    സ്ഥിരം ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    തിരികെ props_dev->name;
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    തിരികെ ggml_metal_device_get_props(ctx_dev)->desc;
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, സ്വതന്ത്രം, total);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    തിരികെ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ സത്യം,
        /* .host_buffer          = */ അസത്യം,
        /* .buffer_from_host_ptr = */ സത്യം,
        /* .events               = */ സത്യം,
        /* .mmap_support         = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    എങ്കിൽ (ctx == ശൂന്യ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        തിരികെ ശൂന്യ;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(വലുപ്പം(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    തിരികെ backend;

    GGML_UNUSED(params);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    സ്ഥിരം ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    തിരികെ props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, ശൂന്യം * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    സ്ഥിരം ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    തിരികെ ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

സ്ഥിതം തർക്കം ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    തിരികെ ggml_metal_device_supports_op(ctx_dev, op);
}

സ്ഥിതം തർക്കം ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    തിരികെ
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

സ്ഥിതം int64_t get_op_batch_size(സ്ഥിരം ggml_tensor * op) {
    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_MUL_MAT:
            തിരികെ op->ne[1];
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            തിരികെ op->ne[2];
        സ്ഥിരസ്ഥിതി:
            തിരികെ ggml_nrows(op);
    }
}

സ്ഥിതം തർക്കം ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    തിരികെ (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

സ്ഥിതം ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = പുതിയ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    തിരികെ ev;
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

സ്ഥിതം ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ ശൂന്യ,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ഘടന ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

തരനിർവചനം ഘടന ggml_backend_metal_reg * ggml_backend_metal_reg_t;

സ്ഥിതം ggml_backend_metal_reg_t ggml_backend_metal_reg_init(ശൂന്യം) {
    ggml_backend_metal_reg_t ctx = പുതിയ ഘടന ggml_backend_metal_reg;

    തിരികെ ctx;
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

ഘടന ggml_backend_metal_reg_deleter {
    ശൂന്യം operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

തരനിർവചനം std::unique_ptr<ഘടന ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    തിരികെ GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

സ്ഥിതം size_t ggml_backend_metal_reg_device_എണ്ണം(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    തിരികെ ctx->devices.size();
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    തിരികെ ctx->devices[index];
}

സ്ഥിതം ggml_backend_feature g_ggml_backend_metal_features[] = {
#എങ്കിൽ defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { ശൂന്യ, ശൂന്യ },
};

സ്ഥിതം ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    തിരികെ g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
സ്ഥിതം ശൂന്യം ggml_backend_metal_tuning_set_fa_vec_override(പൂർണ്ണസംഖ്യ Q, പൂർണ്ണസംഖ്യ NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_tuning_clear_fa_vec_override(ശൂന്യം) {
    ggml_metal_tuning::fa_vec_clear_override();
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    തിരികെ ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    തിരികെ ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_metal_tuning_fa_vec_baseline_ne(പൂർണ്ണസംഖ്യ dk, പൂർണ്ണസംഖ്യ dv) {
    തിരികെ ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    തിരികെ ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
തരനിർവചനം ശൂന്യം * ggml_backend_fusion_t;

സ്ഥിതം ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    തിരികെ ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ഘടന ggml_metal_fusion_info *) finfo);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ഘടന ggml_metal_fusion_info *) finfo);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, സ്ഥിരം അക്ഷരം ** labels, uint64_t * counts, പൂർണ്ണസംഖ്യ n) {
    തിരികെ ggml_metal_fusion_info_stats_get((ഘടന ggml_metal_fusion_info *) finfo, labels, counts, n);
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, തർക്കം enabled) {
    ggml_metal_fusion_info_set_enabled((ഘടന ggml_metal_fusion_info *) finfo, enabled);
}

സ്ഥിതം ശൂന്യം * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    എങ്കിൽ (strcmp(name, "ggml_backend_get_features") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_get_features;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    എങ്കിൽ (strcmp(name, "ggml_backend_fusion_get") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_fusion_get;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_fusion_stats_init;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_fusion_stats_reset;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_fusion_stats_get;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_metal_fusion_set_enabled;
    }

    തിരികെ ശൂന്യ;

    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

സ്ഥിതം ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, പൂർണ്ണസംഖ്യ device) {
    തിരികെ പുതിയ ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

സ്ഥിതം ശൂന്യം ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

ഘടന ggml_backend_device_deleter {
    ശൂന്യം operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

തരനിർവചനം std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(ശൂന്യം) {
    സ്ഥിതം ggml_backend_reg reg;
    സ്ഥിതം തർക്കം initialized = അസത്യം;

    {
        സ്ഥിതം std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        സ്ഥിരം അക്ഷരം * env = getenv("GGML_METAL_DEVICES");
        എങ്കിൽ (env) {
            g_devices = atoi(env);
        }

        സ്ഥിതം std::vector<ggml_backend_device_ptr> devs;

        എങ്കിൽ (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", സത്യം);

            സ്ഥിതം ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < g_devices; ++i) {
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

        initialized = സത്യം;
    }

    തിരികെ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
