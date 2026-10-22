#અન્તર્ભાવ "ggml-metal.h"

#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ "ggml-metal-device.h"
#અન્તર્ભાવ "ggml-metal-fusion.h"
#અન્તર્ભાવ "ggml-metal-context.h"
#અન્તર્ભાવ "ggml-metal-ops.h"
#અન્તર્ભાવ "ggml-metal-tuning.h"

#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
સ્થિર પૂર્ણાંક g_devices = 1;

// forward declaration
સ્થિર તાર્કિક ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

સ્થિર શૂન્ય ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

સ્થિર શૂન્ય * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    પાછા ggml_metal_buffer_get_base(ctx);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

સ્થિર તાર્કિક ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    જો (!ggml_backend_buffer_is_metal(src->buffer)) {
        પાછા ખોટું;
    }

    પાછા ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

સ્થિર ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ ખાલી,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ ખાલી,
    /* .get_tensor_2d = */ ખાલી,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ ખાલી,
};

// private buffer

સ્થિર શૂન્ય ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

સ્થિર શૂન્ય * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    પાછા ggml_metal_buffer_get_base(ctx);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

સ્થિર તાર્કિક ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    જો (!ggml_backend_buffer_is_metal(src->buffer)) {
        પાછા ખોટું;
    }

    પાછા ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

સ્થિર શૂન્ય ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

સ્થિર ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ ખાલી,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ ખાલી,
    /* .get_tensor_2d = */ ખાલી,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ ખાલી,
};

સ્થિર તાર્કિક ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    પાછા buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

સંરચના ggml_backend_metal_buffer_type {
    પૂર્ણાંક device;
    std::string name;
};

સંરચના ggml_backend_metal_buffer_type_deleter {
    શૂન્ય operator()(ggml_backend_metal_buffer_type * ctx) અચલ {
        delete ctx;
    }
};

પ્રકારવ્યાખ્યા std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
સ્થિર ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, તાર્કિક shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    જો (res == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        પાછા ખાલી;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    પાછા ggml_backend_buffer_init(buft, buf_i, res, size);
}

સ્થિર size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    પસંદગી (tensor->op) {
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CUMSUM:
        સ્થિતિ GGML_OP_ARGSORT:
            {
                res *= 2;
            } તોડો;
        સ્થિતિ GGML_OP_ટોચ_K:
            {
                res = 2*કદ(int32_t)*ggml_nelements(tensor->src[0]);
            } તોડો;
        સ્થિતિ GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } તોડો;
        મૂળભૂત:
            તોડો;
    }

    પાછા res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

સ્થિર અચલ અક્ષર * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    પાછા ctx->name.c_str();
}

સ્થિર ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    પાછા ggml_backend_metal_buffer_type_alloc_buffer(buft, size, સાચું);
}

સ્થિર size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા 32;

    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    પાછા ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

સ્થિર size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    પાછા ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

સ્થિર તાર્કિક ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    પાછા ખોટું;

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(પૂર્ણાંક device) {
    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    સ્થિર std::vector<ggml_backend_buffer_type> bufts;
    સ્થિર std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    સ્થિર તાર્કિક initialized = ખોટું;
    જો (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        માટે (પૂર્ણાંક i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                નવું ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ ખાલી,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ ખાલી,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = સાચું;
    }

    પાછા &bufts[device];
}

// default (private) buffer type

સ્થિર અચલ અક્ષર * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    પાછા ctx->name.c_str();
}

સ્થિર ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    પાછા ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ખોટું);
}

સ્થિર size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા 32;

    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    પાછા ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

સ્થિર size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    પાછા ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

સ્થિર તાર્કિક ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    પાછા ખોટું;

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(પૂર્ણાંક device) {
    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    સ્થિર std::vector<ggml_backend_buffer_type> bufts;
    સ્થિર std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    સ્થિર તાર્કિક initialized = ખોટું;
    જો (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        માટે (પૂર્ણાંક i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = નવું ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ ખાલી,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ ખાલી,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = સાચું;
    }

    પાછા &bufts[device];
}

// mapped buffer type

સ્થિર અચલ અક્ષર * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    પાછા ctx->name.c_str();
}

સ્થિર ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    પાછા ggml_backend_metal_buffer_type_alloc_buffer(buft, size, સાચું);
}

સ્થિર size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા 32;

    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    પાછા ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

સ્થિર size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    પાછા ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

સ્થિર તાર્કિક ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    પાછા ખોટું;

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(પૂર્ણાંક device) {
    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    સ્થિર std::vector<ggml_backend_buffer_type> bufts;
    સ્થિર std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    સ્થિર તાર્કિક initialized = ખોટું;
    જો (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        માટે (પૂર્ણાંક i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = નવું ggml_backend_metal_buffer_type{
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
                    /* .alloc_buffer_n      = */ ખાલી,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ ખાલી,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = સાચું;
    }

    પાછા &bufts[device];
}

// backend

સ્થિર અચલ અક્ષર * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    પાછા ggml_metal_get_name(ctx);
}

સ્થિર શૂન્ય ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    મુક્ત(backend);
}

સ્થિર શૂન્ય ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

સ્થિર શૂન્ય ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

સ્થિર શૂન્ય ggml_backend_metal_get_tensor_async(ggml_backend_t backend, અચલ ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

સ્થિર તાર્કિક ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, અચલ ggml_tensor * src, ggml_tensor * dst) {
    જો (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        પાછા ખોટું;
    }

    જો (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        પાછા ખોટું;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    પાછા ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

સ્થિર ગણના ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    પાછા ggml_metal_graph_compute(ctx, cgraph);
}

સ્થિર શૂન્ય ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

સ્થિર શૂન્ય ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

સ્થિર શૂન્ય ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

સ્થિર શૂન્ય ggml_backend_metal_set_n_cb(ggml_backend_t backend, પૂર્ણાંક n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

સ્થિર ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

સ્થિર ggml_guid_t ggml_backend_metal_guid(શૂન્ય) {
    સ્થિર ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    પાછા &guid;
}

ggml_backend_t ggml_backend_metal_init(શૂન્ય) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    જો (ctx == ખાલી) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        પાછા ખાલી;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(કદ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    પાછા backend;
}

તાર્કિક ggml_backend_is_metal(ggml_backend_t backend) {
    પાછા backend != ખાલી && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

શૂન્ય ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, શૂન્ય * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

તાર્કિક ggml_backend_metal_supports_family(ggml_backend_t backend, પૂર્ણાંક family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    પાછા ggml_metal_supports_family(ctx, family);
}

શૂન્ય ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

સ્થિર અચલ અક્ષર * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    અચલ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    પાછા props_dev->name;
}

સ્થિર અચલ અક્ષર * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    પાછા ggml_metal_device_get_props(ctx_dev)->desc;
}

સ્થિર શૂન્ય ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, મુક્ત, total);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    પાછા GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ સાચું,
        /* .host_buffer          = */ ખોટું,
        /* .buffer_from_host_ptr = */ સાચું,
        /* .events               = */ સાચું,
        /* .mmap_support         = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    જો (ctx == ખાલી) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        પાછા ખાલી;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(કદ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    પાછા backend;

    GGML_UNUSED(params);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    અચલ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    પાછા props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

સ્થિર ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, શૂન્ય * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    અચલ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    પાછા ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

સ્થિર તાર્કિક ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    પાછા ggml_metal_device_supports_op(ctx_dev, op);
}

સ્થિર તાર્કિક ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    પાછા
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

સ્થિર int64_t get_op_batch_size(અચલ ggml_tensor * op) {
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_MUL_MAT:
            પાછા op->ne[1];
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            પાછા op->ne[2];
        મૂળભૂત:
            પાછા ggml_nrows(op);
    }
}

સ્થિર તાર્કિક ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    પાછા (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

સ્થિર ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = નવું ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    પાછા ev;
}

સ્થિર શૂન્ય ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

સ્થિર શૂન્ય ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

સ્થિર ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

સંરચના ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

પ્રકારવ્યાખ્યા સંરચના ggml_backend_metal_reg * ggml_backend_metal_reg_t;

સ્થિર ggml_backend_metal_reg_t ggml_backend_metal_reg_init(શૂન્ય) {
    ggml_backend_metal_reg_t ctx = નવું સંરચના ggml_backend_metal_reg;

    પાછા ctx;
}

સ્થિર શૂન્ય ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

સંરચના ggml_backend_metal_reg_deleter {
    શૂન્ય operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

પ્રકારવ્યાખ્યા std::unique_ptr<સંરચના ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

સ્થિર અચલ અક્ષર * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    પાછા GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

સ્થિર size_t ggml_backend_metal_reg_device_ગણતરી(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    પાછા ctx->devices.size();
}

સ્થિર ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    પાછા ctx->devices[index];
}

સ્થિર ggml_backend_feature g_ggml_backend_metal_features[] = {
#જો defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { ખાલી, ખાલી },
};

સ્થિર ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    પાછા g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
સ્થિર શૂન્ય ggml_backend_metal_tuning_set_fa_vec_override(પૂર્ણાંક Q, પૂર્ણાંક NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

સ્થિર શૂન્ય ggml_backend_metal_tuning_clear_fa_vec_override(શૂન્ય) {
    ggml_metal_tuning::fa_vec_clear_override();
}

સ્થિર પૂર્ણાંક ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    પાછા ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

સ્થિર પૂર્ણાંક ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    પાછા ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

સ્થિર પૂર્ણાંક ggml_backend_metal_tuning_fa_vec_baseline_ne(પૂર્ણાંક dk, પૂર્ણાંક dv) {
    પાછા ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

સ્થિર અચલ અક્ષર * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    પાછા ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
પ્રકારવ્યાખ્યા શૂન્ય * ggml_backend_fusion_t;

સ્થિર ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    પાછા ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

સ્થિર શૂન્ય ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((સંરચના ggml_metal_fusion_info *) finfo);
}

સ્થિર શૂન્ય ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((સંરચના ggml_metal_fusion_info *) finfo);
}

સ્થિર પૂર્ણાંક ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, અચલ અક્ષર ** labels, uint64_t * counts, પૂર્ણાંક n) {
    પાછા ggml_metal_fusion_info_stats_get((સંરચના ggml_metal_fusion_info *) finfo, labels, counts, n);
}

સ્થિર શૂન્ય ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, તાર્કિક enabled) {
    ggml_metal_fusion_info_set_enabled((સંરચના ggml_metal_fusion_info *) finfo, enabled);
}

સ્થિર શૂન્ય * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    જો (strcmp(name, "ggml_backend_get_features") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_get_features;
    }
    જો (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    જો (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    જો (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    જો (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    જો (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    જો (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    જો (strcmp(name, "ggml_backend_fusion_get") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_fusion_get;
    }
    જો (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_fusion_stats_init;
    }
    જો (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_fusion_stats_reset;
    }
    જો (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_fusion_stats_get;
    }
    જો (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        પાછા (શૂન્ય *)ggml_backend_metal_fusion_set_enabled;
    }

    પાછા ખાલી;

    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_ગણતરી,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

સ્થિર ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, પૂર્ણાંક device) {
    પાછા નવું ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

સ્થિર શૂન્ય ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

સંરચના ggml_backend_device_deleter {
    શૂન્ય operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

પ્રકારવ્યાખ્યા std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(શૂન્ય) {
    સ્થિર ggml_backend_reg reg;
    સ્થિર તાર્કિક initialized = ખોટું;

    {
        સ્થિર std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        અચલ અક્ષર * env = getenv("GGML_METAL_DEVICES");
        જો (env) {
            g_devices = atoi(env);
        }

        સ્થિર std::vector<ggml_backend_device_ptr> devs;

        જો (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", સાચું);

            સ્થિર ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            માટે (પૂર્ણાંક i = 0; i < g_devices; ++i) {
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

        initialized = સાચું;
    }

    પાછા &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
