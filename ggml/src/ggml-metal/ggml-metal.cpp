#అన్తర్భావ "ggml-metal.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ "ggml-metal-device.h"
#అన్తర్భావ "ggml-metal-fusion.h"
#అన్తర్భావ "ggml-metal-context.h"
#అన్తర్భావ "ggml-metal-ops.h"
#అన్తర్భావ "ggml-metal-tuning.h"

#అన్తర్భావ <mutex>
#అన్తర్భావ <string>

#నిర్వచించు GGML_METAL_NAME "MTL"
#నిర్వచించు GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
స్థిర పూర్ణాంకం g_devices = 1;

// forward declaration
స్థిర తార్కిక ggml_backend_బఫరు_is_metal(ggml_backend_బఫరు_t బఫరు);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

స్థిర శూన్య ggml_backend_metal_బఫరు_shared_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_free(ctx);
}

స్థిర శూన్య * ggml_backend_metal_బఫరు_shared_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    తిరిగి ggml_metal_బఫరు_get_పునాది(ctx);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_shared_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_memset_tensor(ctx, tensor, value, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_shared_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_సమితి_tensor(ctx, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_shared_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_get_tensor(ctx, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర తార్కిక ggml_backend_metal_బఫరు_shared_cpy_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    ఒకవేళ (!ggml_backend_బఫరు_is_metal(src->బఫరు)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి ggml_metal_బఫరు_cpy_tensor(ctx, src, dst);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_shared_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_clear(ctx, value);
}

స్థిర ggml_backend_బఫరు_i ggml_backend_metal_బఫరు_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_బఫరు_shared_free_బఫరు,
    /* .get_base      = */ ggml_backend_metal_బఫరు_shared_get_పునాది,
    /* .init_tensor   = */ ఖాళీ,
    /* .memset_tensor = */ ggml_backend_metal_బఫరు_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_బఫరు_shared_సమితి_tensor,
    /* .get_tensor    = */ ggml_backend_metal_బఫరు_shared_get_tensor,
    /* .set_tensor_2d = */ ఖాళీ,
    /* .get_tensor_2d = */ ఖాళీ,
    /* .cpy_tensor    = */ ggml_backend_metal_బఫరు_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_బఫరు_shared_clear,
    /* .reset         = */ ఖాళీ,
};

// private buffer

స్థిర శూన్య ggml_backend_metal_బఫరు_వ్యక్తిగత_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_free(ctx);
}

స్థిర శూన్య * ggml_backend_metal_బఫరు_వ్యక్తిగత_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    తిరిగి ggml_metal_బఫరు_get_పునాది(ctx);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_వ్యక్తిగత_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_memset_tensor(ctx, tensor, value, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_వ్యక్తిగత_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_సమితి_tensor(ctx, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_వ్యక్తిగత_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_get_tensor(ctx, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర తార్కిక ggml_backend_metal_బఫరు_వ్యక్తిగత_cpy_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    ఒకవేళ (!ggml_backend_బఫరు_is_metal(src->బఫరు)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి ggml_metal_బఫరు_cpy_tensor(ctx, src, dst);
}

స్థిర శూన్య ggml_backend_metal_బఫరు_వ్యక్తిగత_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_metal_బఫరు_t ctx = (ggml_metal_బఫరు_t)బఫరు->context;

    GGML_ASSERT(!ggml_metal_బఫరు_is_shared(ctx));

    ggml_metal_బఫరు_clear(ctx, value);
}

స్థిర ggml_backend_బఫరు_i ggml_backend_metal_బఫరు_వ్యక్తిగత_i = {
    /* .free_buffer   = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_free_బఫరు,
    /* .get_base      = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_get_పునాది,
    /* .init_tensor   = */ ఖాళీ,
    /* .memset_tensor = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_సమితి_tensor,
    /* .get_tensor    = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_get_tensor,
    /* .set_tensor_2d = */ ఖాళీ,
    /* .get_tensor_2d = */ ఖాళీ,
    /* .cpy_tensor    = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_బఫరు_వ్యక్తిగత_clear,
    /* .reset         = */ ఖాళీ,
};

స్థిర తార్కిక ggml_backend_బఫరు_is_metal(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి బఫరు->iface.free_బఫరు == ggml_backend_metal_బఫరు_shared_free_బఫరు ||
           బఫరు->iface.free_బఫరు == ggml_backend_metal_బఫరు_వ్యక్తిగత_free_బఫరు;
}

//
// buffer types
//

నిర్మాణం ggml_backend_metal_బఫరు_type {
    పూర్ణాంకం device;
    std::string name;
};

నిర్మాణం ggml_backend_metal_బఫరు_type_deleter {
    శూన్య క్రియాకారకం()(ggml_backend_metal_బఫరు_type * ctx) స్థిరరాశి {
        తొలగించు ctx;
    }
};

రకనిర్వచనం std::unique_ptr<ggml_backend_metal_బఫరు_type, ggml_backend_metal_బఫరు_type_deleter> ggml_backend_metal_బఫరు_type_ptr;

// common method for allocating shread or private Metal buffers
స్థిర ggml_backend_బఫరు_t ggml_backend_metal_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size, తార్కిక shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_బఫరు_t res = ggml_metal_బఫరు_init(ctx_dev, size, shared);

    ఒకవేళ (res == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        తిరిగి ఖాళీ;
    }

    ggml_backend_బఫరు_i buf_i = ggml_metal_బఫరు_is_shared(res)
        ? ggml_backend_metal_బఫరు_shared_i
        : ggml_backend_metal_బఫరు_వ్యక్తిగత_i;

    తిరిగి ggml_backend_బఫరు_init(buft, buf_i, res, size);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    ఎంపిక (tensor->op) {
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } ఆపు;
        స్థితి GGML_OP_CUMSUM:
        స్థితి GGML_OP_ARGSORT:
            {
                res *= 2;
            } ఆపు;
        స్థితి GGML_OP_శిఖరం_K:
            {
                res = 2*పరిమాణం(int32_t)*ggml_nelements(tensor->src[0]);
            } ఆపు;
        స్థితి GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } ఆపు;
        అప్రమేయ:
            ఆపు;
    }

    తిరిగి res;

    GGML_వాడని(buft);
}

// default (shared) buffer type

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_బఫరు_type_shared_get_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_metal_బఫరు_type * ctx = (ggml_backend_metal_బఫరు_type *)buft->context;

    తిరిగి ctx->name.c_str();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_metal_బఫరు_type_shared_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    తిరిగి ggml_backend_metal_బఫరు_type_alloc_బఫరు(buft, size, నిజం);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_shared_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 32;

    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_shared_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    తిరిగి ggml_metal_device_get_props(ctx_dev)->max_బఫరు_కొలత;
}

స్థిర size_t ggml_backend_metal_బఫరు_type_shared_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_backend_metal_బఫరు_type_get_alloc_కొలత(buft, tensor);
}

స్థిర తార్కిక ggml_backend_metal_బఫరు_type_shared_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి అబద్ధం;

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_metal_బఫరు_type_shared(పూర్ణాంకం device) {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);

    స్థిర std::vector<ggml_backend_బఫరు_type> bufts;
    స్థిర std::vector<ggml_backend_metal_బఫరు_type_ptr> ctxs;

    స్థిర తార్కిక initialized = అబద్ధం;
    ఒకవేళ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        కొరకు (పూర్ణాంకం i = 0; i < g_devices; ++i) {
            ggml_backend_metal_బఫరు_type * raw_ctx =
                కొత్త ggml_backend_metal_బఫరు_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_బఫరు_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_బఫరు_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_బఫరు_type_shared_alloc_బఫరు,
                    /* .alloc_buffer_n      = */ ఖాళీ,
                    /* .get_alignment       = */ ggml_backend_metal_బఫరు_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_బఫరు_type_shared_get_max_కొలత,
                    /* .get_alloc_size      = */ ggml_backend_metal_బఫరు_type_shared_get_alloc_కొలత,
                    /* .get_alloc_size_n    = */ ఖాళీ,
                    /* .is_host             = */ ggml_backend_metal_బఫరు_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = నిజం;
    }

    తిరిగి &bufts[device];
}

// default (private) buffer type

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_metal_బఫరు_type * ctx = (ggml_backend_metal_బఫరు_type *)buft->context;

    తిరిగి ctx->name.c_str();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_metal_బఫరు_type_వ్యక్తిగత_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    తిరిగి ggml_backend_metal_బఫరు_type_alloc_బఫరు(buft, size, అబద్ధం);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 32;

    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    తిరిగి ggml_metal_device_get_props(ctx_dev)->max_బఫరు_కొలత;
}

స్థిర size_t ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_backend_metal_బఫరు_type_get_alloc_కొలత(buft, tensor);
}

స్థిర తార్కిక ggml_backend_metal_బఫరు_type_వ్యక్తిగత_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి అబద్ధం;

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_metal_బఫరు_type_వ్యక్తిగత(పూర్ణాంకం device) {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);

    స్థిర std::vector<ggml_backend_బఫరు_type> bufts;
    స్థిర std::vector<ggml_backend_metal_బఫరు_type_ptr> ctxs;

    స్థిర తార్కిక initialized = అబద్ధం;
    ఒకవేళ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        కొరకు (పూర్ణాంకం i = 0; i < g_devices; ++i) {
            ggml_backend_metal_బఫరు_type * raw_ctx = కొత్త ggml_backend_metal_బఫరు_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_బఫరు_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_బఫరు_type_వ్యక్తిగత_alloc_బఫరు,
                    /* .alloc_buffer_n      = */ ఖాళీ,
                    /* .get_alignment       = */ ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_max_కొలత,
                    /* .get_alloc_size      = */ ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_alloc_కొలత,
                    /* .get_alloc_size_n    = */ ఖాళీ,
                    /* .is_host             = */ ggml_backend_metal_బఫరు_type_వ్యక్తిగత_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = నిజం;
    }

    తిరిగి &bufts[device];
}

// mapped buffer type

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_బఫరు_type_mapped_get_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_metal_బఫరు_type * ctx = (ggml_backend_metal_బఫరు_type *)buft->context;

    తిరిగి ctx->name.c_str();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_metal_బఫరు_type_mapped_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    తిరిగి ggml_backend_metal_బఫరు_type_alloc_బఫరు(buft, size, నిజం);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_mapped_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 32;

    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_metal_బఫరు_type_mapped_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    తిరిగి ggml_metal_device_get_props(ctx_dev)->max_బఫరు_కొలత;
}

స్థిర size_t ggml_backend_metal_బఫరు_type_mapped_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_backend_metal_బఫరు_type_get_alloc_కొలత(buft, tensor);
}

స్థిర తార్కిక ggml_backend_metal_బఫరు_type_mapped_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి అబద్ధం;

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_metal_బఫరు_type_mapped(పూర్ణాంకం device) {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);

    స్థిర std::vector<ggml_backend_బఫరు_type> bufts;
    స్థిర std::vector<ggml_backend_metal_బఫరు_type_ptr> ctxs;

    స్థిర తార్కిక initialized = అబద్ధం;
    ఒకవేళ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        కొరకు (పూర్ణాంకం i = 0; i < g_devices; ++i) {
            ggml_backend_metal_బఫరు_type * raw_ctx = కొత్త ggml_backend_metal_బఫరు_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_బఫరు_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_బఫరు_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_బఫరు_type_mapped_alloc_బఫరు,
                    /* .alloc_buffer_n      = */ ఖాళీ,
                    /* .get_alignment       = */ ggml_backend_metal_బఫరు_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_బఫరు_type_mapped_get_max_కొలత,
                    /* .get_alloc_size      = */ ggml_backend_metal_బఫరు_type_mapped_get_alloc_కొలత,
                    /* .get_alloc_size_n    = */ ఖాళీ,
                    /* .is_host             = */ ggml_backend_metal_బఫరు_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = నిజం;
    }

    తిరిగి &bufts[device];
}

// backend

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    తిరిగి ggml_metal_get_name(ctx);
}

స్థిర శూన్య ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    విడుదల(backend);
}

స్థిర శూన్య ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

స్థిర శూన్య ggml_backend_metal_సమితి_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_సమితి_tensor_async(ctx, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_metal_get_tensor_async(ggml_backend_t backend, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర తార్కిక ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ఒకవేళ (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_backend_బఫరు_is_metal(src->బఫరు) || !ggml_backend_బఫరు_is_metal(dst->బఫరు)) {
        తిరిగి అబద్ధం;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    తిరిగి ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

స్థిర గణన ggml_స్థితి ggml_backend_metal_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    తిరిగి ggml_metal_లేఖాచిత్రం_compute(ctx, cgraph);
}

స్థిర శూన్య ggml_backend_metal_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_రికార్డు(ctx, ev);
}

స్థిర శూన్య ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

స్థిర శూన్య ggml_backend_metal_లేఖాచిత్రం_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_లేఖాచిత్రం_optimize_params * params) {
    GGML_ASSERT(params && params->కూడు_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_కూడు_alloc_deps(params->user_దత్తాంశం, params->కూడు_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_లేఖాచిత్రం_optimize(ctx, cgraph);
}

స్థిర శూన్య ggml_backend_metal_సమితి_n_cb(ggml_backend_t backend, పూర్ణాంకం n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_సమితి_n_cb(ctx, n_cb);
}

స్థిర ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_metal_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_metal_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_లేఖాచిత్రం_optimize,
};

స్థిర ggml_guid_t ggml_backend_metal_guid(శూన్య) {
    స్థిర ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    తిరిగి &guid;
}

ggml_backend_t ggml_backend_metal_init(శూన్య) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ఒకవేళ (ctx == ఖాళీ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        తిరిగి ఖాళీ;
    }

    ggml_backend_t backend = (ggml_backend_t) జ్ఞాపకం_ఇవ్వు(పరిమాణం(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_సమితి_n_cb(backend, 1);

    తిరిగి backend;
}

తార్కిక ggml_backend_is_metal(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

శూన్య ggml_backend_metal_సమితి_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, శూన్య * user_దత్తాంశం) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_సమితి_abort_callback(ctx, abort_callback, user_దత్తాంశం);
}

తార్కిక ggml_backend_metal_supports_family(ggml_backend_t backend, పూర్ణాంకం family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    తిరిగి ggml_metal_supports_family(ctx, family);
}

శూన్య ggml_backend_metal_capture_తదుపరి_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_తదుపరి_compute(ctx);
}

// backend device

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    స్థిరరాశి ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    తిరిగి props_dev->name;
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    తిరిగి ggml_metal_device_get_props(ctx_dev)->desc;
}

స్థిర శూన్య ggml_backend_metal_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_జ్ఞాపకం(ctx_dev, విడుదల, మొత్తం);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);

    props->caps = {
        /* .async                = */ నిజం,
        /* .host_buffer          = */ అబద్ధం,
        /* .buffer_from_host_ptr = */ నిజం,
        /* .events               = */ నిజం,
        /* .mmap_support         = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ఒకవేళ (ctx == ఖాళీ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        తిరిగి ఖాళీ;
    }

    ggml_backend_t backend = (ggml_backend_t) జ్ఞాపకం_ఇవ్వు(పరిమాణం(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_సమితి_n_cb(backend, 1);

    తిరిగి backend;

    GGML_వాడని(params);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_metal_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    స్థిరరాశి ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    తిరిగి props_dev->use_shared_buffers ? ggml_backend_metal_బఫరు_type_shared(props_dev->device) : ggml_backend_metal_బఫరు_type_వ్యక్తిగత(props_dev->device);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_metal_device_బఫరు_mapped(ggml_backend_dev_t dev, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_బఫరు_t res = ggml_metal_బఫరు_map(ctx_dev, ptr, size, max_tensor_కొలత);

    స్థిరరాశి ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    తిరిగి ggml_backend_బఫరు_init(ggml_backend_metal_బఫరు_type_mapped(props_dev->device), ggml_backend_metal_బఫరు_shared_i, res, size);
}

స్థిర తార్కిక ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    తిరిగి ggml_metal_device_supports_op(ctx_dev, op);
}

స్థిర తార్కిక ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    తిరిగి
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_బఫరు_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_బఫరు_type_వ్యక్తిగత_get_name ||
        buft->iface.get_name == ggml_backend_metal_బఫరు_type_mapped_get_name);

    GGML_వాడని(dev);
}

స్థిర int64_t get_op_batch_కొలత(స్థిరరాశి ggml_tensor * op) {
    ఎంపిక (op->op) {
        స్థితి GGML_OP_MUL_MAT:
            తిరిగి op->ne[1];
        స్థితి GGML_OP_MUL_MAT_ID:
            తిరిగి op->ne[2];
        అప్రమేయ:
            తిరిగి ggml_nrows(op);
    }
}

స్థిర తార్కిక ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    తిరిగి (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_కొలత(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_కొలత;
}

స్థిర ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = కొత్త ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    తిరిగి ev;
}

స్థిర శూన్య ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    తొలగించు event;
}

స్థిర శూన్య ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

స్థిర ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_బఫరు_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

నిర్మాణం ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

రకనిర్వచనం నిర్మాణం ggml_backend_metal_reg * ggml_backend_metal_reg_t;

స్థిర ggml_backend_metal_reg_t ggml_backend_metal_reg_init(శూన్య) {
    ggml_backend_metal_reg_t ctx = కొత్త నిర్మాణం ggml_backend_metal_reg;

    తిరిగి ctx;
}

స్థిర శూన్య ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    తొలగించు ctx;
}

నిర్మాణం ggml_backend_metal_reg_deleter {
    శూన్య క్రియాకారకం()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

రకనిర్వచనం std::unique_ptr<నిర్మాణం ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    తిరిగి GGML_METAL_NAME;

    GGML_వాడని(reg);
}

స్థిర size_t ggml_backend_metal_reg_device_లెక్క(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    తిరిగి ctx->devices.size();
}

స్థిర ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t సూచీ) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(సూచీ < ctx->devices.size());
    తిరిగి ctx->devices[సూచీ];
}

స్థిర ggml_backend_feature g_ggml_backend_metal_features[] = {
#ఒకవేళ defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#అయితే_ముగింపు
    { ఖాళీ, ఖాళీ },
};

స్థిర ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    తిరిగి g_ggml_backend_metal_features;

    GGML_వాడని(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
స్థిర శూన్య ggml_backend_metal_tuning_సమితి_fa_vec_override(పూర్ణాంకం Q, పూర్ణాంకం NE) {
    ggml_metal_tuning::fa_vec_సమితి_override({ (int8_t) Q, (int8_t) NE });
}

స్థిర శూన్య ggml_backend_metal_tuning_clear_fa_vec_override(శూన్య) {
    ggml_metal_tuning::fa_vec_clear_override();
}

స్థిర పూర్ణాంకం ggml_backend_metal_tuning_fa_vec_ne11_పాత్ర(int64_t ne11) {
    తిరిగి ggml_metal_tuning::fa_vec_ne11_పాత్ర(ne11);
}

స్థిర పూర్ణాంకం ggml_backend_metal_tuning_fa_vec_ne01_పాత్ర(int64_t ne01) {
    తిరిగి ggml_metal_tuning::fa_vec_ne01_పాత్ర(ne01);
}

స్థిర పూర్ణాంకం ggml_backend_metal_tuning_fa_vec_baseline_ne(పూర్ణాంకం dk, పూర్ణాంకం dv) {
    తిరిగి ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    తిరిగి ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
రకనిర్వచనం శూన్య * ggml_backend_fusion_t;

స్థిర ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    తిరిగి ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

స్థిర శూన్య ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((నిర్మాణం ggml_metal_fusion_info *) finfo);
}

స్థిర శూన్య ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((నిర్మాణం ggml_metal_fusion_info *) finfo);
}

స్థిర పూర్ణాంకం ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, స్థిరరాశి అక్షరం ** labels, uint64_t * counts, పూర్ణాంకం n) {
    తిరిగి ggml_metal_fusion_info_stats_get((నిర్మాణం ggml_metal_fusion_info *) finfo, labels, counts, n);
}

స్థిర శూన్య ggml_backend_metal_fusion_సమితి_enabled(ggml_backend_fusion_t finfo, తార్కిక enabled) {
    ggml_metal_fusion_info_సమితి_enabled((నిర్మాణం ggml_metal_fusion_info *) finfo, enabled);
}

స్థిర శూన్య * ggml_backend_metal_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (strcmp(name, "ggml_backend_get_features") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_get_features;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_tuning_సమితి_fa_vec_override;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_tuning_fa_vec_ne11_పాత్ర;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_tuning_fa_vec_ne01_పాత్ర;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    ఒకవేళ (strcmp(name, "ggml_backend_fusion_get") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_fusion_get;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_fusion_stats_init;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_fusion_stats_reset;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_fusion_stats_get;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        తిరిగి (శూన్య *)ggml_backend_metal_fusion_సమితి_enabled;
    }

    తిరిగి ఖాళీ;

    GGML_వాడని(reg);
}

స్థిర ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_లెక్క,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_చిరునామా,
};

స్థిర ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, పూర్ణాంకం device) {
    తిరిగి కొత్త ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

స్థిర శూన్య ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    తొలగించు dev;
}

నిర్మాణం ggml_backend_device_deleter {
    శూన్య క్రియాకారకం()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

రకనిర్వచనం std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(శూన్య) {
    స్థిర ggml_backend_reg reg;
    స్థిర తార్కిక initialized = అబద్ధం;

    {
        స్థిర std::mutex mutex;
        std::lock_guard<std::mutex> తాళం(mutex);

        స్థిరరాశి అక్షరం * env = getenv("GGML_METAL_DEVICES");
        ఒకవేళ (env) {
            g_devices = atoi(env);
        }

        స్థిర std::vector<ggml_backend_device_ptr> devs;

        ఒకవేళ (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", నిజం);

            స్థిర ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            కొరకు (పూర్ణాంకం i = 0; i < g_devices; ++i) {
                స్వయంచాలక * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = నిజం;
    }

    తిరిగి &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
