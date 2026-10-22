#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-ops.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-device.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-fusion.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-tuning.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>

// threadgroups needed to fill the GPU
// TODO: dedup
ਸਥਿਰ constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

ਸਥਿਰ ggml_metal_ਬਫਰ_id ggml_metal_get_ਬਫਰ_id(ਅਚਲ ggml_tensor * t) {
    ਜੇ (!t) {
        ਵਾਪਸ { ਖਾਲੀ_ਸੂਚਕ, 0 };
    }

    ggml_backend_ਬਫਰ_t ਬਫਰ = t->view_src ? t->view_src->ਬਫਰ : t->ਬਫਰ;

    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t) ਬਫਰ->context;

    ਵਾਪਸ ggml_metal_ਬਫਰ_get_id(ctx, t);
}

ਬਣਤਰ ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        ਪੂਰਨ  idx_ਸ਼ੁਰੂ,
        ਪੂਰਨ  idx_ਸਿਰਾ,
        ਤਾਰਕਿਕ use_concurrency,
        ਤਾਰਕਿਕ use_capture,
        ਪੂਰਨ  debug_ਗ੍ਰਾਫ) {
        ਇਹ->dev             = dev;
        ਇਹ->lib             = ggml_metal_device_get_library(dev);
        ਇਹ->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        ਇਹ->mem_ranges      = ggml_mem_ranges_init(debug_ਗ੍ਰਾਫ);
        ਇਹ->finfo           = finfo;
        ਇਹ->idx_ਸ਼ੁਰੂ       = idx_ਸ਼ੁਰੂ;
        ਇਹ->idx_ਸਿਰਾ         = idx_ਸਿਰਾ;
        ਇਹ->use_concurrency = use_concurrency;
        ਇਹ->use_capture     = use_capture;
        ਇਹ->debug_ਗ੍ਰਾਫ     = debug_ਗ੍ਰਾਫ;
        ਇਹ->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        ਲਈ (ਪੂਰਨ i = idx_ਸ਼ੁਰੂ; i < idx_ਸਿਰਾ; i++) {
            ਜੇ (!ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(gf->nodes[i]->op) && !ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_ਸਿਰਾ_encoding(ਇਹ->enc);
        ggml_metal_encoder_free(ਇਹ->enc);
        ggml_mem_ranges_free(ਇਹ->mem_ranges);
    }

    ਪੂਰਨ n_nodes() ਅਚਲ {
        ਵਾਪਸ idxs.size();
    }

    ggml_tensor * ਗੰਢ(ਪੂਰਨ i) ਅਚਲ {
        ਪ੍ਰਤਿਜ੍ਞਾ(i >= 0 && i < (ਪੂਰਨ) idxs.size());
        ਵਾਪਸ ggml_ਗ੍ਰਾਫ_ਗੰਢ(gf, idxs[i]);
    }

    // consult the fusion table for the longest pattern starting at i0
    // returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes
    ਅਚਲ ggml_metal_fusion * can_fuse(ਪੂਰਨ i0, ਗਿਣਤੀ ggml_metal_fusion_ਬਹੁਲਕ mode, ਪੂਰਨ * n_ਬਾਹਰਲਾ) ਅਚਲ {
        ਪ੍ਰਤਿਜ੍ਞਾ(use_fusion());
        ਪ੍ਰਤਿਜ੍ਞਾ(i0 >= 0 && i0 < n_nodes());

        ਵਾਪਸ ggml_metal_fusion_ਅਗਲਾ(gf, idxs.ਅੰਕੜੇ(), (ਪੂਰਨ) idxs.size(), i0, mode, n_ਬਾਹਰਲਾ);
    }

    // whether to attempt fusion; the toggle lives in the shared fusion debugging context owned
    // by the device (initialized from GGML_METAL_FUSION_DISABLE, overridable by the test)
    ਤਾਰਕਿਕ use_fusion() ਅਚਲ {
        ਵਾਪਸ ggml_metal_fusion_info_enabled(finfo);
    }

    // record that a fusion fired, indexed by the matching table entry
    ਖਾਲੀ ਗਿਣਤੀ_fusions(ਅਚਲ ggml_metal_fusion * fusion) ਅਚਲ {
        ggml_metal_fusion_info_ਗਿਣਤੀ_fusion(finfo, fusion);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    // shared fusion debugging context
    ggml_metal_fusion_info * finfo;

    ਤਾਰਕਿਕ use_concurrency;
    ਤਾਰਕਿਕ use_capture;

    ਪੂਰਨ debug_ਗ੍ਰਾਫ;

ਨਿਜੀ:
    ggml_cgraph * gf;

    ਪੂਰਨ idx_ਸ਼ੁਰੂ;
    ਪੂਰਨ idx_ਸਿਰਾ;

    // non-empty node indices
    std::vector<ਪੂਰਨ> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        ਪੂਰਨ idx_ਸ਼ੁਰੂ,
        ਪੂਰਨ idx_ਸਿਰਾ,
        ਤਾਰਕਿਕ use_concurrency,
        ਤਾਰਕਿਕ use_capture,
        ਪੂਰਨ debug_ਗ੍ਰਾਫ) {
    ggml_metal_op_t res = ਨਵਾਂ ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_ਸ਼ੁਰੂ,
        idx_ਸਿਰਾ,
        use_concurrency,
        use_capture,
        debug_ਗ੍ਰਾਫ);

    ਵਾਪਸ res;
}

ਖਾਲੀ ggml_metal_op_free(ggml_metal_op_t ctx) {
    ਮਿਟਾਓ ctx;
}

ਪੂਰਨ ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    ਵਾਪਸ ctx->n_nodes();
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    ਜੇ (!ctx->mem_ranges) {
        ਵਾਪਸ ਸੱਚ;
    }

    ggml_metal_encoder_ਸਮ੍ਰਿਤੀ_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ਜੇ (!ctx->mem_ranges) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ggml_mem_ranges_check(ctx->mem_ranges, ਗੰਢ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_op_concurrency_ਜੋੜੋ(ggml_metal_op_t ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ਜੇ (!ctx->mem_ranges) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਵਾਪਸ ggml_mem_ranges_ਜੋੜੋ(ctx->mem_ranges, ਗੰਢ);
}

ਸਥਿਰ ਪੂਰਨ ggml_metal_op_encode_impl(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ਬਣਤਰ ggml_tensor * ਗੰਢ = ctx->ਗੰਢ(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ)) {
        ਵਾਪਸ 1;
    }

    ਚੋਣ (ਗੰਢ->op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            {
                // noop -> next node
                ਜੇ (ctx->debug_ਗ੍ਰਾਫ > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(ਗੰਢ->op), "(noop)");
                }
            } ਵਾਪਸ 1;
        ਮੂਲ:
            {
            } ਰੋਕੋ;
    }

    ਜੇ (!ggml_metal_device_supports_op(ctx->dev, ਗੰਢ)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(ਗੰਢ));
        GGML_ABORT("unsupported op");
    }

    ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ਵਾਪਸ 1;
    }

    ਪੂਰਨ n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        ਤਾਰਕਿਕ is_concurrent = ggml_metal_op_concurrency_check(ctx, ਗੰਢ);

        ਜੇ (is_concurrent && ctx->use_fusion()) {
            ਪੂਰਨ n_fuse = 1;
            ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n_fuse);
            ਜੇ (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->ਗੰਢ(idx + n_fuse - 1));
            }
        }

        ਜੇ (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        ਜੇ (ctx->debug_ਗ੍ਰਾਫ > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(ਗੰਢ->op), ggml_get_name(ਗੰਢ), is_concurrent ? "(concurrent)" : "");
        }
        ਜੇ (ctx->debug_ਗ੍ਰਾਫ > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, ਗੰਢ->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, ਗੰਢ->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, ਗੰਢ->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, ਗੰਢ->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, ਗੰਢ->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, ਗੰਢ->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, ਗੰਢ->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, ਗੰਢ->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  ਗੰਢ,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  ਗੰਢ,         nb);

            ਜੇ (ਗੰਢ->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ਗੰਢ->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(ਗੰਢ->src[0]), ਗੰਢ->src[0]->name);
            }
            ਜੇ (ਗੰਢ->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ਗੰਢ->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(ਗੰਢ->src[1]), ਗੰਢ->src[1]->name);
            }
            ਜੇ (ਗੰਢ->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ਗੰਢ->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(ਗੰਢ->src[2]), ਗੰਢ->src[2]->name);
            }
            ਜੇ (ਗੰਢ->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ਗੰਢ->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(ਗੰਢ->src[3]), ਗੰਢ->src[3]->name);
            }
            ਜੇ (ਗੰਢ) {
                GGML_LOG_DEBUG("%s: node - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(ਗੰਢ->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        ਗੰਢ->name);
            }
        }
    }

    ਚੋਣ (ਗੰਢ->op) {
        ਹਾਲਤ GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            {
                n_fuse = ggml_metal_op_ਜੋੜੋ_id(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE:
        ਹਾਲਤ GGML_OP_FILL:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
            {
                n_fuse = ggml_metal_op_silu_ਪਿੱਠ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            {
                n_fuse = ggml_metal_op_ਜੋੜ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            {
                n_fuse = ggml_metal_op_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            {
                n_fuse = ggml_metal_op_ssm_ਜਾਂਚ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            {
                n_fuse = ggml_metal_op_get_ਪੰਕਤੀਆਂ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                n_fuse = ggml_metal_op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            {
                n_fuse = ggml_metal_op_ਸਮੂਹ_norm(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            {
                n_fuse = ggml_metal_op_ਸਿਖਰ_k(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            {
                n_fuse = ggml_metal_op_ਸੈੱਟ(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_ਕਦਮ_adamw(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            {
                n_fuse = ggml_metal_op_opt_ਕਦਮ_sgd(ctx, idx);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
            {
                n_fuse = ggml_metal_op_ਗਿਣਤੀ_equal(ctx, idx);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(ਗੰਢ->op));
                GGML_ABORT("fatal error");
            }
    }

    ਜੇ (ctx->debug_ਗ੍ਰਾਫ > 0) {
        ਜੇ (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    ਲਈ (ਪੂਰਨ i = 0; i < n_fuse; ++i) {
        ਜੇ (!ggml_metal_op_concurrency_ਜੋੜੋ(ctx, ctx->ਗੰਢ(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_encode(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ਜੇ (ctx->use_capture) {
        ggml_metal_encoder_debug_ਸਮੂਹ_ਧੱਕੋ(ctx->enc, ggml_op_desc(ctx->ਗੰਢ(idx)));
    }

    ਪੂਰਨ res = ggml_metal_op_encode_impl(ctx, idx);
    ਜੇ (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    ਜੇ (ctx->use_capture) {
        ggml_metal_encoder_debug_ਸਮੂਹ_ਕੱਢੋ(ctx->enc);
    }

    ਵਾਪਸ res;
}

ਪੂਰਨ ggml_metal_op_concat(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t dim = ((ਅਚਲ int32_t *) op->op_params)[0];

    ਅਚਲ ਤਾਰਕਿਕ is_q = ggml_is_quantized(op->type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    int32_t ne00_arg = ne00;
    int32_t ne10_arg = ne10;
    int32_t ne0_arg  = ne0;
    ਜੇ (is_q) {
        ਅਚਲ int32_t blck = ggml_blck_ਮਾਪ(op->type);
        GGML_ASSERT(ne00 % blck == 0);
        GGML_ASSERT(ne10 % blck == 0);
        GGML_ASSERT(ne0  % blck == 0);
        ne00_arg = ne00/blck;
        ne10_arg = ne10/blck;
        ne0_arg  = ne0/blck;
    }

    ਪੂਰਨ nth = std::min(256, ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    ਪੂਰਨ nrptg = 1;
    ਜੇ (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        ਜੇ (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    ਅਚਲ ਪੂਰਨ nw0 = (ne1 + nrptg - 1) / nrptg;

    // split long rows across threadgroups when there are too few rows to fill the GPU
    ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ = (int64_t) nw0*ne2*ne3;

    ਪੂਰਨ nc0 = 1;
    ਜੇ (nrptg == 1) {
        nc0 = (ਪੂਰਨ) std::max<int64_t>(1, std::min<int64_t>((ne0_arg + nth - 1)/nth, (GGML_METAL_MIN_THREADGROUPS + n_ਪੰਕਤੀਆਂ - 1)/n_ਪੰਕਤੀਆਂ));
    }

    ggml_metal_kargs_concat ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00_arg,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10_arg,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0_arg,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
        /*.nc0  =*/ nc0,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*nc0, ne2, ne3, nth, nrptg, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_repeat(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // TODO: optimize for degenerate cases such as ggml_nelements(op->src[0]) == 1 and others

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_acc(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[1]));

    ਅਚਲ size_t pnb1 = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਅਚਲ size_t pnb2 = ((ਅਚਲ int32_t *) op->op_params)[1];
    ਅਚਲ size_t pnb3 = ((ਅਚਲ int32_t *) op->op_params)[2];
    ਅਚਲ size_t offs = ((ਅਚਲ int32_t *) op->op_params)[3];

    ਅਚਲ ਤਾਰਕਿਕ inplace = (ਤਾਰਕਿਕ) ((ਅਚਲ int32_t *) op->op_params)[4];

    ਜੇ (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy ਦਲੀਲਾਂ_2 = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

        ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ਜੋੜੋ);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ਅਚਲ ਪੂਰਨ nth_max = MIN(256, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

    ਪੂਰਨ nth = 1;

    ਜਦੋਂ (2*nth < ਦਲੀਲਾਂ_2.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_unary(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_kargs_unary ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    ਜੇ (op->op == GGML_OP_LEAKY_RELU) {
        ਦਲੀਲਾਂ_2.ਢਲਾਨ = ggml_get_op_params_f32(op, 0);
    }

    ਜੇ (op->op == GGML_OP_SCALE) {
        ਦਲੀਲਾਂ_2.scale = ggml_get_op_params_f32(op, 0);
        ਦਲੀਲਾਂ_2.bias  = ggml_get_op_params_f32(op, 1);
    }

    ਜੇ (op->op == GGML_OP_FILL) {
        ਦਲੀਲਾਂ_2.val = ggml_get_op_params_f32(op, 0);
    }

    ਜੇ (op->op == GGML_OP_CLAMP) {
        ਦਲੀਲਾਂ_2.min = ggml_get_op_params_f32(op, 0);
        ਦਲੀਲਾਂ_2.max = ggml_get_op_params_f32(op, 1);
    }

    ਜੇ (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        ਦਲੀਲਾਂ_2.ਢਲਾਨ = ggml_get_op_params_f32(op, 1); // alpha_n
        ਦਲੀਲਾਂ_2.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        ਦਲੀਲਾਂ_2.bias  = ggml_get_op_params_f32(op, 3); // beta
        ਦਲੀਲਾਂ_2.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    ਜੇ (pipeline.c4) {
        ਦਲੀਲਾਂ_2.ne00 = ne00/4;
        ਦਲੀਲਾਂ_2.ne0  = ne0/4;
    }

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

    ਜੇ (pipeline.cnt) {
        ਅਚਲ ਪੂਰਨ n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ nth_max = MIN(256, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
        ਅਚਲ ਪੂਰਨ nth = MIN(ਦਲੀਲਾਂ_2.ne00, nth_max);
        ਅਚਲ ਪੂਰਨ nk0 = (ਦਲੀਲਾਂ_2.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_glu(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਜੇ (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    ਅਚਲ int32_t swp = ggml_get_op_params_i32(op, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = ggml_get_op_params_f32(op, 2);
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ggml_get_op_params_f32(op, 3);

    ਅਚਲ int32_t i00 = swp ? ne0 : 0;
    ਅਚਲ int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ ਹੱਦਬੰਦੀ
    };

    ਅਚਲ int64_t nrows = ggml_nrows(op->src[0]);

    ਅਚਲ int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne00/2));

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ਜੇ (op->src[1]) {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    } ਨਹੀਂਤਾਂ {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 2);
    }
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਜੋੜ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op  = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_ਜੋੜ ਦਲੀਲਾਂ_2 = {
        /*.np =*/ n,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਜੋੜ(lib, op);

    ਪੂਰਨ nth = 32; // SIMD width

    ਜਦੋਂ (nth < (ਪੂਰਨ) n && nth < ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
    nth = std::min(nth, (ਪੂਰਨ) n);

    ਅਚਲ ਪੂਰਨ nsg = (nth + 31) / 32;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, GGML_PAD(nsg * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 16), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_kargs_ਜੋੜ_ਪੰਕਤੀਆਂ ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ(lib, op);

    ਜੇ (pipeline.c4) {
        ਦਲੀਲਾਂ_2.ne00 = ne00/4;
        ਦਲੀਲਾਂ_2.ne0  = ne0/4;
    }

    ਪੂਰਨ nth = 32; // SIMD width

    ਜਦੋਂ (nth < ਦਲੀਲਾਂ_2.ne00 && nth < ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
    nth = std::min(nth, (ਪੂਰਨ) ਦਲੀਲਾਂ_2.ne00);

    ਅਚਲ size_t smem = pipeline.smem;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_cumsum(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    ਪੂਰਨ nth = 1;
    ਜਦੋਂ (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    ਅਚਲ int64_t net0 = (ne00 + nth - 1) / nth;
    ਅਚਲ int64_t net1 = ne01;
    ਅਚਲ int64_t net2 = ne02;
    ਅਚਲ int64_t net3 = ne03;

    ਅਚਲ uint64_t nbt0 = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint64_t nbt1 = net0*nbt0;
    ਅਚਲ uint64_t nbt2 = net1*nbt1;
    ਅਚਲ uint64_t nbt3 = net2*nbt2;

    ਅਚਲ size_t smem = GGML_PAD(32*ਆਕਾਰ(ਦਸ਼ਮਲਵ), 16);

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_ਬਫਰ_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk ਦਲੀਲਾਂ_2 = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp,  2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  3);

        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    ਜੇ (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk ਦਲੀਲਾਂ_2 = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ ਝੂਠ,
            };

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 2);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 3);

            ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            ਸਵੈਚਾਲਿਤ pipeline_ਜੋੜੋ = ggml_metal_library_get_pipeline_cumsum_ਜੋੜੋ(lib, op);

            ggml_metal_kargs_cumsum_ਜੋੜੋ ਦਲੀਲਾਂ_2 = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_ਜੋੜੋ);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_get_ਪੰਕਤੀਆਂ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_get_ਪੰਕਤੀਆਂ(lib, op->src[0]->type);

    ggml_metal_kargs_get_ਪੰਕਤੀਆਂ ਦਲੀਲਾਂ_2 = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    ਅਚਲ ਪੂਰਨ nth = std::min(ਦਲੀਲਾਂ_2.ne00t, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

    ਅਚਲ ਪੂਰਨ nw0 = (ਦਲੀਲਾਂ_2.ne00t + nth - 1)/nth;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ(lib, op);

    ਅਚਲ int32_t nk0 = ne0/ggml_blck_ਮਾਪ(op->type);

    ਪੂਰਨ nth = 32; // SIMD width

    ਜਦੋਂ (nth < nk0 && nth < ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    ਪੂਰਨ nrptg = 1;
    ਜੇ (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        ਜੇ (nrptg*nth > ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_ਸੈੱਟ_ਪੰਕਤੀਆਂ ਦਲੀਲਾਂ_2 = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_diag(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    ਅਚਲ ggml_tensor * q = op->src[0];
    ਅਚਲ ggml_tensor * k = op->src[1];
    ਅਚਲ ggml_tensor * w = op->src[2];
    ਅਚਲ ggml_tensor * m = op->src[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32  ||
                k->type == GGML_TYPE_F16  ||
                k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q4_0 ||
                k->type == GGML_TYPE_Q4_1 ||
                k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q5_1 ||
                k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    GGML_ASSERT(q->ne[0] == OP_LIGHTNING_INDEXER_DK);

    ggml_metal_kargs_lightning_indexer ਦਲੀਲਾਂ_2 = {
        /*.n_kv      =*/ (int32_t) k->ne[2],
        /*.n_batch   =*/ (int32_t) q->ne[2],
        /*.mask_ne3  =*/ (int32_t) m->ne[3],
        /*.nb1       =*/ op->nb[1],
        /*.nb3       =*/ op->nb[3],
        /*.nbq1      =*/ q->nb[1],
        /*.nbq2      =*/ q->nb[2],
        /*.nbq3      =*/ q->nb[3],
        /*.nbk2      =*/ k->nb[2],
        /*.nbk3      =*/ k->nb[3],
        /*.nbw1      =*/ w->nb[1],
        /*.nbw3      =*/ w->nb[3],
        /*.nbm1      =*/ m->nb[1],
        /*.nbm3      =*/ m->nb[3],
    };

    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(q),  1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(k),  2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(w),  3);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(m),  4);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op), 5);

    ਅਚਲ ਪੂਰਨ nsg   = OP_LIGHTNING_INDEXER_NSG;
    ਅਚਲ ਪੂਰਨ nkptg = OP_LIGHTNING_INDEXER_NKPSG*nsg;
    ਅਚਲ ਪੂਰਨ nbptg = OP_LIGHTNING_INDEXER_NBPTG;

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, op);
    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc,
            (k->ne[2] + nkptg - 1)/nkptg,
            (q->ne[2] + nbptg - 1)/nbptg,
            q->ne[3], 32, nsg, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
            {
                ਅਚਲ ggml_tensor * mixes = op->src[0];
                ਅਚਲ ggml_tensor * scale = op->src[1];
                ਅਚਲ ggml_tensor * ਅਧਾਰ  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(ਅਧਾਰ->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb ਦਲੀਲਾਂ_2 = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ ਅਧਾਰ->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(mixes), 1);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(scale), 2);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(ਅਧਾਰ),  3);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                ਅਚਲ ਪੂਰਨ nsg = std::min(4, ਦਲੀਲਾਂ_2.n_ਟੋਕਨ);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (ਦਲੀਲਾਂ_2.n_ਟੋਕਨ + nsg - 1)/nsg, 1, 1, 32, nsg, 1);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
            {
                ਅਚਲ ggml_tensor * x       = op->src[0];
                ਅਚਲ ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre ਦਲੀਲਾਂ_2 = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[2],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_x2    =*/ x->nb[2],
                    /*.nb_w0    =*/ weights->nb[0],
                    /*.nb_w1    =*/ weights->nb[1],
                    /*.nb_w2    =*/ weights->nb[2],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.scale    =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(x),       1);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(weights), 2);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op),      3);

                ਅਚਲ ਪੂਰਨ n_tiles = (ਦਲੀਲਾਂ_2.n_embd + 31)/32;
                ਅਚਲ ਪੂਰਨ nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, ਦਲੀਲਾਂ_2.n_ਟੋਕਨ, 1, 32, nsg, 1);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            {
                ਅਚਲ ggml_tensor * x        = op->src[0];
                ਅਚਲ ggml_tensor * residual = op->src[1];
                ਅਚਲ ggml_tensor * post     = op->src[2];
                ਅਚਲ ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post ਦਲੀਲਾਂ_2 = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[1],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_r0    =*/ residual->nb[0],
                    /*.nb_r1    =*/ residual->nb[1],
                    /*.nb_r2    =*/ residual->nb[2],
                    /*.nb_p0    =*/ post->nb[0],
                    /*.nb_p1    =*/ post->nb[1],
                    /*.nb_c0    =*/ comb ? comb->nb[0] : 0,
                    /*.nb_c1    =*/ comb ? comb->nb[1] : 0,
                    /*.nb_c2    =*/ comb ? comb->nb[2] : 0,
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                };

                ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(x),        1);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(residual), 2);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(post),     3);
                ਜੇ (comb) {
                    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(comb), 4);
                    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op),   5);
                } ਨਹੀਂਤਾਂ {
                    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op),   4);
                }

                ਅਚਲ ਪੂਰਨ n_tiles = (ਦਲੀਲਾਂ_2.n_embd + 31)/32;
                ਅਚਲ ਪੂਰਨ nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, ਦਲੀਲਾਂ_2.n_ਟੋਕਨ, 1, 32, nsg, 1);
            } ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_soft_max(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ਜੇ (ctx->use_fusion()) {
        ਪੂਰਨ n = 1;
        ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n);
        ਜੇ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            ਵਾਪਸ ggml_metal_op_topk_moe(ctx, idx);
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ max_bias;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    ((ਅਚਲ int32_t *) op->op_params) + 0, ਆਕਾਰ(scale));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, ((ਅਚਲ int32_t *) op->op_params) + 1, ਆਕਾਰ(max_bias));

    ਅਚਲ uint32_t n_ਸਿਰ      = op->src[0]->ne[2];
    ਅਚਲ  int32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floorf(log2f((ਦਸ਼ਮਲਵ) n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    // softmax

    ggml_metal_kargs_soft_max ਦਲੀਲਾਂ_2 = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_ਸਿਰ_log2,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    ਪੂਰਨ nth = 32; // SIMD width

    ਜੇ (ne00%4 == 0) {
        ਜਦੋਂ (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } ਨਹੀਂਤਾਂ {
        ਜਦੋਂ (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    ਅਚਲ size_t smem = pipeline.smem;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ਜੇ (op->src[1]) {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    } ਨਹੀਂਤਾਂ {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 2);
    }
    ਜੇ (op->src[2]) {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), 3);
    } ਨਹੀਂਤਾਂ {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 3);
    }
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op), 4);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਪੂਰਨ n_fuse = 1;
    ਤਾਰਕਿਕ use_silu = ਝੂਠ;

    ਜੇ (ctx->use_fusion()) {
        ਪੂਰਨ n = 1;
        ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n);
        ਜੇ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = ਸੱਚ;

            ctx->ਗਿਣਤੀ_fusions(fusion);
        }
    }

    ggml_metal_kargs_ssm_conv ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    ਅਚਲ ggml_metal_ਬਫਰ_id bid_dst = ggml_metal_get_ਬਫਰ_id(n_fuse > 1 ? ctx->ਗੰਢ(idx + n_fuse - 1) : op);

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    ਅਚਲ ਤਾਰਕਿਕ use_batched = (ne1 > 1);

    ਜੇ (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        ਪੂਰਨ BATCH_ਮਾਪ;
        ਜੇ      (ne1 > 128) BATCH_ਮਾਪ = 256;
        ਨਹੀਂਤਾਂ ਜੇ (ne1 > 64 ) BATCH_ਮਾਪ = 128;
        ਨਹੀਂਤਾਂ ਜੇ (ne1 > 32 ) BATCH_ਮਾਪ = 64;
        ਨਹੀਂਤਾਂ ਜੇ (ne1 > 16 ) BATCH_ਮਾਪ = 32;
        ਨਹੀਂਤਾਂ ਜੇ (ne1 > 8  ) BATCH_ਮਾਪ = 16;
        ਨਹੀਂਤਾਂ ਜੇ (ne1 > 4  ) BATCH_ਮਾਪ = 8;
        ਨਹੀਂਤਾਂ                BATCH_ਮਾਪ = 2;

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_ਮਾਪ, (int32_t) ne10, use_silu);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        ਅਚਲ ਪੂਰਨ n_token_batches = (ne1 + BATCH_ਮਾਪ - 1) / BATCH_ਮਾਪ;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_ਮਾਪ, 1, 1);
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op, (int32_t) ne10, use_silu);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_dst, 3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    ਜੇ (n_fuse > 1 && ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SSM_CONV + UNARY\n", __func__);
    }

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_ssm_ਜਾਂਚ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ ggml_tensor * src3 = op->src[3];
    ਅਚਲ ggml_tensor * src4 = op->src[4];
    ਅਚਲ ggml_tensor * src5 = op->src[5];
    ਅਚਲ ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    ਅਚਲ int64_t d_state      = ne00;
    ਅਚਲ int64_t d_inner      = ne01;
    ਅਚਲ int64_t n_ਸਿਰ       = ne02;
    ਅਚਲ int64_t n_ਸਮੂਹ      = ne41;
    ਅਚਲ int64_t n_seq_ਟੋਕਨ = ne12;
    ਅਚਲ int64_t n_seqs       = ne13;
    ਅਚਲ int64_t K            = ggml_get_op_params_i32(op, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(op->src[1]) + K*d_state*d_inner*n_ਸਿਰ*n_seqs == ggml_nelements(op));

    ggml_metal_kargs_ssm_ਜਾਂਚ ਦਲੀਲਾਂ_2 = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_ਸਿਰ,
        /*.n_group      =*/ n_ਸਮੂਹ,
        /*.n_seq_tokens =*/ n_seq_ਟੋਕਨ,
        /*.n_seq_tokens_total =*/ n_seq_ਟੋਕਨ,
        /*.token_offset =*/ 0,
        /*.n_seqs       =*/ n_seqs,
        /*.K            =*/ K,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    constexpr int64_t CHUNK = OP_SSM_ਜਾਂਚ_SSD_CS;

    ਅਚਲ int64_t snap_reserve = K > 1 ? K : 0; // tokens reserved for sequential kernel rollback snapshots
    ਅਚਲ int64_t mma_ਟੋਕਨ = ((n_seq_ਟੋਕਨ - snap_reserve) / CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    ਅਚਲ ਤਾਰਕਿਕ use_mma =
        mma_ਟੋਕਨ > 0 &&
        ne30 == 1 && // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev->has_simdgroup_mm && // hardware check for M1 or newer
        d_state % 8 == 0 && // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == OP_SSM_ਜਾਂਚ_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    ਅਚਲ ਸਵੈਚਾਲਿਤ dispatch = [&](ggml_metal_pipeline_with_params pipeline, int64_t nth, int64_t n_tg_x) {
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
        GGML_ASSERT(pipeline.smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), 3);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[3]), 4);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[4]), 5);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[5]), 6);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[6]), 7);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         8);
        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg_x, n_ਸਿਰ, n_seqs, nth, 1, 1);
    };

    ਜੇ (!use_mma) {
        dispatch(ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ(lib, op, ਝੂਠ), d_state, d_inner);
        ਵਾਪਸ 1;
    }

    ਦਲੀਲਾਂ_2.n_seq_ਟੋਕਨ = mma_ਟੋਕਨ;
    dispatch(
        ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ_ssd_mma(lib, op),
        OP_SSM_ਜਾਂਚ_SSD_NSG*32,
        1);

    ਜੇ (mma_ਟੋਕਨ < n_seq_ਟੋਕਨ) {
        ggml_metal_op_concurrency_reset(ctx);

        ਦਲੀਲਾਂ_2.n_seq_ਟੋਕਨ = n_seq_ਟੋਕਨ - mma_ਟੋਕਨ;
        ਦਲੀਲਾਂ_2.token_ਵਿੱਥ = mma_ਟੋਕਨ;
        dispatch(ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ(lib, op, ਸੱਚ), d_state, d_inner);
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_rwkv(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    ਅਚਲ int64_t T = op->src[0]->ne[2];
    ਅਚਲ int64_t C = op->ne[0];
    ਅਚਲ int64_t H = op->src[0]->ne[1];

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    ਪੂਰਨ ida = 0;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[3]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[4]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[5]), ida++);
    ਜੇ (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, (ਖਾਲੀ *) &B, ਆਕਾਰ(B), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, (ਖਾਲੀ *) &T, ਆਕਾਰ(T), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, (ਖਾਲੀ *) &C, ਆਕਾਰ(C), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, (ਖਾਲੀ *) &H, ਆਕਾਰ(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ਤਾਰਕਿਕ use_fusion = ctx->use_fusion();
    ਅਚਲ ਪੂਰਨ  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    // when fused with the trailing cache cpy, the snapshots are written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_CACHE)
    ggml_metal_ਬਫਰ_id bid_ਬਾਹਰਲਾ = ggml_metal_get_ਬਫਰ_id(op);
    uint64_t nb_ਬਾਹਰਲਾ = 0;
    ਪੂਰਨ n_fuse = 1;

    ਜੇ (use_fusion) {
        ਪੂਰਨ n = 1;
        ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n);

        ਜੇ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_GDN_CACHE) {
            ਅਚਲ ggml_tensor * dst_cache = ctx->ਗੰਢ(idx + 1)->src[1]; // cache view

            bid_ਬਾਹਰਲਾ = ggml_metal_get_ਬਫਰ_id(dst_cache);
            nb_ਬਾਹਰਲਾ  = dst_cache->nb[2]/ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            n_fuse = 2;

            ctx->ਗਿਣਤੀ_fusions(fusion);

            ਜੇ (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: GATED_DELTA_NET + CPY\n", __func__);
            }
        }
    }

    ਪੂਰਨ ida = 0;

    ggml_metal_kargs_gated_delta_net ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
        /*.ns12 =*/ (int32_t) (nb12/ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
        /*.ns22 =*/ (int32_t) (nb22/ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.nb_out =*/ nb_ਬਾਹਰਲਾ,
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2),                  ida++); //ਦਲੀਲਾਂ
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), ida++); // q
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), ida++); // k
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), ida++); // v
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[5]), ida++); // state
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         ida++); // dst (attn)
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_ਬਾਹਰਲਾ,                              ida++); // state_out

    ਅਚਲ ਪੂਰਨ nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_solve_tri(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ਅਚਲ ਪੂਰਨ nsg = pipeline.nsg;

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਸੈੱਟ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_src1 = ggml_metal_get_ਬਫਰ_id(op->src[1]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ਅਚਲ size_t pnb1 = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਅਚਲ size_t pnb2 = ((ਅਚਲ int32_t *) op->op_params)[1];
    ਅਚਲ size_t pnb3 = ((ਅਚਲ int32_t *) op->op_params)[2];
    ਅਚਲ size_t offs = ((ਅਚਲ int32_t *) op->op_params)[3];

    ਅਚਲ ਤਾਰਕਿਕ inplace = (ਤਾਰਕਿਕ) ((ਅਚਲ int32_t *) op->op_params)[4];

    ਜੇ (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy ਦਲੀਲਾਂ_2 = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

        ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_ਮਾਪ(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    ਜੇ (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_ਮਾਪ(op->type);
    }

    ਪੂਰਨ nth = std::min<ਪੂਰਨ>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    ਪੂਰਨ nrptg = 1;

    // TODO: relax this constraint in the future
    ਜੇ (ggml_blck_ਮਾਪ(op->src[1]->type) == 1 && ggml_blck_ਮਾਪ(op->type) == 1) {
        ਜੇ (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            ਜੇ (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<ਪੂਰਨ>(nth, nk0);

    ggml_metal_kargs_cpy ਦਲੀਲਾਂ_2 = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_ਮਾਪ(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    ਅਚਲ ਪੂਰਨ nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src1, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_cpy(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_ਮਾਪ(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    ਜੇ (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_ਮਾਪ(op->type);
    }

    ਪੂਰਨ nth = std::min<ਪੂਰਨ>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    ਪੂਰਨ nrptg = 1;

    // TODO: relax this constraint in the future
    ਜੇ (ggml_blck_ਮਾਪ(op->src[0]->type) == 1 && ggml_blck_ਮਾਪ(op->type) == 1) {
        ਜੇ (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            ਜੇ (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<ਪੂਰਨ>(nth, nk0);

    ggml_metal_kargs_cpy ਦਲੀਲਾਂ_2 = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ਅਚਲ ਪੂਰਨ nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_pool_1d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    ਅਚਲ int32_t k0 = opts[1];
    ਅਚਲ int32_t s0 = opts[2];
    ਅਚਲ int32_t p0 = opts[3];

    ਅਚਲ int64_t IW = op->src[0]->ne[0];
    ਅਚਲ int64_t OW = op->ne[0];

    ਅਚਲ int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d ਦਲੀਲਾਂ_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    ਅਚਲ ਪੂਰਨ nth = std::min(256, std::max(1, k0));
    ਅਚਲ ਪੂਰਨ ntg = (ਪੂਰਨ) np;

    ਅਚਲ size_t smem = op_pool == GGML_OP_POOL_AVG ?
        2*32*ਆਕਾਰ(ਦਸ਼ਮਲਵ) :
        32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_pool_1d, ਆਕਾਰ(ਦਲੀਲਾਂ_pool_1d),  0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);
    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_fwht(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_tensor * src1 = op->src[1];

    ਅਚਲ int64_t n = src1->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(src1);

    ggml_metal_kargs_fwht ਦਲੀਲਾਂ_2 = {
        /*.nrows = */ (int32_t) nrows,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, src1->type);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(src1), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op), 2);

    ਅਚਲ ਪੂਰਨ th_max = ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline);
    ਅਚਲ ਪੂਰਨ simd_ਮਾਪ = 32;

    ਜੇ (n >= GGML_METAL_FWHT_TG_MIN_N) {
        GGML_ASSERT(th_max >= GGML_METAL_FWHT_TG_NT);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, GGML_METAL_FWHT_TG_NT, 1, 1);

        ਵਾਪਸ 1;
    }

    ਪੂਰਨ sg_ਪ੍ਰਤੀ_tg = 2;
    sg_ਪ੍ਰਤੀ_tg = std::min(sg_ਪ੍ਰਤੀ_tg, th_max/simd_ਮਾਪ);
    sg_ਪ੍ਰਤੀ_tg = std::max(sg_ਪ੍ਰਤੀ_tg, 1);

    ਅਚਲ int64_t n_tg = (nrows + sg_ਪ੍ਰਤੀ_tg - 1) / sg_ਪ੍ਰਤੀ_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, 32*sg_ਪ੍ਰਤੀ_tg, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_pool_2d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    ਅਚਲ int32_t k0 = opts[1];
    ਅਚਲ int32_t k1 = opts[2];
    ਅਚਲ int32_t s0 = opts[3];
    ਅਚਲ int32_t s1 = opts[4];
    ਅਚਲ int32_t p0 = opts[5];
    ਅਚਲ int32_t p1 = opts[6];

    ਅਚਲ int64_t IH = op->src[0]->ne[1];
    ਅਚਲ int64_t IW = op->src[0]->ne[0];

    ਅਚਲ int64_t N  = op->ne[3];
    ਅਚਲ int64_t OC = op->ne[2];
    ਅਚਲ int64_t OH = op->ne[1];
    ਅਚਲ int64_t OW = op->ne[0];

    ਅਚਲ int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d ਦਲੀਲਾਂ_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), (ਪੂਰਨ) np);
    ਅਚਲ ਪੂਰਨ ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_pool_2d, ਆਕਾਰ(ਦਲੀਲਾਂ_pool_2d), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

// the number of nodes from idx on that the fusion table fuses as pattern id into one dispatch, 1 if it fuses none
ਸਥਿਰ ਪੂਰਨ ggml_metal_op_try_fusion(ggml_metal_op_t ctx, ਪੂਰਨ idx, ggml_metal_fusion_id id) {
    ਜੇ (!ctx->use_fusion()) {
        ਵਾਪਸ 1;
    }

    ਪੂਰਨ n = 1;
    ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n);
    ਜੇ (fusion == ਖਾਲੀ_ਸੂਚਕ || ggml_metal_fusion_get_id(fusion) != id) {
        ਵਾਪਸ 1;
    }

    ctx->ਗਿਣਤੀ_fusions(fusion);

    ਜੇ (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: %s to %s, %d nodes\n", __func__, ggml_op_name(ctx->ਗੰਢ(idx)->op), ggml_op_name(ctx->ਗੰਢ(idx + n - 1)->op), n);
    }

    ਵਾਪਸ n;
}

ਸਥਿਰ ਪੂਰਨ ggml_metal_op_mul_mat_mma(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    // the MMA store adds the residual when the table fuses the ADD after this mat-mul
    ਅਚਲ ਪੂਰਨ  n_fuse   = ggml_metal_op_try_fusion(ctx, idx, GGML_METAL_FUSION_MUL_MAT_ਜੋੜੋ);
    ਅਚਲ ਤਾਰਕਿਕ fuse_ਜੋੜੋ = n_fuse > 1;

    ਅਚਲ ggml_tensor * dst = op;
    ਅਚਲ ggml_tensor * res = dst;

    ਜੇ (fuse_ਜੋੜੋ) {
        dst = ctx->ਗੰਢ(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
    }

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_ਜੋੜੋ);

    ggml_metal_kargs_mul_mv_ext ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.ne11  =*/ ne11,
        /*.ne12  =*/ ne12,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb13  =*/ nb13,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.r2    =*/ (int16_t) (ne12/ne02),
        /*.r3    =*/ (int16_t) (ne13/ne03),
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(dst),        3);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(res),        4);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, pipeline.smem, 0);

    ਅਚਲ ਪੂਰਨ rows0 = pipeline.nr0;
    ਅਚਲ ਪੂਰਨ rows1 = pipeline.nr1;

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows0 - 1)/rows0, (ne11 + rows1 - 1)/rows1, ne12*ne13, 32, pipeline.nsg, 1);

    ਵਾਪਸ n_fuse;
}

// the generic mat-vec kernel, or its 2-row Q4_0 variant if nc
ਸਥਿਰ ਪੂਰਨ ggml_metal_op_mul_mat_mv(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    ਅਚਲ int16_t r2 = ne12/ne02;
    ਅਚਲ int16_t r3 = ne13/ne03;

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

    ਅਚਲ ਪੂਰਨ nr0 = pipeline.nr0;
    ਅਚਲ ਪੂਰਨ nr1 = pipeline.nr1;
    ਅਚਲ ਪੂਰਨ nsg = pipeline.nsg;

    ਅਚਲ size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ਜੇ (op->src[0]->type == GGML_TYPE_F32 ||
        op->src[0]->type == GGML_TYPE_F16 ||
        op->src[0]->type == GGML_TYPE_BF16 ||
        op->src[0]->type == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    } ਨਹੀਂਤਾਂ {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_mul_mat(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    ਜੇ (ggml_metal_op_mul_mat_use_fwht(op, props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ)) {
        ਵਾਪਸ ggml_metal_op_fwht(ctx, idx);
    }

    ਜੇ (props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_mma(op)) {
        ਵਾਪਸ ggml_metal_op_mul_mat_mma(ctx, idx);
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    ਅਚਲ int16_t r2 = ne12/ne02;
    ਅਚਲ int16_t r3 = ne13/ne03;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    ਜੇ (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q2_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           ਝੂਠ) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           ਝੂਠ) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        ਅਚਲ ਪੂਰਨ nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        ਜੇ (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } ਨਹੀਂਤਾਂ ਜੇ (ne00 % 128 == 0) {
            nxpsg = 8;
        } ਨਹੀਂਤਾਂ {
            nxpsg = 4;
        }

        ਅਚਲ int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        ਅਚਲ int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        ਚੋਣ (ne11) {
            ਹਾਲਤ 2:
                r1ptg = 2; ਰੋਕੋ;
            ਹਾਲਤ 3:
            ਹਾਲਤ 6:
                r1ptg = 3; ਰੋਕੋ;
            ਹਾਲਤ 4:
            ਹਾਲਤ 7:
            ਹਾਲਤ 8:
                r1ptg = 4; ਰੋਕੋ;
            ਹਾਲਤ 5:
                r1ptg = 5; ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("unsupported ne11");
        };

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext ਦਲੀਲਾਂ_2 = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_metal_op_mul_mat_use_mm(op, props_dev->has_simdgroup_mm)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm ਦਲੀਲਾਂ_2 = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

        ਅਚਲ size_t smem = pipeline.smem;

        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

        ਅਚਲ ਪੂਰਨ nr0 = pipeline.nr0;
        ਅਚਲ ਪੂਰਨ nr1 = pipeline.nr1;
        ਅਚਲ ਪੂਰਨ nsg = pipeline.nsg;

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ ggml_metal_op_mul_mat_mv(ctx, idx);
    }

    ਵਾਪਸ 1;
}

// upper bound on the number of entry tiles emitted by kernel_mul_mm_id_map0: the full tiles
// over all entries plus at most one partial tile per expert
ਸਥਿਰ int64_t ggml_metal_op_mul_mat_id_max_tiles(ਅਚਲ ggml_tensor * op) {
    ਅਚਲ int64_t ne02 = op->src[0]->ne[2]; // n_expert
    ਅਚਲ int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    ਅਚਲ int64_t ne21 = op->src[2]->ne[1]; // n_token

    ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ = ne20*ne21;
    ਅਚਲ int64_t tiles  = (n_ਪੰਕਤੀਆਂ + 31)/32 + (ne02 < n_ਪੰਕਤੀਆਂ ? ne02 : n_ਪੰਕਤੀਆਂ) - 1;

    ਵਾਪਸ n_ਪੰਕਤੀਆਂ < tiles ? n_ਪੰਕਤੀਆਂ : tiles;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_MUL_MAT_ID);

    // row map of kernel_mul_mm_id: total tile count + one
    // (expert, first entry, valid rows) descriptor per tile
    ਅਚਲ int64_t max_tiles = ggml_metal_op_mul_mat_id_max_tiles(op);

    ਵਾਪਸ ggml_type_ਮਾਪ(GGML_TYPE_I32)*(1 + 3*max_tiles);
}

size_t ggml_metal_op_mul_mat_id_extra_ids(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_MUL_MAT_ID);

    ਅਚਲ int64_t ne02 = op->src[0]->ne[2]; // n_expert
    ਅਚਲ int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    ਅਚਲ int64_t ne21 = op->src[2]->ne[1]; // n_token

    // one entry per (expert used, token) pair of ids plus one padding slot per expert
    // (see kernel_mul_mm_id_map0)
    ਵਾਪਸ ggml_type_ਮਾਪ(GGML_TYPE_I32)*(ne20*ne21 + ne02);
}

size_t ggml_metal_op_mul_mat_id_extra_amax(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_MUL_MAT_ID);

    GGML_ਅਣਵਰਤਿਆ(op);

    // 2 scaling factors (8 bytes) + N_MM_NPART_AMAX per-threadgroup scales for stage-1
    ਵਾਪਸ 8 + N_MM_NPART_AMAX*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
}

ਪੂਰਨ ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_src1 = ggml_metal_get_ਬਫਰ_id(op->src[1]);
    ggml_metal_ਬਫਰ_id bid_src2 = ggml_metal_get_ਬਫਰ_id(op->src[2]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ਅਚਲ uint32_t r2 = 1;
    ਅਚਲ uint32_t r3 = 1;

    ਜੇ (ggml_metal_op_mul_mat_id_use_mm(op, props_dev->has_simdgroup_mm)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_ਬਫਰ_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_ਬਫਰ_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        ggml_metal_ਬਫਰ_id bid_amax = bid_ids;
        bid_amax.offs += ggml_metal_op_mul_mat_id_extra_ids(op);

        // src1 prec [TAG_GGML_PREC]
        ਅਚਲ ਤਾਰਕਿਕ use_amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

        // src1 rescale factors, computed before the matmul
        // ref: https://github.com/ggml-org/llama.cpp/pull/26223
        ਜੇ (use_amax) {
            ggml_metal_kargs_mul_mm_id_amax ਦਲੀਲਾਂ_2 = {
                /*.ne00 =*/ ne10,
                /*.ne01 =*/ ne11,
                /*.ne02 =*/ ne12,
                /*.nb01 =*/ nb11,
                /*.nb02 =*/ nb12,
            };

            ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax_part(lib);

            ਅਚਲ size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src1, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_amax, 2);

            ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, N_MM_NPART_AMAX, 1, 1, 256, 1, 1);
        }

        {
            ggml_metal_kargs_mul_mm_id_map0 ਦਲੀਲਾਂ_2 = {
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            ਅਚਲ size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src2, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tpe,  2);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_ids,  3);

            ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        ਜੇ (use_amax) {
            ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax(lib);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_amax, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 32, 1, 1);

            // the next kernel has to wait for the amax data
            ggml_metal_op_concurrency_reset(ctx);
        }

        {
            ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id ਦਲੀਲਾਂ_2 = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src1, 2);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tpe,  3);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_ids,  4);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  5);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_amax, 6);

            ਅਚਲ size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

            ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

            ਅਚਲ ਪੂਰਨ max_tiles = (ਪੂਰਨ) ggml_metal_op_mul_mat_id_max_tiles(op);

            ggml_metal_encoder_dispatch_threadgroups(enc, max_tiles, (ne01 + 63)/64, 1, 128, 1, 1);
        }
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        ਅਚਲ ਪੂਰਨ nr0 = pipeline.nr0;
        ਅਚਲ ਪੂਰਨ nr1 = pipeline.nr1;
        ਅਚਲ ਪੂਰਨ nsg = pipeline.nsg;

        ਅਚਲ size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id ਦਲੀਲਾਂ_2 = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        ਜੇ (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_src1, 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_dst,  3);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_src2, 4);

        ਅਚਲ int64_t _ne1 = 1;
        ਅਚਲ int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

        ਜੇ (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } ਨਹੀਂਤਾਂ {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਜੋੜੋ_id(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ggml_metal_kargs_ਜੋੜੋ_id ਦਲੀਲਾਂ_2 = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਅਧਾਰ(lib, GGML_OP_ਜੋੜੋ_ID);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), 3);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         4);

    ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਤਾਰਕਿਕ ggml_metal_op_flash_attn_ext_use_vec(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅਚਲ int64_t ne00 = op->src[0]->ne[0]; // head size
    ਅਚਲ int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    ਵਾਪਸ (ne01 < 20) && (ne00 % 32 == 0);
}

// ref: https://github.com/ggml-org/llama.cpp/pull/27390
// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_op_flash_attn_ext_use_kv_f16(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    // large heads need the upfront dequant to fit the non-vec threadgroup memory
    ਜੇ (op->src[0]->ne[1] < 32 &&
        (op->src[0]->ne[0] < 512 || ggml_metal_op_flash_attn_ext_use_vec(op))) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਚੋਣ (op->src[1]->type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_op_flash_attn_ext_use_tensor(ਅਚਲ ggml_tensor * op, ਤਾਰਕਿਕ has_tensor) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਜੇ (!has_tensor || ggml_metal_op_flash_attn_ext_use_vec(op)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t ne01 = op->src[0]->ne[1];
    ਅਚਲ int64_t ne02 = op->src[0]->ne[2];
    ਅਚਲ int64_t ne03 = op->src[0]->ne[3];

    ਅਚਲ int64_t dk = op->src[1]->ne[0];
    ਅਚਲ int64_t dv = op->src[2]->ne[0];

    ਅਚਲ ਤਾਰਕਿਕ dk_dv_ok = (dk == 64  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 192 && dv == 128) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    ਜੇ (!dk_dv_ok) {
        ਵਾਪਸ ਝੂਠ;
    }

    // large heads use fewer queries per threadgroup, so that the queries fit in threadgroup memory
    ਅਚਲ int64_t nqptg = dk >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG;

    // few heads and small batches do not fill the GPU - the half8x8 kernel is faster there
    // TODO: tune per device
    ਜੇ (((ne01 + nqptg - 1)/nqptg)*ne02*ne03*dk < 8192) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_BF16 && !ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ op->src[1]->ne[1] % OP_FLASH_ATTN_EXT_TENSOR_NCPSG == 0;
}

// returns the n_kv_max hint if the sparse path is available for this op, or 0 otherwise
// the mask (src[3]) remains the single source of truth: finite entries are the valid KV positions,
// n_kv_max is only an upper bound on their number per mask row, used to size the index lists
ਸਥਿਰ ਪੂਰਨ ggml_metal_op_flash_attn_ext_n_kv_max_sparse(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    int32_t n_kv_max = 0;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&n_kv_max, ((ਅਚਲ int32_t *) op->op_params) + 4, ਆਕਾਰ(n_kv_max));

    ਜੇ (n_kv_max <= 0) {
        ਵਾਪਸ 0;
    }

    // the sparse indices are gathered from the mask
    ਜੇ (!op->src[3]) {
        ਵਾਪਸ 0;
    }

    // bound the size of the index lists
    ਜੇ (n_kv_max > 4096) {
        ਵਾਪਸ 0;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    ਅਚਲ int64_t dk = op->src[1]->ne[0];
    ਅਚਲ int64_t dv = op->src[2]->ne[0];

    ਅਚਲ ਤਾਰਕਿਕ dk_dv_ok = (dk == 32  && dv == 32)  ||
                          (dk == 64  && dv == 64)  ||
                          (dk == 96  && dv == 96)  ||
                          (dk == 96  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 128 && dv == 96)  ||
                          (dk == 192 && dv == 128) ||
                          (dk == 192 && dv == 192) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 320 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    ਜੇ (!dk_dv_ok) {
        ਵਾਪਸ 0;
    }

    ਚੋਣ (op->src[1]->type) {
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ 0;
    }

    ਵਾਪਸ n_kv_max;
}

// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_op_flash_attn_ext_v_is_view_ਦਾ_k(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅਚਲ ggml_tensor * K = op->src[1];
    ਅਚਲ ggml_tensor * V = op->src[2];

    ਵਾਪਸ V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
ਸਥਿਰ size_t ggml_metal_op_flash_attn_ext_kv_f16_k_ਮਾਪ(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    ਵਾਪਸ GGML_PAD(ਆਕਾਰ(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    ਅਚਲ ਤਾਰਕਿਕ has_ਨਕਾਬ = op->src[3] != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਤਾਰਕਿਕ use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    ਅਚਲ ਤਾਰਕਿਕ v_is_view_ਦਾ_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_ਦਾ_k(op);
    uint64_t nb11_pad = nb11;
    uint64_t nb21_pad = nb21;

    ਜੇ (use_kv_f16) {
        nb11_pad = ਆਕਾਰ(ggml_fp16_t)*ne10;
        nb21_pad = ਆਕਾਰ(ggml_fp16_t)*(v_is_view_ਦਾ_k ? ne10 : ne20);
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    ਜੇ (ਝੂਠ) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        ਅਚਲ ਤਾਰਕਿਕ has_kvpad = ਸੱਚ;

        ਜੇ (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_ਨਕਾਬ ? ggml_type_ਮਾਪ(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } ਨਹੀਂਤਾਂ {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        ਅਚਲ ਤਾਰਕਿਕ has_kvpad = ਸੱਚ;

        ਜੇ (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_ਨਕਾਬ ? ggml_type_ਮਾਪ(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    ਵਾਪਸ res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    ਅਚਲ ਤਾਰਕਿਕ has_ਨਕਾਬ = op->src[3] != ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (!has_ਨਕਾਬ) {
        ਵਾਪਸ res;
    }

    ਅਚਲ ਤਾਰਕਿਕ is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    ਅਚਲ ਪੂਰਨ nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    ਅਚਲ ਪੂਰਨ ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    ਅਚਲ int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    ਅਚਲ int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_ਮਾਪ(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    ਵਾਪਸ res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    ਜੇ (ਸੱਚ) {
        ਅਚਲ int64_t nwg = 32;
        ਅਚਲ int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_ਮਾਪ(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    ਵਾਪਸ res;
}

size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
    //    return 0;
    //}

    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    ਅਚਲ size_t k_ਮਾਪ = ggml_metal_op_flash_attn_ext_kv_f16_k_ਮਾਪ(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    ਅਚਲ ਤਾਰਕਿਕ v_is_view_ਦਾ_k = ggml_metal_op_flash_attn_ext_v_is_view_ਦਾ_k(op);
    ਜੇ (v_is_view_ਦਾ_k) {
        ਵਾਪਸ k_ਮਾਪ;
    }

    ਅਚਲ size_t v_ਮਾਪ = GGML_PAD(ਆਕਾਰ(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);

    ਵਾਪਸ k_ਮਾਪ + v_ਮਾਪ;
}

// size of the sparse index lists: one list of KV indices per mask row,
// padded with -1 up to a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG
size_t ggml_metal_op_flash_attn_ext_extra_idx(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);

    ਅਚਲ ਪੂਰਨ n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);

    ਜੇ (n_kv_max <= 0) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ਪੂਰਨ n_kv_max_padded = GGML_PAD(n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG);

    ਵਾਪਸ GGML_PAD(ਆਕਾਰ(int32_t)*(size_t) n_kv_max_padded*ne31*ne32*ne33, 16);
}

ਪੂਰਨ ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ max_bias;
    ਦਸ਼ਮਲਵ logit_softcap;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         ((ਅਚਲ int32_t *) op->op_params) + 0, ਆਕਾਰ(scale));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      ((ਅਚਲ int32_t *) op->op_params) + 1, ਆਕਾਰ(max_bias));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, ((ਅਚਲ int32_t *) op->op_params) + 2, ਆਕਾਰ(logit_softcap));

    ਜੇ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ਅਚਲ ਤਾਰਕਿਕ has_ਨਕਾਬ  = op->src[3] != ਨਿਰਾ;
    ਅਚਲ ਤਾਰਕਿਕ has_sinks = op->src[4] != ਨਿਰਾ;
    ਅਚਲ ਤਾਰਕਿਕ has_bias  = max_bias != 0.0f;
    ਅਚਲ ਤਾਰਕਿਕ has_scap  = logit_softcap != 0.0f;

    ਅਚਲ uint32_t n_ਸਿਰ      = op->src[0]->ne[2];
    ਅਚਲ  int32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floorf(log2f((ਦਸ਼ਮਲਵ) n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_src1 = ggml_metal_get_ਬਫਰ_id(op->src[1]);
    ggml_metal_ਬਫਰ_id bid_src2 = ggml_metal_get_ਬਫਰ_id(op->src[2]);
    ggml_metal_ਬਫਰ_id bid_src3 = has_ਨਕਾਬ  ? ggml_metal_get_ਬਫਰ_id(op->src[3]) : bid_src0;
    ggml_metal_ਬਫਰ_id bid_src4 = has_sinks ? ggml_metal_get_ਬਫਰ_id(op->src[4]) : bid_src0;

    ggml_metal_ਬਫਰ_id bid_dst = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_ਬਫਰ_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_ਬਫਰ_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_ਬਫਰ_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    ggml_metal_ਬਫਰ_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    ਅਚਲ ਪੂਰਨ n_kv_max_sparse = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    ਅਚਲ ਤਾਰਕਿਕ use_sparse = n_kv_max_sparse > 0;
    ਅਚਲ ਪੂਰਨ n_kv_max_padded = use_sparse ? GGML_PAD(n_kv_max_sparse, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    ਅਚਲ ਤਾਰਕਿਕ use_kv_f16 = !use_sparse && ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    ggml_metal_ਬਫਰ_id bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    ggml_metal_ਬਫਰ_id bid_k = bid_src1;
    ggml_metal_ਬਫਰ_id bid_v = bid_src2;

    uint64_t nb10_attn = nb10;
    uint64_t nb11_attn = nb11;
    uint64_t nb12_attn = nb12;
    uint64_t nb13_attn = nb13;
    uint64_t nb20_attn = nb20;
    uint64_t nb21_attn = nb21;
    uint64_t nb22_attn = nb22;
    uint64_t nb23_attn = nb23;

    ਜੇ (use_kv_f16) {
        ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);

        ਅਚਲ ਤਾਰਕਿਕ v_is_view_ਦਾ_k = ggml_metal_op_flash_attn_ext_v_is_view_ਦਾ_k(op);

        ਅਚਲ int64_t nblocks1_64 = (ne10/ggml_blck_ਮਾਪ(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
        ਅਚਲ int32_t nblocks1 = nblocks1_64;

        ggml_metal_ਬਫਰ_id bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_ਮਾਪ(op);

        ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline0), 256);

        // K
        ggml_metal_kargs_flash_attn_ext_kv_f16 ਦਲੀਲਾਂ_k = {
            /*.ne0    =*/ ne10,
            /*.ne1    =*/ ne11,
            /*.ne2    =*/ ne12,
            /*.ne3    =*/ ne13,
            /*.nb0    =*/ nb10,
            /*.nb1    =*/ nb11,
            /*.nb2    =*/ nb12,
            /*.nb3    =*/ nb13,
            /*.nblocks =*/ nblocks1,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_k, ਆਕਾਰ(ਦਲੀਲਾਂ_k), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src1,        1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        ਜੇ (!v_is_view_ਦਾ_k) {
            ਅਚਲ int64_t nblocks2_64 = (ne20/ggml_blck_ਮਾਪ(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
            ਅਚਲ int32_t nblocks2 = nblocks2_64;

            ggml_metal_kargs_flash_attn_ext_kv_f16 ਦਲੀਲਾਂ_v = {
                /*.ne0    =*/ ne20,
                /*.ne1    =*/ ne21,
                /*.ne2    =*/ ne22,
                /*.ne3    =*/ ne23,
                /*.nb0    =*/ nb20,
                /*.nb1    =*/ nb21,
                /*.nb2    =*/ nb22,
                /*.nb3    =*/ nb23,
                /*.nblocks =*/ nblocks2,
            };

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_v, ਆਕਾਰ(ਦਲੀਲਾਂ_v), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src2,        1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_v_f16,       2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        ggml_metal_op_concurrency_reset(ctx);

        bid_k = bid_kv_f16;
        bid_v = v_is_view_ਦਾ_k ? bid_k : bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = ਆਕਾਰ(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        ਜੇ (v_is_view_ਦਾ_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } ਨਹੀਂਤਾਂ {
            // contiguous F16 layout of the dequantized V
            nb20_attn = ਆਕਾਰ(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne21;
            nb23_attn = nb22_attn*ne22;
        }
    }

    ਜੇ (!use_sparse && ggml_metal_op_flash_attn_ext_use_tensor(op, props_dev->has_tensor)) {
        // tensor API kernel
        ਅਚਲ ਪੂਰਨ nqptg = ne00 >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG; // queries per threadgroup
        ਅਚਲ ਪੂਰਨ ncpsg = OP_FLASH_ATTN_EXT_TENSOR_NCPSG; // cache values per threadgroup
        ਅਚਲ ਪੂਰਨ nsg   = OP_FLASH_ATTN_EXT_TENSOR_NSG;

        ਜੇ (has_ਨਕਾਬ) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &args0, ਆਕਾਰ(args0), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_blk,  2);

            ਅਚਲ int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            ਅਚਲ int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);
        }

        ਅਚਲ int32_t ns10 = nb11_attn/nb10_attn;
        ਅਚਲ int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext ਦਲੀਲਾਂ_2 = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_ਸਿਰ_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // shared memory layout: queries (16-bit), scores (float), probabilities (16-bit), row scale (float), rescale flag (int)
        ਅਚਲ size_t smem = GGML_PAD(nqptg*ne00*ਆਕਾਰ(ggml_fp16_t) + nqptg*ncpsg*(ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(ggml_fp16_t)) + nqptg*ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int32_t), 16);

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_tensor(lib, op, has_ਨਕਾਬ, has_sinks, has_bias, has_scap);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
        GGML_ASSERT(smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_k,    2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_v,    3);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 4);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src4, 5);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_blk,  6);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  7);

        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } ਨਹੀਂਤਾਂ ਜੇ (!use_sparse && !ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        ਅਚਲ ਪੂਰਨ nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        ਅਚਲ ਪੂਰਨ ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        ਤਾਰਕਿਕ need_sync = ਝੂਠ;

        ਅਚਲ ਤਾਰਕਿਕ has_kvpad = ne11 % ncpsg != 0;

        ਜੇ (has_kvpad) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_ਨਕਾਬ, ncpsg);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &args0, ਆਕਾਰ(args0), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_k,    1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_v,    2);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 3);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_pad,  4);

            ਪ੍ਰਤਿਜ੍ਞਾ(ne12 == ne22);
            ਪ੍ਰਤਿਜ੍ਞਾ(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = ਸੱਚ;
        }

        ਜੇ (has_ਨਕਾਬ) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &args0, ਆਕਾਰ(args0), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_blk,  2);

            ਅਚਲ int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            ਅਚਲ int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = ਸੱਚ;
        }

        ਜੇ (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        ਅਚਲ ਪੂਰਨ is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        ਅਚਲ int64_t dv_pad = GGML_PAD(ne20, 64);

        ਸਵੈਚਾਲਿਤ fa_smem = [&](int32_t nsg) -> size_t {
            ਅਚਲ size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
            ਵਾਪਸ GGML_PAD(smem_half*ਆਕਾਰ(ggml_fp16_t), 16);
        };

        // simdgroups per threadgroup (a.k.a. warps)
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        ਅਚਲ size_t smem = fa_smem(nsg);

        ਅਚਲ int32_t ns10 = nb11_attn/nb10_attn;
        ਅਚਲ int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext ਦਲੀਲਾਂ_2 = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_ਸਿਰ_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_ਨਕਾਬ, has_sinks, has_bias, has_scap, has_kvpad, nsg, use_kv_f16, ns10, ns20);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_k,    2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_v,    3);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 4);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src4, 5);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_pad,  6);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_blk,  7);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  8);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } ਨਹੀਂਤਾਂ {
        // half4x4 kernel
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        ਸਵੈਚਾਲਿਤ cfg = use_sparse
                ? ggml_metal_tuning::fa_vec_baseline_cfg((ਪੂਰਨ) ne00, (ਪੂਰਨ) ne20)
                : ggml_metal_tuning::fa_vec_pick(
                          props_dev->gpu_family,
                          (ਪੂਰਨ) op->src[1]->type,
                          (ਪੂਰਨ) ne00, (ਪੂਰਨ) ne20,   // dk, dv (ne00 == dk for FA)
                          ne11, ne01);

        ਪੂਰਨ nqptg = cfg.Q; // queries per threadgroup

        ਅਚਲ ਪੂਰਨ ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        ਅਚਲ ਪੂਰਨ nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg == 1 || nqptg == 2 || nqptg == 4);  // only instantiated Q values
        GGML_ASSERT(ncpsg  % 32 == 0);

        ਤਾਰਕਿਕ need_sync = ਝੂਠ;

        ਅਚਲ ਤਾਰਕਿਕ has_kvpad = !use_sparse && ne11 % ncpsg != 0;

        ਜੇ (use_sparse) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0);

            GGML_ASSERT(ne30 == ne11);

            ggml_metal_kargs_flash_attn_ext_vec_idx args0 = {
                /*.ne30              =*/ ne30,
                /*.ne31              =*/ ne31,
                /*.ne32              =*/ ne32,
                /*.ne33              =*/ ne33,
                /*.nb31              =*/ nb31,
                /*.nb32              =*/ nb32,
                /*.nb33              =*/ nb33,
                /*.n_kv_max          =*/ n_kv_max_sparse,
                /*.n_kv_max_padded   =*/ n_kv_max_padded,
            };

            ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &args0, ਆਕਾਰ(args0), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_idx,  2);

            ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline0), 256);
            nth = std::max(32, (nth/32)*32);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = ਸੱਚ;
        }

        ਜੇ (has_kvpad) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_ਨਕਾਬ, ncpsg);

            ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &args0, ਆਕਾਰ(args0), 0);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_k,    1);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_v,    2);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 3);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_pad,  4);

            ਪ੍ਰਤਿਜ੍ਞਾ(ne12 == ne22);
            ਪ੍ਰਤਿਜ੍ਞਾ(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = ਸੱਚ;
        }

        ਜੇ (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (only when use_sparse)
        ਅਚਲ int64_t dk_pad = GGML_PAD(ne00, 128);
        ਅਚਲ int64_t dv_pad = GGML_PAD(ne20, 128);

        ਸਵੈਚਾਲਿਤ fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
            ਅਚਲ size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
            ਵਾਪਸ GGML_PAD(smem_half*ਆਕਾਰ(ggml_fp16_t) + (use_sparse ? (size_t) nsg*ncpsg*ਆਕਾਰ(ਪੂਰਨ) : 0), 16);
        };

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        ਜੇ (use_sparse) {
            ਜੇ (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                ਜੇ (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } ਨਹੀਂਤਾਂ {
                    ਜਦੋਂ (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                        nsg *= 2;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                ਜਦੋਂ (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                    nsg *= 2;
                }
            }
        } ਨਹੀਂਤਾਂ {
            nwg = 32;
            nsg = 1;
            ਜਦੋਂ (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        ਜੇ (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ) {
            cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((ਪੂਰਨ) ne00, (ਪੂਰਨ) ne20);
            nqptg = cfg.Q;  // = 1
        }

        ਅਚਲ int32_t ns10 = nb11_attn/nb10_attn;
        ਅਚਲ int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext_vec ਦਲੀਲਾਂ_2 = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ use_sparse ? n_kv_max_padded : ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_ਸਿਰ_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.n_kv_max_padded =*/ n_kv_max_padded,
        };

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_ਨਕਾਬ, has_sinks, has_bias, has_scap, has_kvpad, use_sparse, nqptg, cfg.NE, nsg, nwg, use_kv_f16, ns10, ns20);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_k,    2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_v,    3);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src3, 4);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src4, 5);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, use_sparse ? bid_idx : bid_src0, 8);

        ਅਚਲ size_t smem = fa_vec_smem(nsg, nqptg);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ);

        ਜੇ (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_pad, 6);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_dst, 7);

            ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } ਨਹੀਂਤਾਂ {
            // sanity checks
            ਪ੍ਰਤਿਜ੍ਞਾ(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_pad, 6);
            ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, bid_tmp, 7);

            ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                ਅਚਲ int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                ਸਵੈਚਾਲਿਤ pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline0);
                ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &args0, ਆਕਾਰ(args0), 0);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 1);
                ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_bin(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ਪੂਰਨ n_fuse = 1;
    ਅਚਲ ggml_metal_fusion * fusion = ਖਾਲੀ_ਸੂਚਕ;

    ਅਚਲ ਤਾਰਕਿਕ use_fusion = ctx->use_fusion();

    ਜੇ (use_fusion) {
        ਪੂਰਨ n = 1;
        fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        ਜੇ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SNAKE) {
            ctx->ਗਿਣਤੀ_fusions(fusion);
            ਵਾਪਸ ggml_metal_op_snake_fused(ctx, idx);
        }

        // MoE output reduction: experts * weights -> weighted sum
        ਜੇ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MOE_REDUCE) {
            ctx->ਗਿਣਤੀ_fusions(fusion);
            ਵਾਪਸ ggml_metal_op_moe_reduce(ctx, idx);
        }
    }

    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ਪੂਰਨ debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[1]));

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_src1 = ggml_metal_get_ਬਫਰ_id(op->src[1]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_kargs_bin ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    ਜੇ (use_fusion && fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_ਜੋੜੋ_CHAIN) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        ਲਈ (ਪੂਰਨ i = 1; i < n_fuse; i++) {
            ਦਲੀਲਾਂ_2.o1[i] = ggml_metal_get_ਬਫਰ_id(ctx->ਗੰਢ(idx + i)->src[1]).offs;
        }

        ctx->ਗਿਣਤੀ_fusions(fusion);

        ਜੇ (debug_fusion > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    ਬਣਤਰ ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    ਜੇ (n_fuse > 1) {
        bid_dst = ggml_metal_get_ਬਫਰ_id(ctx->ਗੰਢ(idx + n_fuse - 1));

        ਲਈ (ਪੂਰਨ i = 1; i < n_fuse; ++i) {
            ਜੇ (!ggml_metal_op_concurrency_check(ctx, ctx->ਗੰਢ(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (pipeline.c4) {
        ਦਲੀਲਾਂ_2.ne00 = ne00/4;
        ਦਲੀਲਾਂ_2.ne10 = ne10/4;
        ਦਲੀਲਾਂ_2.ne0  = ne0/4;
    }

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src1, 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  3);

    ਜੇ (pipeline.cnt) {
        ggml_metal_encoder_dispatch_threadgroups(enc, ਦਲੀਲਾਂ_2.ne0, ggml_nrows(op), 1, 1, 1, 1);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ nth_max = MIN(256, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

        ਪੂਰਨ nth = 1;

        ਜਦੋਂ (2*nth < ਦਲੀਲਾਂ_2.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_silu_ਪਿੱਠ(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_silu_ਪਿੱਠ(lib, op);

    ਅਚਲ int64_t ne = ggml_nelements(op);

    ggml_metal_kargs_silu_ਪਿੱਠ ਦਲੀਲਾਂ_2 = {
        /*.ne =*/ ne,
    };

    ਪੂਰਨ arg_idx{0};

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), arg_idx++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), arg_idx++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), arg_idx++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op), arg_idx++);

    ਅਚਲ ਪੂਰਨ nth = std::min<int64_t>(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne);
    ਅਚਲ int64_t n = (ne + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_l2_norm(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, op->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_metal_kargs_l2_norm ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    ਜੇ (pipeline.c4) {
        ਦਲੀਲਾਂ_2.ne00 = ne00/4;
        ਦਲੀਲਾਂ_2.ne0  = ne0/4;
    }

    ਪੂਰਨ nth = 32; // SIMD width

    ਜਦੋਂ (nth < ne00 && nth < ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

    ਅਚਲ size_t smem = pipeline.smem;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਸਮੂਹ_norm(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t ngrp = ((ਅਚਲ int32_t *) op->op_params)[0];

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, op->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_metal_kargs_ਸਮੂਹ_norm ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਸਮੂਹ_norm(lib, op);

    ਪੂਰਨ nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    ਅਚਲ size_t smem = pipeline.smem;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_norm(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ਤਾਰਕਿਕ use_fusion = ctx->use_fusion();

    ਅਚਲ ਪੂਰਨ debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, op->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_kargs_norm ਦਲੀਲਾਂ_2 = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
        /*.scale =*/ 1.0f,
    };

    ਪੂਰਨ n_fuse = 1;
    ਤਾਰਕਿਕ fused_norm_scale = ਝੂਠ;

    ggml_metal_ਬਫਰ_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    ਜੇ (use_fusion) {
        ਪੂਰਨ n = 1;
        ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n);

        ਜੇ (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ਜੋੜੋ)) {
            n_fuse = n;

            ctx->ਗਿਣਤੀ_fusions(fusion);

            ਲਈ (ਪੂਰਨ i = 1; i < n_fuse; i++) {
                ਅਚਲ ggml_tensor * fn = ctx->ਗੰਢ(idx + i);

                bid_fuse[i - 1] = ggml_metal_get_ਬਫਰ_id(fn->src[1]);

                ਦਲੀਲਾਂ_2.nef1[i] = fn->src[1]->ne[1];
                ਦਲੀਲਾਂ_2.nef2[i] = fn->src[1]->ne[2];
                ਦਲੀਲਾਂ_2.nef3[i] = fn->src[1]->ne[3];

                ਦਲੀਲਾਂ_2.nbf1[i] = fn->src[1]->nb[1];
                ਦਲੀਲਾਂ_2.nbf2[i] = fn->src[1]->nb[2];
                ਦਲੀਲਾਂ_2.nbf3[i] = fn->src[1]->nb[3];
            }

            ਜੇ (debug_fusion > 1) {
                ਜੇ (n_fuse == 2) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
                }
                ਜੇ (n_fuse == 3) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
                }
            }
        }

        ਜੇ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_SCALE) {
            n_fuse = n;
            fused_norm_scale = ਸੱਚ;

            ctx->ਗਿਣਤੀ_fusions(fusion);

            ਅਚਲ ggml_tensor * scale_ਗੰਢ = ctx->ਗੰਢ(idx + 1);
            ਦਲੀਲਾਂ_2.scale = ggml_get_op_params_f32(scale_ਗੰਢ, 0);

            ਜੇ (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + SCALE\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    ਜੇ (n_fuse > 1) {
        bid_dst = ggml_metal_get_ਬਫਰ_id(ctx->ਗੰਢ(idx + n_fuse - 1));

        ਲਈ (ਪੂਰਨ i = 1; i < n_fuse; ++i) {
            ਜੇ (!ggml_metal_op_concurrency_check(ctx, ctx->ਗੰਢ(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                ਰੋਕੋ;
            }
        }
    }

    ਸਵੈਚਾਲਿਤ pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    ਪੂਰਨ nth = 32; // SIMD width

    ਜਦੋਂ (nth < ਦਲੀਲਾਂ_2.ne00_t && nth < ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
    nth = std::min(nth, (ਦਲੀਲਾਂ_2.ne00_t + 31)/32*32);

    ਅਚਲ size_t smem = pipeline.smem;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0,    1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,     4);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_rope(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    ਅਚਲ ਪੂਰਨ nth = std::min(1024, ne00);

    ਅਚਲ ਪੂਰਨ n_past     = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਅਚਲ ਪੂਰਨ n_dims     = ((ਅਚਲ int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    ਅਚਲ ਪੂਰਨ n_ctx_orig = ((ਅਚਲ int32_t *) op->op_params)[4];

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ freq_scale;
    ਦਸ਼ਮਲਵ ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ beta_fast;
    ਦਸ਼ਮਲਵ beta_slow;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ,   (ਅਚਲ int32_t *) op->op_params +  5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale,  (ਅਚਲ int32_t *) op->op_params +  6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ,  (ਅਚਲ int32_t *) op->op_params +  7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (ਅਚਲ int32_t *) op->op_params +  8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast,   (ਅਚਲ int32_t *) op->op_params +  9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow,   (ਅਚਲ int32_t *) op->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // mrope
    ਅਚਲ ਪੂਰਨ sect_0 = ((ਅਚਲ int32_t *) op->op_params)[11];
    ਅਚਲ ਪੂਰਨ sect_1 = ((ਅਚਲ int32_t *) op->op_params)[12];
    ਅਚਲ ਪੂਰਨ sect_2 = ((ਅਚਲ int32_t *) op->op_params)[13];
    ਅਚਲ ਪੂਰਨ sect_3 = ((ਅਚਲ int32_t *) op->op_params)[14];

    ਅਚਲ ਪੂਰਨ n_offs = ((ਅਚਲ int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    ਅਚਲ ਤਾਰਕਿਕ inplace = op->ਅੰਕੜੇ == op->src[0]->ਅੰਕੜੇ;

    ggml_metal_kargs_rope ਦਲੀਲਾਂ_2 = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_offs      =*/ n_offs,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_ਅਧਾਰ,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_ਗੁਣਕ,
        /*.attn_factor =*/ attn_ਗੁਣਕ,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != ਖਾਲੀ_ਸੂਚਕ,
        /* inplace     =*/ inplace,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ਜੇ (op->src[2]) {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), 3);
    } ਨਹੀਂਤਾਂ {
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 3);
    }
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_im2col(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(op->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(op->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(op->op_params))[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(op->op_params))[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(op->op_params))[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(op->op_params))[5];

    ਅਚਲ ਤਾਰਕਿਕ is_2D = ((ਅਚਲ int32_t *)(op->op_params))[6] == 1;

    ਅਚਲ int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    ਅਚਲ int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    ਅਚਲ int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    ਅਚਲ int32_t IW =         op->src[1]->ne[0];

    ਅਚਲ int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    ਅਚਲ int32_t KW =         op->src[0]->ne[0];

    ਅਚਲ int32_t OH = is_2D ? op->ne[2] : 1;
    ਅਚਲ int32_t OW =         op->ne[1];

    ਅਚਲ int32_t CHW = IC * KH * KW;

    ਅਚਲ uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    ਅਚਲ uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col ਦਲੀਲਾਂ_2 = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    ਜੇ (KH*KW <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        ਅਚਲ uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ uint64_t n_ਧਾਗੇ = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), N);
        ਅਚਲ int64_t  quotient  = N / n_ਧਾਗੇ + (N % n_ਧਾਗੇ > 0 ? 1 : 0);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_ਧਾਗੇ, 1, 1);
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_conv_2d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *) op->op_params)[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *) op->op_params)[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *) op->op_params)[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *) op->op_params)[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d ਦਲੀਲਾਂ_2 = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    ਪੂਰਨ nth = ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    ਅਚਲ uint64_t n_ਬਾਹਰਲਾ = ggml_nelements(op);

    uint64_t tg = (n_ਬਾਹਰਲਾ + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<ਪੂਰਨ>::max());

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *) op->op_params)[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *) op->op_params)[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *) op->op_params)[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *) op->op_params)[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw ਦਲੀਲਾਂ_2 = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.C    =*/ ne12,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne13,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    ਅਚਲ ਤਾਰਕਿਕ use_tiled = (nb12 < nb10);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    ਪੂਰਨ nth = ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    ਅਚਲ int32_t OW = ne0;
    ਅਚਲ int32_t OH = ne1;
    ਅਚਲ int32_t C  = ne12;
    ਅਚਲ int32_t N  = ne13;

    ਅਚਲ ਪੂਰਨ tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    ਅਚਲ ਪੂਰਨ tg_y = OH;
    ਅਚਲ ਪੂਰਨ tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_conv_3d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(op->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(op->op_params))[1];
    ਅਚਲ int32_t s2 = ((ਅਚਲ int32_t *)(op->op_params))[2];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(op->op_params))[3];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(op->op_params))[4];
    ਅਚਲ int32_t p2 = ((ਅਚਲ int32_t *)(op->op_params))[5];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(op->op_params))[6];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(op->op_params))[7];
    ਅਚਲ int32_t d2 = ((ਅਚਲ int32_t *)(op->op_params))[8];
    ਅਚਲ int32_t IC = ((ਅਚਲ int32_t *)(op->op_params))[9];
    ਅਚਲ int32_t N  = ((ਅਚਲ int32_t *)(op->op_params))[10];
    ਅਚਲ int32_t OC = ((ਅਚਲ int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d ਦਲੀਲਾਂ_2 = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    ਪੂਰਨ nth0 = 32; // Standard SIMD width for Apple Silicon
    ਪੂਰਨ nth1 = 1;
    ਪੂਰਨ nth2 = 1;

    int64_t spatial_volume = ਦਲੀਲਾਂ_2.OW * ਦਲੀਲਾਂ_2.OH * ਦਲੀਲਾਂ_2.OD;

    ਪੂਰਨ ntg0 = (spatial_volume + nth0 - 1) / nth0;
    ਪੂਰਨ ntg1 = ਦਲੀਲਾਂ_2.OC;
    ਪੂਰਨ ntg2 = ਦਲੀਲਾਂ_2.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(op->op_params))[0];

    ਅਚਲ int32_t IC = op->src[1]->ne[1];
    ਅਚਲ int32_t IL = op->src[1]->ne[0];

    ਅਚਲ int32_t K  = op->src[0]->ne[0];

    ਅਚਲ int32_t OL = op->ne[0];
    ਅਚਲ int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d ਦਲੀਲਾਂ_2 = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(op->op_params))[0];
    ਅਚਲ int32_t OC = ((ਅਚਲ int32_t *)(op->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(op->op_params))[2];

    ਅਚਲ int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    ਅਚਲ int32_t T_ਵਿੱਚ  = (int32_t) op->src[0]->ne[1];
    ਅਚਲ int32_t K     = K_OC / OC;
    ਅਚਲ int32_t T_ਬਾਹਰਲਾ = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d ਦਲੀਲਾਂ_2 = {
        /*.T_in  =*/ T_ਵਿੱਚ,
        /*.T_out =*/ T_ਬਾਹਰਲਾ,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    ਅਚਲ ਪੂਰਨ ਕੁੱਲ = T_ਬਾਹਰਲਾ * OC;
    ਅਚਲ ਪੂਰਨ nth   = 256;
    ਅਚਲ ਪੂਰਨ ntg   = (ਕੁੱਲ + nth - 1) / nth;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

// Dispatch the fused snake kernel from the matched mul -> sin -> sqr -> mul -> add chain.
// idx points at the leading mul. The caller has validated the chain.
ਪੂਰਨ ggml_metal_op_snake_fused(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਅਚਲ ggml_tensor * mul0 = ctx->ਗੰਢ(idx + 0);
    ਅਚਲ ggml_tensor * sqr  = ctx->ਗੰਢ(idx + 2);
    ਅਚਲ ggml_tensor * mul1 = ctx->ਗੰਢ(idx + 3);
    ggml_tensor *       ਜੋੜੋ_2  = ctx->ਗੰਢ(idx + 4);

    ਅਚਲ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ਅਚਲ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];
    ਅਚਲ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    ਅਚਲ ਪੂਰਨ T     = (ਪੂਰਨ) x->ne[0];
    ਅਚਲ ਪੂਰਨ C     = (ਪੂਰਨ) x->ne[1];
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ = T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    ਲਈ (ਪੂਰਨ i = 1; i < 5; ++i) {
        ਜੇ (!ggml_metal_op_concurrency_check(ctx, ctx->ਗੰਢ(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);

            ਰੋਕੋ;
        }
    }

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_snake(lib, x->type);

    ggml_metal_kargs_snake ਦਲੀਲਾਂ_2 = {
        /*.T =*/ T,
        /*.C =*/ C,
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(x),     1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(a),     2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(inv_b), 3);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(ਜੋੜੋ_2),   4);

    ਅਚਲ ਪੂਰਨ nth = 256;
    ਅਚਲ ਪੂਰਨ ntg = (ਕੁੱਲ + nth - 1) / nth;
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ਵਾਪਸ 5;
}

ਪੂਰਨ ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(op->op_params))[0];

    ਅਚਲ int32_t IC = op->src[1]->ne[2];
    ਅਚਲ int32_t IH = op->src[1]->ne[1];
    ਅਚਲ int32_t IW = op->src[1]->ne[0];

    ਅਚਲ int32_t KH = op->src[0]->ne[1];
    ਅਚਲ int32_t KW = op->src[0]->ne[0];

    ਅਚਲ int32_t OW = op->ne[0];
    ਅਚਲ int32_t OH = op->ne[1];
    ਅਚਲ int32_t OC = op->ne[2];
    ਅਚਲ int32_t N  = op->src[1]->ne[3];

    ggml_metal_kargs_conv_transpose_2d ਦਲੀਲਾਂ_2 = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    ਅਚਲ size_t smem = GGML_PAD(KW * KH * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 16);
    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_upscale(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਦਸ਼ਮਲਵ sf0 = (ਦਸ਼ਮਲਵ)ne0/op->src[0]->ne[0];
    ਦਸ਼ਮਲਵ sf1 = (ਦਸ਼ਮਲਵ)ne1/op->src[0]->ne[1];
    ਦਸ਼ਮਲਵ sf2 = (ਦਸ਼ਮਲਵ)ne2/op->src[0]->ne[2];
    ਦਸ਼ਮਲਵ sf3 = (ਦਸ਼ਮਲਵ)ne3/op->src[0]->ne[3];

    ਅਚਲ int32_t ਬਹੁਲਕ_flags = ggml_get_op_params_i32(op, 0);

    ਦਸ਼ਮਲਵ poffs = 0.5f;

    ਜੇ (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ਦਸ਼ਮਲਵ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ਦਸ਼ਮਲਵ)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne0);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_roll(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int32_t s0 = ggml_get_op_params_i32(op, 0);
    ਅਚਲ int32_t s1 = ggml_get_op_params_i32(op, 1);
    ਅਚਲ int32_t s2 = ggml_get_op_params_i32(op, 2);
    ਅਚਲ int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    ਅਚਲ ਪੂਰਨ nth = std::min(1024, ne0);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_pad(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.lp0  =*/ ggml_get_op_params_i32(op, 0),
        /*.lp1  =*/ ggml_get_op_params_i32(op, 2),
        /*.lp2  =*/ ggml_get_op_params_i32(op, 4),
        /*.lp3  =*/ ggml_get_op_params_i32(op, 6),
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    ਅਚਲ ਪੂਰਨ nth_max = MIN(64, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
    ਅਚਲ ਪੂਰਨ nth = MIN(ਦਲੀਲਾਂ_2.ne0, nth_max);
    ਅਚਲ ਪੂਰਨ nk0 = (ਦਲੀਲਾਂ_2.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((ਅਚਲ int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((ਅਚਲ int32_t *)(op->op_params))[1]
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    ਅਚਲ ਪੂਰਨ nth = std::min(1024, ne0);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_arange(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਦਸ਼ਮਲਵ start;
    ਦਸ਼ਮਲਵ ਕਦਮ;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&start, ((ਅਚਲ int32_t *) op->op_params) + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਕਦਮ,  ((ਅਚਲ int32_t *) op->op_params) + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_metal_kargs_arange ਦਲੀਲਾਂ_2 = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ ਕਦਮ
    };

    ਅਚਲ ਪੂਰਨ nth = std::min(1024, ne0);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ ਪੂਰਨ dim        = op->op_params[0];
    ਅਚਲ ਪੂਰਨ max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding ਦਲੀਲਾਂ_2 = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    ਅਚਲ ਪੂਰਨ nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

size_t ggml_metal_op_argmax_extra(ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ARGMAX);

    ਅਚਲ int64_t ne00 = op->src[0]->ne[0];
    ਅਚਲ int64_t ne01 = op->src[0]->ne[1];

    ਪੂਰਨ npart = 1;
    ਜੇ (ne00 > N_ARGMAX_CHUNK) {
        npart = (ਪੂਰਨ) std::min<int64_t>(N_ARGMAX_NPART_MAX, (ne00 + N_ARGMAX_CHUNK - 1)/N_ARGMAX_CHUNK);
    }

    ਜੇ (npart == 1) {
        ਵਾਪਸ 0;
    }

    // partial max + partial argmax for every row
    ਵਾਪਸ (ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int32_t)) * npart * ne01;
}

ਪੂਰਨ ggml_metal_op_argmax(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਅਚਲ int64_t nrows = ggml_nrows(op->src[0]);

    ਪੂਰਨ npart = 1;
    ਜੇ (ne00 > N_ARGMAX_CHUNK) {
        npart = (ਪੂਰਨ) std::min<int64_t>(N_ARGMAX_NPART_MAX, (ne00 + N_ARGMAX_CHUNK - 1)/N_ARGMAX_CHUNK);
    }

    ggml_metal_kargs_argmax ਦਲੀਲਾਂ_2 = {
        /*.ne00  = */ ne00,
        /*.nb01  = */ nb01,
        /*.npart = */ npart,
        /*.ne01  = */ ne01,
    };

    ਅਚਲ size_t smem = 32*(ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int32_t));

    ਜੇ (npart > 1) {
        ਸਵੈਚਾਲਿਤ pipeline_part = ggml_metal_library_get_pipeline_argmax_part(lib);
        ਸਵੈਚਾਲਿਤ pipeline_final = ggml_metal_library_get_pipeline_argmax_final(lib);

        ggml_metal_ਬਫਰ_id bid_dst = ggml_metal_get_ਬਫਰ_id(op);
        ggml_metal_ਬਫਰ_id bid_tmp = bid_dst;
        bid_tmp.offs += ggml_nbytes(op);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_part);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 2);
        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, npart, nrows, 1, 256, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);

        ਪੂਰਨ nth = 32; // SIMD width
        ਜਦੋਂ (nth < npart && nth < 256) {
            nth *= 2;
        }

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_final);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst, 2);
        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

        ਵਾਪਸ 1;
    }

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    ਪੂਰਨ nth = 32; // SIMD width
    ਜਦੋਂ (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_argsort(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    ਪੂਰਨ nth = 1;
    ਜਦੋਂ (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    ਅਚਲ ਪੂਰਨ npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    ਅਚਲ size_t smem = GGML_PAD(nth*ਆਕਾਰ(int32_t), 16);

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_ਬਫਰ_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    ਜੇ ((ਪੂਰਨ) ceil(std::ਲਘੂਗਣਕ(npr) / std::ਲਘੂਗਣਕ(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    ਸਵੈਚਾਲਿਤ pipeline_ਮਿਲਾਓ = ggml_metal_library_get_pipeline_argsort_ਮਿਲਾਓ(lib, op);

    ਪੂਰਨ len = nth;

    ਜਦੋਂ (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_ਮਿਲਾਓ ਦਲੀਲਾਂ_ਮਿਲਾਓ = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        ਅਚਲ ਪੂਰਨ nm = (ne00 + 2*len - 1) / (2*len);

        ਅਚਲ ਪੂਰਨ nth = std::min(512, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline_ਮਿਲਾਓ));

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_ਮਿਲਾਓ);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_ਮਿਲਾਓ, ਆਕਾਰ(ਦਲੀਲਾਂ_ਮਿਲਾਓ), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    ਵਾਪਸ 1;
}

// bitonic-sort + merge fallback: efficient when k is small and there are few rows,
// where the single-workgroup-per-row radix-select cannot reach enough parallelism
ਸਥਿਰ ਖਾਲੀ ggml_metal_op_ਸਿਖਰ_k_bitonic(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਸਿਖਰ_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    ਪੂਰਨ nth = 1;
    ਜਦੋਂ (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    ਅਚਲ ਪੂਰਨ npr = (ne00 + nth - 1)/nth;

    ਅਚਲ size_t smem = GGML_PAD(nth*ਆਕਾਰ(int32_t), 16);

    ggml_metal_ਬਫਰ_id bid_src0 = ggml_metal_get_ਬਫਰ_id(op->src[0]);
    ggml_metal_ਬਫਰ_id bid_dst  = ggml_metal_get_ਬਫਰ_id(op);

    ggml_metal_ਬਫਰ_id bid_tmp = bid_dst;
    bid_tmp.offs += ਆਕਾਰ(int32_t)*ggml_nelements(op->src[0]);

    ਜੇ ((ਪੂਰਨ) ceil(std::ਲਘੂਗਣਕ(npr) / std::ਲਘੂਗਣਕ(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ਅਚਲ ਪੂਰਨ ਸਿਖਰ_k = ne0;

    ggml_metal_kargs_argsort ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, ਸਿਖਰ_k), // for each block, keep just the top_k indices
    };

    ਜੇ (npr > 1) {
        ਦਲੀਲਾਂ_2.ne0 = (npr - 1)*ਦਲੀਲਾਂ_2.ਸਿਖਰ_k + std::min(ne00 - (npr - 1)*nth, ਦਲੀਲਾਂ_2.ਸਿਖਰ_k);
    }

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    ਸਵੈਚਾਲਿਤ pipeline_ਮਿਲਾਓ = ggml_metal_library_get_pipeline_ਸਿਖਰ_k_ਮਿਲਾਓ(lib, op);

    ਪੂਰਨ len = ਦਲੀਲਾਂ_2.ਸਿਖਰ_k;

    ਜਦੋਂ (len < ਦਲੀਲਾਂ_2.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        ਅਚਲ ਪੂਰਨ nm = (ਦਲੀਲਾਂ_2.ne0 + 2*len - 1) / (2*len);

        ਅਚਲ ਪੂਰਨ nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline_ਮਿਲਾਓ)));

        ggml_metal_kargs_argsort_ਮਿਲਾਓ ਦਲੀਲਾਂ_ਮਿਲਾਓ = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ਦਲੀਲਾਂ_2.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? ਸਿਖਰ_k : ਦਲੀਲਾਂ_2.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline_ਮਿਲਾਓ);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_ਮਿਲਾਓ, ਆਕਾਰ(ਦਲੀਲਾਂ_ਮਿਲਾਓ), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_src0, 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_dst,  2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }
}

// radix-select: one workgroup per row. Maps each float to an order-preserving unsigned
// key, finds the k-th largest via 4 radix-8 histogram passes, then compacts the top-k
// indices. Fast for large k and/or many rows.
ਸਥਿਰ ਖਾਲੀ ggml_metal_op_ਸਿਖਰ_k_radix(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਸਿਖਰ_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    ਅਚਲ ਪੂਰਨ nth = std::min(1024, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

    ggml_metal_kargs_ਸਿਖਰ_k ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.top_k =*/ (int32_t) op->ne[0],
    };

    // shared memory: 256-entry histogram + bucket/above scalars + output counter
    ਅਚਲ size_t smem_histo  = GGML_PAD(256*ਆਕਾਰ(uint32_t), 16);
    ਅਚਲ size_t smem_ਬਾਲਟੀ = GGML_PAD(    ਆਕਾਰ(uint32_t), 16);
    ਅਚਲ size_t smem_above  = GGML_PAD(    ਆਕਾਰ(uint32_t), 16);
    ਅਚਲ size_t smem_ਬਾਹਰਲਾ    = GGML_PAD(    ਆਕਾਰ(uint32_t), 16);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem_histo,  0);
    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem_ਬਾਲਟੀ, 1);
    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem_above,  2);
    ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem_ਬਾਹਰਲਾ,    3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
}

ਪੂਰਨ ggml_metal_op_topk_moe(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਪੂਰਨ n_fuse = 1;
    ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n_fuse);
    ਜੇ (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_TOPK_MOE) {
        ਵਾਪਸ 1;
    }

    ggml_tensor * softmax  = ctx->ਗੰਢ(idx);
    ggml_tensor * logits   = softmax->src[0];
    ggml_tensor * get_ਪੰਕਤੀਆਂ = ctx->ਗੰਢ(idx + 2);
    ggml_tensor * ids      = get_ਪੰਕਤੀਆਂ->src[1];
    ggml_tensor * weights  = ctx->ਗੰਢ(idx + n_fuse - 1);

    ਅਚਲ int64_t n_expert      = logits->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ      = logits->ne[1];
    ਅਚਲ int64_t n_expert_ਵਰਤਿਆ = ids->ne[0];

    ਅਚਲ ਤਾਰਕਿਕ with_norm  = n_fuse >= 6;
    ਅਚਲ ਤਾਰਕਿਕ with_scale = n_fuse == 4 || n_fuse == 7;

    ਦਸ਼ਮਲਵ clamp = -INFINITY;
    ਜੇ (with_norm) {
        ggml_tensor * clamp_ਗੰਢ = ctx->ਗੰਢ(idx + 4);
        clamp = ggml_get_op_params_f32(clamp_ਗੰਢ, 0);
    }

    ਦਸ਼ਮਲਵ scale = 1.0f;
    ਜੇ (with_scale) {
        ggml_tensor * scale_ਗੰਢ = ctx->ਗੰਢ(idx + n_fuse - 1);
        scale = ggml_get_op_params_f32(scale_ਗੰਢ, 0);
    }

    ggml_metal_kargs_topk_moe ਦਲੀਲਾਂ_2 = {
        /*.ne01      =*/ (int32_t) n_ਟੋਕਨ,
        /*.nb01      =*/ logits->nb[1],
        /*.nb1_ids   =*/ ids->nb[1],
        /*.clamp =*/ clamp,
        /*.scale =*/ scale,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_ਵਰਤਿਆ, with_norm);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(logits),  1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(weights), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(ids),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) n_ਟੋਕਨ, 1, 1, 32, 1, 1);

    ctx->ਗਿਣਤੀ_fusions(fusion);

    ਜੇ (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SOFT_MAX + ARGSORT + GET_ROWS\n", __func__);
    }

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_moe_reduce(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ਪੂਰਨ n_fuse = 1;
    ਅਚਲ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_ਭਰਿਆ, &n_fuse);
    ਜੇ (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MOE_REDUCE) {
        ਵਾਪਸ 1;
    }

    ggml_tensor * mul     = ctx->ਗੰਢ(idx);
    ggml_tensor * experts = mul->src[0];
    ggml_tensor * weights = mul->src[1];
    ggml_tensor * dst     = ctx->ਗੰਢ(idx + n_fuse - 1);

    ggml_metal_kargs_moe_reduce ਦਲੀਲਾਂ_2 = {
        /*.ne00 =*/ (int32_t) experts->ne[0],
        /*.ne02 =*/ (int32_t) experts->ne[2],
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    ਅਚਲ ਪੂਰਨ nth = std::min(256, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles = (ਦਲੀਲਾਂ_2.ne00 + nth - 1) / nth;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(experts), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(weights), 2);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(dst),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) ਦਲੀਲਾਂ_2.ne02, (uint32_t) n_ਖਾਨਾ_tiles, 1, nth, 1, 1);

    ctx->ਗਿਣਤੀ_fusions(fusion);

    ਜੇ (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\n", __func__);
    }

    ਵਾਪਸ n_fuse;
}

ਪੂਰਨ ggml_metal_op_ਸਿਖਰ_k(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    ਅਚਲ ਪੂਰਨ ncols = op->src[0]->ne[0];
    ਅਚਲ ਪੂਰਨ k     = op->ne[0];
    ਅਚਲ ਪੂਰਨ nrows = ggml_nrows(op->src[0]);

    ਅਚਲ ਤਾਰਕਿਕ use_radix =
        ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));

    ਜੇ (use_radix) {
        ggml_metal_op_ਸਿਖਰ_k_radix(ctx, idx);
    } ਨਹੀਂਤਾਂ {
        ggml_metal_op_ਸਿਖਰ_k_bitonic(ctx, idx);
    }

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_tri(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri ਦਲੀਲਾਂ_2 = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    ਪੂਰਨ nth = 32; // SIMD width

    ਜਦੋਂ (nth < ne00 && nth < ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_opt_ਕਦਮ_adamw(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_opt_ਕਦਮ_adamw(lib, op);

    ਅਚਲ int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_ਕਦਮ_adamw ਦਲੀਲਾਂ_2 = {
        /*.np =*/ np,
    };

    ਪੂਰਨ ida = 0;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[3]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[4]), ida++);

    ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne0);
    ਅਚਲ int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_opt_ਕਦਮ_sgd(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_opt_ਕਦਮ_sgd(lib, op);

    ਅਚਲ int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_ਕਦਮ_sgd ਦਲੀਲਾਂ_2 = {
        /*.np =*/ np,
    };

    ਪੂਰਨ ida = 0;

    ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
    ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ   (enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), ida++);
    ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ  (enc, ggml_metal_get_ਬਫਰ_id(op->src[2]), ida++);

    ਅਚਲ ਪੂਰਨ nth = std::min(ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline), ne0);
    ਅਚਲ int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    ਵਾਪਸ 1;
}

ਪੂਰਨ ggml_metal_op_ਗਿਣਤੀ_equal(ggml_metal_op_t ctx, ਪੂਰਨ idx) {
    ggml_tensor * op = ctx->ਗੰਢ(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset ਦਲੀਲਾਂ_2 = { /*.val =*/ 0 };

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_ਗਿਣਤੀ_equal ਦਲੀਲਾਂ_2 = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_ਗਿਣਤੀ_equal(lib, op);

        ਅਚਲ size_t smem = pipeline.smem;

        ਅਚਲ ਪੂਰਨ nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline));

        ggml_metal_encoder_ਸੈੱਟ_pipeline(enc, pipeline);
        ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ(enc, &ਦਲੀਲਾਂ_2, ਆਕਾਰ(ਦਲੀਲਾਂ_2), 0);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[0]), 1);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op->src[1]), 2);
        ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(enc, ggml_metal_get_ਬਫਰ_id(op), 3);

        ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    ਵਾਪਸ 1;
}
