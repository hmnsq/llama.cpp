#ಅನ್ತರ್ಭಾವ "ggml-metal-ops.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ "ggml-metal-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-fusion.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-tuning.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <cmath>

// threadgroups needed to fill the GPU
// TODO: dedup
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

ಸ್ಥಿರ ggml_metal_buffer_id ggml_metal_get_buffer_id(ಧ್ರುವ ggml_tensor * t) {
    ಯದಿ (!t) {
        ಪ್ರತ್ಯಾಗಮ { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_buffer_get_id(ctx, t);
}

ಸಂರಚನಾ ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        ಪೂರ್ಣಾಙ್ಕ  idx_start,
        ಪೂರ್ಣಾಙ್ಕ  idx_end,
        ತಾರ್ಕಿಕ use_concurrency,
        ತಾರ್ಕಿಕ use_capture,
        ಪೂರ್ಣಾಙ್ಕ  debug_graph) {
        ಏತತ್->dev             = dev;
        ಏತತ್->lib             = ggml_metal_device_get_library(dev);
        ಏತತ್->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        ಏತತ್->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        ಏತತ್->finfo           = finfo;
        ಏತತ್->idx_start       = idx_start;
        ಏತತ್->idx_end         = idx_end;
        ಏತತ್->use_concurrency = use_concurrency;
        ಏತತ್->use_capture     = use_capture;
        ಏತತ್->debug_graph     = debug_graph;
        ಏತತ್->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = idx_start; i < idx_end; i++) {
            ಯದಿ (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(ಏತತ್->enc);
        ggml_metal_encoder_free(ಏತತ್->enc);
        ggml_mem_ranges_free(ಏತತ್->mem_ranges);
    }

    ಪೂರ್ಣಾಙ್ಕ n_nodes() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ idxs.size();
    }

    ggml_tensor * ಗಂಟು(ಪೂರ್ಣಾಙ್ಕ i) ಧ್ರುವ {
        ಪ್ರತಿಜ್ಞಾ(i >= 0 && i < (ಪೂರ್ಣಾಙ್ಕ) idxs.size());
        ಪ್ರತ್ಯಾಗಮ ggml_graph_ಗಂಟು(gf, idxs[i]);
    }

    // consult the fusion table for the longest pattern starting at i0
    // returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes
    ಧ್ರುವ ggml_metal_fusion * can_fuse(ಪೂರ್ಣಾಙ್ಕ i0, ಗಣನಾ ggml_metal_fusion_mode mode, ಪೂರ್ಣಾಙ್ಕ * n_out) ಧ್ರುವ {
        ಪ್ರತಿಜ್ಞಾ(use_fusion());
        ಪ್ರತಿಜ್ಞಾ(i0 >= 0 && i0 < n_nodes());

        ಪ್ರತ್ಯಾಗಮ ggml_metal_fusion_next(gf, idxs.data(), (ಪೂರ್ಣಾಙ್ಕ) idxs.size(), i0, mode, n_out);
    }

    // whether to attempt fusion; the toggle lives in the shared fusion debugging context owned
    // by the device (initialized from GGML_METAL_FUSION_DISABLE, overridable by the test)
    ತಾರ್ಕಿಕ use_fusion() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ggml_metal_fusion_info_enabled(finfo);
    }

    // record that a fusion fired, indexed by the matching table entry
    ಶೂನ್ಯ ಎಣಿಕೆ_fusions(ಧ್ರುವ ggml_metal_fusion * fusion) ಧ್ರುವ {
        ggml_metal_fusion_info_ಎಣಿಕೆ_fusion(finfo, fusion);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    // shared fusion debugging context
    ggml_metal_fusion_info * finfo;

    ತಾರ್ಕಿಕ use_concurrency;
    ತಾರ್ಕಿಕ use_capture;

    ಪೂರ್ಣಾಙ್ಕ debug_graph;

ಗುಪ್ತ:
    ggml_cgraph * gf;

    ಪೂರ್ಣಾಙ್ಕ idx_start;
    ಪೂರ್ಣಾಙ್ಕ idx_end;

    // non-empty node indices
    std::vector<ಪೂರ್ಣಾಙ್ಕ> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        ಪೂರ್ಣಾಙ್ಕ idx_start,
        ಪೂರ್ಣಾಙ್ಕ idx_end,
        ತಾರ್ಕಿಕ use_concurrency,
        ತಾರ್ಕಿಕ use_capture,
        ಪೂರ್ಣಾಙ್ಕ debug_graph) {
    ggml_metal_op_t res = ನೂತನ ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_start,
        idx_end,
        use_concurrency,
        use_capture,
        debug_graph);

    ಪ್ರತ್ಯಾಗಮ res;
}

ಶೂನ್ಯ ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->n_nodes();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    ಯದಿ (!ctx->mem_ranges) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    ಯದಿ (!ctx->mem_ranges) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_mem_ranges_check(ctx->mem_ranges, ಗಂಟು);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    ಯದಿ (!ctx->mem_ranges) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_mem_ranges_add(ctx->mem_ranges, ಗಂಟು);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_encode_impl(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ಸಂರಚನಾ ggml_tensor * ಗಂಟು = ctx->ಗಂಟು(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    ಯದಿ (ggml_is_empty(ಗಂಟು)) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ವಿಕಲ್ಪ (ಗಂಟು->op) {
        ಪಕ್ಷ GGML_OP_NONE:
        ಪಕ್ಷ GGML_OP_RESHAPE:
        ಪಕ್ಷ GGML_OP_VIEW:
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
        ಪಕ್ಷ GGML_OP_PERMUTE:
            {
                // noop -> next node
                ಯದಿ (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(ಗಂಟು->op), "(noop)");
                }
            } ಪ್ರತ್ಯಾಗಮ 1;
        ಸ್ವಭಾವ:
            {
            } ಭಙ್ಗ;
    }

    ಯದಿ (!ggml_metal_device_supports_op(ctx->dev, ಗಂಟು)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(ಗಂಟು));
        GGML_ABORT("unsupported op");
    }

    ಯದಿ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        ತಾರ್ಕಿಕ is_concurrent = ggml_metal_op_concurrency_check(ctx, ಗಂಟು);

        ಯದಿ (is_concurrent && ctx->use_fusion()) {
            ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;
            ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
            ಯದಿ (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->ಗಂಟು(idx + n_fuse - 1));
            }
        }

        ಯದಿ (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        ಯದಿ (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(ಗಂಟು->op), ggml_get_name(ಗಂಟು), is_concurrent ? "(concurrent)" : "");
        }
        ಯದಿ (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, ಗಂಟು->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, ಗಂಟು->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, ಗಂಟು->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, ಗಂಟು->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, ಗಂಟು->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, ಗಂಟು->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, ಗಂಟು->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, ಗಂಟು->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  ಗಂಟು,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  ಗಂಟು,         nb);

            ಯದಿ (ಗಂಟು->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ಗಂಟು->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(ಗಂಟು->src[0]), ಗಂಟು->src[0]->name);
            }
            ಯದಿ (ಗಂಟು->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ಗಂಟು->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(ಗಂಟು->src[1]), ಗಂಟು->src[1]->name);
            }
            ಯದಿ (ಗಂಟು->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ಗಂಟು->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(ಗಂಟು->src[2]), ಗಂಟು->src[2]->name);
            }
            ಯದಿ (ಗಂಟು->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(ಗಂಟು->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(ಗಂಟು->src[3]), ಗಂಟು->src[3]->name);
            }
            ಯದಿ (ಗಂಟು) {
                GGML_LOG_DEBUG("%s: node - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(ಗಂಟು->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        ಗಂಟು->name);
            }
        }
    }

    ವಿಕಲ್ಪ (ಗಂಟು->op) {
        ಪಕ್ಷ GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ADD:
        ಪಕ್ಷ GGML_OP_SUB:
        ಪಕ್ಷ GGML_OP_MUL:
        ಪಕ್ಷ GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SCALE:
        ಪಕ್ಷ GGML_OP_FILL:
        ಪಕ್ಷ GGML_OP_CLAMP:
        ಪಕ್ಷ GGML_OP_LEAKY_RELU:
        ಪಕ್ಷ GGML_OP_SQR:
        ಪಕ್ಷ GGML_OP_SQRT:
        ಪಕ್ಷ GGML_OP_SIN:
        ಪಕ್ಷ GGML_OP_COS:
        ಪಕ್ಷ GGML_OP_LOG:
        ಪಕ್ಷ GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SILU_BACK:
            {
                n_fuse = ggml_metal_op_silu_back(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUM_ROWS:
        ಪಕ್ಷ GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DSV4_HC_COMB:
        ಪಕ್ಷ GGML_OP_DSV4_HC_PRE:
        ಪಕ್ಷ GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_RWKV_WKV6:
        ಪಕ್ಷ GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_NORM:
        ಪಕ್ಷ GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ROPE:
        ಪಕ್ಷ GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ತುದಿ_K:
            {
                n_fuse = ggml_metal_op_ತುದಿ_k(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DUP:
        ಪಕ್ಷ GGML_OP_CPY:
        ಪಕ್ಷ GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ಎಣಿಕೆ_EQUAL:
            {
                n_fuse = ggml_metal_op_ಎಣಿಕೆ_equal(ctx, idx);
            } ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(ಗಂಟು->op));
                GGML_ABORT("fatal error");
            }
    }

    ಯದಿ (ctx->debug_graph > 0) {
        ಯದಿ (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_fuse; ++i) {
        ಯದಿ (!ggml_metal_op_concurrency_add(ctx, ctx->ಗಂಟು(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_encode(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ಯದಿ (ctx->use_capture) {
        ggml_metal_encoder_debug_group_ತಳ್ಳು(ctx->enc, ggml_op_desc(ctx->ಗಂಟು(idx)));
    }

    ಪೂರ್ಣಾಙ್ಕ res = ggml_metal_op_encode_impl(ctx, idx);
    ಯದಿ (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    ಯದಿ (ctx->use_capture) {
        ggml_metal_encoder_debug_group_ತೆಗೆ(ctx->enc);
    }

    ಪ್ರತ್ಯಾಗಮ res;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_concat(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t dim = ((ಧ್ರುವ int32_t *) op->op_params)[0];

    ಧ್ರುವ ತಾರ್ಕಿಕ is_q = ggml_is_quantized(op->type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    int32_t ne00_arg = ne00;
    int32_t ne10_arg = ne10;
    int32_t ne0_arg  = ne0;
    ಯದಿ (is_q) {
        ಧ್ರುವ int32_t blck = ggml_blck_size(op->type);
        GGML_ASSERT(ne00 % blck == 0);
        GGML_ASSERT(ne10 % blck == 0);
        GGML_ASSERT(ne0  % blck == 0);
        ne00_arg = ne00/blck;
        ne10_arg = ne10/blck;
        ne0_arg  = ne0/blck;
    }

    ಪೂರ್ಣಾಙ್ಕ nth = std::min(256, ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    ಪೂರ್ಣಾಙ್ಕ nrptg = 1;
    ಯದಿ (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        ಯದಿ (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nw0 = (ne1 + nrptg - 1) / nrptg;

    // split long rows across threadgroups when there are too few rows to fill the GPU
    ಧ್ರುವ int64_t n_rows = (int64_t) nw0*ne2*ne3;

    ಪೂರ್ಣಾಙ್ಕ nc0 = 1;
    ಯದಿ (nrptg == 1) {
        nc0 = (ಪೂರ್ಣಾಙ್ಕ) std::max<int64_t>(1, std::min<int64_t>((ne0_arg + nth - 1)/nth, (GGML_METAL_MIN_THREADGROUPS + n_rows - 1)/n_rows));
    }

    ggml_metal_kargs_concat args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*nc0, ne2, ne3, nth, nrptg, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_repeat(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // TODO: optimize for degenerate cases such as ggml_nelements(op->src[0]) == 1 and others

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
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

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_acc(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

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

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ಧ್ರುವ size_t pnb1 = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಧ್ರುವ size_t pnb2 = ((ಧ್ರುವ int32_t *) op->op_params)[1];
    ಧ್ರುವ size_t pnb3 = ((ಧ್ರುವ int32_t *) op->op_params)[2];
    ಧ್ರುವ size_t offs = ((ಧ್ರುವ int32_t *) op->op_params)[3];

    ಧ್ರುವ ತಾರ್ಕಿಕ inplace = (ತಾರ್ಕಿಕ) ((ಧ್ರುವ int32_t *) op->op_params)[4];

    ಯದಿ (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
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

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ಪೂರ್ಣಾಙ್ಕ nth = 1;

    ಯಾವತ್ (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_unary(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
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

    ಯದಿ (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    ಯದಿ (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    ಯದಿ (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    ಯದಿ (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    ಯದಿ (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    ಯದಿ (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ಯದಿ (pipeline.cnt) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = MIN(args.ne00, nth_max);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_glu(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಯದಿ (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    ಧ್ರುವ int32_t swp = ggml_get_op_params_i32(op, 1);
    ಧ್ರುವ ಭಿನ್ನ alpha = ggml_get_op_params_f32(op, 2);
    ಧ್ರುವ ಭಿನ್ನ limit = ggml_get_op_params_f32(op, 3);

    ಧ್ರುವ int32_t i00 = swp ? ne0 : 0;
    ಧ್ರುವ int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    ಧ್ರುವ int64_t nrows = ggml_nrows(op->src[0]);

    ಧ್ರುವ int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ಯದಿ (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } ಅನ್ಯಥಾ {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_sum(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op  = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯಾವತ್ (nth < (ಪೂರ್ಣಾಙ್ಕ) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (ಪೂರ್ಣಾಙ್ಕ) n);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = (nth + 31) / 32;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, GGML_PAD(nsg * ಪರಿಮಾಣ(ಭಿನ್ನ), 16), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_sum_rows(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    ಯದಿ (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯಾವತ್ (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (ಪೂರ್ಣಾಙ್ಕ) args.ne00);

    ಧ್ರುವ size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_cumsum(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = 1;
    ಯಾವತ್ (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    ಧ್ರುವ int64_t net0 = (ne00 + nth - 1) / nth;
    ಧ್ರುವ int64_t net1 = ne01;
    ಧ್ರುವ int64_t net2 = ne02;
    ಧ್ರುವ int64_t net3 = ne03;

    ಧ್ರುವ uint64_t nbt0 = ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint64_t nbt1 = net0*nbt0;
    ಧ್ರುವ uint64_t nbt2 = net1*nbt1;
    ಧ್ರುವ uint64_t nbt3 = net2*nbt2;

    ಧ್ರುವ size_t smem = GGML_PAD(32*ಪರಿಮಾಣ(ಭಿನ್ನ), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
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

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    ಯದಿ (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
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
                /*.outb =*/ ಅಸತ್ಯ,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
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

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_get_rows(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
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

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_set_rows(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op);

    ಧ್ರುವ int32_t nk0 = ne0/ggml_blck_size(op->type);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯಾವತ್ (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    ಪೂರ್ಣಾಙ್ಕ nrptg = 1;
    ಯದಿ (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        ಯದಿ (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
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

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_diag(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    ಧ್ರುವ ggml_tensor * q = op->src[0];
    ಧ್ರುವ ggml_tensor * k = op->src[1];
    ಧ್ರುವ ggml_tensor * w = op->src[2];
    ಧ್ರುವ ggml_tensor * m = op->src[3];

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

    ggml_metal_kargs_lightning_indexer args = {
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

    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(q),  1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(k),  2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w),  3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m),  4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg   = OP_LIGHTNING_INDEXER_NSG;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nkptg = OP_LIGHTNING_INDEXER_NKPSG*nsg;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nbptg = OP_LIGHTNING_INDEXER_NBPTG;

    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, op);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc,
            (k->ne[2] + nkptg - 1)/nkptg,
            (q->ne[2] + nbptg - 1)/nbptg,
            q->ne[3], 32, nsg, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);

    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_DSV4_HC_COMB:
            {
                ಧ್ರುವ ggml_tensor * mixes = op->src[0];
                ಧ್ರುವ ggml_tensor * scale = op->src[1];
                ಧ್ರುವ ggml_tensor * base  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(base->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb args = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ base->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, ಪರಿಮಾಣ(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mixes), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scale), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(base),  3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = std::min(4, args.n_tokens);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (args.n_tokens + nsg - 1)/nsg, 1, 1, 32, nsg, 1);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DSV4_HC_PRE:
            {
                ಧ್ರುವ ggml_tensor * x       = op->src[0];
                ಧ್ರುವ ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre args = {
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

                ggml_metal_encoder_set_bytes (enc, &args, ಪರಿಮಾಣ(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      3);

                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_tiles = (args.n_embd + 31)/32;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DSV4_HC_POST:
            {
                ಧ್ರುವ ggml_tensor * x        = op->src[0];
                ಧ್ರುವ ggml_tensor * residual = op->src[1];
                ಧ್ರುವ ggml_tensor * post     = op->src[2];
                ಧ್ರುವ ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post args = {
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

                ggml_metal_encoder_set_bytes (enc, &args, ಪರಿಮಾಣ(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(post),     3);
                ಯದಿ (comb) {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(comb), 4);
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   5);
                } ಅನ್ಯಥಾ {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);
                }

                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_tiles = (args.n_embd + 31)/32;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("fatal error");
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_soft_max(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ಯದಿ (ctx->use_fusion()) {
        ಪೂರ್ಣಾಙ್ಕ n = 1;
        ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        ಯದಿ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            ಪ್ರತ್ಯಾಗಮ ggml_metal_op_topk_moe(ctx, idx);
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

    ಭಿನ್ನ scale;
    ಭಿನ್ನ max_bias;

    memcpy(&scale,    ((ಧ್ರುವ int32_t *) op->op_params) + 0, ಪರಿಮಾಣ(scale));
    memcpy(&max_bias, ((ಧ್ರುವ int32_t *) op->op_params) + 1, ಪರಿಮಾಣ(max_bias));

    ಧ್ರುವ uint32_t n_head      = op->src[0]->ne[2];
    ಧ್ರುವ  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ಭಿನ್ನ) n_head));

    ಧ್ರುವ ಭಿನ್ನ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ಧ್ರುವ ಭಿನ್ನ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
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
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯದಿ (ne00%4 == 0) {
        ಯಾವತ್ (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } ಅನ್ಯಥಾ {
        ಯಾವತ್ (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    ಧ್ರುವ size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ಯದಿ (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } ಅನ್ಯಥಾ {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ಯದಿ (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } ಅನ್ಯಥಾ {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;
    ತಾರ್ಕಿಕ use_silu = ಅಸತ್ಯ;

    ಯದಿ (ctx->use_fusion()) {
        ಪೂರ್ಣಾಙ್ಕ n = 1;
        ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        ಯದಿ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = ಸತ್ಯ;

            ctx->ಎಣಿಕೆ_fusions(fusion);
        }
    }

    ggml_metal_kargs_ssm_conv args = {
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

    ಧ್ರುವ ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(n_fuse > 1 ? ctx->ಗಂಟು(idx + n_fuse - 1) : op);

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    ಧ್ರುವ ತಾರ್ಕಿಕ use_batched = (ne1 > 1);

    ಯದಿ (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        ಪೂರ್ಣಾಙ್ಕ BATCH_SIZE;
        ಯದಿ      (ne1 > 128) BATCH_SIZE = 256;
        ಅನ್ಯಥಾ ಯದಿ (ne1 > 64 ) BATCH_SIZE = 128;
        ಅನ್ಯಥಾ ಯದಿ (ne1 > 32 ) BATCH_SIZE = 64;
        ಅನ್ಯಥಾ ಯದಿ (ne1 > 16 ) BATCH_SIZE = 32;
        ಅನ್ಯಥಾ ಯದಿ (ne1 > 8  ) BATCH_SIZE = 16;
        ಅನ್ಯಥಾ ಯದಿ (ne1 > 4  ) BATCH_SIZE = 8;
        ಅನ್ಯಥಾ                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } ಅನ್ಯಥಾ {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    ಯದಿ (n_fuse > 1 && ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SSM_CONV + UNARY\n", __func__);
    }

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

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

    ಧ್ರುವ ggml_tensor * src3 = op->src[3];
    ಧ್ರುವ ggml_tensor * src4 = op->src[4];
    ಧ್ರುವ ggml_tensor * src5 = op->src[5];
    ಧ್ರುವ ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    ಧ್ರುವ int64_t d_state      = ne00;
    ಧ್ರುವ int64_t d_inner      = ne01;
    ಧ್ರುವ int64_t n_head       = ne02;
    ಧ್ರುವ int64_t n_group      = ne41;
    ಧ್ರುವ int64_t n_seq_tokens = ne12;
    ಧ್ರುವ int64_t n_seqs       = ne13;
    ಧ್ರುವ int64_t K            = ggml_get_op_params_i32(op, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op));

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seq_tokens_total =*/ n_seq_tokens,
        /*.token_offset =*/ 0,
        /*.n_seqs       =*/ n_seqs,
        /*.K            =*/ K,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * ಪರಿಮಾಣ(ಭಿನ್ನ),
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

    constexpr int64_t CHUNK = OP_SSM_SCAN_SSD_CS;

    ಧ್ರುವ int64_t snap_reserve = K > 1 ? K : 0; // tokens reserved for sequential kernel rollback snapshots
    ಧ್ರುವ int64_t mma_tokens = ((n_seq_tokens - snap_reserve) / CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    ಧ್ರುವ ತಾರ್ಕಿಕ use_mma =
        mma_tokens > 0 &&
        ne30 == 1 && // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev->has_simdgroup_mm && // hardware check for M1 or newer
        d_state % 8 == 0 && // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == OP_SSM_SCAN_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    ಧ್ರುವ auto dispatch = [&](ggml_metal_pipeline_with_params pipeline, int64_t nth, int64_t n_tg_x) {
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(pipeline.smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg_x, n_head, n_seqs, nth, 1, 1);
    };

    ಯದಿ (!use_mma) {
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, ಅಸತ್ಯ), d_state, d_inner);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    args.n_seq_tokens = mma_tokens;
    dispatch(
        ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(lib, op),
        OP_SSM_SCAN_SSD_NSG*32,
        1);

    ಯದಿ (mma_tokens < n_seq_tokens) {
        ggml_metal_op_concurrency_reset(ctx);

        args.n_seq_tokens = n_seq_tokens - mma_tokens;
        args.token_offset = mma_tokens;
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, ಸತ್ಯ), d_state, d_inner);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_rwkv(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    ಧ್ರುವ int64_t T = op->src[0]->ne[2];
    ಧ್ರುವ int64_t C = op->ne[0];
    ಧ್ರುವ int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    ಪೂರ್ಣಾಙ್ಕ ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    ಯದಿ (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (ಶೂನ್ಯ *) &B, ಪರಿಮಾಣ(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (ಶೂನ್ಯ *) &T, ಪರಿಮಾಣ(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (ಶೂನ್ಯ *) &C, ಪರಿಮಾಣ(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (ಶೂನ್ಯ *) &H, ಪರಿಮಾಣ(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ತಾರ್ಕಿಕ use_fusion = ctx->use_fusion();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    // when fused with the trailing cache cpy, the snapshots are written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_CACHE)
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);
    uint64_t nb_out = 0;
    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;

    ಯದಿ (use_fusion) {
        ಪೂರ್ಣಾಙ್ಕ n = 1;
        ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        ಯದಿ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_GDN_CACHE) {
            ಧ್ರುವ ggml_tensor * dst_cache = ctx->ಗಂಟು(idx + 1)->src[1]; // cache view

            bid_out = ggml_metal_get_buffer_id(dst_cache);
            nb_out  = dst_cache->nb[2]/ಪರಿಮಾಣ(ಭಿನ್ನ);
            n_fuse = 2;

            ctx->ಎಣಿಕೆ_fusions(fusion);

            ಯದಿ (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: GATED_DELTA_NET + CPY\n", __func__);
            }
        }
    }

    ಪೂರ್ಣಾಙ್ಕ ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
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
        /*.ns02 =*/ (int32_t) (nb02/ಪರಿಮಾಣ(ಭಿನ್ನ)),
        /*.ns12 =*/ (int32_t) (nb12/ಪರಿಮಾಣ(ಭಿನ್ನ)),
        /*.ns22 =*/ (int32_t) (nb22/ಪರಿಮಾಣ(ಭಿನ್ನ)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.nb_out =*/ nb_out,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args),                  ida++); // args
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst (attn)
    ggml_metal_encoder_set_buffer  (enc, bid_out,                              ida++); // state_out

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_solve_tri(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_set(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ಧ್ರುವ size_t pnb1 = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಧ್ರುವ size_t pnb2 = ((ಧ್ರುವ int32_t *) op->op_params)[1];
    ಧ್ರುವ size_t pnb3 = ((ಧ್ರುವ int32_t *) op->op_params)[2];
    ಧ್ರುವ size_t offs = ((ಧ್ರುವ int32_t *) op->op_params)[3];

    ಧ್ರುವ ತಾರ್ಕಿಕ inplace = (ತಾರ್ಕಿಕ) ((ಧ್ರುವ int32_t *) op->op_params)[4];

    ಯದಿ (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
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

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    ಯದಿ (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } ಅನ್ಯಥಾ ಯದಿ (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    ಪೂರ್ಣಾಙ್ಕ nth = std::min<ಪೂರ್ಣಾಙ್ಕ>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    ಪೂರ್ಣಾಙ್ಕ nrptg = 1;

    // TODO: relax this constraint in the future
    ಯದಿ (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        ಯದಿ (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            ಯದಿ (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<ಪೂರ್ಣಾಙ್ಕ>(nth, nk0);

    ggml_metal_kargs_cpy args = {
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
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_cpy(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    ಯದಿ (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } ಅನ್ಯಥಾ ಯದಿ (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    ಪೂರ್ಣಾಙ್ಕ nth = std::min<ಪೂರ್ಣಾಙ್ಕ>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    ಪೂರ್ಣಾಙ್ಕ nrptg = 1;

    // TODO: relax this constraint in the future
    ಯದಿ (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        ಯದಿ (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            ಯದಿ (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<ಪೂರ್ಣಾಙ್ಕ>(nth, nk0);

    ggml_metal_kargs_cpy args = {
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

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_pool_1d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    ಧ್ರುವ int32_t k0 = opts[1];
    ಧ್ರುವ int32_t s0 = opts[2];
    ಧ್ರುವ int32_t p0 = opts[3];

    ಧ್ರುವ int64_t IW = op->src[0]->ne[0];
    ಧ್ರುವ int64_t OW = op->ne[0];

    ಧ್ರುವ int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(256, std::max(1, k0));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntg = (ಪೂರ್ಣಾಙ್ಕ) np;

    ಧ್ರುವ size_t smem = op_pool == GGML_OP_POOL_AVG ?
        2*32*ಪರಿಮಾಣ(ಭಿನ್ನ) :
        32*ಪರಿಮಾಣ(ಭಿನ್ನ);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, ಪರಿಮಾಣ(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_fwht(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_tensor * src1 = op->src[1];

    ಧ್ರುವ int64_t n = src1->ne[0];
    ಧ್ರುವ int64_t nrows = ggml_nrows(src1);

    ggml_metal_kargs_fwht args = {
        /*.nrows = */ (int32_t) nrows,
    };

    auto pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, src1->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ th_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ simd_size = 32;

    ಯದಿ (n >= GGML_METAL_FWHT_TG_MIN_N) {
        GGML_ASSERT(th_max >= GGML_METAL_FWHT_TG_NT);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, GGML_METAL_FWHT_TG_NT, 1, 1);

        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ಪೂರ್ಣಾಙ್ಕ sg_per_tg = 2;
    sg_per_tg = std::min(sg_per_tg, th_max/simd_size);
    sg_per_tg = std::max(sg_per_tg, 1);

    ಧ್ರುವ int64_t n_tg = (nrows + sg_per_tg - 1) / sg_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, 32*sg_per_tg, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_pool_2d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    ಧ್ರುವ int32_t k0 = opts[1];
    ಧ್ರುವ int32_t k1 = opts[2];
    ಧ್ರುವ int32_t s0 = opts[3];
    ಧ್ರುವ int32_t s1 = opts[4];
    ಧ್ರುವ int32_t p0 = opts[5];
    ಧ್ರುವ int32_t p1 = opts[6];

    ಧ್ರುವ int64_t IH = op->src[0]->ne[1];
    ಧ್ರುವ int64_t IW = op->src[0]->ne[0];

    ಧ್ರುವ int64_t N  = op->ne[3];
    ಧ್ರುವ int64_t OC = op->ne[2];
    ಧ್ರುವ int64_t OH = op->ne[1];
    ಧ್ರುವ int64_t OW = op->ne[0];

    ಧ್ರುವ int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
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

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (ಪೂರ್ಣಾಙ್ಕ) np);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, ಪರಿಮಾಣ(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

// the number of nodes from idx on that the fusion table fuses as pattern id into one dispatch, 1 if it fuses none
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_try_fusion(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx, ggml_metal_fusion_id id) {
    ಯದಿ (!ctx->use_fusion()) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ಪೂರ್ಣಾಙ್ಕ n = 1;
    ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
    ಯದಿ (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != id) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ctx->ಎಣಿಕೆ_fusions(fusion);

    ಯದಿ (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: %s to %s, %d nodes\n", __func__, ggml_op_name(ctx->ಗಂಟು(idx)->op), ggml_op_name(ctx->ಗಂಟು(idx + n - 1)->op), n);
    }

    ಪ್ರತ್ಯಾಗಮ n;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_mul_mat_mma(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    // the MMA store adds the residual when the table fuses the ADD after this mat-mul
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  n_fuse   = ggml_metal_op_try_fusion(ctx, idx, GGML_METAL_FUSION_MUL_MAT_ADD);
    ಧ್ರುವ ತಾರ್ಕಿಕ fuse_add = n_fuse > 1;

    ಧ್ರುವ ggml_tensor * dst = op;
    ಧ್ರುವ ggml_tensor * res = dst;

    ಯದಿ (fuse_add) {
        dst = ctx->ಗಂಟು(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_add);

    ggml_metal_kargs_mul_mv_ext args = {
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

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),        3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(res),        4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ rows0 = pipeline.nr0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ rows1 = pipeline.nr1;

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows0 - 1)/rows0, (ne11 + rows1 - 1)/rows1, ne12*ne13, 32, pipeline.nsg, 1);

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

// the generic mat-vec kernel, or its 2-row Q4_0 variant if nc
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_mul_mat_mv(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    ಧ್ರುವ int16_t r2 = ne12/ne02;
    ಧ್ರುವ int16_t r3 = ne13/ne03;

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr0 = pipeline.nr0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr1 = pipeline.nr1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = pipeline.nsg;

    ಧ್ರುವ size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
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

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ಯದಿ (op->src[0]->type == GGML_TYPE_F32 ||
        op->src[0]->type == GGML_TYPE_F16 ||
        op->src[0]->type == GGML_TYPE_BF16 ||
        op->src[0]->type == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    } ಅನ್ಯಥಾ {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_mul_mat(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    ಯದಿ (ggml_metal_op_mul_mat_use_fwht(op, props_dev->max_theadgroup_memory_size)) {
        ಪ್ರತ್ಯಾಗಮ ggml_metal_op_fwht(ctx, idx);
    }

    ಯದಿ (props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_mma(op)) {
        ಪ್ರತ್ಯಾಗಮ ggml_metal_op_mul_mat_mma(ctx, idx);
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

    ಧ್ರುವ int16_t r2 = ne12/ne02;
    ಧ್ರುವ int16_t r3 = ne13/ne03;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    ಯದಿ (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
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
           ಅಸತ್ಯ) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           ಅಸತ್ಯ) && (ne11 >= 4 && ne11 <= 8)
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
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        ಯದಿ (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } ಅನ್ಯಥಾ ಯದಿ (ne00 % 128 == 0) {
            nxpsg = 8;
        } ಅನ್ಯಥಾ {
            nxpsg = 4;
        }

        ಧ್ರುವ int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        ಧ್ರುವ int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        ವಿಕಲ್ಪ (ne11) {
            ಪಕ್ಷ 2:
                r1ptg = 2; ಭಙ್ಗ;
            ಪಕ್ಷ 3:
            ಪಕ್ಷ 6:
                r1ptg = 3; ಭಙ್ಗ;
            ಪಕ್ಷ 4:
            ಪಕ್ಷ 7:
            ಪಕ್ಷ 8:
                r1ptg = 4; ಭಙ್ಗ;
            ಪಕ್ಷ 5:
                r1ptg = 5; ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("unsupported ne11");
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
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

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } ಅನ್ಯಥಾ ಯದಿ (ggml_metal_op_mul_mat_use_mm(op, props_dev->has_simdgroup_mm)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
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

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ಧ್ರುವ size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr0 = pipeline.nr0;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr1 = pipeline.nr1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = pipeline.nsg;

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ ggml_metal_op_mul_mat_mv(ctx, idx);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

// upper bound on the number of entry tiles emitted by kernel_mul_mm_id_map0: the full tiles
// over all entries plus at most one partial tile per expert
ಸ್ಥಿರ int64_t ggml_metal_op_mul_mat_id_max_tiles(ಧ್ರುವ ggml_tensor * op) {
    ಧ್ರುವ int64_t ne02 = op->src[0]->ne[2]; // n_expert
    ಧ್ರುವ int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    ಧ್ರುವ int64_t ne21 = op->src[2]->ne[1]; // n_token

    ಧ್ರುವ int64_t n_rows = ne20*ne21;
    ಧ್ರುವ int64_t tiles  = (n_rows + 31)/32 + (ne02 < n_rows ? ne02 : n_rows) - 1;

    ಪ್ರತ್ಯಾಗಮ n_rows < tiles ? n_rows : tiles;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_MUL_MAT_ID);

    // row map of kernel_mul_mm_id: total tile count + one
    // (expert, first entry, valid rows) descriptor per tile
    ಧ್ರುವ int64_t max_tiles = ggml_metal_op_mul_mat_id_max_tiles(op);

    ಪ್ರತ್ಯಾಗಮ ggml_type_size(GGML_TYPE_I32)*(1 + 3*max_tiles);
}

size_t ggml_metal_op_mul_mat_id_extra_ids(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_MUL_MAT_ID);

    ಧ್ರುವ int64_t ne02 = op->src[0]->ne[2]; // n_expert
    ಧ್ರುವ int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    ಧ್ರುವ int64_t ne21 = op->src[2]->ne[1]; // n_token

    // one entry per (expert used, token) pair of ids plus one padding slot per expert
    // (see kernel_mul_mm_id_map0)
    ಪ್ರತ್ಯಾಗಮ ggml_type_size(GGML_TYPE_I32)*(ne20*ne21 + ne02);
}

size_t ggml_metal_op_mul_mat_id_extra_amax(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(op);

    // 2 scaling factors (8 bytes) + N_MM_NPART_AMAX per-threadgroup scales for stage-1
    ಪ್ರತ್ಯಾಗಮ 8 + N_MM_NPART_AMAX*ಪರಿಮಾಣ(ಭಿನ್ನ);
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

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

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ಧ್ರುವ uint32_t r2 = 1;
    ಧ್ರುವ uint32_t r3 = 1;

    ಯದಿ (ggml_metal_op_mul_mat_id_use_mm(op, props_dev->has_simdgroup_mm)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        ggml_metal_buffer_id bid_amax = bid_ids;
        bid_amax.offs += ggml_metal_op_mul_mat_id_extra_ids(op);

        // src1 prec [TAG_GGML_PREC]
        ಧ್ರುವ ತಾರ್ಕಿಕ use_amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

        // src1 rescale factors, computed before the matmul
        // ref: https://github.com/ggml-org/llama.cpp/pull/26223
        ಯದಿ (use_amax) {
            ggml_metal_kargs_mul_mm_id_amax args = {
                /*.ne00 =*/ ne10,
                /*.ne01 =*/ ne11,
                /*.ne02 =*/ ne12,
                /*.nb01 =*/ nb11,
                /*.nb02 =*/ nb12,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax_part(lib);

            ಧ್ರುವ size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 2);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, N_MM_NPART_AMAX, 1, 1, 256, 1, 1);
        }

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            ಧ್ರುವ size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        ಯದಿ (use_amax) {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax(lib);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 32, 1, 1);

            // the next kernel has to wait for the amax data
            ggml_metal_op_concurrency_reset(ctx);
        }

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
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

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 6);

            ಧ್ರುವ size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max_tiles = (ಪೂರ್ಣಾಙ್ಕ) ggml_metal_op_mul_mat_id_max_tiles(op);

            ggml_metal_encoder_dispatch_threadgroups(enc, max_tiles, (ne01 + 63)/64, 1, 128, 1, 1);
        }
    } ಅನ್ಯಥಾ {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr0 = pipeline.nr0;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr1 = pipeline.nr1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg = pipeline.nsg;

        ಧ್ರುವ size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
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

        ಯದಿ (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        ಧ್ರುವ int64_t _ne1 = 1;
        ಧ್ರುವ int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ಯದಿ (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } ಅನ್ಯಥಾ {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_add_id(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

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

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ತಾರ್ಕಿಕ ggml_metal_op_flash_attn_ext_use_vec(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಧ್ರುವ int64_t ne00 = op->src[0]->ne[0]; // head size
    ಧ್ರುವ int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    ಪ್ರತ್ಯಾಗಮ (ne01 < 20) && (ne00 % 32 == 0);
}

// ref: https://github.com/ggml-org/llama.cpp/pull/27390
// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_op_flash_attn_ext_use_kv_f16(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    // large heads need the upfront dequant to fit the non-vec threadgroup memory
    ಯದಿ (op->src[0]->ne[1] < 32 &&
        (op->src[0]->ne[0] < 512 || ggml_metal_op_flash_attn_ext_use_vec(op))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ವಿಕಲ್ಪ (op->src[1]->type) {
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_op_flash_attn_ext_use_tensor(ಧ್ರುವ ggml_tensor * op, ತಾರ್ಕಿಕ has_tensor) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಯದಿ (!has_tensor || ggml_metal_op_flash_attn_ext_use_vec(op)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t ne01 = op->src[0]->ne[1];
    ಧ್ರುವ int64_t ne02 = op->src[0]->ne[2];
    ಧ್ರುವ int64_t ne03 = op->src[0]->ne[3];

    ಧ್ರುವ int64_t dk = op->src[1]->ne[0];
    ಧ್ರುವ int64_t dv = op->src[2]->ne[0];

    ಧ್ರುವ ತಾರ್ಕಿಕ dk_dv_ok = (dk == 64  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 192 && dv == 128) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    ಯದಿ (!dk_dv_ok) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // large heads use fewer queries per threadgroup, so that the queries fit in threadgroup memory
    ಧ್ರುವ int64_t nqptg = dk >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG;

    // few heads and small batches do not fill the GPU - the half8x8 kernel is faster there
    // TODO: tune per device
    ಯದಿ (((ne01 + nqptg - 1)/nqptg)*ne02*ne03*dk < 8192) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_BF16 && !ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ op->src[1]->ne[1] % OP_FLASH_ATTN_EXT_TENSOR_NCPSG == 0;
}

// returns the n_kv_max hint if the sparse path is available for this op, or 0 otherwise
// the mask (src[3]) remains the single source of truth: finite entries are the valid KV positions,
// n_kv_max is only an upper bound on their number per mask row, used to size the index lists
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_flash_attn_ext_n_kv_max_sparse(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    int32_t n_kv_max = 0;
    memcpy(&n_kv_max, ((ಧ್ರುವ int32_t *) op->op_params) + 4, ಪರಿಮಾಣ(n_kv_max));

    ಯದಿ (n_kv_max <= 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // the sparse indices are gathered from the mask
    ಯದಿ (!op->src[3]) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // bound the size of the index lists
    ಯದಿ (n_kv_max > 4096) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    ಧ್ರುವ int64_t dk = op->src[1]->ne[0];
    ಧ್ರುವ int64_t dv = op->src[2]->ne[0];

    ಧ್ರುವ ತಾರ್ಕಿಕ dk_dv_ok = (dk == 32  && dv == 32)  ||
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

    ಯದಿ (!dk_dv_ok) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ವಿಕಲ್ಪ (op->src[1]->type) {
        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_BF16:
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪ್ರತ್ಯಾಗಮ n_kv_max;
}

// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_op_flash_attn_ext_v_is_view_of_k(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಧ್ರುವ ggml_tensor * K = op->src[1];
    ಧ್ರುವ ggml_tensor * V = op->src[2];

    ಪ್ರತ್ಯಾಗಮ V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
ಸ್ಥಿರ size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    ಪ್ರತ್ಯಾಗಮ GGML_PAD(ಪರಿಮಾಣ(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    ಧ್ರುವ ತಾರ್ಕಿಕ has_mask = op->src[3] != nullptr;
    ಧ್ರುವ ತಾರ್ಕಿಕ use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    ಧ್ರುವ ತಾರ್ಕಿಕ v_is_view_of_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    uint64_t nb11_pad = nb11;
    uint64_t nb21_pad = nb21;

    ಯದಿ (use_kv_f16) {
        nb11_pad = ಪರಿಮಾಣ(ggml_fp16_t)*ne10;
        nb21_pad = ಪರಿಮಾಣ(ggml_fp16_t)*(v_is_view_of_k ? ne10 : ne20);
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    ಯದಿ (ಅಸತ್ಯ) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        ಧ್ರುವ ತಾರ್ಕಿಕ has_kvpad = ಸತ್ಯ;

        ಯದಿ (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } ಅನ್ಯಥಾ {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        ಧ್ರುವ ತಾರ್ಕಿಕ has_kvpad = ಸತ್ಯ;

        ಯದಿ (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    ಪ್ರತ್ಯಾಗಮ res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    ಧ್ರುವ ತಾರ್ಕಿಕ has_mask = op->src[3] != nullptr;

    ಯದಿ (!has_mask) {
        ಪ್ರತ್ಯಾಗಮ res;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    ಧ್ರುವ int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    ಧ್ರುವ int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    ಪ್ರತ್ಯಾಗಮ res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

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
    ಯದಿ (ಸತ್ಯ) {
        ಧ್ರುವ int64_t nwg = 32;
        ಧ್ರುವ int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    ಪ್ರತ್ಯಾಗಮ res;
}

size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
    //    return 0;
    //}

    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    ಧ್ರುವ size_t k_size = ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    ಧ್ರುವ ತಾರ್ಕಿಕ v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    ಯದಿ (v_is_view_of_k) {
        ಪ್ರತ್ಯಾಗಮ k_size;
    }

    ಧ್ರುವ size_t v_size = GGML_PAD(ಪರಿಮಾಣ(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);

    ಪ್ರತ್ಯಾಗಮ k_size + v_size;
}

// size of the sparse index lists: one list of KV indices per mask row,
// padded with -1 up to a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG
size_t ggml_metal_op_flash_attn_ext_extra_idx(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);

    ಯದಿ (n_kv_max <= 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_kv_max_padded = GGML_PAD(n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG);

    ಪ್ರತ್ಯಾಗಮ GGML_PAD(ಪರಿಮಾಣ(int32_t)*(size_t) n_kv_max_padded*ne31*ne32*ne33, 16);
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

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

    ಭಿನ್ನ scale;
    ಭಿನ್ನ max_bias;
    ಭಿನ್ನ logit_softcap;

    memcpy(&scale,         ((ಧ್ರುವ int32_t *) op->op_params) + 0, ಪರಿಮಾಣ(scale));
    memcpy(&max_bias,      ((ಧ್ರುವ int32_t *) op->op_params) + 1, ಪರಿಮಾಣ(max_bias));
    memcpy(&logit_softcap, ((ಧ್ರುವ int32_t *) op->op_params) + 2, ಪರಿಮಾಣ(logit_softcap));

    ಯದಿ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ has_mask  = op->src[3] != ರಿಕ್ತ;
    ಧ್ರುವ ತಾರ್ಕಿಕ has_sinks = op->src[4] != ರಿಕ್ತ;
    ಧ್ರುವ ತಾರ್ಕಿಕ has_bias  = max_bias != 0.0f;
    ಧ್ರುವ ತಾರ್ಕಿಕ has_scap  = logit_softcap != 0.0f;

    ಧ್ರುವ uint32_t n_head      = op->src[0]->ne[2];
    ಧ್ರುವ  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ಭಿನ್ನ) n_head));

    ಧ್ರುವ ಭಿನ್ನ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ಧ್ರುವ ಭಿನ್ನ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_kv_max_sparse = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    ಧ್ರುವ ತಾರ್ಕಿಕ use_sparse = n_kv_max_sparse > 0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_kv_max_padded = use_sparse ? GGML_PAD(n_kv_max_sparse, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    ಧ್ರುವ ತಾರ್ಕಿಕ use_kv_f16 = !use_sparse && ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    ggml_metal_buffer_id bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    ggml_metal_buffer_id bid_k = bid_src1;
    ggml_metal_buffer_id bid_v = bid_src2;

    uint64_t nb10_attn = nb10;
    uint64_t nb11_attn = nb11;
    uint64_t nb12_attn = nb12;
    uint64_t nb13_attn = nb13;
    uint64_t nb20_attn = nb20;
    uint64_t nb21_attn = nb21;
    uint64_t nb22_attn = nb22;
    uint64_t nb23_attn = nb23;

    ಯದಿ (use_kv_f16) {
        ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);

        ಧ್ರುವ ತಾರ್ಕಿಕ v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        ಧ್ರುವ int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
        ಧ್ರುವ int32_t nblocks1 = nblocks1_64;

        ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);

        // K
        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
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

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args_k, ಪರಿಮಾಣ(args_k), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src1,        1);
        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        ಯದಿ (!v_is_view_of_k) {
            ಧ್ರುವ int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
            ಧ್ರುವ int32_t nblocks2 = nblocks2_64;

            ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
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

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args_v, ಪರಿಮಾಣ(args_v), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2,        1);
            ggml_metal_encoder_set_buffer  (enc, bid_v_f16,       2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        ggml_metal_op_concurrency_reset(ctx);

        bid_k = bid_kv_f16;
        bid_v = v_is_view_of_k ? bid_k : bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = ಪರಿಮಾಣ(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        ಯದಿ (v_is_view_of_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } ಅನ್ಯಥಾ {
            // contiguous F16 layout of the dequantized V
            nb20_attn = ಪರಿಮಾಣ(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne21;
            nb23_attn = nb22_attn*ne22;
        }
    }

    ಯದಿ (!use_sparse && ggml_metal_op_flash_attn_ext_use_tensor(op, props_dev->has_tensor)) {
        // tensor API kernel
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nqptg = ne00 >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG; // queries per threadgroup
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncpsg = OP_FLASH_ATTN_EXT_TENSOR_NCPSG; // cache values per threadgroup
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsg   = OP_FLASH_ATTN_EXT_TENSOR_NSG;

        ಯದಿ (has_mask) {
            ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

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

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, ಪರಿಮಾಣ(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            ಧ್ರುವ int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            ಧ್ರುವ int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);
        }

        ಧ್ರುವ int32_t ns10 = nb11_attn/nb10_attn;
        ಧ್ರುವ int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
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
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // shared memory layout: queries (16-bit), scores (float), probabilities (16-bit), row scale (float), rescale flag (int)
        ಧ್ರುವ size_t smem = GGML_PAD(nqptg*ne00*ಪರಿಮಾಣ(ggml_fp16_t) + nqptg*ncpsg*(ಪರಿಮಾಣ(ಭಿನ್ನ) + ಪರಿಮಾಣ(ggml_fp16_t)) + nqptg*ಪರಿಮಾಣ(ಭಿನ್ನ) + ಪರಿಮಾಣ(int32_t), 16);

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_tensor(lib, op, has_mask, has_sinks, has_bias, has_scap);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } ಅನ್ಯಥಾ ಯದಿ (!use_sparse && !ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        ತಾರ್ಕಿಕ need_sync = ಅಸತ್ಯ;

        ಧ್ರುವ ತಾರ್ಕಿಕ has_kvpad = ne11 % ncpsg != 0;

        ಯದಿ (has_kvpad) {
            ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

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

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, ಪರಿಮಾಣ(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            ಪ್ರತಿಜ್ಞಾ(ne12 == ne22);
            ಪ್ರತಿಜ್ಞಾ(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = ಸತ್ಯ;
        }

        ಯದಿ (has_mask) {
            ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

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

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, ಪರಿಮಾಣ(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            ಧ್ರುವ int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            ಧ್ರುವ int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = ಸತ್ಯ;
        }

        ಯದಿ (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        ಧ್ರುವ int64_t dv_pad = GGML_PAD(ne20, 64);

        auto fa_smem = [&](int32_t nsg) -> size_t {
            ಧ್ರುವ size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
            ಪ್ರತ್ಯಾಗಮ GGML_PAD(smem_half*ಪರಿಮಾಣ(ggml_fp16_t), 16);
        };

        // simdgroups per threadgroup (a.k.a. warps)
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        ಧ್ರುವ size_t smem = fa_smem(nsg);

        ಧ್ರುವ int32_t ns10 = nb11_attn/nb10_attn;
        ಧ್ರುವ int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
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
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, use_kv_f16, ns10, ns20);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } ಅನ್ಯಥಾ {
        // half4x4 kernel
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        auto cfg = use_sparse
                ? ggml_metal_tuning::fa_vec_baseline_cfg((ಪೂರ್ಣಾಙ್ಕ) ne00, (ಪೂರ್ಣಾಙ್ಕ) ne20)
                : ggml_metal_tuning::fa_vec_pick(
                          props_dev->gpu_family,
                          (ಪೂರ್ಣಾಙ್ಕ) op->src[1]->type,
                          (ಪೂರ್ಣಾಙ್ಕ) ne00, (ಪೂರ್ಣಾಙ್ಕ) ne20,   // dk, dv (ne00 == dk for FA)
                          ne11, ne01);

        ಪೂರ್ಣಾಙ್ಕ nqptg = cfg.Q; // queries per threadgroup

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg == 1 || nqptg == 2 || nqptg == 4);  // only instantiated Q values
        GGML_ASSERT(ncpsg  % 32 == 0);

        ತಾರ್ಕಿಕ need_sync = ಅಸತ್ಯ;

        ಧ್ರುವ ತಾರ್ಕಿಕ has_kvpad = !use_sparse && ne11 % ncpsg != 0;

        ಯದಿ (use_sparse) {
            ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0);

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

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, ಪರಿಮಾಣ(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_idx,  2);

            ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);
            nth = std::max(32, (nth/32)*32);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = ಸತ್ಯ;
        }

        ಯದಿ (has_kvpad) {
            ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

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

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, ಪರಿಮಾಣ(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            ಪ್ರತಿಜ್ಞಾ(ne12 == ne22);
            ಪ್ರತಿಜ್ಞಾ(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = ಸತ್ಯ;
        }

        ಯದಿ (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (only when use_sparse)
        ಧ್ರುವ int64_t dk_pad = GGML_PAD(ne00, 128);
        ಧ್ರುವ int64_t dv_pad = GGML_PAD(ne20, 128);

        auto fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
            ಧ್ರುವ size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
            ಪ್ರತ್ಯಾಗಮ GGML_PAD(smem_half*ಪರಿಮಾಣ(ggml_fp16_t) + (use_sparse ? (size_t) nsg*ncpsg*ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) : 0), 16);
        };

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        ಯದಿ (use_sparse) {
            ಯದಿ (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                ಯದಿ (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } ಅನ್ಯಥಾ {
                    ಯಾವತ್ (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                        nsg *= 2;
                    }
                }
            } ಅನ್ಯಥಾ {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                ಯಾವತ್ (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                    nsg *= 2;
                }
            }
        } ಅನ್ಯಥಾ {
            nwg = 32;
            nsg = 1;
            ಯಾವತ್ (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        ಯದಿ (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_memory_size) {
            cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((ಪೂರ್ಣಾಙ್ಕ) ne00, (ಪೂರ್ಣಾಙ್ಕ) ne20);
            nqptg = cfg.Q;  // = 1
        }

        ಧ್ರುವ int32_t ns10 = nb11_attn/nb10_attn;
        ಧ್ರುವ int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext_vec args = {
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
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.n_kv_max_padded =*/ n_kv_max_padded,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, use_sparse, nqptg, cfg.NE, nsg, nwg, use_kv_f16, ns10, ns20);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, use_sparse ? bid_idx : bid_src0, 8);

        ಧ್ರುವ size_t smem = fa_vec_smem(nsg, nqptg);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ಯದಿ (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } ಅನ್ಯಥಾ {
            // sanity checks
            ಪ್ರತಿಜ್ಞಾ(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                ಧ್ರುವ int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, ಪರಿಮಾಣ(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_bin(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;
    ಧ್ರುವ ggml_metal_fusion * fusion = nullptr;

    ಧ್ರುವ ತಾರ್ಕಿಕ use_fusion = ctx->use_fusion();

    ಯದಿ (use_fusion) {
        ಪೂರ್ಣಾಙ್ಕ n = 1;
        fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        ಯದಿ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SNAKE) {
            ctx->ಎಣಿಕೆ_fusions(fusion);
            ಪ್ರತ್ಯಾಗಮ ggml_metal_op_snake_fused(ctx, idx);
        }

        // MoE output reduction: experts * weights -> weighted sum
        ಯದಿ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MOE_REDUCE) {
            ctx->ಎಣಿಕೆ_fusions(fusion);
            ಪ್ರತ್ಯಾಗಮ ggml_metal_op_moe_reduce(ctx, idx);
        }
    }

    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
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
    ಯದಿ (use_fusion && fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_ADD_CHAIN) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_fuse; i++) {
            args.o1[i] = ggml_metal_get_buffer_id(ctx->ಗಂಟು(idx + i)->src[1]).offs;
        }

        ctx->ಎಣಿಕೆ_fusions(fusion);

        ಯದಿ (debug_fusion > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    ಸಂರಚನಾ ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    ಯದಿ (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->ಗಂಟು(idx + n_fuse - 1));

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_fuse; ++i) {
            ಯದಿ (!ggml_metal_op_concurrency_check(ctx, ctx->ಗಂಟು(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                ಭಙ್ಗ;
            }
        }
    }

    ಯದಿ (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    ಯದಿ (pipeline.cnt) {
        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, ggml_nrows(op), 1, 1, 1, 1);
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ಪೂರ್ಣಾಙ್ಕ nth = 1;

        ಯಾವತ್ (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_silu_back(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_silu_back(lib, op);

    ಧ್ರುವ int64_t ne = ggml_nelements(op);

    ggml_metal_kargs_silu_back args = {
        /*.ne =*/ ne,
    };

    ಪೂರ್ಣಾಙ್ಕ arg_idx{0};

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), arg_idx++);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min<int64_t>(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne);
    ಧ್ರುವ int64_t n = (ne + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_l2_norm(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ಭಿನ್ನ eps;
    memcpy(&eps, op->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ggml_metal_kargs_l2_norm args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    ಯದಿ (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯಾವತ್ (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ಧ್ರುವ size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_group_norm(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t ngrp = ((ಧ್ರುವ int32_t *) op->op_params)[0];

    ಭಿನ್ನ eps;
    memcpy(&eps, op->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    ಧ್ರುವ size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_norm(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ತಾರ್ಕಿಕ use_fusion = ctx->use_fusion();

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಭಿನ್ನ eps;
    memcpy(&eps, op->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
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

    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;
    ತಾರ್ಕಿಕ fused_norm_scale = ಅಸತ್ಯ;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    ಯದಿ (use_fusion) {
        ಪೂರ್ಣಾಙ್ಕ n = 1;
        ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        ಯದಿ (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD)) {
            n_fuse = n;

            ctx->ಎಣಿಕೆ_fusions(fusion);

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_fuse; i++) {
                ಧ್ರುವ ggml_tensor * fn = ctx->ಗಂಟು(idx + i);

                bid_fuse[i - 1] = ggml_metal_get_buffer_id(fn->src[1]);

                args.nef1[i] = fn->src[1]->ne[1];
                args.nef2[i] = fn->src[1]->ne[2];
                args.nef3[i] = fn->src[1]->ne[3];

                args.nbf1[i] = fn->src[1]->nb[1];
                args.nbf2[i] = fn->src[1]->nb[2];
                args.nbf3[i] = fn->src[1]->nb[3];
            }

            ಯದಿ (debug_fusion > 1) {
                ಯದಿ (n_fuse == 2) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
                }
                ಯದಿ (n_fuse == 3) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
                }
            }
        }

        ಯದಿ (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_SCALE) {
            n_fuse = n;
            fused_norm_scale = ಸತ್ಯ;

            ctx->ಎಣಿಕೆ_fusions(fusion);

            ಧ್ರುವ ggml_tensor * scale_ಗಂಟು = ctx->ಗಂಟು(idx + 1);
            args.scale = ggml_get_op_params_f32(scale_ಗಂಟು, 0);

            ಯದಿ (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + SCALE\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    ಯದಿ (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->ಗಂಟು(idx + n_fuse - 1));

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_fuse; ++i) {
            ಯದಿ (!ggml_metal_op_concurrency_check(ctx, ctx->ಗಂಟು(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                ಭಙ್ಗ;
            }
        }
    }

    auto pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯಾವತ್ (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (args.ne00_t + 31)/32*32);

    ಧ್ರುವ size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_rope(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

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

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(1024, ne00);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_past     = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims     = ((ಧ್ರುವ int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig = ((ಧ್ರುವ int32_t *) op->op_params)[4];

    ಭಿನ್ನ freq_base;
    ಭಿನ್ನ freq_scale;
    ಭಿನ್ನ ext_factor;
    ಭಿನ್ನ attn_factor;
    ಭಿನ್ನ beta_fast;
    ಭಿನ್ನ beta_slow;

    memcpy(&freq_base,   (ಧ್ರುವ int32_t *) op->op_params +  5, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&freq_scale,  (ಧ್ರುವ int32_t *) op->op_params +  6, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&ext_factor,  (ಧ್ರುವ int32_t *) op->op_params +  7, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&attn_factor, (ಧ್ರುವ int32_t *) op->op_params +  8, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&beta_fast,   (ಧ್ರುವ int32_t *) op->op_params +  9, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&beta_slow,   (ಧ್ರುವ int32_t *) op->op_params + 10, ಪರಿಮಾಣ(ಭಿನ್ನ));

    // mrope
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sect_0 = ((ಧ್ರುವ int32_t *) op->op_params)[11];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sect_1 = ((ಧ್ರುವ int32_t *) op->op_params)[12];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sect_2 = ((ಧ್ರುವ int32_t *) op->op_params)[13];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sect_3 = ((ಧ್ರುವ int32_t *) op->op_params)[14];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs = ((ಧ್ರುವ int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    ಧ್ರುವ ತಾರ್ಕಿಕ inplace = op->data == op->src[0]->data;

    ggml_metal_kargs_rope args = {
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
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
        /* inplace     =*/ inplace,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ಯದಿ (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } ಅನ್ಯಥಾ {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_im2col(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(op->op_params))[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t *)(op->op_params))[1];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *)(op->op_params))[2];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t *)(op->op_params))[3];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t *)(op->op_params))[4];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t *)(op->op_params))[5];

    ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = ((ಧ್ರುವ int32_t *)(op->op_params))[6] == 1;

    ಧ್ರುವ int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    ಧ್ರುವ int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    ಧ್ರುವ int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    ಧ್ರುವ int32_t IW =         op->src[1]->ne[0];

    ಧ್ರುವ int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    ಧ್ರುವ int32_t KW =         op->src[0]->ne[0];

    ಧ್ರುವ int32_t OH = is_2D ? op->ne[2] : 1;
    ಧ್ರುವ int32_t OW =         op->ne[1];

    ಧ್ರುವ int32_t CHW = IC * KH * KW;

    ಧ್ರುವ uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    ಧ್ರುವ uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    ಯದಿ (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        ಧ್ರುವ uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
        ಧ್ರುವ int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_conv_2d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

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

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t *) op->op_params)[1];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *) op->op_params)[2];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t *) op->op_params)[3];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t *) op->op_params)[4];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    ಧ್ರುವ uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<ಪೂರ್ಣಾಙ್ಕ>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

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

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t *) op->op_params)[1];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *) op->op_params)[2];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t *) op->op_params)[3];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t *) op->op_params)[4];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw args = {
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

    ಧ್ರುವ ತಾರ್ಕಿಕ use_tiled = (nb12 < nb10);

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    ಪೂರ್ಣಾಙ್ಕ nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    ಧ್ರುವ int32_t OW = ne0;
    ಧ್ರುವ int32_t OH = ne1;
    ಧ್ರುವ int32_t C  = ne12;
    ಧ್ರುವ int32_t N  = ne13;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tg_y = OH;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_conv_3d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(op->op_params))[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t *)(op->op_params))[1];
    ಧ್ರುವ int32_t s2 = ((ಧ್ರುವ int32_t *)(op->op_params))[2];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *)(op->op_params))[3];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t *)(op->op_params))[4];
    ಧ್ರುವ int32_t p2 = ((ಧ್ರುವ int32_t *)(op->op_params))[5];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t *)(op->op_params))[6];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t *)(op->op_params))[7];
    ಧ್ರುವ int32_t d2 = ((ಧ್ರುವ int32_t *)(op->op_params))[8];
    ಧ್ರುವ int32_t IC = ((ಧ್ರುವ int32_t *)(op->op_params))[9];
    ಧ್ರುವ int32_t N  = ((ಧ್ರುವ int32_t *)(op->op_params))[10];
    ಧ್ರುವ int32_t OC = ((ಧ್ರುವ int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
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
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    ಪೂರ್ಣಾಙ್ಕ nth0 = 32; // Standard SIMD width for Apple Silicon
    ಪೂರ್ಣಾಙ್ಕ nth1 = 1;
    ಪೂರ್ಣಾಙ್ಕ nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    ಪೂರ್ಣಾಙ್ಕ ntg0 = (spatial_volume + nth0 - 1) / nth0;
    ಪೂರ್ಣಾಙ್ಕ ntg1 = args.OC;
    ಪೂರ್ಣಾಙ್ಕ ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(op->op_params))[0];

    ಧ್ರುವ int32_t IC = op->src[1]->ne[1];
    ಧ್ರುವ int32_t IL = op->src[1]->ne[0];

    ಧ್ರುವ int32_t K  = op->src[0]->ne[0];

    ಧ್ರುವ int32_t OL = op->ne[0];
    ಧ್ರುವ int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(op->op_params))[0];
    ಧ್ರುವ int32_t OC = ((ಧ್ರುವ int32_t *)(op->op_params))[1];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *)(op->op_params))[2];

    ಧ್ರುವ int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    ಧ್ರುವ int32_t T_in  = (int32_t) op->src[0]->ne[1];
    ಧ್ರುವ int32_t K     = K_OC / OC;
    ಧ್ರುವ int32_t T_out = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d args = {
        /*.T_in  =*/ T_in,
        /*.T_out =*/ T_out,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total = T_out * OC;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth   = 256;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntg   = (total + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

// Dispatch the fused snake kernel from the matched mul -> sin -> sqr -> mul -> add chain.
// idx points at the leading mul. The caller has validated the chain.
ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_snake_fused(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಧ್ರುವ ggml_tensor * mul0 = ctx->ಗಂಟು(idx + 0);
    ಧ್ರುವ ggml_tensor * sqr  = ctx->ಗಂಟು(idx + 2);
    ಧ್ರುವ ggml_tensor * mul1 = ctx->ಗಂಟು(idx + 3);
    ggml_tensor *       add  = ctx->ಗಂಟು(idx + 4);

    ಧ್ರುವ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಧ್ರುವ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];
    ಧ್ರುವ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ T     = (ಪೂರ್ಣಾಙ್ಕ) x->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ C     = (ಪೂರ್ಣಾಙ್ಕ) x->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total = T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < 5; ++i) {
        ಯದಿ (!ggml_metal_op_concurrency_check(ctx, ctx->ಗಂಟು(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);

            ಭಙ್ಗ;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, x->type);

    ggml_metal_kargs_snake args = {
        /*.T =*/ T,
        /*.C =*/ C,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(inv_b), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add),   4);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = 256;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntg = (total + nth - 1) / nth;
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 5;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(op->op_params))[0];

    ಧ್ರುವ int32_t IC = op->src[1]->ne[2];
    ಧ್ರುವ int32_t IH = op->src[1]->ne[1];
    ಧ್ರುವ int32_t IW = op->src[1]->ne[0];

    ಧ್ರುವ int32_t KH = op->src[0]->ne[1];
    ಧ್ರುವ int32_t KW = op->src[0]->ne[0];

    ಧ್ರುವ int32_t OW = op->ne[0];
    ಧ್ರುವ int32_t OH = op->ne[1];
    ಧ್ರುವ int32_t OC = op->ne[2];
    ಧ್ರುವ int32_t N  = op->src[1]->ne[3];

    ggml_metal_kargs_conv_transpose_2d args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    ಧ್ರುವ size_t smem = GGML_PAD(KW * KH * ಪರಿಮಾಣ(ಭಿನ್ನ), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_upscale(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಭಿನ್ನ sf0 = (ಭಿನ್ನ)ne0/op->src[0]->ne[0];
    ಭಿನ್ನ sf1 = (ಭಿನ್ನ)ne1/op->src[0]->ne[1];
    ಭಿನ್ನ sf2 = (ಭಿನ್ನ)ne2/op->src[0]->ne[2];
    ಭಿನ್ನ sf3 = (ಭಿನ್ನ)ne3/op->src[0]->ne[3];

    ಧ್ರುವ int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    ಭಿನ್ನ poffs = 0.5f;

    ಯದಿ (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ಭಿನ್ನ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ಭಿನ್ನ)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_roll(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int32_t s0 = ggml_get_op_params_i32(op, 0);
    ಧ್ರುವ int32_t s1 = ggml_get_op_params_i32(op, 1);
    ಧ್ರುವ int32_t s2 = ggml_get_op_params_i32(op, 2);
    ಧ್ರುವ int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_pad(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth_max = MIN(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = MIN(args.ne0, nth_max);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nk0 = (args.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
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
        /*.p0 =*/ ((ಧ್ರುವ int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((ಧ್ರುವ int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_arange(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಭಿನ್ನ start;
    ಭಿನ್ನ step;

    memcpy(&start, ((ಧ್ರುವ int32_t *) op->op_params) + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&step,  ((ಧ್ರುವ int32_t *) op->op_params) + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dim        = op->op_params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

size_t ggml_metal_op_argmax_extra(ಧ್ರುವ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ARGMAX);

    ಧ್ರುವ int64_t ne00 = op->src[0]->ne[0];
    ಧ್ರುವ int64_t ne01 = op->src[0]->ne[1];

    ಪೂರ್ಣಾಙ್ಕ npart = 1;
    ಯದಿ (ne00 > N_ARGMAX_CHUNK) {
        npart = (ಪೂರ್ಣಾಙ್ಕ) std::min<int64_t>(N_ARGMAX_NPART_MAX, (ne00 + N_ARGMAX_CHUNK - 1)/N_ARGMAX_CHUNK);
    }

    ಯದಿ (npart == 1) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // partial max + partial argmax for every row
    ಪ್ರತ್ಯಾಗಮ (ಪರಿಮಾಣ(ಭಿನ್ನ) + ಪರಿಮಾಣ(int32_t)) * npart * ne01;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_argmax(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ಧ್ರುವ int64_t nrows = ggml_nrows(op->src[0]);

    ಪೂರ್ಣಾಙ್ಕ npart = 1;
    ಯದಿ (ne00 > N_ARGMAX_CHUNK) {
        npart = (ಪೂರ್ಣಾಙ್ಕ) std::min<int64_t>(N_ARGMAX_NPART_MAX, (ne00 + N_ARGMAX_CHUNK - 1)/N_ARGMAX_CHUNK);
    }

    ggml_metal_kargs_argmax args = {
        /*.ne00  = */ ne00,
        /*.nb01  = */ nb01,
        /*.npart = */ npart,
        /*.ne01  = */ ne01,
    };

    ಧ್ರುವ size_t smem = 32*(ಪರಿಮಾಣ(ಭಿನ್ನ) + ಪರಿಮಾಣ(int32_t));

    ಯದಿ (npart > 1) {
        auto pipeline_part = ggml_metal_library_get_pipeline_argmax_part(lib);
        auto pipeline_final = ggml_metal_library_get_pipeline_argmax_final(lib);

        ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
        ggml_metal_buffer_id bid_tmp = bid_dst;
        bid_tmp.offs += ggml_nbytes(op);

        ggml_metal_encoder_set_pipeline(enc, pipeline_part);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, npart, nrows, 1, 256, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);

        ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width
        ಯಾವತ್ (nth < npart && nth < 256) {
            nth *= 2;
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline_final);
        ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

        ಪ್ರತ್ಯಾಗಮ 1;
    }

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width
    ಯಾವತ್ (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_argsort(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    ಪೂರ್ಣಾಙ್ಕ nth = 1;
    ಯಾವತ್ (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    ಧ್ರುವ size_t smem = GGML_PAD(nth*ಪರಿಮಾಣ(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    ಯದಿ ((ಪೂರ್ಣಾಙ್ಕ) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
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

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    ಪೂರ್ಣಾಙ್ಕ len = nth;

    ಯಾವತ್ (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
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
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nm = (ne00 + 2*len - 1) / (2*len);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, ಪರಿಮಾಣ(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

// bitonic-sort + merge fallback: efficient when k is small and there are few rows,
// where the single-workgroup-per-row radix-select cannot reach enough parallelism
ಸ್ಥಿರ ಶೂನ್ಯ ggml_metal_op_ತುದಿ_k_bitonic(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_ತುದಿ_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    ಪೂರ್ಣಾಙ್ಕ nth = 1;
    ಯಾವತ್ (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ npr = (ne00 + nth - 1)/nth;

    ಧ್ರುವ size_t smem = GGML_PAD(nth*ಪರಿಮಾಣ(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ಪರಿಮಾಣ(int32_t)*ggml_nelements(op->src[0]);

    ಯದಿ ((ಪೂರ್ಣಾಙ್ಕ) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ತುದಿ_k = ne0;

    ggml_metal_kargs_argsort args = {
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
        /*.top_k =*/ std::min(nth, ತುದಿ_k), // for each block, keep just the top_k indices
    };

    ಯದಿ (npr > 1) {
        args.ne0 = (npr - 1)*args.ತುದಿ_k + std::min(ne00 - (npr - 1)*nth, args.ತುದಿ_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_ತುದಿ_k_merge(lib, op);

    ಪೂರ್ಣಾಙ್ಕ len = args.ತುದಿ_k;

    ಯಾವತ್ (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nm = (args.ne0 + 2*len - 1) / (2*len);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? ತುದಿ_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, ಪರಿಮಾಣ(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }
}

// radix-select: one workgroup per row. Maps each float to an order-preserving unsigned
// key, finds the k-th largest via 4 radix-8 histogram passes, then compacts the top-k
// indices. Fast for large k and/or many rows.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_metal_op_ತುದಿ_k_radix(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);

    auto pipeline = ggml_metal_library_get_pipeline_ತುದಿ_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_kargs_ತುದಿ_k args = {
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
    ಧ್ರುವ size_t smem_histo  = GGML_PAD(256*ಪರಿಮಾಣ(uint32_t), 16);
    ಧ್ರುವ size_t smem_bucket = GGML_PAD(    ಪರಿಮಾಣ(uint32_t), 16);
    ಧ್ರುವ size_t smem_above  = GGML_PAD(    ಪರಿಮಾಣ(uint32_t), 16);
    ಧ್ರುವ size_t smem_out    = GGML_PAD(    ಪರಿಮಾಣ(uint32_t), 16);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_histo,  0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_bucket, 1);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_above,  2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_out,    3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_topk_moe(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;
    ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    ಯದಿ (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_TOPK_MOE) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ggml_tensor * softmax  = ctx->ಗಂಟು(idx);
    ggml_tensor * logits   = softmax->src[0];
    ggml_tensor * get_rows = ctx->ಗಂಟು(idx + 2);
    ggml_tensor * ids      = get_rows->src[1];
    ggml_tensor * weights  = ctx->ಗಂಟು(idx + n_fuse - 1);

    ಧ್ರುವ int64_t n_expert      = logits->ne[0];
    ಧ್ರುವ int64_t n_tokens      = logits->ne[1];
    ಧ್ರುವ int64_t n_expert_used = ids->ne[0];

    ಧ್ರುವ ತಾರ್ಕಿಕ with_norm  = n_fuse >= 6;
    ಧ್ರುವ ತಾರ್ಕಿಕ with_scale = n_fuse == 4 || n_fuse == 7;

    ಭಿನ್ನ clamp = -INFINITY;
    ಯದಿ (with_norm) {
        ggml_tensor * clamp_ಗಂಟು = ctx->ಗಂಟು(idx + 4);
        clamp = ggml_get_op_params_f32(clamp_ಗಂಟು, 0);
    }

    ಭಿನ್ನ scale = 1.0f;
    ಯದಿ (with_scale) {
        ggml_tensor * scale_ಗಂಟು = ctx->ಗಂಟು(idx + n_fuse - 1);
        scale = ggml_get_op_params_f32(scale_ಗಂಟು, 0);
    }

    ggml_metal_kargs_topk_moe args = {
        /*.ne01      =*/ (int32_t) n_tokens,
        /*.nb01      =*/ logits->nb[1],
        /*.nb1_ids   =*/ ids->nb[1],
        /*.clamp =*/ clamp,
        /*.scale =*/ scale,
    };

    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_used, with_norm);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(logits),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) n_tokens, 1, 1, 32, 1, 1);

    ctx->ಎಣಿಕೆ_fusions(fusion);

    ಯದಿ (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SOFT_MAX + ARGSORT + GET_ROWS\n", __func__);
    }

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_moe_reduce(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ಪೂರ್ಣಾಙ್ಕ n_fuse = 1;
    ಧ್ರುವ ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    ಯದಿ (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MOE_REDUCE) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ggml_tensor * mul     = ctx->ಗಂಟು(idx);
    ggml_tensor * experts = mul->src[0];
    ggml_tensor * weights = mul->src[1];
    ggml_tensor * dst     = ctx->ಗಂಟು(idx + n_fuse - 1);

    ggml_metal_kargs_moe_reduce args = {
        /*.ne00 =*/ (int32_t) experts->ne[0],
        /*.ne02 =*/ (int32_t) experts->ne[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) args.ne02, (uint32_t) n_col_tiles, 1, nth, 1, 1);

    ctx->ಎಣಿಕೆ_fusions(fusion);

    ಯದಿ (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\n", __func__);
    }

    ಪ್ರತ್ಯಾಗಮ n_fuse;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_ತುದಿ_k(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncols = op->src[0]->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k     = op->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nrows = ggml_nrows(op->src[0]);

    ಧ್ರುವ ತಾರ್ಕಿಕ use_radix =
        ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));

    ಯದಿ (use_radix) {
        ggml_metal_op_ತುದಿ_k_radix(ctx, idx);
    } ಅನ್ಯಥಾ {
        ggml_metal_op_ತುದಿ_k_bitonic(ctx, idx);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_tri(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    ಪೂರ್ಣಾಙ್ಕ nth = 32; // SIMD width

    ಯಾವತ್ (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    ಧ್ರುವ int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    ಪೂರ್ಣಾಙ್ಕ ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    ಧ್ರುವ int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    ಧ್ರುವ int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    ಪೂರ್ಣಾಙ್ಕ ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, ಪರಿಮಾಣ(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    ಧ್ರುವ int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    ಪ್ರತ್ಯಾಗಮ 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_op_ಎಣಿಕೆ_equal(ggml_metal_op_t ctx, ಪೂರ್ಣಾಙ್ಕ idx) {
    ggml_tensor * op = ctx->ಗಂಟು(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_ಎಣಿಕೆ_equal args = {
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

        auto pipeline = ggml_metal_library_get_pipeline_ಎಣಿಕೆ_equal(lib, op);

        ಧ್ರುವ size_t smem = pipeline.smem;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, ಪರಿಮಾಣ(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    ಪ್ರತ್ಯಾಗಮ 1;
}
