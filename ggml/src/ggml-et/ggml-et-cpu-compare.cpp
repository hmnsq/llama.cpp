#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-cpu-compare.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu/ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu/ops.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

ਤਾਰਕਿਕ ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ਅਚਲ ggml_tensor * ਗੰਢ, ggml_op op) {
    ਜੇ (!ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Clear context
    memset(ctx, 0, ਆਕਾਰ(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    ਸਵੈਚਾਲਿਤ get_tensor_ਬਫਰ_ਮਾਪ = [](ਅਚਲ ggml_tensor * tensor) -> size_t {
        ਜੇ (!tensor) {
            ਵਾਪਸ 0;
        }

        ਜੇ (tensor->ਬਫਰ) {
            // Get actual backend buffer size
            size_t ਬਫਰ_ਮਾਪ = ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);

            // Use the full buffer size to avoid any truncation issues
            ਵਾਪਸ ਬਫਰ_ਮਾਪ;
        } ਨਹੀਂਤਾਂ {
            // Fallback to logical size if no buffer
            ਵਾਪਸ ggml_nbytes(tensor);
        }
    };

    ctx->src0_ਮਾਪ = get_tensor_ਬਫਰ_ਮਾਪ(ਗੰਢ->src[0]);
    ctx->src1_ਮਾਪ = get_tensor_ਬਫਰ_ਮਾਪ(ਗੰਢ->src[1]);
    ctx->src2_ਮਾਪ = get_tensor_ਬਫਰ_ਮਾਪ(ਗੰਢ->src[2]);
    ctx->dst_ਮਾਪ  = get_tensor_ਬਫਰ_ਮਾਪ(ਗੰਢ);

    // Allocate CPU buffers for all tensors
    ਜੇ (ctx->src0_ਮਾਪ > 0) {
        ctx->cpu_src0_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ctx->src0_ਮਾਪ);
        ਜੇ (!ctx->cpu_src0_ਅੰਕੜੇ) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            ਜਾਓ cleanup;
        }
    }

    ਜੇ (ctx->src1_ਮਾਪ > 0) {
        ctx->cpu_src1_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ctx->src1_ਮਾਪ);
        ਜੇ (!ctx->cpu_src1_ਅੰਕੜੇ) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            ਜਾਓ cleanup;
        }
    }

    ਜੇ (ctx->src2_ਮਾਪ > 0) {
        ctx->cpu_src2_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ctx->src2_ਮਾਪ);
        ਜੇ (!ctx->cpu_src2_ਅੰਕੜੇ) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            ਜਾਓ cleanup;
        }
    }

    ctx->cpu_dst_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ctx->dst_ਮਾਪ);
    ਜੇ (!ctx->cpu_dst_ਅੰਕੜੇ) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        ਜਾਓ cleanup;
    }

    ctx->et_dst_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ctx->dst_ਮਾਪ);
    ਜੇ (!ctx->et_dst_ਅੰਕੜੇ) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        ਜਾਓ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    ਜੇ (ctx->src0_ਮਾਪ > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_ਮਾਪ = ggml_nbytes(ਗੰਢ->src[0]);
        ggml_backend_tensor_get(ਗੰਢ->src[0], ctx->cpu_src0_ਅੰਕੜੇ, 0, logical_ਮਾਪ);
    }
    ਜੇ (ctx->src1_ਮਾਪ > 0) {
        size_t logical_ਮਾਪ = ggml_nbytes(ਗੰਢ->src[1]);
        ggml_backend_tensor_get(ਗੰਢ->src[1], ctx->cpu_src1_ਅੰਕੜੇ, 0, logical_ਮਾਪ);
    }
    ਜੇ (ctx->src2_ਮਾਪ > 0) {
        size_t logical_ਮਾਪ = ggml_nbytes(ਗੰਢ->src[2]);
        ggml_backend_tensor_get(ਗੰਢ->src[2], ctx->cpu_src2_ਅੰਕੜੇ, 0, logical_ਮਾਪ);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_ਮਾਪ = ggml_nbytes(ਗੰਢ);
        ggml_backend_tensor_get(ਗੰਢ, ctx->cpu_dst_ਅੰਕੜੇ, 0, logical_ਮਾਪ);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    ਜੇ (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        ਜਾਓ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_ਮਾਪ   = ggml_tensor_overhead() * 4 + ggml_ਗ੍ਰਾਫ_overhead();  // up to 4 tensors + graph
    ctx_params.mem_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    ctx_params.no_alloc   = ਸੱਚ;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    ਜੇ (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        ਜਾਓ cleanup;
    }

    // Create CPU tensors with proper context
    ਜੇ (ਗੰਢ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ਗੰਢ->src[0]->type, GGML_MAX_DIMS, ਗੰਢ->src[0]->ne);
        ਜੇ (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            ਜਾਓ cleanup;
        }
        ctx->cpu_src0->ਅੰਕੜੇ = ctx->cpu_src0_ਅੰਕੜੇ;
        // Copy stride array (nb) for correct memory layout
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_src0->nb, ਗੰਢ->src[0]->nb, ਆਕਾਰ(ਗੰਢ->src[0]->nb));
        // Copy op_params if present
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_src0->op_params, ਗੰਢ->src[0]->op_params, ਆਕਾਰ(ਗੰਢ->src[0]->op_params));
    }

    ਜੇ (ਗੰਢ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ਗੰਢ->src[1]->type, GGML_MAX_DIMS, ਗੰਢ->src[1]->ne);
        ਜੇ (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            ਜਾਓ cleanup;
        }
        ctx->cpu_src1->ਅੰਕੜੇ = ctx->cpu_src1_ਅੰਕੜੇ;
        // Copy stride array (nb) for correct memory layout
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_src1->nb, ਗੰਢ->src[1]->nb, ਆਕਾਰ(ਗੰਢ->src[1]->nb));
        // Copy op_params if present
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_src1->op_params, ਗੰਢ->src[1]->op_params, ਆਕਾਰ(ਗੰਢ->src[1]->op_params));
    }

    ਜੇ (ਗੰਢ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ਗੰਢ->src[2]->type, GGML_MAX_DIMS, ਗੰਢ->src[2]->ne);
        ਜੇ (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            ਜਾਓ cleanup;
        }
        ctx->cpu_src2->ਅੰਕੜੇ = ctx->cpu_src2_ਅੰਕੜੇ;
        // Copy stride array (nb) for correct memory layout
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_src2->nb, ਗੰਢ->src[2]->nb, ਆਕਾਰ(ਗੰਢ->src[2]->nb));
        // Copy op_params if present
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_src2->op_params, ਗੰਢ->src[2]->op_params, ਆਕਾਰ(ਗੰਢ->src[2]->op_params));
    }

    ਵਾਪਸ ਸੱਚ;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    ਵਾਪਸ ਝੂਠ;
}

ਤਾਰਕਿਕ ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ਅਚਲ ggml_tensor *                ਗੰਢ,
                                           ਅਚਲ ggml_et_cpu_compare_config * config) {
    ਜੇ (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ਗੰਢ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ਗੰਢ->op;
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
            ctx->cpu_dst = ggml_ਜੋੜੋ(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
            {
                ਅਚਲ int32_t * op_params  = (ਅਚਲ int32_t *) ਗੰਢ->op_params;
                ਅਚਲ int32_t   n_dims     = op_params[1];
                ਅਚਲ int32_t   mode       = op_params[2];
                ਅਚਲ int32_t   n_ctx_orig = op_params[4];
                ਅਚਲ ਦਸ਼ਮਲਵ     freq_ਅਧਾਰ  = *((ਅਚਲ ਦਸ਼ਮਲਵ *) (op_params + 5));
                ਅਚਲ ਦਸ਼ਮਲਵ     freq_scale = *((ਅਚਲ ਦਸ਼ਮਲਵ *) (op_params + 6));
                ਅਚਲ ਦਸ਼ਮਲਵ     ext_ਗੁਣਕ = *((ਅਚਲ ਦਸ਼ਮਲਵ *) (op_params + 7));
                ਅਚਲ ਦਸ਼ਮਲਵ     attn_ਗੁਣਕ = *((ਅਚਲ ਦਸ਼ਮਲਵ *) (op_params + 8));
                ਅਚਲ ਦਸ਼ਮਲਵ     beta_fast  = *((ਅਚਲ ਦਸ਼ਮਲਵ *) (op_params + 9));
                ਅਚਲ ਦਸ਼ਮਲਵ     beta_slow  = *((ਅਚਲ ਦਸ਼ਮਲਵ *) (op_params + 10));

                ਜੇ (mode & GGML_ROPE_TYPE_MROPE) {
                    ਪੂਰਨ sections[GGML_MROPE_SECTIONS];
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(sections, op_params + 11, ਆਕਾਰ(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
                                                   ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
                } ਨਹੀਂਤਾਂ {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ,
                                                 attn_ਗੁਣਕ, beta_fast, beta_slow);
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                ਦਸ਼ਮਲਵ eps;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ਗੰਢ, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ctx->cpu_dst = ggml_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx->ggml_ctx, ctx->cpu_src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            ctx->cpu_dst = ggml_ਔਸਤ(ctx->ggml_ctx, ctx->cpu_src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            {
                ਦਸ਼ਮਲਵ clamp_min, clamp_max;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&clamp_min, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&clamp_max, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ਗੰਢ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                ਜੇ (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ggml_get_op_params_f32(ਗੰਢ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ਹੱਦਬੰਦੀ);
                } ਨਹੀਂਤਾਂ {
                    ctx->cpu_dst = ggml_glu_ਵੰਡੋ(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                ਦਸ਼ਮਲਵ scale    = 1.0f;
                ਦਸ਼ਮਲਵ max_bias = 0.0f;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ਜੇ (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } ਨਹੀਂਤਾਂ {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                ਜੇ (ctx->cpu_src2) {
                    ggml_soft_max_ਜੋੜੋ_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ctx->cpu_dst = ggml_get_ਪੰਕਤੀਆਂ(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_ਅਧਾਰ = ggml_new_tensor(ctx->ggml_ctx, ਗੰਢ->type, GGML_MAX_DIMS, ਗੰਢ->ne);
                ਜੇ (!cpu_dst_ਅਧਾਰ) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    ਵਾਪਸ ਝੂਠ;
                }
                cpu_dst_ਅਧਾਰ->ਅੰਕੜੇ = ctx->cpu_dst_ਅੰਕੜੇ;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpu_dst_ਅਧਾਰ->nb, ਗੰਢ->nb, ਆਕਾਰ(ਗੰਢ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx->ggml_ctx, cpu_dst_ਅਧਾਰ, ctx->cpu_src0, ctx->cpu_src1);
            }
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        ਵਾਪਸ ਝੂਠ;
    }

    ctx->cpu_dst->ਅੰਕੜੇ = ctx->cpu_dst_ਅੰਕੜੇ;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    ਜੇ (op != GGML_OP_CONT) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ctx->cpu_dst->nb, ਗੰਢ->nb, ਆਕਾਰ(ਗੰਢ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_ਗ੍ਰਾਫ = ggml_new_ਗ੍ਰਾਫ_custom(ctx->ggml_ctx, 1, ਝੂਠ);
    ਜੇ (!ctx->cpu_ਗ੍ਰਾਫ) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        ਵਾਪਸ ਝੂਠ;
    }
    ctx->cpu_ਗ੍ਰਾਫ->nodes[0] = ctx->cpu_dst;
    ctx->cpu_ਗ੍ਰਾਫ->n_nodes  = 1;

    // Log input data for debugging if enabled
    ਜੇ (config && config->log_differences) {
        ਜੇ (ctx->cpu_src0_ਅੰਕੜੇ && ctx->src0_ਮਾਪ >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_ਅੰਕੜੇ)[0],
                           ((uint8_t *) ctx->cpu_src0_ਅੰਕੜੇ)[1], ((uint8_t *) ctx->cpu_src0_ਅੰਕੜੇ)[2],
                           ((uint8_t *) ctx->cpu_src0_ਅੰਕੜੇ)[3]);
        }
        ਜੇ (ctx->cpu_src1_ਅੰਕੜੇ && ctx->src1_ਮਾਪ >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((ਦਸ਼ਮਲਵ *) ctx->cpu_src1_ਅੰਕੜੇ)[0],
                           ((ਦਸ਼ਮਲਵ *) ctx->cpu_src1_ਅੰਕੜੇ)[1], ((ਦਸ਼ਮਲਵ *) ctx->cpu_src1_ਅੰਕੜੇ)[2],
                           ((ਦਸ਼ਮਲਵ *) ctx->cpu_src1_ਅੰਕੜੇ)[3]);
        }
    }

    // Compute using CPU backend
    ggml_ਸਥਿਤੀ cpu_ਨਤੀਜਾ = ggml_backend_ਗ੍ਰਾਫ_compute(ctx->cpu_backend, ctx->cpu_ਗ੍ਰਾਫ);

    ਜੇ (cpu_ਨਤੀਜਾ != GGML_ਸਥਿਤੀ_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_ਨਤੀਜਾ);
        ਵਾਪਸ ਝੂਠ;
    }

    // Log output data for debugging if enabled
    ਜੇ (config && config->log_differences && ctx->dst_ਮਾਪ >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((ਦਸ਼ਮਲਵ *) ctx->cpu_dst_ਅੰਕੜੇ)[0], ((ਦਸ਼ਮਲਵ *) ctx->cpu_dst_ਅੰਕੜੇ)[1],
                       ((ਦਸ਼ਮਲਵ *) ctx->cpu_dst_ਅੰਕੜੇ)[2], ((ਦਸ਼ਮਲਵ *) ctx->cpu_dst_ਅੰਕੜੇ)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_ਮਾਪ = ggml_nbytes(ਗੰਢ);
    ggml_backend_tensor_get(ਗੰਢ, ctx->et_dst_ਅੰਕੜੇ, 0, dst_logical_ਮਾਪ);

    ਜੇ (config->log_differences) {
        size_t ਅੰਕ_elements = ggml_nelements(ਗੰਢ);
        size_t max_log      = std::min(ਅੰਕ_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        ਤਾਰਕਿਕ    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ਜੋੜੋ || op == GGML_OP_GLU);
        ਦਸ਼ਮਲਵ * cpu_src0_float = is_elementwise ? (ਦਸ਼ਮਲਵ *) ctx->cpu_src0_ਅੰਕੜੇ : ਖਾਲੀ_ਸੂਚਕ;
        ਦਸ਼ਮਲਵ * cpu_src1_float = is_elementwise ? (ਦਸ਼ਮਲਵ *) ctx->cpu_src1_ਅੰਕੜੇ : ਖਾਲੀ_ਸੂਚਕ;

        // Helper to get float value from tensor data (handles f16 and f32)
        ਸਵੈਚਾਲਿਤ get_float = [](ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t idx, ggml_type type) -> ਦਸ਼ਮਲਵ {
            ਜੇ (type == GGML_TYPE_F16) {
                ਅਚਲ ggml_fp16_t * fp16_ਅੰਕੜੇ = (ਅਚਲ ggml_fp16_t *) ਅੰਕੜੇ;
                ਵਾਪਸ ggml_fp16_ਮੰਜ਼ਿਲ_fp32(fp16_ਅੰਕੜੇ[idx]);
            }

            ਅਚਲ ਦਸ਼ਮਲਵ * float_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ਅੰਕੜੇ;
            ਵਾਪਸ float_ਅੰਕੜੇ[idx];
        };

        // Compare all elements but log only the first max_log_elements
        ਤਾਰਕਿਕ   matches          = ਸੱਚ;
        size_t ਕੁੱਲ_mismatches = 0;

        // First pass: check all elements for mismatches
        ਲਈ (size_t i = 0; i < ਅੰਕ_elements; i++) {
            ਦਸ਼ਮਲਵ cpu_val  = get_float(ctx->cpu_dst_ਅੰਕੜੇ, i, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ et_val   = get_float(ctx->et_dst_ਅੰਕੜੇ, i, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ diff     = fabsf(cpu_val - et_val);
            ਦਸ਼ਮਲਵ rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            ਜੇ (rel_diff > config->tolerance) {
                matches = ਝੂਠ;
                ਕੁੱਲ_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        ਲਈ (size_t i = 0; i < max_log; i++) {
            ਦਸ਼ਮਲਵ cpu_val = get_float(ctx->cpu_dst_ਅੰਕੜੇ, i, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ et_val  = get_float(ctx->et_dst_ਅੰਕੜੇ, i, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ diff    = fabsf(cpu_val - et_val);

            ਜੇ (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } ਨਹੀਂਤਾਂ ਜੇ (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } ਨਹੀਂਤਾਂ {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        ਜੇ (ਅੰਕ_elements > max_log) {
            size_t mid     = ਅੰਕ_elements / 2;
            size_t end     = ਅੰਕ_elements - 1;
            ਦਸ਼ਮਲਵ  cpu_mid = get_float(ctx->cpu_dst_ਅੰਕੜੇ, mid, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ  et_mid  = get_float(ctx->et_dst_ਅੰਕੜੇ, mid, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ  cpu_ਸਿਰਾ = get_float(ctx->cpu_dst_ਅੰਕੜੇ, end, ਗੰਢ->type);
            ਦਸ਼ਮਲਵ  et_ਸਿਰਾ  = get_float(ctx->et_dst_ਅੰਕੜੇ, end, ਗੰਢ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_ਸਿਰਾ, et_ਸਿਰਾ);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       ਅੰਕ_elements - ਕੁੱਲ_mismatches, ਅੰਕ_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    ਜੇ (config->use_cpu_ਨਤੀਜਾ) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_ਮਾਪ = ggml_nbytes(ਗੰਢ);
        ggml_backend_tensor_ਸੈੱਟ(const_cast<ggml_tensor *>(ਗੰਢ), ctx->cpu_dst_ਅੰਕੜੇ, 0, dst_logical_ਮਾਪ);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    ਜੇ (!ctx) {
        ਵਾਪਸ;
    }

    ਜੇ (ctx->cpu_src0_ਅੰਕੜੇ) {
        ਮੁਕਤ(ctx->cpu_src0_ਅੰਕੜੇ);
        ctx->cpu_src0_ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ctx->cpu_src1_ਅੰਕੜੇ) {
        ਮੁਕਤ(ctx->cpu_src1_ਅੰਕੜੇ);
        ctx->cpu_src1_ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ctx->cpu_src2_ਅੰਕੜੇ) {
        ਮੁਕਤ(ctx->cpu_src2_ਅੰਕੜੇ);
        ctx->cpu_src2_ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ctx->cpu_dst_ਅੰਕੜੇ) {
        ਮੁਕਤ(ctx->cpu_dst_ਅੰਕੜੇ);
        ctx->cpu_dst_ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ctx->et_dst_ਅੰਕੜੇ) {
        ਮੁਕਤ(ctx->et_dst_ਅੰਕੜੇ);
        ctx->et_dst_ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = ਖਾਲੀ_ਸੂਚਕ;
    }

    // Clear pointers
    ctx->cpu_src0  = ਖਾਲੀ_ਸੂਚਕ;
    ctx->cpu_src1  = ਖਾਲੀ_ਸੂਚਕ;
    ctx->cpu_src2  = ਖਾਲੀ_ਸੂਚਕ;
    ctx->cpu_dst   = ਖਾਲੀ_ਸੂਚਕ;
    ctx->cpu_ਗ੍ਰਾਫ = ਖਾਲੀ_ਸੂਚਕ;
}
