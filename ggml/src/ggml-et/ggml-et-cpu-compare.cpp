#ಅನ್ತರ್ಭಾವ "ggml-et-cpu-compare.h"

#ಅನ್ತರ್ಭಾವ "ggml-cpu/ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu/ops.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>

ತಾರ್ಕಿಕ ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು, ggml_op op) {
    ಆದರೆ (!ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Clear context
    memset(ctx, 0, ಗಾತ್ರ(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    ಸ್ವಯಂಚಾಲಿತ get_tensor_ಬಫರ್_ಗಾತ್ರ = [](ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) -> size_t {
        ಆದರೆ (!tensor) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಆದರೆ (tensor->ಬಫರ್) {
            // Get actual backend buffer size
            size_t ಬಫರ್_ಗಾತ್ರ = ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);

            // Use the full buffer size to avoid any truncation issues
            ಹಿಂತಿರುಗಿ ಬಫರ್_ಗಾತ್ರ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Fallback to logical size if no buffer
            ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
        }
    };

    ctx->src0_ಗಾತ್ರ = get_tensor_ಬಫರ್_ಗಾತ್ರ(ಗಂಟು->src[0]);
    ctx->src1_ಗಾತ್ರ = get_tensor_ಬಫರ್_ಗಾತ್ರ(ಗಂಟು->src[1]);
    ctx->src2_ಗಾತ್ರ = get_tensor_ಬಫರ್_ಗಾತ್ರ(ಗಂಟು->src[2]);
    ctx->dst_ಗಾತ್ರ  = get_tensor_ಬಫರ್_ಗಾತ್ರ(ಗಂಟು);

    // Allocate CPU buffers for all tensors
    ಆದರೆ (ctx->src0_ಗಾತ್ರ > 0) {
        ctx->cpu_src0_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(ctx->src0_ಗಾತ್ರ);
        ಆದರೆ (!ctx->cpu_src0_ದತ್ತಾಂಶ) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            ಹೋಗು cleanup;
        }
    }

    ಆದರೆ (ctx->src1_ಗಾತ್ರ > 0) {
        ctx->cpu_src1_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(ctx->src1_ಗಾತ್ರ);
        ಆದರೆ (!ctx->cpu_src1_ದತ್ತಾಂಶ) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            ಹೋಗು cleanup;
        }
    }

    ಆದರೆ (ctx->src2_ಗಾತ್ರ > 0) {
        ctx->cpu_src2_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(ctx->src2_ಗಾತ್ರ);
        ಆದರೆ (!ctx->cpu_src2_ದತ್ತಾಂಶ) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            ಹೋಗು cleanup;
        }
    }

    ctx->cpu_dst_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(ctx->dst_ಗಾತ್ರ);
    ಆದರೆ (!ctx->cpu_dst_ದತ್ತಾಂಶ) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        ಹೋಗು cleanup;
    }

    ctx->et_dst_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(ctx->dst_ಗಾತ್ರ);
    ಆದರೆ (!ctx->et_dst_ದತ್ತಾಂಶ) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        ಹೋಗು cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    ಆದರೆ (ctx->src0_ಗಾತ್ರ > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು->src[0]);
        ggml_backend_tensor_get(ಗಂಟು->src[0], ctx->cpu_src0_ದತ್ತಾಂಶ, 0, logical_ಗಾತ್ರ);
    }
    ಆದರೆ (ctx->src1_ಗಾತ್ರ > 0) {
        size_t logical_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು->src[1]);
        ggml_backend_tensor_get(ಗಂಟು->src[1], ctx->cpu_src1_ದತ್ತಾಂಶ, 0, logical_ಗಾತ್ರ);
    }
    ಆದರೆ (ctx->src2_ಗಾತ್ರ > 0) {
        size_t logical_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು->src[2]);
        ggml_backend_tensor_get(ಗಂಟು->src[2], ctx->cpu_src2_ದತ್ತಾಂಶ, 0, logical_ಗಾತ್ರ);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು);
        ggml_backend_tensor_get(ಗಂಟು, ctx->cpu_dst_ದತ್ತಾಂಶ, 0, logical_ಗಾತ್ರ);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    ಆದರೆ (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        ಹೋಗು cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_ಗಾತ್ರ   = ggml_tensor_overhead() * 4 + ggml_ನಕ್ಷೆ_overhead();  // up to 4 tensors + graph
    ctx_params.mem_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    ctx_params.no_alloc   = ನಿಜ;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    ಆದರೆ (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        ಹೋಗು cleanup;
    }

    // Create CPU tensors with proper context
    ಆದರೆ (ಗಂಟು->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->src[0]->type, GGML_MAX_DIMS, ಗಂಟು->src[0]->ne);
        ಆದರೆ (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            ಹೋಗು cleanup;
        }
        ctx->cpu_src0->ದತ್ತಾಂಶ = ctx->cpu_src0_ದತ್ತಾಂಶ;
        // Copy stride array (nb) for correct memory layout
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_src0->nb, ಗಂಟು->src[0]->nb, ಗಾತ್ರ(ಗಂಟು->src[0]->nb));
        // Copy op_params if present
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_src0->op_params, ಗಂಟು->src[0]->op_params, ಗಾತ್ರ(ಗಂಟು->src[0]->op_params));
    }

    ಆದರೆ (ಗಂಟು->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->src[1]->type, GGML_MAX_DIMS, ಗಂಟು->src[1]->ne);
        ಆದರೆ (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            ಹೋಗು cleanup;
        }
        ctx->cpu_src1->ದತ್ತಾಂಶ = ctx->cpu_src1_ದತ್ತಾಂಶ;
        // Copy stride array (nb) for correct memory layout
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_src1->nb, ಗಂಟು->src[1]->nb, ಗಾತ್ರ(ಗಂಟು->src[1]->nb));
        // Copy op_params if present
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_src1->op_params, ಗಂಟು->src[1]->op_params, ಗಾತ್ರ(ಗಂಟು->src[1]->op_params));
    }

    ಆದರೆ (ಗಂಟು->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->src[2]->type, GGML_MAX_DIMS, ಗಂಟು->src[2]->ne);
        ಆದರೆ (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            ಹೋಗು cleanup;
        }
        ctx->cpu_src2->ದತ್ತಾಂಶ = ctx->cpu_src2_ದತ್ತಾಂಶ;
        // Copy stride array (nb) for correct memory layout
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_src2->nb, ಗಂಟು->src[2]->nb, ಗಾತ್ರ(ಗಂಟು->src[2]->nb));
        // Copy op_params if present
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_src2->op_params, ಗಂಟು->src[2]->op_params, ಗಾತ್ರ(ಗಂಟು->src[2]->op_params));
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ತಾರ್ಕಿಕ ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ಸ್ಥಿರಾಂಕ ggml_tensor *                ಗಂಟು,
                                           ಸ್ಥಿರಾಂಕ ggml_et_cpu_compare_config * config) {
    ಆದರೆ (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ಗಂಟು || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ಗಂಟು->op;
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
            ctx->cpu_dst = ggml_ಕೂಡಿಸು(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            {
                ಸ್ಥಿರಾಂಕ int32_t * op_params  = (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params;
                ಸ್ಥಿರಾಂಕ int32_t   n_dims     = op_params[1];
                ಸ್ಥಿರಾಂಕ int32_t   mode       = op_params[2];
                ಸ್ಥಿರಾಂಕ int32_t   n_ctx_orig = op_params[4];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     freq_ತಳ  = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op_params + 5));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     freq_scale = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op_params + 6));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     ext_ಗುಣಕ = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op_params + 7));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     attn_ಗುಣಕ = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op_params + 8));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     beta_fast  = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op_params + 9));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     beta_slow  = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op_params + 10));

                ಆದರೆ (mode & GGML_ROPE_TYPE_MROPE) {
                    ಪೂರ್ಣಾಂಕ sections[GGML_MROPE_SECTIONS];
                    ಸ್ಮರಣೆ_ನಕಲು(sections, op_params + 11, ಗಾತ್ರ(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_ತಳ, freq_scale,
                                                   ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_ತಳ, freq_scale, ext_ಗುಣಕ,
                                                 attn_ಗುಣಕ, beta_fast, beta_slow);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                ದಶಮಾಂಶ eps;
                ಸ್ಮರಣೆ_ನಕಲು(&eps, ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ಗಂಟು, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ctx->cpu_dst = ggml_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ctx->ggml_ctx, ctx->cpu_src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            ctx->cpu_dst = ggml_ಸರಾಸರಿ(ctx->ggml_ctx, ctx->cpu_src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            {
                ದಶಮಾಂಶ clamp_min, clamp_max;
                ಸ್ಮರಣೆ_ನಕಲು(&clamp_min, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&clamp_max, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ಗಂಟು, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                ಆದರೆ (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = ggml_get_op_params_f32(ಗಂಟು, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ಎಲ್ಲೆ);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ctx->cpu_dst = ggml_glu_ವಿಭಜಿಸು(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                ದಶಮಾಂಶ scale    = 1.0f;
                ದಶಮಾಂಶ max_bias = 0.0f;
                ಸ್ಮರಣೆ_ನಕಲು(&scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

                ಆದರೆ (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                ಆದರೆ (ctx->cpu_src2) {
                    ggml_soft_max_ಕೂಡಿಸು_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ctx->cpu_dst = ggml_get_ಅಡ್ಡಸಾಲುಗಳು(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_ತಳ = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->type, GGML_MAX_DIMS, ಗಂಟು->ne);
                ಆದರೆ (!cpu_dst_ತಳ) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                cpu_dst_ತಳ->ದತ್ತಾಂಶ = ctx->cpu_dst_ದತ್ತಾಂಶ;
                ಸ್ಮರಣೆ_ನಕಲು(cpu_dst_ತಳ->nb, ಗಂಟು->nb, ಗಾತ್ರ(ಗಂಟು->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx->ggml_ctx, cpu_dst_ತಳ, ctx->cpu_src0, ctx->cpu_src1);
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ctx->cpu_dst->ದತ್ತಾಂಶ = ctx->cpu_dst_ದತ್ತಾಂಶ;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    ಆದರೆ (op != GGML_OP_CONT) {
        ಸ್ಮರಣೆ_ನಕಲು(ctx->cpu_dst->nb, ಗಂಟು->nb, ಗಾತ್ರ(ಗಂಟು->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_ನಕ್ಷೆ = ggml_new_ನಕ್ಷೆ_custom(ctx->ggml_ctx, 1, ಸುಳ್ಳು);
    ಆದರೆ (!ctx->cpu_ನಕ್ಷೆ) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ctx->cpu_ನಕ್ಷೆ->nodes[0] = ctx->cpu_dst;
    ctx->cpu_ನಕ್ಷೆ->n_nodes  = 1;

    // Log input data for debugging if enabled
    ಆದರೆ (config && config->log_differences) {
        ಆದರೆ (ctx->cpu_src0_ದತ್ತಾಂಶ && ctx->src0_ಗಾತ್ರ >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_ದತ್ತಾಂಶ)[0],
                           ((uint8_t *) ctx->cpu_src0_ದತ್ತಾಂಶ)[1], ((uint8_t *) ctx->cpu_src0_ದತ್ತಾಂಶ)[2],
                           ((uint8_t *) ctx->cpu_src0_ದತ್ತಾಂಶ)[3]);
        }
        ಆದರೆ (ctx->cpu_src1_ದತ್ತಾಂಶ && ctx->src1_ಗಾತ್ರ >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((ದಶಮಾಂಶ *) ctx->cpu_src1_ದತ್ತಾಂಶ)[0],
                           ((ದಶಮಾಂಶ *) ctx->cpu_src1_ದತ್ತಾಂಶ)[1], ((ದಶಮಾಂಶ *) ctx->cpu_src1_ದತ್ತಾಂಶ)[2],
                           ((ದಶಮಾಂಶ *) ctx->cpu_src1_ದತ್ತಾಂಶ)[3]);
        }
    }

    // Compute using CPU backend
    ggml_ಸ್ಥಿತಿ cpu_ಫಲಿತಾಂಶ = ggml_backend_ನಕ್ಷೆ_compute(ctx->cpu_backend, ctx->cpu_ನಕ್ಷೆ);

    ಆದರೆ (cpu_ಫಲಿತಾಂಶ != GGML_ಸ್ಥಿತಿ_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_ಫಲಿತಾಂಶ);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Log output data for debugging if enabled
    ಆದರೆ (config && config->log_differences && ctx->dst_ಗಾತ್ರ >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((ದಶಮಾಂಶ *) ctx->cpu_dst_ದತ್ತಾಂಶ)[0], ((ದಶಮಾಂಶ *) ctx->cpu_dst_ದತ್ತಾಂಶ)[1],
                       ((ದಶಮಾಂಶ *) ctx->cpu_dst_ದತ್ತಾಂಶ)[2], ((ದಶಮಾಂಶ *) ctx->cpu_dst_ದತ್ತಾಂಶ)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು);
    ggml_backend_tensor_get(ಗಂಟು, ctx->et_dst_ದತ್ತಾಂಶ, 0, dst_logical_ಗಾತ್ರ);

    ಆದರೆ (config->log_differences) {
        size_t ಅಂಕೆ_elements = ggml_nelements(ಗಂಟು);
        size_t max_log      = std::min(ಅಂಕೆ_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        ತಾರ್ಕಿಕ    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ಕೂಡಿಸು || op == GGML_OP_GLU);
        ದಶಮಾಂಶ * cpu_src0_float = is_elementwise ? (ದಶಮಾಂಶ *) ctx->cpu_src0_ದತ್ತಾಂಶ : ಶೂನ್ಯಸೂಚಕ;
        ದಶಮಾಂಶ * cpu_src1_float = is_elementwise ? (ದಶಮಾಂಶ *) ctx->cpu_src1_ದತ್ತಾಂಶ : ಶೂನ್ಯಸೂಚಕ;

        // Helper to get float value from tensor data (handles f16 and f32)
        ಸ್ವಯಂಚಾಲಿತ get_float = [](ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t idx, ggml_type type) -> ದಶಮಾಂಶ {
            ಆದರೆ (type == GGML_TYPE_F16) {
                ಸ್ಥಿರಾಂಕ ggml_fp16_t * fp16_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ದತ್ತಾಂಶ;
                ಹಿಂತಿರುಗಿ ggml_fp16_ಗಮ್ಯ_fp32(fp16_ದತ್ತಾಂಶ[idx]);
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * float_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ದತ್ತಾಂಶ;
            ಹಿಂತಿರುಗಿ float_ದತ್ತಾಂಶ[idx];
        };

        // Compare all elements but log only the first max_log_elements
        ತಾರ್ಕಿಕ   matches          = ನಿಜ;
        size_t ಒಟ್ಟು_mismatches = 0;

        // First pass: check all elements for mismatches
        ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_elements; i++) {
            ದಶಮಾಂಶ cpu_val  = get_float(ctx->cpu_dst_ದತ್ತಾಂಶ, i, ಗಂಟು->type);
            ದಶಮಾಂಶ et_val   = get_float(ctx->et_dst_ದತ್ತಾಂಶ, i, ಗಂಟು->type);
            ದಶಮಾಂಶ diff     = fabsf(cpu_val - et_val);
            ದಶಮಾಂಶ rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            ಆದರೆ (rel_diff > config->tolerance) {
                matches = ಸುಳ್ಳು;
                ಒಟ್ಟು_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        ಗಾಗಿ (size_t i = 0; i < max_log; i++) {
            ದಶಮಾಂಶ cpu_val = get_float(ctx->cpu_dst_ದತ್ತಾಂಶ, i, ಗಂಟು->type);
            ದಶಮಾಂಶ et_val  = get_float(ctx->et_dst_ದತ್ತಾಂಶ, i, ಗಂಟು->type);
            ದಶಮಾಂಶ diff    = fabsf(cpu_val - et_val);

            ಆದರೆ (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        ಆದರೆ (ಅಂಕೆ_elements > max_log) {
            size_t mid     = ಅಂಕೆ_elements / 2;
            size_t end     = ಅಂಕೆ_elements - 1;
            ದಶಮಾಂಶ  cpu_mid = get_float(ctx->cpu_dst_ದತ್ತಾಂಶ, mid, ಗಂಟು->type);
            ದಶಮಾಂಶ  et_mid  = get_float(ctx->et_dst_ದತ್ತಾಂಶ, mid, ಗಂಟು->type);
            ದಶಮಾಂಶ  cpu_ಕೊನೆ = get_float(ctx->cpu_dst_ದತ್ತಾಂಶ, end, ಗಂಟು->type);
            ದಶಮಾಂಶ  et_ಕೊನೆ  = get_float(ctx->et_dst_ದತ್ತಾಂಶ, end, ಗಂಟು->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_ಕೊನೆ, et_ಕೊನೆ);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       ಅಂಕೆ_elements - ಒಟ್ಟು_mismatches, ಅಂಕೆ_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    ಆದರೆ (config->use_cpu_ಫಲಿತಾಂಶ) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು);
        ggml_backend_tensor_ಗಣ(const_cast<ggml_tensor *>(ಗಂಟು), ctx->cpu_dst_ದತ್ತಾಂಶ, 0, dst_logical_ಗಾತ್ರ);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    ಆದರೆ (!ctx) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ctx->cpu_src0_ದತ್ತಾಂಶ) {
        ಬಿಡುಗಡೆ(ctx->cpu_src0_ದತ್ತಾಂಶ);
        ctx->cpu_src0_ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ctx->cpu_src1_ದತ್ತಾಂಶ) {
        ಬಿಡುಗಡೆ(ctx->cpu_src1_ದತ್ತಾಂಶ);
        ctx->cpu_src1_ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ctx->cpu_src2_ದತ್ತಾಂಶ) {
        ಬಿಡುಗಡೆ(ctx->cpu_src2_ದತ್ತಾಂಶ);
        ctx->cpu_src2_ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ctx->cpu_dst_ದತ್ತಾಂಶ) {
        ಬಿಡುಗಡೆ(ctx->cpu_dst_ದತ್ತಾಂಶ);
        ctx->cpu_dst_ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ctx->et_dst_ದತ್ತಾಂಶ) {
        ಬಿಡುಗಡೆ(ctx->et_dst_ದತ್ತಾಂಶ);
        ctx->et_dst_ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = ಶೂನ್ಯಸೂಚಕ;
    }

    // Clear pointers
    ctx->cpu_src0  = ಶೂನ್ಯಸೂಚಕ;
    ctx->cpu_src1  = ಶೂನ್ಯಸೂಚಕ;
    ctx->cpu_src2  = ಶೂನ್ಯಸೂಚಕ;
    ctx->cpu_dst   = ಶೂನ್ಯಸೂಚಕ;
    ctx->cpu_ನಕ್ಷೆ = ಶೂನ್ಯಸೂಚಕ;
}
