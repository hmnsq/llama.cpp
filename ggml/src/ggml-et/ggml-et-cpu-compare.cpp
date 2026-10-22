#అన్తర్భావ "ggml-et-cpu-compare.h"

#అన్తర్భావ "ggml-cpu/ggml-cpu-impl.h"
#అన్తర్భావ "ggml-cpu/ops.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>

తార్కిక ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, స్థిరరాశి ggml_tensor * కణుపు, ggml_op op) {
    ఒకవేళ (!ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        తిరిగి అబద్ధం;
    }

    // Clear context
    memset(ctx, 0, పరిమాణం(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    స్వయంచాలక get_tensor_బఫరు_కొలత = [](స్థిరరాశి ggml_tensor * tensor) -> size_t {
        ఒకవేళ (!tensor) {
            తిరిగి 0;
        }

        ఒకవేళ (tensor->బఫరు) {
            // Get actual backend buffer size
            size_t బఫరు_కొలత = ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);

            // Use the full buffer size to avoid any truncation issues
            తిరిగి బఫరు_కొలత;
        } లేకపోతే {
            // Fallback to logical size if no buffer
            తిరిగి ggml_nbytes(tensor);
        }
    };

    ctx->src0_కొలత = get_tensor_బఫరు_కొలత(కణుపు->src[0]);
    ctx->src1_కొలత = get_tensor_బఫరు_కొలత(కణుపు->src[1]);
    ctx->src2_కొలత = get_tensor_బఫరు_కొలత(కణుపు->src[2]);
    ctx->dst_కొలత  = get_tensor_బఫరు_కొలత(కణుపు);

    // Allocate CPU buffers for all tensors
    ఒకవేళ (ctx->src0_కొలత > 0) {
        ctx->cpu_src0_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(ctx->src0_కొలత);
        ఒకవేళ (!ctx->cpu_src0_దత్తాంశం) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            వెళ్ళు cleanup;
        }
    }

    ఒకవేళ (ctx->src1_కొలత > 0) {
        ctx->cpu_src1_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(ctx->src1_కొలత);
        ఒకవేళ (!ctx->cpu_src1_దత్తాంశం) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            వెళ్ళు cleanup;
        }
    }

    ఒకవేళ (ctx->src2_కొలత > 0) {
        ctx->cpu_src2_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(ctx->src2_కొలత);
        ఒకవేళ (!ctx->cpu_src2_దత్తాంశం) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            వెళ్ళు cleanup;
        }
    }

    ctx->cpu_dst_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(ctx->dst_కొలత);
    ఒకవేళ (!ctx->cpu_dst_దత్తాంశం) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        వెళ్ళు cleanup;
    }

    ctx->et_dst_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(ctx->dst_కొలత);
    ఒకవేళ (!ctx->et_dst_దత్తాంశం) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        వెళ్ళు cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    ఒకవేళ (ctx->src0_కొలత > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_కొలత = ggml_nbytes(కణుపు->src[0]);
        ggml_backend_tensor_get(కణుపు->src[0], ctx->cpu_src0_దత్తాంశం, 0, logical_కొలత);
    }
    ఒకవేళ (ctx->src1_కొలత > 0) {
        size_t logical_కొలత = ggml_nbytes(కణుపు->src[1]);
        ggml_backend_tensor_get(కణుపు->src[1], ctx->cpu_src1_దత్తాంశం, 0, logical_కొలత);
    }
    ఒకవేళ (ctx->src2_కొలత > 0) {
        size_t logical_కొలత = ggml_nbytes(కణుపు->src[2]);
        ggml_backend_tensor_get(కణుపు->src[2], ctx->cpu_src2_దత్తాంశం, 0, logical_కొలత);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_కొలత = ggml_nbytes(కణుపు);
        ggml_backend_tensor_get(కణుపు, ctx->cpu_dst_దత్తాంశం, 0, logical_కొలత);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    ఒకవేళ (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        వెళ్ళు cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_కొలత   = ggml_tensor_overhead() * 4 + ggml_లేఖాచిత్రం_overhead();  // up to 4 tensors + graph
    ctx_params.mem_బఫరు = శూన్యసూచిక;
    ctx_params.no_alloc   = నిజం;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    ఒకవేళ (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        వెళ్ళు cleanup;
    }

    // Create CPU tensors with proper context
    ఒకవేళ (కణుపు->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, కణుపు->src[0]->type, GGML_MAX_DIMS, కణుపు->src[0]->ne);
        ఒకవేళ (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            వెళ్ళు cleanup;
        }
        ctx->cpu_src0->దత్తాంశం = ctx->cpu_src0_దత్తాంశం;
        // Copy stride array (nb) for correct memory layout
        జ్ఞాపకం_నకలు(ctx->cpu_src0->nb, కణుపు->src[0]->nb, పరిమాణం(కణుపు->src[0]->nb));
        // Copy op_params if present
        జ్ఞాపకం_నకలు(ctx->cpu_src0->op_params, కణుపు->src[0]->op_params, పరిమాణం(కణుపు->src[0]->op_params));
    }

    ఒకవేళ (కణుపు->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, కణుపు->src[1]->type, GGML_MAX_DIMS, కణుపు->src[1]->ne);
        ఒకవేళ (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            వెళ్ళు cleanup;
        }
        ctx->cpu_src1->దత్తాంశం = ctx->cpu_src1_దత్తాంశం;
        // Copy stride array (nb) for correct memory layout
        జ్ఞాపకం_నకలు(ctx->cpu_src1->nb, కణుపు->src[1]->nb, పరిమాణం(కణుపు->src[1]->nb));
        // Copy op_params if present
        జ్ఞాపకం_నకలు(ctx->cpu_src1->op_params, కణుపు->src[1]->op_params, పరిమాణం(కణుపు->src[1]->op_params));
    }

    ఒకవేళ (కణుపు->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, కణుపు->src[2]->type, GGML_MAX_DIMS, కణుపు->src[2]->ne);
        ఒకవేళ (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            వెళ్ళు cleanup;
        }
        ctx->cpu_src2->దత్తాంశం = ctx->cpu_src2_దత్తాంశం;
        // Copy stride array (nb) for correct memory layout
        జ్ఞాపకం_నకలు(ctx->cpu_src2->nb, కణుపు->src[2]->nb, పరిమాణం(కణుపు->src[2]->nb));
        // Copy op_params if present
        జ్ఞాపకం_నకలు(ctx->cpu_src2->op_params, కణుపు->src[2]->op_params, పరిమాణం(కణుపు->src[2]->op_params));
    }

    తిరిగి నిజం;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    తిరిగి అబద్ధం;
}

తార్కిక ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           స్థిరరాశి ggml_tensor *                కణుపు,
                                           స్థిరరాశి ggml_et_cpu_compare_config * config) {
    ఒకవేళ (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !కణుపు || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        తిరిగి అబద్ధం;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = కణుపు->op;
    ఎంపిక (op) {
        స్థితి GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ఆపు;
        స్థితి GGML_OP_కూడు:
            ctx->cpu_dst = ggml_కూడు(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            ఆపు;
        స్థితి GGML_OP_ROPE:
            {
                స్థిరరాశి int32_t * op_params  = (స్థిరరాశి int32_t *) కణుపు->op_params;
                స్థిరరాశి int32_t   n_dims     = op_params[1];
                స్థిరరాశి int32_t   mode       = op_params[2];
                స్థిరరాశి int32_t   n_ctx_orig = op_params[4];
                స్థిరరాశి దశాంశం     freq_పునాది  = *((స్థిరరాశి దశాంశం *) (op_params + 5));
                స్థిరరాశి దశాంశం     freq_scale = *((స్థిరరాశి దశాంశం *) (op_params + 6));
                స్థిరరాశి దశాంశం     ext_గుణకం = *((స్థిరరాశి దశాంశం *) (op_params + 7));
                స్థిరరాశి దశాంశం     attn_గుణకం = *((స్థిరరాశి దశాంశం *) (op_params + 8));
                స్థిరరాశి దశాంశం     beta_fast  = *((స్థిరరాశి దశాంశం *) (op_params + 9));
                స్థిరరాశి దశాంశం     beta_slow  = *((స్థిరరాశి దశాంశం *) (op_params + 10));

                ఒకవేళ (mode & GGML_ROPE_TYPE_MROPE) {
                    పూర్ణాంకం sections[GGML_MROPE_SECTIONS];
                    జ్ఞాపకం_నకలు(sections, op_params + 11, పరిమాణం(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_పునాది, freq_scale,
                                                   ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
                } లేకపోతే {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_పునాది, freq_scale, ext_గుణకం,
                                                 attn_గుణకం, beta_fast, beta_slow);
                }
            }
            ఆపు;
        స్థితి GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                దశాంశం eps;
                జ్ఞాపకం_నకలు(&eps, కణుపు->op_params, పరిమాణం(దశాంశం));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            ఆపు;
        స్థితి GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            ఆపు;
        స్థితి GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(కణుపు, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            ctx->cpu_dst = ggml_కూడిక_అడ్డువరుసలు(ctx->ggml_ctx, ctx->cpu_src0);
            ఆపు;
        స్థితి GGML_OP_సగటు:
            ctx->cpu_dst = ggml_సగటు(ctx->ggml_ctx, ctx->cpu_src0);
            ఆపు;
        స్థితి GGML_OP_CLAMP:
            {
                దశాంశం clamp_min, clamp_max;
                జ్ఞాపకం_నకలు(&clamp_min, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 0, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&clamp_max, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(దశాంశం));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            ఆపు;
        స్థితి GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(కణుపు, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                ఒకవేళ (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    స్థిరరాశి దశాంశం హద్దు = ggml_get_op_params_f32(కణుపు, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, హద్దు);
                } లేకపోతే {
                    ctx->cpu_dst = ggml_glu_విభజించు(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                దశాంశం scale    = 1.0f;
                దశాంశం max_bias = 0.0f;
                జ్ఞాపకం_నకలు(&scale, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 0, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(దశాంశం));

                ఒకవేళ (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } లేకపోతే {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                ఒకవేళ (ctx->cpu_src2) {
                    ggml_soft_max_కూడు_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            ctx->cpu_dst = ggml_get_అడ్డువరుసలు(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ఆపు;
        స్థితి GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_పునాది = ggml_new_tensor(ctx->ggml_ctx, కణుపు->type, GGML_MAX_DIMS, కణుపు->ne);
                ఒకవేళ (!cpu_dst_పునాది) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    తిరిగి అబద్ధం;
                }
                cpu_dst_పునాది->దత్తాంశం = ctx->cpu_dst_దత్తాంశం;
                జ్ఞాపకం_నకలు(cpu_dst_పునాది->nb, కణుపు->nb, పరిమాణం(కణుపు->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_సమితి_అడ్డువరుసలు(ctx->ggml_ctx, cpu_dst_పునాది, ctx->cpu_src0, ctx->cpu_src1);
            }
            ఆపు;
        అప్రమేయ:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        తిరిగి అబద్ధం;
    }

    ctx->cpu_dst->దత్తాంశం = ctx->cpu_dst_దత్తాంశం;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    ఒకవేళ (op != GGML_OP_CONT) {
        జ్ఞాపకం_నకలు(ctx->cpu_dst->nb, కణుపు->nb, పరిమాణం(కణుపు->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_లేఖాచిత్రం = ggml_new_లేఖాచిత్రం_custom(ctx->ggml_ctx, 1, అబద్ధం);
    ఒకవేళ (!ctx->cpu_లేఖాచిత్రం) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        తిరిగి అబద్ధం;
    }
    ctx->cpu_లేఖాచిత్రం->nodes[0] = ctx->cpu_dst;
    ctx->cpu_లేఖాచిత్రం->n_nodes  = 1;

    // Log input data for debugging if enabled
    ఒకవేళ (config && config->log_differences) {
        ఒకవేళ (ctx->cpu_src0_దత్తాంశం && ctx->src0_కొలత >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_దత్తాంశం)[0],
                           ((uint8_t *) ctx->cpu_src0_దత్తాంశం)[1], ((uint8_t *) ctx->cpu_src0_దత్తాంశం)[2],
                           ((uint8_t *) ctx->cpu_src0_దత్తాంశం)[3]);
        }
        ఒకవేళ (ctx->cpu_src1_దత్తాంశం && ctx->src1_కొలత >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((దశాంశం *) ctx->cpu_src1_దత్తాంశం)[0],
                           ((దశాంశం *) ctx->cpu_src1_దత్తాంశం)[1], ((దశాంశం *) ctx->cpu_src1_దత్తాంశం)[2],
                           ((దశాంశం *) ctx->cpu_src1_దత్తాంశం)[3]);
        }
    }

    // Compute using CPU backend
    ggml_స్థితి cpu_ఫలితం = ggml_backend_లేఖాచిత్రం_compute(ctx->cpu_backend, ctx->cpu_లేఖాచిత్రం);

    ఒకవేళ (cpu_ఫలితం != GGML_స్థితి_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_ఫలితం);
        తిరిగి అబద్ధం;
    }

    // Log output data for debugging if enabled
    ఒకవేళ (config && config->log_differences && ctx->dst_కొలత >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((దశాంశం *) ctx->cpu_dst_దత్తాంశం)[0], ((దశాంశం *) ctx->cpu_dst_దత్తాంశం)[1],
                       ((దశాంశం *) ctx->cpu_dst_దత్తాంశం)[2], ((దశాంశం *) ctx->cpu_dst_దత్తాంశం)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_కొలత = ggml_nbytes(కణుపు);
    ggml_backend_tensor_get(కణుపు, ctx->et_dst_దత్తాంశం, 0, dst_logical_కొలత);

    ఒకవేళ (config->log_differences) {
        size_t అంకె_elements = ggml_nelements(కణుపు);
        size_t max_log      = std::min(అంకె_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        తార్కిక    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_కూడు || op == GGML_OP_GLU);
        దశాంశం * cpu_src0_float = is_elementwise ? (దశాంశం *) ctx->cpu_src0_దత్తాంశం : శూన్యసూచిక;
        దశాంశం * cpu_src1_float = is_elementwise ? (దశాంశం *) ctx->cpu_src1_దత్తాంశం : శూన్యసూచిక;

        // Helper to get float value from tensor data (handles f16 and f32)
        స్వయంచాలక get_float = [](స్థిరరాశి శూన్య * దత్తాంశం, size_t idx, ggml_type type) -> దశాంశం {
            ఒకవేళ (type == GGML_TYPE_F16) {
                స్థిరరాశి ggml_fp16_t * fp16_దత్తాంశం = (స్థిరరాశి ggml_fp16_t *) దత్తాంశం;
                తిరిగి ggml_fp16_గమ్యం_fp32(fp16_దత్తాంశం[idx]);
            }

            స్థిరరాశి దశాంశం * float_దత్తాంశం = (స్థిరరాశి దశాంశం *) దత్తాంశం;
            తిరిగి float_దత్తాంశం[idx];
        };

        // Compare all elements but log only the first max_log_elements
        తార్కిక   matches          = నిజం;
        size_t మొత్తం_mismatches = 0;

        // First pass: check all elements for mismatches
        కొరకు (size_t i = 0; i < అంకె_elements; i++) {
            దశాంశం cpu_val  = get_float(ctx->cpu_dst_దత్తాంశం, i, కణుపు->type);
            దశాంశం et_val   = get_float(ctx->et_dst_దత్తాంశం, i, కణుపు->type);
            దశాంశం diff     = fabsf(cpu_val - et_val);
            దశాంశం rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            ఒకవేళ (rel_diff > config->tolerance) {
                matches = అబద్ధం;
                మొత్తం_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        కొరకు (size_t i = 0; i < max_log; i++) {
            దశాంశం cpu_val = get_float(ctx->cpu_dst_దత్తాంశం, i, కణుపు->type);
            దశాంశం et_val  = get_float(ctx->et_dst_దత్తాంశం, i, కణుపు->type);
            దశాంశం diff    = fabsf(cpu_val - et_val);

            ఒకవేళ (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } లేకపోతే ఒకవేళ (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } లేకపోతే {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        ఒకవేళ (అంకె_elements > max_log) {
            size_t mid     = అంకె_elements / 2;
            size_t end     = అంకె_elements - 1;
            దశాంశం  cpu_mid = get_float(ctx->cpu_dst_దత్తాంశం, mid, కణుపు->type);
            దశాంశం  et_mid  = get_float(ctx->et_dst_దత్తాంశం, mid, కణుపు->type);
            దశాంశం  cpu_చివర = get_float(ctx->cpu_dst_దత్తాంశం, end, కణుపు->type);
            దశాంశం  et_చివర  = get_float(ctx->et_dst_దత్తాంశం, end, కణుపు->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_చివర, et_చివర);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       అంకె_elements - మొత్తం_mismatches, అంకె_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    ఒకవేళ (config->use_cpu_ఫలితం) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_కొలత = ggml_nbytes(కణుపు);
        ggml_backend_tensor_సమితి(const_cast<ggml_tensor *>(కణుపు), ctx->cpu_dst_దత్తాంశం, 0, dst_logical_కొలత);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    తిరిగి నిజం;
}

శూన్య ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    ఒకవేళ (!ctx) {
        తిరిగి;
    }

    ఒకవేళ (ctx->cpu_src0_దత్తాంశం) {
        విడుదల(ctx->cpu_src0_దత్తాంశం);
        ctx->cpu_src0_దత్తాంశం = శూన్యసూచిక;
    }
    ఒకవేళ (ctx->cpu_src1_దత్తాంశం) {
        విడుదల(ctx->cpu_src1_దత్తాంశం);
        ctx->cpu_src1_దత్తాంశం = శూన్యసూచిక;
    }
    ఒకవేళ (ctx->cpu_src2_దత్తాంశం) {
        విడుదల(ctx->cpu_src2_దత్తాంశం);
        ctx->cpu_src2_దత్తాంశం = శూన్యసూచిక;
    }
    ఒకవేళ (ctx->cpu_dst_దత్తాంశం) {
        విడుదల(ctx->cpu_dst_దత్తాంశం);
        ctx->cpu_dst_దత్తాంశం = శూన్యసూచిక;
    }
    ఒకవేళ (ctx->et_dst_దత్తాంశం) {
        విడుదల(ctx->et_dst_దత్తాంశం);
        ctx->et_dst_దత్తాంశం = శూన్యసూచిక;
    }

    ఒకవేళ (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = శూన్యసూచిక;
    }

    ఒకవేళ (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = శూన్యసూచిక;
    }

    // Clear pointers
    ctx->cpu_src0  = శూన్యసూచిక;
    ctx->cpu_src1  = శూన్యసూచిక;
    ctx->cpu_src2  = శూన్యసూచిక;
    ctx->cpu_dst   = శూన్యసూచిక;
    ctx->cpu_లేఖాచిత్రం = శూన్యసూచిక;
}
