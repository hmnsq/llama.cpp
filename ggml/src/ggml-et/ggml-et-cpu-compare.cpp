#অন্তর্ভাব "ggml-et-cpu-compare.h"

#অন্তর্ভাব "ggml-cpu/ggml-cpu-impl.h"
#অন্তর্ভাব "ggml-cpu/ops.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>

যৌক্তিক ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ধ্রুবক ggml_tensor * গ্রন্থি, ggml_op op) {
    যদি (!ctx || !গ্রন্থি) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        ফেরত মিথ্যা;
    }

    // Clear context
    memset(ctx, 0, আকার(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_বাফার_আকার = [](ধ্রুবক ggml_tensor * tensor) -> size_t {
        যদি (!tensor) {
            ফেরত 0;
        }

        যদি (tensor->বাফার) {
            // Get actual backend buffer size
            size_t বাফার_আকার = ggml_backend_বাফার_get_আকার(tensor->বাফার);

            // Use the full buffer size to avoid any truncation issues
            ফেরত বাফার_আকার;
        } অন্যথা {
            // Fallback to logical size if no buffer
            ফেরত ggml_nbytes(tensor);
        }
    };

    ctx->src0_আকার = get_tensor_বাফার_আকার(গ্রন্থি->src[0]);
    ctx->src1_আকার = get_tensor_বাফার_আকার(গ্রন্থি->src[1]);
    ctx->src2_আকার = get_tensor_বাফার_আকার(গ্রন্থি->src[2]);
    ctx->dst_আকার  = get_tensor_বাফার_আকার(গ্রন্থি);

    // Allocate CPU buffers for all tensors
    যদি (ctx->src0_আকার > 0) {
        ctx->cpu_src0_তথ্য = malloc(ctx->src0_আকার);
        যদি (!ctx->cpu_src0_তথ্য) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            যাও cleanup;
        }
    }

    যদি (ctx->src1_আকার > 0) {
        ctx->cpu_src1_তথ্য = malloc(ctx->src1_আকার);
        যদি (!ctx->cpu_src1_তথ্য) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            যাও cleanup;
        }
    }

    যদি (ctx->src2_আকার > 0) {
        ctx->cpu_src2_তথ্য = malloc(ctx->src2_আকার);
        যদি (!ctx->cpu_src2_তথ্য) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            যাও cleanup;
        }
    }

    ctx->cpu_dst_তথ্য = malloc(ctx->dst_আকার);
    যদি (!ctx->cpu_dst_তথ্য) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        যাও cleanup;
    }

    ctx->et_dst_তথ্য = malloc(ctx->dst_আকার);
    যদি (!ctx->et_dst_তথ্য) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        যাও cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    যদি (ctx->src0_আকার > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_আকার = ggml_nbytes(গ্রন্থি->src[0]);
        ggml_backend_tensor_get(গ্রন্থি->src[0], ctx->cpu_src0_তথ্য, 0, logical_আকার);
    }
    যদি (ctx->src1_আকার > 0) {
        size_t logical_আকার = ggml_nbytes(গ্রন্থি->src[1]);
        ggml_backend_tensor_get(গ্রন্থি->src[1], ctx->cpu_src1_তথ্য, 0, logical_আকার);
    }
    যদি (ctx->src2_আকার > 0) {
        size_t logical_আকার = ggml_nbytes(গ্রন্থি->src[2]);
        ggml_backend_tensor_get(গ্রন্থি->src[2], ctx->cpu_src2_তথ্য, 0, logical_আকার);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_আকার = ggml_nbytes(গ্রন্থি);
        ggml_backend_tensor_get(গ্রন্থি, ctx->cpu_dst_তথ্য, 0, logical_আকার);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    যদি (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        যাও cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_আকার   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_বাফার = nullptr;
    ctx_params.no_alloc   = সত্য;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    যদি (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        যাও cleanup;
    }

    // Create CPU tensors with proper context
    যদি (গ্রন্থি->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->src[0]->type, GGML_MAX_DIMS, গ্রন্থি->src[0]->ne);
        যদি (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            যাও cleanup;
        }
        ctx->cpu_src0->তথ্য = ctx->cpu_src0_তথ্য;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, গ্রন্থি->src[0]->nb, আকার(গ্রন্থি->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, গ্রন্থি->src[0]->op_params, আকার(গ্রন্থি->src[0]->op_params));
    }

    যদি (গ্রন্থি->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->src[1]->type, GGML_MAX_DIMS, গ্রন্থি->src[1]->ne);
        যদি (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            যাও cleanup;
        }
        ctx->cpu_src1->তথ্য = ctx->cpu_src1_তথ্য;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, গ্রন্থি->src[1]->nb, আকার(গ্রন্থি->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, গ্রন্থি->src[1]->op_params, আকার(গ্রন্থি->src[1]->op_params));
    }

    যদি (গ্রন্থি->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->src[2]->type, GGML_MAX_DIMS, গ্রন্থি->src[2]->ne);
        যদি (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            যাও cleanup;
        }
        ctx->cpu_src2->তথ্য = ctx->cpu_src2_তথ্য;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, গ্রন্থি->src[2]->nb, আকার(গ্রন্থি->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, গ্রন্থি->src[2]->op_params, আকার(গ্রন্থি->src[2]->op_params));
    }

    ফেরত সত্য;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    ফেরত মিথ্যা;
}

যৌক্তিক ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ধ্রুবক ggml_tensor *                গ্রন্থি,
                                           ধ্রুবক ggml_et_cpu_compare_config * config) {
    যদি (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !গ্রন্থি || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        ফেরত মিথ্যা;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = গ্রন্থি->op;
    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            বিরতি;
        ক্ষেত্র GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            বিরতি;
        ক্ষেত্র GGML_OP_ROPE:
            {
                ধ্রুবক int32_t * op_params  = (ধ্রুবক int32_t *) গ্রন্থি->op_params;
                ধ্রুবক int32_t   n_dims     = op_params[1];
                ধ্রুবক int32_t   mode       = op_params[2];
                ধ্রুবক int32_t   n_ctx_orig = op_params[4];
                ধ্রুবক দশমিক     freq_base  = *((ধ্রুবক দশমিক *) (op_params + 5));
                ধ্রুবক দশমিক     freq_scale = *((ধ্রুবক দশমিক *) (op_params + 6));
                ধ্রুবক দশমিক     ext_factor = *((ধ্রুবক দশমিক *) (op_params + 7));
                ধ্রুবক দশমিক     attn_factor = *((ধ্রুবক দশমিক *) (op_params + 8));
                ধ্রুবক দশমিক     beta_fast  = *((ধ্রুবক দশমিক *) (op_params + 9));
                ধ্রুবক দশমিক     beta_slow  = *((ধ্রুবক দশমিক *) (op_params + 10));

                যদি (mode & GGML_ROPE_TYPE_MROPE) {
                    পূর্ণসংখ্যা sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, আকার(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } অন্যথা {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                দশমিক eps;
                memcpy(&eps, গ্রন্থি->op_params, আকার(দশমিক));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            বিরতি;
        ক্ষেত্র GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            বিরতি;
        ক্ষেত্র GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(গ্রন্থি, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            বিরতি;
        ক্ষেত্র GGML_OP_যোগ_ROWS:
            ctx->cpu_dst = ggml_যোগ_rows(ctx->ggml_ctx, ctx->cpu_src0);
            বিরতি;
        ক্ষেত্র GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            বিরতি;
        ক্ষেত্র GGML_OP_CLAMP:
            {
                দশমিক clamp_min, clamp_max;
                memcpy(&clamp_min, (ধ্রুবক দশমিক *) গ্রন্থি->op_params + 0, আকার(দশমিক));
                memcpy(&clamp_max, (ধ্রুবক দশমিক *) গ্রন্থি->op_params + 1, আকার(দশমিক));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            বিরতি;
        ক্ষেত্র GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(গ্রন্থি, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                যদি (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    ফেরত মিথ্যা;
                }
                যদি (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ধ্রুবক দশমিক limit = ggml_get_op_params_f32(গ্রন্থি, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } অন্যথা {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                দশমিক scale    = 1.0f;
                দশমিক max_bias = 0.0f;
                memcpy(&scale, (ধ্রুবক দশমিক *) গ্রন্থি->op_params + 0, আকার(দশমিক));
                memcpy(&max_bias, (ধ্রুবক দশমিক *) গ্রন্থি->op_params + 1, আকার(দশমিক));

                যদি (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } অন্যথা {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                যদি (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            বিরতি;
        ক্ষেত্র GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            বিরতি;
        ক্ষেত্র GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->type, GGML_MAX_DIMS, গ্রন্থি->ne);
                যদি (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    ফেরত মিথ্যা;
                }
                cpu_dst_base->তথ্য = ctx->cpu_dst_তথ্য;
                memcpy(cpu_dst_base->nb, গ্রন্থি->nb, আকার(গ্রন্থি->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            বিরতি;
        পূর্বনির্ধারিত:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            ফেরত মিথ্যা;
    }

    যদি (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        ফেরত মিথ্যা;
    }

    ctx->cpu_dst->তথ্য = ctx->cpu_dst_তথ্য;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    যদি (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, গ্রন্থি->nb, আকার(গ্রন্থি->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, মিথ্যা);
    যদি (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        ফেরত মিথ্যা;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    যদি (config && config->log_differences) {
        যদি (ctx->cpu_src0_তথ্য && ctx->src0_আকার >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_তথ্য)[0],
                           ((uint8_t *) ctx->cpu_src0_তথ্য)[1], ((uint8_t *) ctx->cpu_src0_তথ্য)[2],
                           ((uint8_t *) ctx->cpu_src0_তথ্য)[3]);
        }
        যদি (ctx->cpu_src1_তথ্য && ctx->src1_আকার >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((দশমিক *) ctx->cpu_src1_তথ্য)[0],
                           ((দশমিক *) ctx->cpu_src1_তথ্য)[1], ((দশমিক *) ctx->cpu_src1_তথ্য)[2],
                           ((দশমিক *) ctx->cpu_src1_তথ্য)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_ফলাফল = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    যদি (cpu_ফলাফল != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_ফলাফল);
        ফেরত মিথ্যা;
    }

    // Log output data for debugging if enabled
    যদি (config && config->log_differences && ctx->dst_আকার >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((দশমিক *) ctx->cpu_dst_তথ্য)[0], ((দশমিক *) ctx->cpu_dst_তথ্য)[1],
                       ((দশমিক *) ctx->cpu_dst_তথ্য)[2], ((দশমিক *) ctx->cpu_dst_তথ্য)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_আকার = ggml_nbytes(গ্রন্থি);
    ggml_backend_tensor_get(গ্রন্থি, ctx->et_dst_তথ্য, 0, dst_logical_আকার);

    যদি (config->log_differences) {
        size_t অঙ্ক_elements = ggml_nelements(গ্রন্থি);
        size_t max_log      = std::min(অঙ্ক_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        যৌক্তিক    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        দশমিক * cpu_src0_float = is_elementwise ? (দশমিক *) ctx->cpu_src0_তথ্য : nullptr;
        দশমিক * cpu_src1_float = is_elementwise ? (দশমিক *) ctx->cpu_src1_তথ্য : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ধ্রুবক শূন্য * তথ্য, size_t idx, ggml_type type) -> দশমিক {
            যদি (type == GGML_TYPE_F16) {
                ধ্রুবক ggml_fp16_t * fp16_তথ্য = (ধ্রুবক ggml_fp16_t *) তথ্য;
                ফেরত ggml_fp16_to_fp32(fp16_তথ্য[idx]);
            }

            ধ্রুবক দশমিক * float_তথ্য = (ধ্রুবক দশমিক *) তথ্য;
            ফেরত float_তথ্য[idx];
        };

        // Compare all elements but log only the first max_log_elements
        যৌক্তিক   matches          = সত্য;
        size_t যোগ_mismatches = 0;

        // First pass: check all elements for mismatches
        জন্য (size_t i = 0; i < অঙ্ক_elements; i++) {
            দশমিক cpu_val  = get_float(ctx->cpu_dst_তথ্য, i, গ্রন্থি->type);
            দশমিক et_val   = get_float(ctx->et_dst_তথ্য, i, গ্রন্থি->type);
            দশমিক diff     = fabsf(cpu_val - et_val);
            দশমিক rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            যদি (rel_diff > config->tolerance) {
                matches = মিথ্যা;
                যোগ_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        জন্য (size_t i = 0; i < max_log; i++) {
            দশমিক cpu_val = get_float(ctx->cpu_dst_তথ্য, i, গ্রন্থি->type);
            দশমিক et_val  = get_float(ctx->et_dst_তথ্য, i, গ্রন্থি->type);
            দশমিক diff    = fabsf(cpu_val - et_val);

            যদি (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } অন্যথা যদি (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } অন্যথা {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        যদি (অঙ্ক_elements > max_log) {
            size_t mid     = অঙ্ক_elements / 2;
            size_t end     = অঙ্ক_elements - 1;
            দশমিক  cpu_mid = get_float(ctx->cpu_dst_তথ্য, mid, গ্রন্থি->type);
            দশমিক  et_mid  = get_float(ctx->et_dst_তথ্য, mid, গ্রন্থি->type);
            দশমিক  cpu_end = get_float(ctx->cpu_dst_তথ্য, end, গ্রন্থি->type);
            দশমিক  et_end  = get_float(ctx->et_dst_তথ্য, end, গ্রন্থি->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       অঙ্ক_elements - যোগ_mismatches, অঙ্ক_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    যদি (config->use_cpu_ফলাফল) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_আকার = ggml_nbytes(গ্রন্থি);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(গ্রন্থি), ctx->cpu_dst_তথ্য, 0, dst_logical_আকার);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    ফেরত সত্য;
}

শূন্য ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    যদি (!ctx) {
        ফেরত;
    }

    যদি (ctx->cpu_src0_তথ্য) {
        মুক্ত(ctx->cpu_src0_তথ্য);
        ctx->cpu_src0_তথ্য = nullptr;
    }
    যদি (ctx->cpu_src1_তথ্য) {
        মুক্ত(ctx->cpu_src1_তথ্য);
        ctx->cpu_src1_তথ্য = nullptr;
    }
    যদি (ctx->cpu_src2_তথ্য) {
        মুক্ত(ctx->cpu_src2_তথ্য);
        ctx->cpu_src2_তথ্য = nullptr;
    }
    যদি (ctx->cpu_dst_তথ্য) {
        মুক্ত(ctx->cpu_dst_তথ্য);
        ctx->cpu_dst_তথ্য = nullptr;
    }
    যদি (ctx->et_dst_তথ্য) {
        মুক্ত(ctx->et_dst_তথ্য);
        ctx->et_dst_তথ্য = nullptr;
    }

    যদি (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    যদি (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }

    // Clear pointers
    ctx->cpu_src0  = nullptr;
    ctx->cpu_src1  = nullptr;
    ctx->cpu_src2  = nullptr;
    ctx->cpu_dst   = nullptr;
    ctx->cpu_graph = nullptr;
}
