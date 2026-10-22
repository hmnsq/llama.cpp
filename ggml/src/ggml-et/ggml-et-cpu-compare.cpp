#অন্তর্ভাব "ggml-et-cpu-compare.h"

#অন্তর্ভাব "ggml-cpu/ggml-cpu-impl.h"
#অন্তর্ভাব "ggml-cpu/ops.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>

তার্কিক ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ধ্রুব ggml_tensor * গ্রন্থি, ggml_op op) {
    যদি (!ctx || !গ্রন্থি) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        প্রত্যাগম অসত্য;
    }

    // Clear context
    memset(ctx, 0, পরিমাণ(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](ধ্রুব ggml_tensor * tensor) -> size_t {
        যদি (!tensor) {
            প্রত্যাগম 0;
        }

        যদি (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            প্রত্যাগম buffer_size;
        } অন্যথা {
            // Fallback to logical size if no buffer
            প্রত্যাগম ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(গ্রন্থি->src[0]);
    ctx->src1_size = get_tensor_buffer_size(গ্রন্থি->src[1]);
    ctx->src2_size = get_tensor_buffer_size(গ্রন্থি->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(গ্রন্থি);

    // Allocate CPU buffers for all tensors
    যদি (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        যদি (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            গচ্ছ cleanup;
        }
    }

    যদি (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        যদি (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            গচ্ছ cleanup;
        }
    }

    যদি (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        যদি (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            গচ্ছ cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    যদি (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        গচ্ছ cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    যদি (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        গচ্ছ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    যদি (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(গ্রন্থি->src[0]);
        ggml_backend_tensor_get(গ্রন্থি->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    যদি (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(গ্রন্থি->src[1]);
        ggml_backend_tensor_get(গ্রন্থি->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    যদি (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(গ্রন্থি->src[2]);
        ggml_backend_tensor_get(গ্রন্থি->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(গ্রন্থি);
        ggml_backend_tensor_get(গ্রন্থি, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    যদি (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        গচ্ছ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = সত্য;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    যদি (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        গচ্ছ cleanup;
    }

    // Create CPU tensors with proper context
    যদি (গ্রন্থি->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->src[0]->type, GGML_MAX_DIMS, গ্রন্থি->src[0]->ne);
        যদি (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            গচ্ছ cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, গ্রন্থি->src[0]->nb, পরিমাণ(গ্রন্থি->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, গ্রন্থি->src[0]->op_params, পরিমাণ(গ্রন্থি->src[0]->op_params));
    }

    যদি (গ্রন্থি->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->src[1]->type, GGML_MAX_DIMS, গ্রন্থি->src[1]->ne);
        যদি (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            গচ্ছ cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, গ্রন্থি->src[1]->nb, পরিমাণ(গ্রন্থি->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, গ্রন্থি->src[1]->op_params, পরিমাণ(গ্রন্থি->src[1]->op_params));
    }

    যদি (গ্রন্থি->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->src[2]->type, GGML_MAX_DIMS, গ্রন্থি->src[2]->ne);
        যদি (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            গচ্ছ cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, গ্রন্থি->src[2]->nb, পরিমাণ(গ্রন্থি->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, গ্রন্থি->src[2]->op_params, পরিমাণ(গ্রন্থি->src[2]->op_params));
    }

    প্রত্যাগম সত্য;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    প্রত্যাগম অসত্য;
}

তার্কিক ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ধ্রুব ggml_tensor *                গ্রন্থি,
                                           ধ্রুব ggml_et_cpu_compare_config * config) {
    যদি (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !গ্রন্থি || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        প্রত্যাগম অসত্য;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = গ্রন্থি->op;
    বিকল্প (op) {
        পক্ষ GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ভঙ্গ;
        পক্ষ GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            ভঙ্গ;
        পক্ষ GGML_OP_ROPE:
            {
                ধ্রুব int32_t * op_params  = (ধ্রুব int32_t *) গ্রন্থি->op_params;
                ধ্রুব int32_t   n_dims     = op_params[1];
                ধ্রুব int32_t   mode       = op_params[2];
                ধ্রুব int32_t   n_ctx_orig = op_params[4];
                ধ্রুব ভিন্ন     freq_base  = *((ধ্রুব ভিন্ন *) (op_params + 5));
                ধ্রুব ভিন্ন     freq_scale = *((ধ্রুব ভিন্ন *) (op_params + 6));
                ধ্রুব ভিন্ন     ext_factor = *((ধ্রুব ভিন্ন *) (op_params + 7));
                ধ্রুব ভিন্ন     attn_factor = *((ধ্রুব ভিন্ন *) (op_params + 8));
                ধ্রুব ভিন্ন     beta_fast  = *((ধ্রুব ভিন্ন *) (op_params + 9));
                ধ্রুব ভিন্ন     beta_slow  = *((ধ্রুব ভিন্ন *) (op_params + 10));

                যদি (mode & GGML_ROPE_TYPE_MROPE) {
                    পূর্ণাঙ্ক sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, পরিমাণ(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } অন্যথা {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            ভঙ্গ;
        পক্ষ GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                ভিন্ন eps;
                memcpy(&eps, গ্রন্থি->op_params, পরিমাণ(ভিন্ন));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            ভঙ্গ;
        পক্ষ GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(গ্রন্থি, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            ভঙ্গ;
        পক্ষ GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            ভঙ্গ;
        পক্ষ GGML_OP_CLAMP:
            {
                ভিন্ন clamp_min, clamp_max;
                memcpy(&clamp_min, (ধ্রুব ভিন্ন *) গ্রন্থি->op_params + 0, পরিমাণ(ভিন্ন));
                memcpy(&clamp_max, (ধ্রুব ভিন্ন *) গ্রন্থি->op_params + 1, পরিমাণ(ভিন্ন));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(গ্রন্থি, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                যদি (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    প্রত্যাগম অসত্য;
                }
                যদি (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ধ্রুব ভিন্ন limit = ggml_get_op_params_f32(গ্রন্থি, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } অন্যথা {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                ভিন্ন scale    = 1.0f;
                ভিন্ন max_bias = 0.0f;
                memcpy(&scale, (ধ্রুব ভিন্ন *) গ্রন্থি->op_params + 0, পরিমাণ(ভিন্ন));
                memcpy(&max_bias, (ধ্রুব ভিন্ন *) গ্রন্থি->op_params + 1, পরিমাণ(ভিন্ন));

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
            ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ভঙ্গ;
        পক্ষ GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            ভঙ্গ;
        পক্ষ GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, গ্রন্থি->type, GGML_MAX_DIMS, গ্রন্থি->ne);
                যদি (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    প্রত্যাগম অসত্য;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, গ্রন্থি->nb, পরিমাণ(গ্রন্থি->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            ভঙ্গ;
        স্বভাব:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            প্রত্যাগম অসত্য;
    }

    যদি (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        প্রত্যাগম অসত্য;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    যদি (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, গ্রন্থি->nb, পরিমাণ(গ্রন্থি->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, অসত্য);
    যদি (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        প্রত্যাগম অসত্য;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    যদি (config && config->log_differences) {
        যদি (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        যদি (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((ভিন্ন *) ctx->cpu_src1_data)[0],
                           ((ভিন্ন *) ctx->cpu_src1_data)[1], ((ভিন্ন *) ctx->cpu_src1_data)[2],
                           ((ভিন্ন *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    যদি (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        প্রত্যাগম অসত্য;
    }

    // Log output data for debugging if enabled
    যদি (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((ভিন্ন *) ctx->cpu_dst_data)[0], ((ভিন্ন *) ctx->cpu_dst_data)[1],
                       ((ভিন্ন *) ctx->cpu_dst_data)[2], ((ভিন্ন *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(গ্রন্থি);
    ggml_backend_tensor_get(গ্রন্থি, ctx->et_dst_data, 0, dst_logical_size);

    যদি (config->log_differences) {
        size_t অঙ্ক_elements = ggml_nelements(গ্রন্থি);
        size_t max_log      = std::min(অঙ্ক_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        তার্কিক    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        ভিন্ন * cpu_src0_float = is_elementwise ? (ভিন্ন *) ctx->cpu_src0_data : nullptr;
        ভিন্ন * cpu_src1_float = is_elementwise ? (ভিন্ন *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ধ্রুব শূন্য * data, size_t idx, ggml_type type) -> ভিন্ন {
            যদি (type == GGML_TYPE_F16) {
                ধ্রুব ggml_fp16_t * fp16_data = (ধ্রুব ggml_fp16_t *) data;
                প্রত্যাগম ggml_fp16_to_fp32(fp16_data[idx]);
            }

            ধ্রুব ভিন্ন * float_data = (ধ্রুব ভিন্ন *) data;
            প্রত্যাগম float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        তার্কিক   matches          = সত্য;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        প্রতি (size_t i = 0; i < অঙ্ক_elements; i++) {
            ভিন্ন cpu_val  = get_float(ctx->cpu_dst_data, i, গ্রন্থি->type);
            ভিন্ন et_val   = get_float(ctx->et_dst_data, i, গ্রন্থি->type);
            ভিন্ন diff     = fabsf(cpu_val - et_val);
            ভিন্ন rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            যদি (rel_diff > config->tolerance) {
                matches = অসত্য;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        প্রতি (size_t i = 0; i < max_log; i++) {
            ভিন্ন cpu_val = get_float(ctx->cpu_dst_data, i, গ্রন্থি->type);
            ভিন্ন et_val  = get_float(ctx->et_dst_data, i, গ্রন্থি->type);
            ভিন্ন diff    = fabsf(cpu_val - et_val);

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
            ভিন্ন  cpu_mid = get_float(ctx->cpu_dst_data, mid, গ্রন্থি->type);
            ভিন্ন  et_mid  = get_float(ctx->et_dst_data, mid, গ্রন্থি->type);
            ভিন্ন  cpu_end = get_float(ctx->cpu_dst_data, end, গ্রন্থি->type);
            ভিন্ন  et_end  = get_float(ctx->et_dst_data, end, গ্রন্থি->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       অঙ্ক_elements - total_mismatches, অঙ্ক_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    যদি (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(গ্রন্থি);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(গ্রন্থি), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    প্রত্যাগম সত্য;
}

শূন্য ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    যদি (!ctx) {
        প্রত্যাগম;
    }

    যদি (ctx->cpu_src0_data) {
        মুক্ত(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    যদি (ctx->cpu_src1_data) {
        মুক্ত(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    যদি (ctx->cpu_src2_data) {
        মুক্ত(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    যদি (ctx->cpu_dst_data) {
        মুক্ত(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    যদি (ctx->et_dst_data) {
        মুক্ত(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
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
