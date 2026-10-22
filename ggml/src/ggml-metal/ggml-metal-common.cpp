#अन्तर्भाव "ggml-metal-common.h"
#अन्तर्भाव "ggml-metal-fusion.h"

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
स्थिर तार्किक ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    सचे (n == 64 || n == 128 || n == 256 || n == 512) {
        पटिनिवत्त सच्च;
    }

    सचे (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        पटिनिवत्त (size_t) n * परिमाण(भिन्न) <= max_tg_mem;
    }

    पटिनिवत्त मिच्छा;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
तार्किक ggml_metal_op_mul_mat_use_fwht(ध्रुव संरचना ggml_tensor * op, size_t max_tg_mem) {
    पटिनिवत्त ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

तार्किक ggml_metal_op_mul_mat_use_mm(ध्रुव संरचना ggml_tensor * op, तार्किक has_simdgroup_mm) {
    ध्रुव int64_t ne00 = op->src[0]->ne[0];
    ध्रुव int64_t ne11 = op->src[1]->ne[1];

    पटिनिवत्त !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

तार्किक ggml_metal_op_mul_mat_id_use_mm(ध्रुव संरचना ggml_tensor * op, तार्किक has_simdgroup_mm) {
    ध्रुव int64_t ne00 = op->src[0]->ne[0];
    ध्रुव int64_t ne02 = op->src[0]->ne[2];
    ध्रुव int64_t ne21 = op->src[2]->ne[1];

    पटिनिवत्त has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
स्थिर constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
स्थिर constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
स्थिर constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

गणना ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(गणना ggml_type type, पूर्णाङ्क rt) {
    सचे (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        पटिनिवत्त GGML_METAL_MMA_KIND_BLK;
    }
    पटिनिवत्त type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

पूर्णाङ्क ggml_metal_mul_mv_mma_rt(ध्रुव संरचना ggml_tensor * op) {
    पटिनिवत्त op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

स्थिर तार्किक ggml_metal_mul_mv_mma_type_supported(गणना ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_TQ2_0:
            पटिनिवत्त सच्च;
        स्वभाव:
            पटिनिवत्त मिच्छा;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(गणना ggml_type type, पूर्णाङ्क rt) {
    सचे (!ggml_metal_mul_mv_mma_type_supported(type)) {
        पटिनिवत्त 0;
    }
    पटिनिवत्त ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

स्थिर तार्किक ggml_metal_mul_mat_mma_type_ok(ध्रुव संरचना ggml_tensor * op) {
    ध्रुव ggml_tensor * src0 = op->src[0];
    ध्रुव int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    पटिनिवत्त step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
स्थिर int64_t ggml_metal_mul_mv_mma_rows_min(गणना ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
            पटिनिवत्त 6;
        पक्ष GGML_TYPE_TQ2_0:
            पटिनिवत्त 5;
        पक्ष GGML_TYPE_BF16:
            पटिनिवत्त 4;
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_MXFP4:
            पटिनिवत्त 3;
        स्वभाव:
            पटिनिवत्त 2;
    }
}

तार्किक ggml_metal_op_mul_mat_use_mma(ध्रुव संरचना ggml_tensor * op) {
    ध्रुव ggml_tensor * src0 = op->src[0];
    ध्रुव ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ध्रुव तार्किक batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    पटिनिवत्त ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == परिमाण(भिन्न) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

तार्किक ggml_metal_op_mul_mat_may_use_mma(ध्रुव संरचना ggml_tensor * op) {
    पटिनिवत्त ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
संरचना ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

संरचना ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    पूर्णाङ्क debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(पूर्णाङ्क debug) {
    auto * res = नूतन ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    पटिनिवत्त res;
}

शून्य ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

शून्य ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

स्थिर तार्किक ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    पटिनिवत्त सच्च;
}

स्थिर ggml_mem_range ggml_mem_range_from_tensor(ध्रुव ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    सचे (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } अञ्ञथा {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    पटिनिवत्त mr;
}

स्थिर ggml_mem_range ggml_mem_range_from_tensor_src(ध्रुव ggml_tensor * tensor) {
    पटिनिवत्त ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

स्थिर ggml_mem_range ggml_mem_range_from_tensor_dst(ध्रुव ggml_tensor * tensor) {
    पटिनिवत्त ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

स्थिर तार्किक ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, ध्रुव ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    सचे (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    पटिनिवत्त ggml_mem_ranges_add(mrs, mr);
}

स्थिर तार्किक ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, ध्रुव ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    सचे (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    पटिनिवत्त ggml_mem_ranges_add(mrs, mr);
}

तार्किक ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ध्रुव ggml_tensor * tensor) {
    अत्थाय (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        सचे (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    पटिनिवत्त ggml_mem_ranges_add_dst(mrs, tensor);
}

स्थिर तार्किक ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    अत्थाय (size_t i = 0; i < mrs->ranges.size(); i++) {
        ध्रुव auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        सचे (mr.pb != cmp.pb) {
            अनुवर्त;
        }

        // intersecting source ranges are allowed
        सचे (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            अनुवर्त;
        }

        सचे (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            सचे (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            पटिनिवत्त मिच्छा;
        }
    }

    पटिनिवत्त सच्च;
}

स्थिर तार्किक ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ध्रुव ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ध्रुव तार्किक res = ggml_mem_ranges_check(mrs, mr);

    पटिनिवत्त res;
}

स्थिर तार्किक ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ध्रुव ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ध्रुव तार्किक res = ggml_mem_ranges_check(mrs, mr);

    पटिनिवत्त res;
}

तार्किक ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ध्रुव ggml_tensor * tensor) {
    अत्थाय (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        सचे (tensor->src[i]) {
            सचे (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                पटिनिवत्त मिच्छा;
            }
        }
    }

    पटिनिवत्त ggml_mem_ranges_check_dst(mrs, tensor);
}

संरचना ग्रन्थिः_info {
    ggml_tensor * ग्रन्थिः;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ध्रुव {
        पटिनिवत्त ग्रन्थिः->op;
    }

    ध्रुव ggml_tensor * dst() ध्रुव {
        पटिनिवत्त fused.empty() ? ग्रन्थिः : fused.back();
    }

    तार्किक is_empty() ध्रुव {
        पटिनिवत्त ggml_op_is_empty(ग्रन्थिः->op);
    }

    शून्य add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    तार्किक is_output(ध्रुव ggml_tensor * t) ध्रुव {
        सचे (t == ग्रन्थिः) {
            पटिनिवत्त सच्च;
        }
        अत्थाय (ध्रुव auto * f : fused) {
            सचे (t == f) {
                पटिनिवत्त सच्च;
            }
        }
        पटिनिवत्त मिच्छा;
    }
};

स्थिर std::vector<पूर्णाङ्क> ggml_metal_graph_optimize_reorder(ध्रुव std::vector<ग्रन्थिः_info> & nodes) {
    // helper to add node src and dst ranges
    ध्रुव auto & h_add = [](ggml_mem_ranges_t mrs, ध्रुव ग्रन्थिः_info & ग्रन्थिः) {
        // only external sources matter: sources produced by the fused group are internal
        अत्थाय (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            ध्रुव ggml_tensor * src = ग्रन्थिः.ग्रन्थिः->src[i];
            सचे (src && !ग्रन्थिः.is_output(src)) {
                सचे (!ggml_mem_ranges_add_src(mrs, src)) {
                    पटिनिवत्त मिच्छा;
                }
            }
        }

        अत्थाय (ध्रुव auto * fused : ग्रन्थिः.fused) {
            अत्थाय (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
                ध्रुव ggml_tensor * src = fused->src[i];
                सचे (src && !ग्रन्थिः.is_output(src)) {
                    सचे (!ggml_mem_ranges_add_src(mrs, src)) {
                        पटिनिवत्त मिच्छा;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        सचे (!ggml_mem_ranges_add_dst(mrs, ग्रन्थिः.ग्रन्थिः)) {
            पटिनिवत्त मिच्छा;
        }
        अत्थाय (ध्रुव auto * fused : ग्रन्थिः.fused) {
            सचे (!ggml_mem_ranges_add_dst(mrs, fused)) {
                पटिनिवत्त मिच्छा;
            }
        }

        पटिनिवत्त सच्च;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ध्रुव auto & h_check = [](ggml_mem_ranges_t mrs, ध्रुव ग्रन्थिः_info & ग्रन्थिः) {
        अत्थाय (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            ध्रुव ggml_tensor * src = ग्रन्थिः.ग्रन्थिः->src[i];
            सचे (src && !ग्रन्थिः.is_output(src)) {
                सचे (!ggml_mem_ranges_check_src(mrs, src)) {
                    पटिनिवत्त मिच्छा;
                }
            }
        }

        अत्थाय (ध्रुव auto * fused : ग्रन्थिः.fused) {
            अत्थाय (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
                ध्रुव ggml_tensor * src = fused->src[i];
                सचे (src && !ग्रन्थिः.is_output(src)) {
                    सचे (!ggml_mem_ranges_check_src(mrs, src)) {
                        पटिनिवत्त मिच्छा;
                    }
                }
            }
        }

        सचे (!ggml_mem_ranges_check_dst(mrs, ग्रन्थिः.ग्रन्थिः)) {
            पटिनिवत्त मिच्छा;
        }
        अत्थाय (ध्रुव auto * fused : ग्रन्थिः.fused) {
            सचे (!ggml_mem_ranges_check_dst(mrs, fused)) {
                पटिनिवत्त मिच्छा;
            }
        }

        पटिनिवत्त सच्च;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ध्रुव auto & h_safe = [](ggml_op op) {
        विकल्प (op) {
            पक्ष GGML_OP_MUL_MAT:
            पक्ष GGML_OP_MUL_MAT_ID:
            पक्ष GGML_OP_ROPE:
            पक्ष GGML_OP_NORM:
            पक्ष GGML_OP_RMS_NORM:
            पक्ष GGML_OP_GROUP_NORM:
            पक्ष GGML_OP_L2_NORM:
            पक्ष GGML_OP_SUM_ROWS:
            पक्ष GGML_OP_SSM_CONV:
            पक्ष GGML_OP_SSM_SCAN:
            पक्ष GGML_OP_CLAMP:
            पक्ष GGML_OP_TRI:
            पक्ष GGML_OP_DIAG:
            पक्ष GGML_OP_MUL:
            पक्ष GGML_OP_ADD:
            पक्ष GGML_OP_SUB:
            पक्ष GGML_OP_DIV:
            पक्ष GGML_OP_GLU:
            पक्ष GGML_OP_SCALE:
            पक्ष GGML_OP_UNARY:
            पक्ष GGML_OP_GET_ROWS:
            पक्ष GGML_OP_SET_ROWS:
            पक्ष GGML_OP_SET:
            पक्ष GGML_OP_CPY:
            पक्ष GGML_OP_CONT:
            पक्ष GGML_OP_REPEAT:
                पटिनिवत्त सच्च;
            स्वभाव:
                पटिनिवत्त ggml_op_is_empty(op);
        }
    };

    ध्रुव पूर्णाङ्क n = nodes.size();

    std::vector<पूर्णाङ्क> res;
    res.reserve(n);

    std::vector<तार्किक> used(n, मिच्छा);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    अत्थाय (पूर्णाङ्क i0 = 0; i0 < n; i0++) {
        सचे (used[i0]) {
            अनुवर्त;
        }

        ध्रुव auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        सचे (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr पूर्णाङ्क N_FORWARD = 64;

            अत्थाय (पूर्णाङ्क i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                सचे (used[i1]) {
                    अनुवर्त;
                }

                ध्रुव auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                सचे (!h_safe(node1.op())) {
                    भङ्ग;
                }

                ध्रुव तार्किक is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                सचे ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = सच्च;
                } अञ्ञथा {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    पटिनिवत्त res;
}

शून्य ggml_graph_optimize(ggml_cgraph * gf) {
    ध्रुव पूर्णाङ्क n = gf->n_nodes;

    std::vector<ग्रन्थिः_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    अत्थाय (पूर्णाङ्क i = 0; i < n; i++) {
        ग्रन्थिः_info ग्रन्थिः = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ध्रुव पूर्णाङ्क f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        अत्थाय (पूर्णाङ्क k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            ग्रन्थिः.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(ग्रन्थिः));
    }

#सचे 1
    // reorder to improve concurrency
    ध्रुव auto order = ggml_metal_graph_optimize_reorder(nodes);
#अञ्ञथा
    std::vector<पूर्णाङ्क> order(nodes.size());
    अत्थाय (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        पूर्णाङ्क j = 0;
        अत्थाय (ध्रुव auto i : order) {
            ध्रुव auto & ग्रन्थिः = nodes[i];

            gf->nodes[j++] = ग्रन्थिः.ग्रन्थिः;

            अत्थाय (auto * fused : ग्रन्थिः.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
