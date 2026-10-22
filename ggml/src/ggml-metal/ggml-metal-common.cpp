#অন্তর্ভাব "ggml-metal-common.h"
#অন্তর্ভাব "ggml-metal-fusion.h"

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
স্থির যৌক্তিক ggml_metal_fwht_supported_আকার(int64_t n, size_t max_tg_mem) {
    যদি (n == 64 || n == 128 || n == 256 || n == 512) {
        ফেরত সত্য;
    }

    যদি (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ফেরত (size_t) n * আকার(দশমিক) <= max_tg_mem;
    }

    ফেরত মিথ্যা;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
যৌক্তিক ggml_metal_op_mul_mat_use_fwht(ধ্রুবক কাঠামো ggml_tensor * op, size_t max_tg_mem) {
    ফেরত ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_আকার(op->src[1]->ne[0], max_tg_mem);
}

যৌক্তিক ggml_metal_op_mul_mat_use_mm(ধ্রুবক কাঠামো ggml_tensor * op, যৌক্তিক has_simdgroup_mm) {
    ধ্রুবক int64_t ne00 = op->src[0]->ne[0];
    ধ্রুবক int64_t ne11 = op->src[1]->ne[1];

    ফেরত !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

যৌক্তিক ggml_metal_op_mul_mat_id_use_mm(ধ্রুবক কাঠামো ggml_tensor * op, যৌক্তিক has_simdgroup_mm) {
    ধ্রুবক int64_t ne00 = op->src[0]->ne[0];
    ধ্রুবক int64_t ne02 = op->src[0]->ne[2];
    ধ্রুবক int64_t ne21 = op->src[2]->ne[1];

    ফেরত has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
স্থির constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
স্থির constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
স্থির constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

গণনা ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(গণনা ggml_type type, পূর্ণসংখ্যা rt) {
    যদি (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ফেরত GGML_METAL_MMA_KIND_BLK;
    }
    ফেরত type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

পূর্ণসংখ্যা ggml_metal_mul_mv_mma_rt(ধ্রুবক কাঠামো ggml_tensor * op) {
    ফেরত op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

স্থির যৌক্তিক ggml_metal_mul_mv_mma_type_supported(গণনা ggml_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_F32:
        ক্ষেত্র GGML_TYPE_F16:
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_Q4_1:
        ক্ষেত্র GGML_TYPE_Q5_0:
        ক্ষেত্র GGML_TYPE_Q5_1:
        ক্ষেত্র GGML_TYPE_Q8_0:
        ক্ষেত্র GGML_TYPE_Q4_K:
        ক্ষেত্র GGML_TYPE_Q5_K:
        ক্ষেত্র GGML_TYPE_Q6_K:
        ক্ষেত্র GGML_TYPE_BF16:
        ক্ষেত্র GGML_TYPE_Q1_0:
        ক্ষেত্র GGML_TYPE_Q2_0:
        ক্ষেত্র GGML_TYPE_MXFP4:
        ক্ষেত্র GGML_TYPE_Q2_K:
        ক্ষেত্র GGML_TYPE_Q3_K:
        ক্ষেত্র GGML_TYPE_IQ2_XXS:
        ক্ষেত্র GGML_TYPE_IQ2_XS:
        ক্ষেত্র GGML_TYPE_IQ2_S:
        ক্ষেত্র GGML_TYPE_IQ3_XXS:
        ক্ষেত্র GGML_TYPE_IQ3_S:
        ক্ষেত্র GGML_TYPE_IQ1_S:
        ক্ষেত্র GGML_TYPE_IQ1_M:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
        ক্ষেত্র GGML_TYPE_IQ4_XS:
        ক্ষেত্র GGML_TYPE_TQ2_0:
            ফেরত সত্য;
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(গণনা ggml_type type, পূর্ণসংখ্যা rt) {
    যদি (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ফেরত 0;
    }
    ফেরত ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_আকার(type) : GGML_METAL_MMA_K_CHUNK;
}

স্থির যৌক্তিক ggml_metal_mul_mat_mma_type_ok(ধ্রুবক কাঠামো ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ধ্রুবক int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ফেরত step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_আকার(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
স্থির int64_t ggml_metal_mul_mv_mma_rows_min(গণনা ggml_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_F32:
            ফেরত 6;
        ক্ষেত্র GGML_TYPE_TQ2_0:
            ফেরত 5;
        ক্ষেত্র GGML_TYPE_BF16:
            ফেরত 4;
        ক্ষেত্র GGML_TYPE_F16:
        ক্ষেত্র GGML_TYPE_Q2_0:
        ক্ষেত্র GGML_TYPE_Q4_K:
        ক্ষেত্র GGML_TYPE_Q5_0:
        ক্ষেত্র GGML_TYPE_Q5_1:
        ক্ষেত্র GGML_TYPE_Q2_K:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
        ক্ষেত্র GGML_TYPE_MXFP4:
            ফেরত 3;
        পূর্বনির্ধারিত:
            ফেরত 2;
    }
}

যৌক্তিক ggml_metal_op_mul_mat_use_mma(ধ্রুবক কাঠামো ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ধ্রুবক ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ধ্রুবক যৌক্তিক batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ফেরত ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == আকার(দশমিক) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

যৌক্তিক ggml_metal_op_mul_mat_may_use_mma(ধ্রুবক কাঠামো ggml_tensor * op) {
    ফেরত ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
কাঠামো ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

কাঠামো ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    পূর্ণসংখ্যা debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(পূর্ণসংখ্যা debug) {
    auto * res = নতুন ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ফেরত res;
}

শূন্য ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

শূন্য ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

স্থির যৌক্তিক ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ফেরত সত্য;
}

স্থির ggml_mem_range ggml_mem_range_from_tensor(ধ্রুবক ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    যদি (tensor->বাফার) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->বাফার,
            /*.p0 =*/ (uint64_t) tensor->তথ্য,
            /*.p1 =*/ (uint64_t) tensor->তথ্য + ggml_backend_buft_get_alloc_আকার(tensor->বাফার->buft, tensor),
            /*.pt =*/ pt,
        };
    } অন্যথা {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    ফেরত mr;
}

স্থির ggml_mem_range ggml_mem_range_from_tensor_src(ধ্রুবক ggml_tensor * tensor) {
    ফেরত ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

স্থির ggml_mem_range ggml_mem_range_from_tensor_dst(ধ্রুবক ggml_tensor * tensor) {
    ফেরত ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

স্থির যৌক্তিক ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, ধ্রুবক ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    যদি (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ফেরত ggml_mem_ranges_add(mrs, mr);
}

স্থির যৌক্তিক ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, ধ্রুবক ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    যদি (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ফেরত ggml_mem_ranges_add(mrs, mr);
}

যৌক্তিক ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ধ্রুবক ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        যদি (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    ফেরত ggml_mem_ranges_add_dst(mrs, tensor);
}

স্থির যৌক্তিক ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    জন্য (size_t i = 0; i < mrs->ranges.size(); i++) {
        ধ্রুবক auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        যদি (mr.pb != cmp.pb) {
            চালু;
        }

        // intersecting source ranges are allowed
        যদি (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            চালু;
        }

        যদি (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            যদি (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ধ্রুবক ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ধ্রুবক যৌক্তিক res = ggml_mem_ranges_check(mrs, mr);

    ফেরত res;
}

স্থির যৌক্তিক ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ধ্রুবক ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ধ্রুবক যৌক্তিক res = ggml_mem_ranges_check(mrs, mr);

    ফেরত res;
}

যৌক্তিক ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ধ্রুবক ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        যদি (tensor->src[i]) {
            যদি (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ফেরত মিথ্যা;
            }
        }
    }

    ফেরত ggml_mem_ranges_check_dst(mrs, tensor);
}

কাঠামো গ্রন্থি_info {
    ggml_tensor * গ্রন্থি;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ধ্রুবক {
        ফেরত গ্রন্থি->op;
    }

    ধ্রুবক ggml_tensor * dst() ধ্রুবক {
        ফেরত fused.empty() ? গ্রন্থি : fused.back();
    }

    যৌক্তিক is_empty() ধ্রুবক {
        ফেরত ggml_op_is_empty(গ্রন্থি->op);
    }

    শূন্য add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    যৌক্তিক is_output(ধ্রুবক ggml_tensor * t) ধ্রুবক {
        যদি (t == গ্রন্থি) {
            ফেরত সত্য;
        }
        জন্য (ধ্রুবক auto * f : fused) {
            যদি (t == f) {
                ফেরত সত্য;
            }
        }
        ফেরত মিথ্যা;
    }
};

স্থির std::vector<পূর্ণসংখ্যা> ggml_metal_graph_optimize_reorder(ধ্রুবক std::vector<গ্রন্থি_info> & nodes) {
    // helper to add node src and dst ranges
    ধ্রুবক auto & h_add = [](ggml_mem_ranges_t mrs, ধ্রুবক গ্রন্থি_info & গ্রন্থি) {
        // only external sources matter: sources produced by the fused group are internal
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
            ধ্রুবক ggml_tensor * src = গ্রন্থি.গ্রন্থি->src[i];
            যদি (src && !গ্রন্থি.is_output(src)) {
                যদি (!ggml_mem_ranges_add_src(mrs, src)) {
                    ফেরত মিথ্যা;
                }
            }
        }

        জন্য (ধ্রুবক auto * fused : গ্রন্থি.fused) {
            জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
                ধ্রুবক ggml_tensor * src = fused->src[i];
                যদি (src && !গ্রন্থি.is_output(src)) {
                    যদি (!ggml_mem_ranges_add_src(mrs, src)) {
                        ফেরত মিথ্যা;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        যদি (!ggml_mem_ranges_add_dst(mrs, গ্রন্থি.গ্রন্থি)) {
            ফেরত মিথ্যা;
        }
        জন্য (ধ্রুবক auto * fused : গ্রন্থি.fused) {
            যদি (!ggml_mem_ranges_add_dst(mrs, fused)) {
                ফেরত মিথ্যা;
            }
        }

        ফেরত সত্য;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ধ্রুবক auto & h_check = [](ggml_mem_ranges_t mrs, ধ্রুবক গ্রন্থি_info & গ্রন্থি) {
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
            ধ্রুবক ggml_tensor * src = গ্রন্থি.গ্রন্থি->src[i];
            যদি (src && !গ্রন্থি.is_output(src)) {
                যদি (!ggml_mem_ranges_check_src(mrs, src)) {
                    ফেরত মিথ্যা;
                }
            }
        }

        জন্য (ধ্রুবক auto * fused : গ্রন্থি.fused) {
            জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
                ধ্রুবক ggml_tensor * src = fused->src[i];
                যদি (src && !গ্রন্থি.is_output(src)) {
                    যদি (!ggml_mem_ranges_check_src(mrs, src)) {
                        ফেরত মিথ্যা;
                    }
                }
            }
        }

        যদি (!ggml_mem_ranges_check_dst(mrs, গ্রন্থি.গ্রন্থি)) {
            ফেরত মিথ্যা;
        }
        জন্য (ধ্রুবক auto * fused : গ্রন্থি.fused) {
            যদি (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ফেরত মিথ্যা;
            }
        }

        ফেরত সত্য;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ধ্রুবক auto & h_safe = [](ggml_op op) {
        নির্বাচন (op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
            ক্ষেত্র GGML_OP_ROPE:
            ক্ষেত্র GGML_OP_NORM:
            ক্ষেত্র GGML_OP_RMS_NORM:
            ক্ষেত্র GGML_OP_GROUP_NORM:
            ক্ষেত্র GGML_OP_L2_NORM:
            ক্ষেত্র GGML_OP_যোগ_ROWS:
            ক্ষেত্র GGML_OP_SSM_CONV:
            ক্ষেত্র GGML_OP_SSM_SCAN:
            ক্ষেত্র GGML_OP_CLAMP:
            ক্ষেত্র GGML_OP_TRI:
            ক্ষেত্র GGML_OP_DIAG:
            ক্ষেত্র GGML_OP_MUL:
            ক্ষেত্র GGML_OP_ADD:
            ক্ষেত্র GGML_OP_SUB:
            ক্ষেত্র GGML_OP_DIV:
            ক্ষেত্র GGML_OP_GLU:
            ক্ষেত্র GGML_OP_SCALE:
            ক্ষেত্র GGML_OP_UNARY:
            ক্ষেত্র GGML_OP_GET_ROWS:
            ক্ষেত্র GGML_OP_SET_ROWS:
            ক্ষেত্র GGML_OP_SET:
            ক্ষেত্র GGML_OP_CPY:
            ক্ষেত্র GGML_OP_CONT:
            ক্ষেত্র GGML_OP_REPEAT:
                ফেরত সত্য;
            পূর্বনির্ধারিত:
                ফেরত ggml_op_is_empty(op);
        }
    };

    ধ্রুবক পূর্ণসংখ্যা n = nodes.size();

    std::vector<পূর্ণসংখ্যা> res;
    res.reserve(n);

    std::vector<যৌক্তিক> used(n, মিথ্যা);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    জন্য (পূর্ণসংখ্যা i0 = 0; i0 < n; i0++) {
        যদি (used[i0]) {
            চালু;
        }

        ধ্রুবক auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        যদি (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr পূর্ণসংখ্যা N_FORWARD = 64;

            জন্য (পূর্ণসংখ্যা i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                যদি (used[i1]) {
                    চালু;
                }

                ধ্রুবক auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                যদি (!h_safe(node1.op())) {
                    বিরতি;
                }

                ধ্রুবক যৌক্তিক is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                যদি ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = সত্য;
                } অন্যথা {
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

    ফেরত res;
}

শূন্য ggml_graph_optimize(ggml_cgraph * gf) {
    ধ্রুবক পূর্ণসংখ্যা n = gf->n_nodes;

    std::vector<গ্রন্থি_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        গ্রন্থি_info গ্রন্থি = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ধ্রুবক পূর্ণসংখ্যা f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        জন্য (পূর্ণসংখ্যা k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            গ্রন্থি.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(গ্রন্থি));
    }

#যদি 1
    // reorder to improve concurrency
    ধ্রুবক auto order = ggml_metal_graph_optimize_reorder(nodes);
#অন্যথা
    std::vector<পূর্ণসংখ্যা> order(nodes.size());
    জন্য (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        পূর্ণসংখ্যা j = 0;
        জন্য (ধ্রুবক auto i : order) {
            ধ্রুবক auto & গ্রন্থি = nodes[i];

            gf->nodes[j++] = গ্রন্থি.গ্রন্থি;

            জন্য (auto * fused : গ্রন্থি.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
