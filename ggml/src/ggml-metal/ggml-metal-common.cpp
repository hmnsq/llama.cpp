#অন্তর্ভাৱ "ggml-metal-common.h"
#অন্তর্ভাৱ "ggml-metal-fusion.h"

#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"

#অন্তর্ভাৱ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
স্থির তার্কিক ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    যদি (n == 64 || n == 128 || n == 256 || n == 512) {
        ঘূৰাই সঁচা;
    }

    যদি (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ঘূৰাই (size_t) n * পরিমাণ(ভিন্ন) <= max_tg_mem;
    }

    ঘূৰাই মিছা;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
তার্কিক ggml_metal_op_mul_mat_use_fwht(ধ্রুৱ সংরচনা ggml_tensor * op, size_t max_tg_mem) {
    ঘূৰাই ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

তার্কিক ggml_metal_op_mul_mat_use_mm(ধ্রুৱ সংরচনা ggml_tensor * op, তার্কিক has_simdgroup_mm) {
    ধ্রুৱ int64_t ne00 = op->src[0]->ne[0];
    ধ্রুৱ int64_t ne11 = op->src[1]->ne[1];

    ঘূৰাই !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

তার্কিক ggml_metal_op_mul_mat_id_use_mm(ধ্রুৱ সংরচনা ggml_tensor * op, তার্কিক has_simdgroup_mm) {
    ধ্রুৱ int64_t ne00 = op->src[0]->ne[0];
    ধ্রুৱ int64_t ne02 = op->src[0]->ne[2];
    ধ্রুৱ int64_t ne21 = op->src[2]->ne[1];

    ঘূৰাই has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
স্থির constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
স্থির constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
স্থির constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

গণনা ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(গণনা ggml_type type, পূর্ণাঙ্ক rt) {
    যদি (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ঘূৰাই GGML_METAL_MMA_KIND_BLK;
    }
    ঘূৰাই type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

পূর্ণাঙ্ক ggml_metal_mul_mv_mma_rt(ধ্রুৱ সংরচনা ggml_tensor * op) {
    ঘূৰাই op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

স্থির তার্কিক ggml_metal_mul_mv_mma_type_supported(গণনা ggml_type type) {
    ৱিকল্প (type) {
        পক্ষ GGML_TYPE_F32:
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_BF16:
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ2_S:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_TQ2_0:
            ঘূৰাই সঁচা;
        স্ৱভাৱ:
            ঘূৰাই মিছা;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(গণনা ggml_type type, পূর্ণাঙ্ক rt) {
    যদি (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ঘূৰাই 0;
    }
    ঘূৰাই ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

স্থির তার্কিক ggml_metal_mul_mat_mma_type_ok(ধ্রুৱ সংরচনা ggml_tensor * op) {
    ধ্রুৱ ggml_tensor * src0 = op->src[0];
    ধ্রুৱ int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ঘূৰাই step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
স্থির int64_t ggml_metal_mul_mv_mma_rows_min(গণনা ggml_type type) {
    ৱিকল্প (type) {
        পক্ষ GGML_TYPE_F32:
            ঘূৰাই 6;
        পক্ষ GGML_TYPE_TQ2_0:
            ঘূৰাই 5;
        পক্ষ GGML_TYPE_BF16:
            ঘূৰাই 4;
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_MXFP4:
            ঘূৰাই 3;
        স্ৱভাৱ:
            ঘূৰাই 2;
    }
}

তার্কিক ggml_metal_op_mul_mat_use_mma(ধ্রুৱ সংরচনা ggml_tensor * op) {
    ধ্রুৱ ggml_tensor * src0 = op->src[0];
    ধ্রুৱ ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ধ্রুৱ তার্কিক batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ঘূৰাই ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == পরিমাণ(ভিন্ন) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

তার্কিক ggml_metal_op_mul_mat_may_use_mma(ধ্রুৱ সংরচনা ggml_tensor * op) {
    ঘূৰাই ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
সংরচনা ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

সংরচনা ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    পূর্ণাঙ্ক debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(পূর্ণাঙ্ক debug) {
    auto * res = নূতন ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ঘূৰাই res;
}

শূন্য ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

শূন্য ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

স্থির তার্কিক ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ঘূৰাই সঁচা;
}

স্থির ggml_mem_range ggml_mem_range_from_tensor(ধ্রুৱ ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    যদি (tensor->buffer) {
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

    ঘূৰাই mr;
}

স্থির ggml_mem_range ggml_mem_range_from_tensor_src(ধ্রুৱ ggml_tensor * tensor) {
    ঘূৰাই ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

স্থির ggml_mem_range ggml_mem_range_from_tensor_dst(ধ্রুৱ ggml_tensor * tensor) {
    ঘূৰাই ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

স্থির তার্কিক ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, ধ্রুৱ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    যদি (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ঘূৰাই ggml_mem_ranges_add(mrs, mr);
}

স্থির তার্কিক ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, ধ্রুৱ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    যদি (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ঘূৰাই ggml_mem_ranges_add(mrs, mr);
}

তার্কিক ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ধ্রুৱ ggml_tensor * tensor) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        যদি (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    ঘূৰাই ggml_mem_ranges_add_dst(mrs, tensor);
}

স্থির তার্কিক ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    বাবে (size_t i = 0; i < mrs->ranges.size(); i++) {
        ধ্রুৱ auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        যদি (mr.pb != cmp.pb) {
            অনুৱর্ত;
        }

        // intersecting source ranges are allowed
        যদি (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            অনুৱর্ত;
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

            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

স্থির তার্কিক ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ধ্রুৱ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ধ্রুৱ তার্কিক res = ggml_mem_ranges_check(mrs, mr);

    ঘূৰাই res;
}

স্থির তার্কিক ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ধ্রুৱ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ধ্রুৱ তার্কিক res = ggml_mem_ranges_check(mrs, mr);

    ঘূৰাই res;
}

তার্কিক ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ধ্রুৱ ggml_tensor * tensor) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        যদি (tensor->src[i]) {
            যদি (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ঘূৰাই মিছা;
            }
        }
    }

    ঘূৰাই ggml_mem_ranges_check_dst(mrs, tensor);
}

সংরচনা গ্ৰন্থি_info {
    ggml_tensor * গ্ৰন্থি;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ধ্রুৱ {
        ঘূৰাই গ্ৰন্থি->op;
    }

    ধ্রুৱ ggml_tensor * dst() ধ্রুৱ {
        ঘূৰাই fused.empty() ? গ্ৰন্থি : fused.back();
    }

    তার্কিক is_empty() ধ্রুৱ {
        ঘূৰাই ggml_op_is_empty(গ্ৰন্থি->op);
    }

    শূন্য add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    তার্কিক is_output(ধ্রুৱ ggml_tensor * t) ধ্রুৱ {
        যদি (t == গ্ৰন্থি) {
            ঘূৰাই সঁচা;
        }
        বাবে (ধ্রুৱ auto * f : fused) {
            যদি (t == f) {
                ঘূৰাই সঁচা;
            }
        }
        ঘূৰাই মিছা;
    }
};

স্থির std::vector<পূর্ণাঙ্ক> ggml_metal_graph_optimize_reorder(ধ্রুৱ std::vector<গ্ৰন্থি_info> & nodes) {
    // helper to add node src and dst ranges
    ধ্রুৱ auto & h_add = [](ggml_mem_ranges_t mrs, ধ্রুৱ গ্ৰন্থি_info & গ্ৰন্থি) {
        // only external sources matter: sources produced by the fused group are internal
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            ধ্রুৱ ggml_tensor * src = গ্ৰন্থি.গ্ৰন্থি->src[i];
            যদি (src && !গ্ৰন্থি.is_output(src)) {
                যদি (!ggml_mem_ranges_add_src(mrs, src)) {
                    ঘূৰাই মিছা;
                }
            }
        }

        বাবে (ধ্রুৱ auto * fused : গ্ৰন্থি.fused) {
            বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
                ধ্রুৱ ggml_tensor * src = fused->src[i];
                যদি (src && !গ্ৰন্থি.is_output(src)) {
                    যদি (!ggml_mem_ranges_add_src(mrs, src)) {
                        ঘূৰাই মিছা;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        যদি (!ggml_mem_ranges_add_dst(mrs, গ্ৰন্থি.গ্ৰন্থি)) {
            ঘূৰাই মিছা;
        }
        বাবে (ধ্রুৱ auto * fused : গ্ৰন্থি.fused) {
            যদি (!ggml_mem_ranges_add_dst(mrs, fused)) {
                ঘূৰাই মিছা;
            }
        }

        ঘূৰাই সঁচা;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ধ্রুৱ auto & h_check = [](ggml_mem_ranges_t mrs, ধ্রুৱ গ্ৰন্থি_info & গ্ৰন্থি) {
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            ধ্রুৱ ggml_tensor * src = গ্ৰন্থি.গ্ৰন্থি->src[i];
            যদি (src && !গ্ৰন্থি.is_output(src)) {
                যদি (!ggml_mem_ranges_check_src(mrs, src)) {
                    ঘূৰাই মিছা;
                }
            }
        }

        বাবে (ধ্রুৱ auto * fused : গ্ৰন্থি.fused) {
            বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
                ধ্রুৱ ggml_tensor * src = fused->src[i];
                যদি (src && !গ্ৰন্থি.is_output(src)) {
                    যদি (!ggml_mem_ranges_check_src(mrs, src)) {
                        ঘূৰাই মিছা;
                    }
                }
            }
        }

        যদি (!ggml_mem_ranges_check_dst(mrs, গ্ৰন্থি.গ্ৰন্থি)) {
            ঘূৰাই মিছা;
        }
        বাবে (ধ্রুৱ auto * fused : গ্ৰন্থি.fused) {
            যদি (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ঘূৰাই মিছা;
            }
        }

        ঘূৰাই সঁচা;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ধ্রুৱ auto & h_safe = [](ggml_op op) {
        ৱিকল্প (op) {
            পক্ষ GGML_OP_MUL_MAT:
            পক্ষ GGML_OP_MUL_MAT_ID:
            পক্ষ GGML_OP_ROPE:
            পক্ষ GGML_OP_NORM:
            পক্ষ GGML_OP_RMS_NORM:
            পক্ষ GGML_OP_GROUP_NORM:
            পক্ষ GGML_OP_L2_NORM:
            পক্ষ GGML_OP_SUM_ROWS:
            পক্ষ GGML_OP_SSM_CONV:
            পক্ষ GGML_OP_SSM_SCAN:
            পক্ষ GGML_OP_CLAMP:
            পক্ষ GGML_OP_TRI:
            পক্ষ GGML_OP_DIAG:
            পক্ষ GGML_OP_MUL:
            পক্ষ GGML_OP_ADD:
            পক্ষ GGML_OP_SUB:
            পক্ষ GGML_OP_DIV:
            পক্ষ GGML_OP_GLU:
            পক্ষ GGML_OP_SCALE:
            পক্ষ GGML_OP_UNARY:
            পক্ষ GGML_OP_GET_ROWS:
            পক্ষ GGML_OP_SET_ROWS:
            পক্ষ GGML_OP_SET:
            পক্ষ GGML_OP_CPY:
            পক্ষ GGML_OP_CONT:
            পক্ষ GGML_OP_REPEAT:
                ঘূৰাই সঁচা;
            স্ৱভাৱ:
                ঘূৰাই ggml_op_is_empty(op);
        }
    };

    ধ্রুৱ পূর্ণাঙ্ক n = nodes.size();

    std::vector<পূর্ণাঙ্ক> res;
    res.reserve(n);

    std::vector<তার্কিক> used(n, মিছা);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    বাবে (পূর্ণাঙ্ক i0 = 0; i0 < n; i0++) {
        যদি (used[i0]) {
            অনুৱর্ত;
        }

        ধ্রুৱ auto & node0 = nodes[i0];

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
            constexpr পূর্ণাঙ্ক N_FORWARD = 64;

            বাবে (পূর্ণাঙ্ক i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                যদি (used[i1]) {
                    অনুৱর্ত;
                }

                ধ্রুৱ auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                যদি (!h_safe(node1.op())) {
                    ভঙ্গ;
                }

                ধ্রুৱ তার্কিক is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                যদি ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = সঁচা;
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

    ঘূৰাই res;
}

শূন্য ggml_graph_optimize(ggml_cgraph * gf) {
    ধ্রুৱ পূর্ণাঙ্ক n = gf->n_nodes;

    std::vector<গ্ৰন্থি_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        গ্ৰন্থি_info গ্ৰন্থি = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ধ্রুৱ পূর্ণাঙ্ক f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        বাবে (পূর্ণাঙ্ক k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            গ্ৰন্থি.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(গ্ৰন্থি));
    }

#যদি 1
    // reorder to improve concurrency
    ধ্রুৱ auto order = ggml_metal_graph_optimize_reorder(nodes);
#অন্যথা
    std::vector<পূর্ণাঙ্ক> order(nodes.size());
    বাবে (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        পূর্ণাঙ্ক j = 0;
        বাবে (ধ্রুৱ auto i : order) {
            ধ্রুৱ auto & গ্ৰন্থি = nodes[i];

            gf->nodes[j++] = গ্ৰন্থি.গ্ৰন্থি;

            বাবে (auto * fused : গ্ৰন্থি.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
