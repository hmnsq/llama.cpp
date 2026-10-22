#ಅನ್ತರ್ಭಾವ "ggml-metal-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-fusion.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    ಯದಿ (n == 64 || n == 128 || n == 256 || n == 512) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ಪ್ರತ್ಯಾಗಮ (size_t) n * ಪರಿಮಾಣ(ಭಿನ್ನ) <= max_tg_mem;
    }

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_use_fwht(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, size_t max_tg_mem) {
    ಪ್ರತ್ಯಾಗಮ ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_use_mm(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, ತಾರ್ಕಿಕ has_simdgroup_mm) {
    ಧ್ರುವ int64_t ne00 = op->src[0]->ne[0];
    ಧ್ರುವ int64_t ne11 = op->src[1]->ne[1];

    ಪ್ರತ್ಯಾಗಮ !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_id_use_mm(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, ತಾರ್ಕಿಕ has_simdgroup_mm) {
    ಧ್ರುವ int64_t ne00 = op->src[0]->ne[0];
    ಧ್ರುವ int64_t ne02 = op->src[0]->ne[2];
    ಧ್ರುವ int64_t ne21 = op->src[2]->ne[1];

    ಪ್ರತ್ಯಾಗಮ has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

ಗಣನಾ ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(ಗಣನಾ ggml_type type, ಪೂರ್ಣಾಙ್ಕ rt) {
    ಯದಿ (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ಪ್ರತ್ಯಾಗಮ GGML_METAL_MMA_KIND_BLK;
    }
    ಪ್ರತ್ಯಾಗಮ type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_mul_mv_mma_rt(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಪ್ರತ್ಯಾಗಮ op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_mul_mv_mma_type_supported(ಗಣನಾ ggml_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_BF16:
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ3_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_M:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
        ಪಕ್ಷ GGML_TYPE_TQ2_0:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(ಗಣನಾ ggml_type type, ಪೂರ್ಣಾಙ್ಕ rt) {
    ಯದಿ (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_mul_mat_mma_type_ok(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ಪ್ರತ್ಯಾಗಮ step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
ಸ್ಥಿರ int64_t ggml_metal_mul_mv_mma_rows_min(ಗಣನಾ ggml_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            ಪ್ರತ್ಯಾಗಮ 6;
        ಪಕ್ಷ GGML_TYPE_TQ2_0:
            ಪ್ರತ್ಯಾಗಮ 5;
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಪ್ರತ್ಯಾಗಮ 4;
        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
            ಪ್ರತ್ಯಾಗಮ 3;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ 2;
    }
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_use_mma(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ಧ್ರುವ ತಾರ್ಕಿಕ batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == ಪರಿಮಾಣ(ಭಿನ್ನ) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_may_use_mma(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಪ್ರತ್ಯಾಗಮ ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
ಸಂರಚನಾ ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

ಸಂರಚನಾ ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    ಪೂರ್ಣಾಙ್ಕ debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(ಪೂರ್ಣಾಙ್ಕ debug) {
    auto * res = ನೂತನ ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ಪ್ರತ್ಯಾಗಮ res;
}

ಶೂನ್ಯ ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

ಶೂನ್ಯ ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ggml_mem_range ggml_mem_range_from_tensor(ಧ್ರುವ ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    ಯದಿ (tensor->buffer) {
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
    } ಅನ್ಯಥಾ {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    ಪ್ರತ್ಯಾಗಮ mr;
}

ಸ್ಥಿರ ggml_mem_range ggml_mem_range_from_tensor_src(ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

ಸ್ಥಿರ ggml_mem_range ggml_mem_range_from_tensor_dst(ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, ಧ್ರುವ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ಯದಿ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ಪ್ರತ್ಯಾಗಮ ggml_mem_ranges_add(mrs, mr);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, ಧ್ರುವ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ಯದಿ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ಪ್ರತ್ಯಾಗಮ ggml_mem_ranges_add(mrs, mr);
}

ತಾರ್ಕಿಕ ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    ಪ್ರತ್ಯಾಗಮ ggml_mem_ranges_add_dst(mrs, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    ಪ್ರತಿ (size_t i = 0; i < mrs->ranges.size(); i++) {
        ಧ್ರುವ auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        ಯದಿ (mr.pb != cmp.pb) {
            ಅನುವರ್ತ;
        }

        // intersecting source ranges are allowed
        ಯದಿ (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            ಯದಿ (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ಧ್ರುವ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ಧ್ರುವ ತಾರ್ಕಿಕ res = ggml_mem_ranges_check(mrs, mr);

    ಪ್ರತ್ಯಾಗಮ res;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ಧ್ರುವ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ಧ್ರುವ ತಾರ್ಕಿಕ res = ggml_mem_ranges_check(mrs, mr);

    ಪ್ರತ್ಯಾಗಮ res;
}

ತಾರ್ಕಿಕ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (tensor->src[i]) {
            ಯದಿ (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ ggml_mem_ranges_check_dst(mrs, tensor);
}

ಸಂರಚನಾ ಗಂಟು_info {
    ggml_tensor * ಗಂಟು;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op;
    }

    ಧ್ರುವ ggml_tensor * dst() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ fused.empty() ? ಗಂಟು : fused.back();
    }

    ತಾರ್ಕಿಕ is_empty() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ggml_op_is_empty(ಗಂಟು->op);
    }

    ಶೂನ್ಯ add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    ತಾರ್ಕಿಕ is_output(ಧ್ರುವ ggml_tensor * t) ಧ್ರುವ {
        ಯದಿ (t == ಗಂಟು) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತಿ (ಧ್ರುವ auto * f : fused) {
            ಯದಿ (t == f) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
};

ಸ್ಥಿರ std::vector<ಪೂರ್ಣಾಙ್ಕ> ggml_metal_graph_optimize_reorder(ಧ್ರುವ std::vector<ಗಂಟು_info> & nodes) {
    // helper to add node src and dst ranges
    ಧ್ರುವ auto & h_add = [](ggml_mem_ranges_t mrs, ಧ್ರುವ ಗಂಟು_info & ಗಂಟು) {
        // only external sources matter: sources produced by the fused group are internal
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಧ್ರುವ ggml_tensor * src = ಗಂಟು.ಗಂಟು->src[i];
            ಯದಿ (src && !ಗಂಟು.is_output(src)) {
                ಯದಿ (!ggml_mem_ranges_add_src(mrs, src)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        }

        ಪ್ರತಿ (ಧ್ರುವ auto * fused : ಗಂಟು.fused) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
                ಧ್ರುವ ggml_tensor * src = fused->src[i];
                ಯದಿ (src && !ಗಂಟು.is_output(src)) {
                    ಯದಿ (!ggml_mem_ranges_add_src(mrs, src)) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        ಯದಿ (!ggml_mem_ranges_add_dst(mrs, ಗಂಟು.ಗಂಟು)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತಿ (ಧ್ರುವ auto * fused : ಗಂಟು.fused) {
            ಯದಿ (!ggml_mem_ranges_add_dst(mrs, fused)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ಧ್ರುವ auto & h_check = [](ggml_mem_ranges_t mrs, ಧ್ರುವ ಗಂಟು_info & ಗಂಟು) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಧ್ರುವ ggml_tensor * src = ಗಂಟು.ಗಂಟು->src[i];
            ಯದಿ (src && !ಗಂಟು.is_output(src)) {
                ಯದಿ (!ggml_mem_ranges_check_src(mrs, src)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        }

        ಪ್ರತಿ (ಧ್ರುವ auto * fused : ಗಂಟು.fused) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
                ಧ್ರುವ ggml_tensor * src = fused->src[i];
                ಯದಿ (src && !ಗಂಟು.is_output(src)) {
                    ಯದಿ (!ggml_mem_ranges_check_src(mrs, src)) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
            }
        }

        ಯದಿ (!ggml_mem_ranges_check_dst(mrs, ಗಂಟು.ಗಂಟು)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತಿ (ಧ್ರುವ auto * fused : ಗಂಟು.fused) {
            ಯದಿ (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ಧ್ರುವ auto & h_safe = [](ggml_op op) {
        ವಿಕಲ್ಪ (op) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            ಪಕ್ಷ GGML_OP_ROPE:
            ಪಕ್ಷ GGML_OP_NORM:
            ಪಕ್ಷ GGML_OP_RMS_NORM:
            ಪಕ್ಷ GGML_OP_GROUP_NORM:
            ಪಕ್ಷ GGML_OP_L2_NORM:
            ಪಕ್ಷ GGML_OP_SUM_ROWS:
            ಪಕ್ಷ GGML_OP_SSM_CONV:
            ಪಕ್ಷ GGML_OP_SSM_SCAN:
            ಪಕ್ಷ GGML_OP_CLAMP:
            ಪಕ್ಷ GGML_OP_TRI:
            ಪಕ್ಷ GGML_OP_DIAG:
            ಪಕ್ಷ GGML_OP_MUL:
            ಪಕ್ಷ GGML_OP_ADD:
            ಪಕ್ಷ GGML_OP_SUB:
            ಪಕ್ಷ GGML_OP_DIV:
            ಪಕ್ಷ GGML_OP_GLU:
            ಪಕ್ಷ GGML_OP_SCALE:
            ಪಕ್ಷ GGML_OP_UNARY:
            ಪಕ್ಷ GGML_OP_GET_ROWS:
            ಪಕ್ಷ GGML_OP_SET_ROWS:
            ಪಕ್ಷ GGML_OP_SET:
            ಪಕ್ಷ GGML_OP_CPY:
            ಪಕ್ಷ GGML_OP_CONT:
            ಪಕ್ಷ GGML_OP_REPEAT:
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ ggml_op_is_empty(op);
        }
    };

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = nodes.size();

    std::vector<ಪೂರ್ಣಾಙ್ಕ> res;
    res.reserve(n);

    std::vector<ತಾರ್ಕಿಕ> used(n, ಅಸತ್ಯ);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < n; i0++) {
        ಯದಿ (used[i0]) {
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        ಯದಿ (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr ಪೂರ್ಣಾಙ್ಕ N_FORWARD = 64;

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                ಯದಿ (used[i1]) {
                    ಅನುವರ್ತ;
                }

                ಧ್ರುವ auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                ಯದಿ (!h_safe(node1.op())) {
                    ಭಙ್ಗ;
                }

                ಧ್ರುವ ತಾರ್ಕಿಕ is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                ಯದಿ ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = ಸತ್ಯ;
                } ಅನ್ಯಥಾ {
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

    ಪ್ರತ್ಯಾಗಮ res;
}

ಶೂನ್ಯ ggml_graph_optimize(ggml_cgraph * gf) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = gf->n_nodes;

    std::vector<ಗಂಟು_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        ಗಂಟು_info ಗಂಟು = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            ಗಂಟು.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(ಗಂಟು));
    }

#ಯದಿ 1
    // reorder to improve concurrency
    ಧ್ರುವ auto order = ggml_metal_graph_optimize_reorder(nodes);
#ಅನ್ಯಥಾ
    std::vector<ಪೂರ್ಣಾಙ್ಕ> order(nodes.size());
    ಪ್ರತಿ (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        ಪೂರ್ಣಾಙ್ಕ j = 0;
        ಪ್ರತಿ (ಧ್ರುವ auto i : order) {
            ಧ್ರುವ auto & ಗಂಟು = nodes[i];

            gf->nodes[j++] = ಗಂಟು.ಗಂಟು;

            ಪ್ರತಿ (auto * fused : ಗಂಟು.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
