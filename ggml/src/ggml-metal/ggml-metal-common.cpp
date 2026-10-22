#ಅನ್ತರ್ಭಾವ "ggml-metal-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-fusion.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fwht_supported_ಗಾತ್ರ(int64_t n, size_t max_tg_mem) {
    ಆದರೆ (n == 64 || n == 128 || n == 256 || n == 512) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ಹಿಂತಿರುಗಿ (size_t) n * ಗಾತ್ರ(ದಶಮಾಂಶ) <= max_tg_mem;
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_use_fwht(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, size_t max_tg_mem) {
    ಹಿಂತಿರುಗಿ ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_ಗಾತ್ರ(op->src[1]->ne[0], max_tg_mem);
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_use_mm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ತಾರ್ಕಿಕ has_simdgroup_mm) {
    ಸ್ಥಿರಾಂಕ int64_t ne00 = op->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne11 = op->src[1]->ne[1];

    ಹಿಂತಿರುಗಿ !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_id_use_mm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ತಾರ್ಕಿಕ has_simdgroup_mm) {
    ಸ್ಥಿರಾಂಕ int64_t ne00 = op->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = op->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne21 = op->src[2]->ne[1];

    ಹಿಂತಿರುಗಿ has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_ಅಡ್ಡಸಾಲುಗಳು_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_TILE_ಅಡ್ಡಸಾಲುಗಳು = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

ಎಣಿಕೆ ggml_metal_mma_ಬಗೆ ggml_metal_mul_mv_mma_ಬಗೆ(ಎಣಿಕೆ ggml_type type, ಪೂರ್ಣಾಂಕ rt) {
    ಆದರೆ (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ಹಿಂತಿರುಗಿ GGML_METAL_MMA_ಬಗೆ_BLK;
    }
    ಹಿಂತಿರುಗಿ type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_ಬಗೆ_Q5_K : GGML_METAL_MMA_ಬಗೆ_GEN;
}

ಪೂರ್ಣಾಂಕ ggml_metal_mul_mv_mma_rt(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಹಿಂತಿರುಗಿ op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ಅಡ್ಡಸಾಲುಗಳು ? 2 : 1;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_mul_mv_mma_type_supported(ಎಣಿಕೆ ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

int64_t ggml_metal_mul_mv_mma_k_ಹೆಜ್ಜೆ(ಎಣಿಕೆ ggml_type type, ಪೂರ್ಣಾಂಕ rt) {
    ಆದರೆ (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಹಿಂತಿರುಗಿ ggml_metal_mul_mv_mma_ಬಗೆ(type, rt) == GGML_METAL_MMA_ಬಗೆ_BLK ? ggml_blck_ಗಾತ್ರ(type) : GGML_METAL_MMA_K_CHUNK;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_mul_mat_mma_type_ok(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ int64_t ಹೆಜ್ಜೆ = ggml_metal_mul_mv_mma_k_ಹೆಜ್ಜೆ(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ಹಿಂತಿರುಗಿ ಹೆಜ್ಜೆ > 0 && src0->ne[0] % ಹೆಜ್ಜೆ == 0 && src0->nb[0] == ggml_type_ಗಾತ್ರ(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
ಸ್ಥಿರ int64_t ggml_metal_mul_mv_mma_ಅಡ್ಡಸಾಲುಗಳು_min(ಎಣಿಕೆ ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ 6;
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            ಹಿಂತಿರುಗಿ 5;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ 4;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            ಹಿಂತಿರುಗಿ 3;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ 2;
    }
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_use_mma(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ಹಿಂತಿರುಗಿ ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_ಅಡ್ಡಸಾಲುಗಳು_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ಅಡ್ಡಸಾಲುಗಳು_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

ತಾರ್ಕಿಕ ggml_metal_op_mul_mat_may_use_mma(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಹಿಂತಿರುಗಿ ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
ರಚನೆ ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; //ಕೊನೆ

    ggml_mem_range_type pt;
};

ರಚನೆ ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    ಪೂರ್ಣಾಂಕ debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(ಪೂರ್ಣಾಂಕ debug) {
    ಸ್ವಯಂಚಾಲಿತ * res = ಹೊಸ ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ಹಿಂತಿರುಗಿ res;
}

ಶೂನ್ಯ ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    ಅಳಿಸು mrs;
}

ಶೂನ್ಯ ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_ಕೂಡಿಸು(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ggml_mem_range ggml_mem_range_from_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    ಆದರೆ (tensor->ಬಫರ್) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->ಬಫರ್,
            /*.p0 =*/ (uint64_t) tensor->ದತ್ತಾಂಶ,
            /*.p1 =*/ (uint64_t) tensor->ದತ್ತಾಂಶ + ggml_backend_buft_get_alloc_ಗಾತ್ರ(tensor->ಬಫರ್->buft, tensor),
            /*.pt =*/ pt,
        };
    } ಇಲ್ಲದಿದ್ದರೆ {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    ಹಿಂತಿರುಗಿ mr;
}

ಸ್ಥಿರ ggml_mem_range ggml_mem_range_from_tensor_src(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

ಸ್ಥಿರ ggml_mem_range ggml_mem_range_from_tensor_dst(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_ಕೂಡಿಸು_src(ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ಆದರೆ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ಹಿಂತಿರುಗಿ ggml_mem_ranges_ಕೂಡಿಸು(mrs, mr);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_ಕೂಡಿಸು_dst(ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ಆದರೆ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ಹಿಂತಿರುಗಿ ggml_mem_ranges_ಕೂಡಿಸು(mrs, mr);
}

ತಾರ್ಕಿಕ ggml_mem_ranges_ಕೂಡಿಸು(ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (tensor->src[i]) {
            ggml_mem_ranges_ಕೂಡಿಸು_src(mrs, tensor->src[i]);
        }
    }

    ಹಿಂತಿರುಗಿ ggml_mem_ranges_ಕೂಡಿಸು_dst(mrs, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    ಗಾಗಿ (size_t i = 0; i < mrs->ranges.size(); i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        ಆದರೆ (mr.pb != cmp.pb) {
            ಮುಂದುವರಿ;
        }

        // intersecting source ranges are allowed
        ಆದರೆ (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            ಆದರೆ (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ res = ggml_mem_ranges_check(mrs, mr);

    ಹಿಂತಿರುಗಿ res;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ res = ggml_mem_ranges_check(mrs, mr);

    ಹಿಂತಿರುಗಿ res;
}

ತಾರ್ಕಿಕ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (tensor->src[i]) {
            ಆದರೆ (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }

    ಹಿಂತಿರುಗಿ ggml_mem_ranges_check_dst(mrs, tensor);
}

ರಚನೆ ಗಂಟು_info {
    ggml_tensor * ಗಂಟು;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ಗಂಟು->op;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * dst() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ fused.ಖಾಲಿಗಳು() ? ಗಂಟು : fused.back();
    }

    ತಾರ್ಕಿಕ is_ಖಾಲಿಗಳು() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ggml_op_is_ಖಾಲಿಗಳು(ಗಂಟು->op);
    }

    ಶೂನ್ಯ ಕೂಡಿಸು_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    ತಾರ್ಕಿಕ is_output(ಸ್ಥಿರಾಂಕ ggml_tensor * t) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (t == ಗಂಟು) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * f : fused) {
            ಆದರೆ (t == f) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
};

ಸ್ಥಿರ std::vector<ಪೂರ್ಣಾಂಕ> ggml_metal_ನಕ್ಷೆ_optimize_reorder(ಸ್ಥಿರಾಂಕ std::vector<ಗಂಟು_info> & nodes) {
    // helper to add node src and dst ranges
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & h_ಕೂಡಿಸು = [](ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ಗಂಟು_info & ಗಂಟು) {
        // only external sources matter: sources produced by the fused group are internal
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * src = ಗಂಟು.ಗಂಟು->src[i];
            ಆದರೆ (src && !ಗಂಟು.is_output(src)) {
                ಆದರೆ (!ggml_mem_ranges_ಕೂಡಿಸು_src(mrs, src)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        }

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src = fused->src[i];
                ಆದರೆ (src && !ಗಂಟು.is_output(src)) {
                    ಆದರೆ (!ggml_mem_ranges_ಕೂಡಿಸು_src(mrs, src)) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        ಆದರೆ (!ggml_mem_ranges_ಕೂಡಿಸು_dst(mrs, ಗಂಟು.ಗಂಟು)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
            ಆದರೆ (!ggml_mem_ranges_ಕೂಡಿಸು_dst(mrs, fused)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & h_check = [](ggml_mem_ranges_t mrs, ಸ್ಥಿರಾಂಕ ಗಂಟು_info & ಗಂಟು) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * src = ಗಂಟು.ಗಂಟು->src[i];
            ಆದರೆ (src && !ಗಂಟು.is_output(src)) {
                ಆದರೆ (!ggml_mem_ranges_check_src(mrs, src)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        }

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src = fused->src[i];
                ಆದರೆ (src && !ಗಂಟು.is_output(src)) {
                    ಆದರೆ (!ggml_mem_ranges_check_src(mrs, src)) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
            }
        }

        ಆದರೆ (!ggml_mem_ranges_check_dst(mrs, ಗಂಟು.ಗಂಟು)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
            ಆದರೆ (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & h_ಸುರಕ್ಷಿತ = [](ggml_op op) {
        ಆಯ್ಕೆ (op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ಸ್ಥಿತಿ GGML_OP_ROPE:
            ಸ್ಥಿತಿ GGML_OP_NORM:
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            ಸ್ಥಿತಿ GGML_OP_CLAMP:
            ಸ್ಥಿತಿ GGML_OP_TRI:
            ಸ್ಥಿತಿ GGML_OP_DIAG:
            ಸ್ಥಿತಿ GGML_OP_MUL:
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
            ಸ್ಥಿತಿ GGML_OP_SUB:
            ಸ್ಥಿತಿ GGML_OP_DIV:
            ಸ್ಥಿತಿ GGML_OP_GLU:
            ಸ್ಥಿತಿ GGML_OP_SCALE:
            ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            ಸ್ಥಿತಿ GGML_OP_CPY:
            ಸ್ಥಿತಿ GGML_OP_CONT:
            ಸ್ಥಿತಿ GGML_OP_REPEAT:
                ಹಿಂತಿರುಗಿ ನಿಜ;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ ggml_op_is_ಖಾಲಿಗಳು(op);
        }
    };

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = nodes.size();

    std::vector<ಪೂರ್ಣಾಂಕ> res;
    res.reserve(n);

    std::vector<ತಾರ್ಕಿಕ> ಬಳಸಿದ(n, ಸುಳ್ಳು);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < n; i0++) {
        ಆದರೆ (ಬಳಸಿದ[i0]) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        ಆದರೆ (!node0.is_ಖಾಲಿಗಳು() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_ಕೂಡಿಸು(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr ಪೂರ್ಣಾಂಕ N_FORWARD = 64;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                ಆದರೆ (ಬಳಸಿದ[i1]) {
                    ಮುಂದುವರಿ;
                }

                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & node1 = nodes[i1];

                // disallow reordering of certain ops
                ಆದರೆ (!h_ಸುರಕ್ಷಿತ(node1.op())) {
                    ನಿಲ್ಲಿಸು;
                }

                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಖಾಲಿಗಳು = node1.is_ಖಾಲಿಗಳು();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                ಆದರೆ ((is_ಖಾಲಿಗಳು || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_ಕೂಡಿಸು(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    ಬಳಸಿದ[i1] = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // expand the set of nodes that haven't been processed yet
                    h_ಕೂಡಿಸು(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_ಕೂಡಿಸು(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    ಹಿಂತಿರುಗಿ res;
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_optimize(ggml_cgraph * gf) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = gf->n_nodes;

    std::vector<ಗಂಟು_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ಗಂಟು_info ಗಂಟು = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            ಗಂಟು.ಕೂಡಿಸು_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(ಗಂಟು));
    }

#ಆದರೆ 1
    // reorder to improve concurrency
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕ್ರಮ = ggml_metal_ನಕ್ಷೆ_optimize_reorder(nodes);
#ಇಲ್ಲದಿದ್ದರೆ
    std::vector<ಪೂರ್ಣಾಂಕ> ಕ್ರಮ(nodes.size());
    ಗಾಗಿ (size_t i = 0; i < nodes.size(); i++) {
        ಕ್ರಮ[i] = i;
    }
#ಆದರೆ_ಅಂತ್ಯ

    // unfuse
    {
        ಪೂರ್ಣಾಂಕ j = 0;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ i : ಕ್ರಮ) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಗಂಟು = nodes[i];

            gf->nodes[j++] = ಗಂಟು.ಗಂಟು;

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
