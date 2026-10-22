#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-fusion.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fwht_supported_ਮਾਪ(int64_t n, size_t max_tg_mem) {
    ਜੇ (n == 64 || n == 128 || n == 256 || n == 512) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ਵਾਪਸ (size_t) n * ਆਕਾਰ(ਦਸ਼ਮਲਵ) <= max_tg_mem;
    }

    ਵਾਪਸ ਝੂਠ;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
ਤਾਰਕਿਕ ggml_metal_op_mul_mat_use_fwht(ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t max_tg_mem) {
    ਵਾਪਸ ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_ਮਾਪ(op->src[1]->ne[0], max_tg_mem);
}

ਤਾਰਕਿਕ ggml_metal_op_mul_mat_use_mm(ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਤਾਰਕਿਕ has_simdgroup_mm) {
    ਅਚਲ int64_t ne00 = op->src[0]->ne[0];
    ਅਚਲ int64_t ne11 = op->src[1]->ne[1];

    ਵਾਪਸ !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

ਤਾਰਕਿਕ ggml_metal_op_mul_mat_id_use_mm(ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਤਾਰਕਿਕ has_simdgroup_mm) {
    ਅਚਲ int64_t ne00 = op->src[0]->ne[0];
    ਅਚਲ int64_t ne02 = op->src[0]->ne[2];
    ਅਚਲ int64_t ne21 = op->src[2]->ne[1];

    ਵਾਪਸ has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
ਸਥਿਰ constexpr int64_t GGML_METAL_MMA_ਪੰਕਤੀਆਂ_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
ਸਥਿਰ constexpr int64_t GGML_METAL_MMA_TILE_ਪੰਕਤੀਆਂ = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
ਸਥਿਰ constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

ਗਿਣਤੀ ggml_metal_mma_ਕਿਸਮ ggml_metal_mul_mv_mma_ਕਿਸਮ(ਗਿਣਤੀ ggml_type type, ਪੂਰਨ rt) {
    ਜੇ (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ਵਾਪਸ GGML_METAL_MMA_ਕਿਸਮ_BLK;
    }
    ਵਾਪਸ type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_ਕਿਸਮ_Q5_K : GGML_METAL_MMA_ਕਿਸਮ_GEN;
}

ਪੂਰਨ ggml_metal_mul_mv_mma_rt(ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਵਾਪਸ op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ਪੰਕਤੀਆਂ ? 2 : 1;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_mul_mv_mma_type_supported(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

int64_t ggml_metal_mul_mv_mma_k_ਕਦਮ(ਗਿਣਤੀ ggml_type type, ਪੂਰਨ rt) {
    ਜੇ (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ਵਾਪਸ 0;
    }
    ਵਾਪਸ ggml_metal_mul_mv_mma_ਕਿਸਮ(type, rt) == GGML_METAL_MMA_ਕਿਸਮ_BLK ? ggml_blck_ਮਾਪ(type) : GGML_METAL_MMA_K_CHUNK;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_mul_mat_mma_type_ok(ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ਅਚਲ int64_t ਕਦਮ = ggml_metal_mul_mv_mma_k_ਕਦਮ(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ਵਾਪਸ ਕਦਮ > 0 && src0->ne[0] % ਕਦਮ == 0 && src0->nb[0] == ggml_type_ਮਾਪ(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
ਸਥਿਰ int64_t ggml_metal_mul_mv_mma_ਪੰਕਤੀਆਂ_min(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ 6;
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            ਵਾਪਸ 5;
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ 4;
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_MXFP4:
            ਵਾਪਸ 3;
        ਮੂਲ:
            ਵਾਪਸ 2;
    }
}

ਤਾਰਕਿਕ ggml_metal_op_mul_mat_use_mma(ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ਅਚਲ ਤਾਰਕਿਕ batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ਵਾਪਸ ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_ਪੰਕਤੀਆਂ_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ਪੰਕਤੀਆਂ_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

ਤਾਰਕਿਕ ggml_metal_op_mul_mat_may_use_mma(ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਵਾਪਸ ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
ਬਣਤਰ ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; //ਸਿਰਾ

    ggml_mem_range_type pt;
};

ਬਣਤਰ ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    ਪੂਰਨ debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(ਪੂਰਨ debug) {
    ਸਵੈਚਾਲਿਤ * res = ਨਵਾਂ ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ਵਾਪਸ res;
}

ਖਾਲੀ ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    ਮਿਟਾਓ mrs;
}

ਖਾਲੀ ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_mem_ranges_ਜੋੜੋ(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ggml_mem_range ggml_mem_range_from_tensor(ਅਚਲ ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    ਜੇ (tensor->ਬਫਰ) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->ਬਫਰ,
            /*.p0 =*/ (uint64_t) tensor->ਅੰਕੜੇ,
            /*.p1 =*/ (uint64_t) tensor->ਅੰਕੜੇ + ggml_backend_buft_get_alloc_ਮਾਪ(tensor->ਬਫਰ->buft, tensor),
            /*.pt =*/ pt,
        };
    } ਨਹੀਂਤਾਂ {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    ਵਾਪਸ mr;
}

ਸਥਿਰ ggml_mem_range ggml_mem_range_from_tensor_src(ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

ਸਥਿਰ ggml_mem_range ggml_mem_range_from_tensor_dst(ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_mem_ranges_ਜੋੜੋ_src(ggml_mem_ranges_t mrs, ਅਚਲ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ਜੇ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ਵਾਪਸ ggml_mem_ranges_ਜੋੜੋ(mrs, mr);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_mem_ranges_ਜੋੜੋ_dst(ggml_mem_ranges_t mrs, ਅਚਲ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ਜੇ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ਵਾਪਸ ggml_mem_ranges_ਜੋੜੋ(mrs, mr);
}

ਤਾਰਕਿਕ ggml_mem_ranges_ਜੋੜੋ(ggml_mem_ranges_t mrs, ਅਚਲ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (tensor->src[i]) {
            ggml_mem_ranges_ਜੋੜੋ_src(mrs, tensor->src[i]);
        }
    }

    ਵਾਪਸ ggml_mem_ranges_ਜੋੜੋ_dst(mrs, tensor);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    ਲਈ (size_t i = 0; i < mrs->ranges.size(); i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        ਜੇ (mr.pb != cmp.pb) {
            ਜਾਰੀ;
        }

        // intersecting source ranges are allowed
        ਜੇ (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            ਜਾਰੀ;
        }

        ਜੇ (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            ਜੇ (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ਅਚਲ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ਅਚਲ ਤਾਰਕਿਕ res = ggml_mem_ranges_check(mrs, mr);

    ਵਾਪਸ res;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ਅਚਲ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ਅਚਲ ਤਾਰਕਿਕ res = ggml_mem_ranges_check(mrs, mr);

    ਵਾਪਸ res;
}

ਤਾਰਕਿਕ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ਅਚਲ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (tensor->src[i]) {
            ਜੇ (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }

    ਵਾਪਸ ggml_mem_ranges_check_dst(mrs, tensor);
}

ਬਣਤਰ ਗੰਢ_info {
    ggml_tensor * ਗੰਢ;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ਅਚਲ {
        ਵਾਪਸ ਗੰਢ->op;
    }

    ਅਚਲ ggml_tensor * dst() ਅਚਲ {
        ਵਾਪਸ fused.ਖਾਲੀ_ਥਾਵਾਂ() ? ਗੰਢ : fused.back();
    }

    ਤਾਰਕਿਕ is_ਖਾਲੀ_ਥਾਵਾਂ() ਅਚਲ {
        ਵਾਪਸ ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ->op);
    }

    ਖਾਲੀ ਜੋੜੋ_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    ਤਾਰਕਿਕ is_output(ਅਚਲ ggml_tensor * t) ਅਚਲ {
        ਜੇ (t == ਗੰਢ) {
            ਵਾਪਸ ਸੱਚ;
        }
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * f : fused) {
            ਜੇ (t == f) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
        ਵਾਪਸ ਝੂਠ;
    }
};

ਸਥਿਰ std::vector<ਪੂਰਨ> ggml_metal_ਗ੍ਰਾਫ_optimize_reorder(ਅਚਲ std::vector<ਗੰਢ_info> & nodes) {
    // helper to add node src and dst ranges
    ਅਚਲ ਸਵੈਚਾਲਿਤ & h_ਜੋੜੋ = [](ggml_mem_ranges_t mrs, ਅਚਲ ਗੰਢ_info & ਗੰਢ) {
        // only external sources matter: sources produced by the fused group are internal
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਅਚਲ ggml_tensor * src = ਗੰਢ.ਗੰਢ->src[i];
            ਜੇ (src && !ਗੰਢ.is_output(src)) {
                ਜੇ (!ggml_mem_ranges_ਜੋੜੋ_src(mrs, src)) {
                    ਵਾਪਸ ਝੂਠ;
                }
            }
        }

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
            ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
                ਅਚਲ ggml_tensor * src = fused->src[i];
                ਜੇ (src && !ਗੰਢ.is_output(src)) {
                    ਜੇ (!ggml_mem_ranges_ਜੋੜੋ_src(mrs, src)) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        ਜੇ (!ggml_mem_ranges_ਜੋੜੋ_dst(mrs, ਗੰਢ.ਗੰਢ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
            ਜੇ (!ggml_mem_ranges_ਜੋੜੋ_dst(mrs, fused)) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        ਵਾਪਸ ਸੱਚ;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ਅਚਲ ਸਵੈਚਾਲਿਤ & h_check = [](ggml_mem_ranges_t mrs, ਅਚਲ ਗੰਢ_info & ਗੰਢ) {
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਅਚਲ ggml_tensor * src = ਗੰਢ.ਗੰਢ->src[i];
            ਜੇ (src && !ਗੰਢ.is_output(src)) {
                ਜੇ (!ggml_mem_ranges_check_src(mrs, src)) {
                    ਵਾਪਸ ਝੂਠ;
                }
            }
        }

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
            ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
                ਅਚਲ ggml_tensor * src = fused->src[i];
                ਜੇ (src && !ਗੰਢ.is_output(src)) {
                    ਜੇ (!ggml_mem_ranges_check_src(mrs, src)) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
            }
        }

        ਜੇ (!ggml_mem_ranges_check_dst(mrs, ਗੰਢ.ਗੰਢ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
            ਜੇ (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        ਵਾਪਸ ਸੱਚ;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ਅਚਲ ਸਵੈਚਾਲਿਤ & h_ਸੁਰੱਖਿਅਤ = [](ggml_op op) {
        ਚੋਣ (op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ਹਾਲਤ GGML_OP_ROPE:
            ਹਾਲਤ GGML_OP_NORM:
            ਹਾਲਤ GGML_OP_RMS_NORM:
            ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            ਹਾਲਤ GGML_OP_L2_NORM:
            ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ਹਾਲਤ GGML_OP_SSM_CONV:
            ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            ਹਾਲਤ GGML_OP_CLAMP:
            ਹਾਲਤ GGML_OP_TRI:
            ਹਾਲਤ GGML_OP_DIAG:
            ਹਾਲਤ GGML_OP_MUL:
            ਹਾਲਤ GGML_OP_ਜੋੜੋ:
            ਹਾਲਤ GGML_OP_SUB:
            ਹਾਲਤ GGML_OP_DIV:
            ਹਾਲਤ GGML_OP_GLU:
            ਹਾਲਤ GGML_OP_SCALE:
            ਹਾਲਤ GGML_OP_UNARY:
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            ਹਾਲਤ GGML_OP_CPY:
            ਹਾਲਤ GGML_OP_CONT:
            ਹਾਲਤ GGML_OP_REPEAT:
                ਵਾਪਸ ਸੱਚ;
            ਮੂਲ:
                ਵਾਪਸ ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(op);
        }
    };

    ਅਚਲ ਪੂਰਨ n = nodes.size();

    std::vector<ਪੂਰਨ> res;
    res.reserve(n);

    std::vector<ਤਾਰਕਿਕ> ਵਰਤਿਆ(n, ਝੂਠ);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    ਲਈ (ਪੂਰਨ i0 = 0; i0 < n; i0++) {
        ਜੇ (ਵਰਤਿਆ[i0]) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        ਜੇ (!node0.is_ਖਾਲੀ_ਥਾਵਾਂ() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_ਜੋੜੋ(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr ਪੂਰਨ N_FORWARD = 64;

            ਲਈ (ਪੂਰਨ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                ਜੇ (ਵਰਤਿਆ[i1]) {
                    ਜਾਰੀ;
                }

                ਅਚਲ ਸਵੈਚਾਲਿਤ & node1 = nodes[i1];

                // disallow reordering of certain ops
                ਜੇ (!h_ਸੁਰੱਖਿਅਤ(node1.op())) {
                    ਰੋਕੋ;
                }

                ਅਚਲ ਤਾਰਕਿਕ is_ਖਾਲੀ_ਥਾਵਾਂ = node1.is_ਖਾਲੀ_ਥਾਵਾਂ();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                ਜੇ ((is_ਖਾਲੀ_ਥਾਵਾਂ || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_ਜੋੜੋ(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    ਵਰਤਿਆ[i1] = ਸੱਚ;
                } ਨਹੀਂਤਾਂ {
                    // expand the set of nodes that haven't been processed yet
                    h_ਜੋੜੋ(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_ਜੋੜੋ(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    ਵਾਪਸ res;
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_optimize(ggml_cgraph * gf) {
    ਅਚਲ ਪੂਰਨ n = gf->n_nodes;

    std::vector<ਗੰਢ_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ਗੰਢ_info ਗੰਢ = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ਅਚਲ ਪੂਰਨ f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        ਲਈ (ਪੂਰਨ k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            ਗੰਢ.ਜੋੜੋ_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(ਗੰਢ));
    }

#ਜੇ 1
    // reorder to improve concurrency
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਕ੍ਰਮ = ggml_metal_ਗ੍ਰਾਫ_optimize_reorder(nodes);
#ਨਹੀਂਤਾਂ
    std::vector<ਪੂਰਨ> ਕ੍ਰਮ(nodes.size());
    ਲਈ (size_t i = 0; i < nodes.size(); i++) {
        ਕ੍ਰਮ[i] = i;
    }
#ਜੇ_ਅੰਤ

    // unfuse
    {
        ਪੂਰਨ j = 0;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ i : ਕ੍ਰਮ) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & ਗੰਢ = nodes[i];

            gf->nodes[j++] = ਗੰਢ.ਗੰਢ;

            ਲਈ (ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
