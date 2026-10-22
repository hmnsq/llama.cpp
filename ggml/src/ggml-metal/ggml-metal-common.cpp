#അന്തര്ഭാവ "ggml-metal-common.h"
#അന്തര്ഭാവ "ggml-metal-fusion.h"

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
സ്ഥിതം തർക്കം ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    എങ്കിൽ (n == 64 || n == 128 || n == 256 || n == 512) {
        തിരികെ സത്യം;
    }

    എങ്കിൽ (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        തിരികെ (size_t) n * വലുപ്പം(ദശാംശം) <= max_tg_mem;
    }

    തിരികെ അസത്യം;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
തർക്കം ggml_metal_op_mul_mat_use_fwht(സ്ഥിരം ഘടന ggml_tensor * op, size_t max_tg_mem) {
    തിരികെ ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

തർക്കം ggml_metal_op_mul_mat_use_mm(സ്ഥിരം ഘടന ggml_tensor * op, തർക്കം has_simdgroup_mm) {
    സ്ഥിരം int64_t ne00 = op->src[0]->ne[0];
    സ്ഥിരം int64_t ne11 = op->src[1]->ne[1];

    തിരികെ !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

തർക്കം ggml_metal_op_mul_mat_id_use_mm(സ്ഥിരം ഘടന ggml_tensor * op, തർക്കം has_simdgroup_mm) {
    സ്ഥിരം int64_t ne00 = op->src[0]->ne[0];
    സ്ഥിരം int64_t ne02 = op->src[0]->ne[2];
    സ്ഥിരം int64_t ne21 = op->src[2]->ne[1];

    തിരികെ has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
സ്ഥിതം constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
സ്ഥിതം constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
സ്ഥിതം constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

എണ്ണൽ ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(എണ്ണൽ ggml_type type, പൂർണ്ണസംഖ്യ rt) {
    എങ്കിൽ (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        തിരികെ GGML_METAL_MMA_KIND_BLK;
    }
    തിരികെ type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

പൂർണ്ണസംഖ്യ ggml_metal_mul_mv_mma_rt(സ്ഥിരം ഘടന ggml_tensor * op) {
    തിരികെ op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

സ്ഥിതം തർക്കം ggml_metal_mul_mv_mma_type_supported(എണ്ണൽ ggml_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_F32:
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_BF16:
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ2_S:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_TQ2_0:
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(എണ്ണൽ ggml_type type, പൂർണ്ണസംഖ്യ rt) {
    എങ്കിൽ (!ggml_metal_mul_mv_mma_type_supported(type)) {
        തിരികെ 0;
    }
    തിരികെ ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

സ്ഥിതം തർക്കം ggml_metal_mul_mat_mma_type_ok(സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    സ്ഥിരം int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    തിരികെ step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
സ്ഥിതം int64_t ggml_metal_mul_mv_mma_rows_min(എണ്ണൽ ggml_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_F32:
            തിരികെ 6;
        അവസ്ഥ GGML_TYPE_TQ2_0:
            തിരികെ 5;
        അവസ്ഥ GGML_TYPE_BF16:
            തിരികെ 4;
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_MXFP4:
            തിരികെ 3;
        സ്ഥിരസ്ഥിതി:
            തിരികെ 2;
    }
}

തർക്കം ggml_metal_op_mul_mat_use_mma(സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    സ്ഥിരം തർക്കം batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    തിരികെ ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == വലുപ്പം(ദശാംശം) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

തർക്കം ggml_metal_op_mul_mat_may_use_mma(സ്ഥിരം ഘടന ggml_tensor * op) {
    തിരികെ ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
ഘടന ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

ഘടന ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    പൂർണ്ണസംഖ്യ debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(പൂർണ്ണസംഖ്യ debug) {
    auto * res = പുതിയ ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    തിരികെ res;
}

ശൂന്യം ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

ശൂന്യം ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

സ്ഥിതം തർക്കം ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    തിരികെ സത്യം;
}

സ്ഥിതം ggml_mem_range ggml_mem_range_from_tensor(സ്ഥിരം ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    എങ്കിൽ (tensor->buffer) {
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
    } അല്ലെങ്കിൽ {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    തിരികെ mr;
}

സ്ഥിതം ggml_mem_range ggml_mem_range_from_tensor_src(സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

സ്ഥിതം ggml_mem_range ggml_mem_range_from_tensor_dst(സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

സ്ഥിതം തർക്കം ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, സ്ഥിരം ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    എങ്കിൽ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    തിരികെ ggml_mem_ranges_add(mrs, mr);
}

സ്ഥിതം തർക്കം ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, സ്ഥിരം ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    എങ്കിൽ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    തിരികെ ggml_mem_ranges_add(mrs, mr);
}

തർക്കം ggml_mem_ranges_add(ggml_mem_ranges_t mrs, സ്ഥിരം ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        എങ്കിൽ (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    തിരികെ ggml_mem_ranges_add_dst(mrs, tensor);
}

സ്ഥിതം തർക്കം ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    വേണ്ടി (size_t i = 0; i < mrs->ranges.size(); i++) {
        സ്ഥിരം auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        എങ്കിൽ (mr.pb != cmp.pb) {
            തുടരുക;
        }

        // intersecting source ranges are allowed
        എങ്കിൽ (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            തുടരുക;
        }

        എങ്കിൽ (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            എങ്കിൽ (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, സ്ഥിരം ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    സ്ഥിരം തർക്കം res = ggml_mem_ranges_check(mrs, mr);

    തിരികെ res;
}

സ്ഥിതം തർക്കം ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, സ്ഥിരം ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    സ്ഥിരം തർക്കം res = ggml_mem_ranges_check(mrs, mr);

    തിരികെ res;
}

തർക്കം ggml_mem_ranges_check(ggml_mem_ranges_t mrs, സ്ഥിരം ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        എങ്കിൽ (tensor->src[i]) {
            എങ്കിൽ (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                തിരികെ അസത്യം;
            }
        }
    }

    തിരികെ ggml_mem_ranges_check_dst(mrs, tensor);
}

ഘടന കണ്ണി_info {
    ggml_tensor * കണ്ണി;

    std::vector<ggml_tensor *> fused;

    ggml_op op() സ്ഥിരം {
        തിരികെ കണ്ണി->op;
    }

    സ്ഥിരം ggml_tensor * dst() സ്ഥിരം {
        തിരികെ fused.empty() ? കണ്ണി : fused.back();
    }

    തർക്കം is_empty() സ്ഥിരം {
        തിരികെ ggml_op_is_empty(കണ്ണി->op);
    }

    ശൂന്യം add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    തർക്കം is_output(സ്ഥിരം ggml_tensor * t) സ്ഥിരം {
        എങ്കിൽ (t == കണ്ണി) {
            തിരികെ സത്യം;
        }
        വേണ്ടി (സ്ഥിരം auto * f : fused) {
            എങ്കിൽ (t == f) {
                തിരികെ സത്യം;
            }
        }
        തിരികെ അസത്യം;
    }
};

സ്ഥിതം std::vector<പൂർണ്ണസംഖ്യ> ggml_metal_graph_optimize_reorder(സ്ഥിരം std::vector<കണ്ണി_info> & nodes) {
    // helper to add node src and dst ranges
    സ്ഥിരം auto & h_add = [](ggml_mem_ranges_t mrs, സ്ഥിരം കണ്ണി_info & കണ്ണി) {
        // only external sources matter: sources produced by the fused group are internal
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            സ്ഥിരം ggml_tensor * src = കണ്ണി.കണ്ണി->src[i];
            എങ്കിൽ (src && !കണ്ണി.is_output(src)) {
                എങ്കിൽ (!ggml_mem_ranges_add_src(mrs, src)) {
                    തിരികെ അസത്യം;
                }
            }
        }

        വേണ്ടി (സ്ഥിരം auto * fused : കണ്ണി.fused) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
                സ്ഥിരം ggml_tensor * src = fused->src[i];
                എങ്കിൽ (src && !കണ്ണി.is_output(src)) {
                    എങ്കിൽ (!ggml_mem_ranges_add_src(mrs, src)) {
                        തിരികെ അസത്യം;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        എങ്കിൽ (!ggml_mem_ranges_add_dst(mrs, കണ്ണി.കണ്ണി)) {
            തിരികെ അസത്യം;
        }
        വേണ്ടി (സ്ഥിരം auto * fused : കണ്ണി.fused) {
            എങ്കിൽ (!ggml_mem_ranges_add_dst(mrs, fused)) {
                തിരികെ അസത്യം;
            }
        }

        തിരികെ സത്യം;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    സ്ഥിരം auto & h_check = [](ggml_mem_ranges_t mrs, സ്ഥിരം കണ്ണി_info & കണ്ണി) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            സ്ഥിരം ggml_tensor * src = കണ്ണി.കണ്ണി->src[i];
            എങ്കിൽ (src && !കണ്ണി.is_output(src)) {
                എങ്കിൽ (!ggml_mem_ranges_check_src(mrs, src)) {
                    തിരികെ അസത്യം;
                }
            }
        }

        വേണ്ടി (സ്ഥിരം auto * fused : കണ്ണി.fused) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
                സ്ഥിരം ggml_tensor * src = fused->src[i];
                എങ്കിൽ (src && !കണ്ണി.is_output(src)) {
                    എങ്കിൽ (!ggml_mem_ranges_check_src(mrs, src)) {
                        തിരികെ അസത്യം;
                    }
                }
            }
        }

        എങ്കിൽ (!ggml_mem_ranges_check_dst(mrs, കണ്ണി.കണ്ണി)) {
            തിരികെ അസത്യം;
        }
        വേണ്ടി (സ്ഥിരം auto * fused : കണ്ണി.fused) {
            എങ്കിൽ (!ggml_mem_ranges_check_dst(mrs, fused)) {
                തിരികെ അസത്യം;
            }
        }

        തിരികെ സത്യം;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    സ്ഥിരം auto & h_safe = [](ggml_op op) {
        തിരഞ്ഞെടുപ്പ് (op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
            അവസ്ഥ GGML_OP_ROPE:
            അവസ്ഥ GGML_OP_NORM:
            അവസ്ഥ GGML_OP_RMS_NORM:
            അവസ്ഥ GGML_OP_GROUP_NORM:
            അവസ്ഥ GGML_OP_L2_NORM:
            അവസ്ഥ GGML_OP_SUM_ROWS:
            അവസ്ഥ GGML_OP_SSM_CONV:
            അവസ്ഥ GGML_OP_SSM_SCAN:
            അവസ്ഥ GGML_OP_CLAMP:
            അവസ്ഥ GGML_OP_TRI:
            അവസ്ഥ GGML_OP_DIAG:
            അവസ്ഥ GGML_OP_MUL:
            അവസ്ഥ GGML_OP_ADD:
            അവസ്ഥ GGML_OP_SUB:
            അവസ്ഥ GGML_OP_DIV:
            അവസ്ഥ GGML_OP_GLU:
            അവസ്ഥ GGML_OP_SCALE:
            അവസ്ഥ GGML_OP_UNARY:
            അവസ്ഥ GGML_OP_GET_ROWS:
            അവസ്ഥ GGML_OP_SET_ROWS:
            അവസ്ഥ GGML_OP_SET:
            അവസ്ഥ GGML_OP_CPY:
            അവസ്ഥ GGML_OP_CONT:
            അവസ്ഥ GGML_OP_REPEAT:
                തിരികെ സത്യം;
            സ്ഥിരസ്ഥിതി:
                തിരികെ ggml_op_is_empty(op);
        }
    };

    സ്ഥിരം പൂർണ്ണസംഖ്യ n = nodes.size();

    std::vector<പൂർണ്ണസംഖ്യ> res;
    res.reserve(n);

    std::vector<തർക്കം> used(n, അസത്യം);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < n; i0++) {
        എങ്കിൽ (used[i0]) {
            തുടരുക;
        }

        സ്ഥിരം auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        എങ്കിൽ (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr പൂർണ്ണസംഖ്യ N_FORWARD = 64;

            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                എങ്കിൽ (used[i1]) {
                    തുടരുക;
                }

                സ്ഥിരം auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                എങ്കിൽ (!h_safe(node1.op())) {
                    നിർത്തുക;
                }

                സ്ഥിരം തർക്കം is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                എങ്കിൽ ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = സത്യം;
                } അല്ലെങ്കിൽ {
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

    തിരികെ res;
}

ശൂന്യം ggml_graph_optimize(ggml_cgraph * gf) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ n = gf->n_nodes;

    std::vector<കണ്ണി_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        കണ്ണി_info കണ്ണി = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        സ്ഥിരം പൂർണ്ണസംഖ്യ f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            കണ്ണി.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(കണ്ണി));
    }

#എങ്കിൽ 1
    // reorder to improve concurrency
    സ്ഥിരം auto order = ggml_metal_graph_optimize_reorder(nodes);
#അല്ലെങ്കിൽ
    std::vector<പൂർണ്ണസംഖ്യ> order(nodes.size());
    വേണ്ടി (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        പൂർണ്ണസംഖ്യ j = 0;
        വേണ്ടി (സ്ഥിരം auto i : order) {
            സ്ഥിരം auto & കണ്ണി = nodes[i];

            gf->nodes[j++] = കണ്ണി.കണ്ണി;

            വേണ്ടി (auto * fused : കണ്ണി.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
