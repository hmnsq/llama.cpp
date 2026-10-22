#ଅନ୍ତର୍ଭାଵ "ggml-metal-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-metal-fusion.h"

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
ସ୍ଥିର ତାର୍କିକ ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    ଯଦି (n == 64 || n == 128 || n == 256 || n == 512) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ଫେରନ୍ତୁ (size_t) n * ଆକାର(ଦଶମିକ) <= max_tg_mem;
    }

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
ତାର୍କିକ ggml_metal_op_mul_mat_use_fwht(ଅଚଳ ସଂରଚନା ggml_tensor * op, size_t max_tg_mem) {
    ଫେରନ୍ତୁ ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

ତାର୍କିକ ggml_metal_op_mul_mat_use_mm(ଅଚଳ ସଂରଚନା ggml_tensor * op, ତାର୍କିକ has_simdgroup_mm) {
    ଅଚଳ int64_t ne00 = op->src[0]->ne[0];
    ଅଚଳ int64_t ne11 = op->src[1]->ne[1];

    ଫେରନ୍ତୁ !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

ତାର୍କିକ ggml_metal_op_mul_mat_id_use_mm(ଅଚଳ ସଂରଚନା ggml_tensor * op, ତାର୍କିକ has_simdgroup_mm) {
    ଅଚଳ int64_t ne00 = op->src[0]->ne[0];
    ଅଚଳ int64_t ne02 = op->src[0]->ne[2];
    ଅଚଳ int64_t ne21 = op->src[2]->ne[1];

    ଫେରନ୍ତୁ has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
ସ୍ଥିର constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
ସ୍ଥିର constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
ସ୍ଥିର constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

ଗଣନା ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(ଗଣନା ggml_type type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rt) {
    ଯଦି (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ଫେରନ୍ତୁ GGML_METAL_MMA_KIND_BLK;
    }
    ଫେରନ୍ତୁ type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_metal_mul_mv_mma_rt(ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଫେରନ୍ତୁ op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

ସ୍ଥିର ତାର୍କିକ ggml_metal_mul_mv_mma_type_supported(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_F32:
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_BF16:
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(ଗଣନା ggml_type type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rt) {
    ଯଦି (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ଫେରନ୍ତୁ 0;
    }
    ଫେରନ୍ତୁ ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

ସ୍ଥିର ତାର୍କିକ ggml_metal_mul_mat_mma_type_ok(ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ଅଚଳ int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ଫେରନ୍ତୁ step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
ସ୍ଥିର int64_t ggml_metal_mul_mv_mma_rows_min(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ 6;
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
            ଫେରନ୍ତୁ 5;
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ 4;
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            ଫେରନ୍ତୁ 3;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ 2;
    }
}

ତାର୍କିକ ggml_metal_op_mul_mat_use_mma(ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ଅଚଳ ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ଅଚଳ ତାର୍କିକ batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ଫେରନ୍ତୁ ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == ଆକାର(ଦଶମିକ) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

ତାର୍କିକ ggml_metal_op_mul_mat_may_use_mma(ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଫେରନ୍ତୁ ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
ସଂରଚନା ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

ସଂରଚନା ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(ପୂର୍ଣ୍ଣସଂଖ୍ୟା debug) {
    auto * res = ନୂତନ ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ଫେରନ୍ତୁ res;
}

ଶୂନ୍ୟ ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

ଶୂନ୍ୟ ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

ସ୍ଥିର ତାର୍କିକ ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ggml_mem_range ggml_mem_range_from_tensor(ଅଚଳ ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    ଯଦି (tensor->buffer) {
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
    } ଅନ୍ୟଥା {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    ଫେରନ୍ତୁ mr;
}

ସ୍ଥିର ggml_mem_range ggml_mem_range_from_tensor_src(ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

ସ୍ଥିର ggml_mem_range ggml_mem_range_from_tensor_dst(ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

ସ୍ଥିର ତାର୍କିକ ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, ଅଚଳ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ଯଦି (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ଫେରନ୍ତୁ ggml_mem_ranges_add(mrs, mr);
}

ସ୍ଥିର ତାର୍କିକ ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, ଅଚଳ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ଯଦି (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ଫେରନ୍ତୁ ggml_mem_ranges_add(mrs, mr);
}

ତାର୍କିକ ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ଅଚଳ ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        ଯଦି (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    ଫେରନ୍ତୁ ggml_mem_ranges_add_dst(mrs, tensor);
}

ସ୍ଥିର ତାର୍କିକ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    ପାଇଁ (size_t i = 0; i < mrs->ranges.size(); i++) {
        ଅଚଳ auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        ଯଦି (mr.pb != cmp.pb) {
            ଜାରି;
        }

        // intersecting source ranges are allowed
        ଯଦି (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            ଜାରି;
        }

        ଯଦି (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            ଯଦି (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ଅଚଳ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ଅଚଳ ତାର୍କିକ res = ggml_mem_ranges_check(mrs, mr);

    ଫେରନ୍ତୁ res;
}

ସ୍ଥିର ତାର୍କିକ ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ଅଚଳ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ଅଚଳ ତାର୍କିକ res = ggml_mem_ranges_check(mrs, mr);

    ଫେରନ୍ତୁ res;
}

ତାର୍କିକ ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ଅଚଳ ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        ଯଦି (tensor->src[i]) {
            ଯଦି (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    }

    ଫେରନ୍ତୁ ggml_mem_ranges_check_dst(mrs, tensor);
}

ସଂରଚନା ଗଣ୍ଠି_info {
    ggml_tensor * ଗଣ୍ଠି;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ଅଚଳ {
        ଫେରନ୍ତୁ ଗଣ୍ଠି->op;
    }

    ଅଚଳ ggml_tensor * dst() ଅଚଳ {
        ଫେରନ୍ତୁ fused.empty() ? ଗଣ୍ଠି : fused.back();
    }

    ତାର୍କିକ is_empty() ଅଚଳ {
        ଫେରନ୍ତୁ ggml_op_is_empty(ଗଣ୍ଠି->op);
    }

    ଶୂନ୍ୟ add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    ତାର୍କିକ is_output(ଅଚଳ ggml_tensor * t) ଅଚଳ {
        ଯଦି (t == ଗଣ୍ଠି) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ପାଇଁ (ଅଚଳ auto * f : fused) {
            ଯଦି (t == f) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
};

ସ୍ଥିର std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> ggml_metal_graph_optimize_reorder(ଅଚଳ std::vector<ଗଣ୍ଠି_info> & nodes) {
    // helper to add node src and dst ranges
    ଅଚଳ auto & h_add = [](ggml_mem_ranges_t mrs, ଅଚଳ ଗଣ୍ଠି_info & ଗଣ୍ଠି) {
        // only external sources matter: sources produced by the fused group are internal
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            ଅଚଳ ggml_tensor * src = ଗଣ୍ଠି.ଗଣ୍ଠି->src[i];
            ଯଦି (src && !ଗଣ୍ଠି.is_output(src)) {
                ଯଦି (!ggml_mem_ranges_add_src(mrs, src)) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
            }
        }

        ପାଇଁ (ଅଚଳ auto * fused : ଗଣ୍ଠି.fused) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
                ଅଚଳ ggml_tensor * src = fused->src[i];
                ଯଦି (src && !ଗଣ୍ଠି.is_output(src)) {
                    ଯଦି (!ggml_mem_ranges_add_src(mrs, src)) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        ଯଦି (!ggml_mem_ranges_add_dst(mrs, ଗଣ୍ଠି.ଗଣ୍ଠି)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ପାଇଁ (ଅଚଳ auto * fused : ଗଣ୍ଠି.fused) {
            ଯଦି (!ggml_mem_ranges_add_dst(mrs, fused)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ଅଚଳ auto & h_check = [](ggml_mem_ranges_t mrs, ଅଚଳ ଗଣ୍ଠି_info & ଗଣ୍ଠି) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            ଅଚଳ ggml_tensor * src = ଗଣ୍ଠି.ଗଣ୍ଠି->src[i];
            ଯଦି (src && !ଗଣ୍ଠି.is_output(src)) {
                ଯଦି (!ggml_mem_ranges_check_src(mrs, src)) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
            }
        }

        ପାଇଁ (ଅଚଳ auto * fused : ଗଣ୍ଠି.fused) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
                ଅଚଳ ggml_tensor * src = fused->src[i];
                ଯଦି (src && !ଗଣ୍ଠି.is_output(src)) {
                    ଯଦି (!ggml_mem_ranges_check_src(mrs, src)) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }
            }
        }

        ଯଦି (!ggml_mem_ranges_check_dst(mrs, ଗଣ୍ଠି.ଗଣ୍ଠି)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ପାଇଁ (ଅଚଳ auto * fused : ଗଣ୍ଠି.fused) {
            ଯଦି (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ଅଚଳ auto & h_safe = [](ggml_op op) {
        ବାଛ (op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            ସ୍ଥିତି GGML_OP_ROPE:
            ସ୍ଥିତି GGML_OP_NORM:
            ସ୍ଥିତି GGML_OP_RMS_NORM:
            ସ୍ଥିତି GGML_OP_GROUP_NORM:
            ସ୍ଥିତି GGML_OP_L2_NORM:
            ସ୍ଥିତି GGML_OP_SUM_ROWS:
            ସ୍ଥିତି GGML_OP_SSM_CONV:
            ସ୍ଥିତି GGML_OP_SSM_SCAN:
            ସ୍ଥିତି GGML_OP_CLAMP:
            ସ୍ଥିତି GGML_OP_TRI:
            ସ୍ଥିତି GGML_OP_DIAG:
            ସ୍ଥିତି GGML_OP_MUL:
            ସ୍ଥିତି GGML_OP_ADD:
            ସ୍ଥିତି GGML_OP_SUB:
            ସ୍ଥିତି GGML_OP_DIV:
            ସ୍ଥିତି GGML_OP_GLU:
            ସ୍ଥିତି GGML_OP_SCALE:
            ସ୍ଥିତି GGML_OP_UNARY:
            ସ୍ଥିତି GGML_OP_GET_ROWS:
            ସ୍ଥିତି GGML_OP_SET_ROWS:
            ସ୍ଥିତି GGML_OP_SET:
            ସ୍ଥିତି GGML_OP_CPY:
            ସ୍ଥିତି GGML_OP_CONT:
            ସ୍ଥିତି GGML_OP_REPEAT:
                ଫେରନ୍ତୁ ସତ୍ୟ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫେରନ୍ତୁ ggml_op_is_empty(op);
        }
    };

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = nodes.size();

    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> res;
    res.reserve(n);

    std::vector<ତାର୍କିକ> used(n, ମିଥ୍ୟା);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < n; i0++) {
        ଯଦି (used[i0]) {
            ଜାରି;
        }

        ଅଚଳ auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        ଯଦି (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା N_FORWARD = 64;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                ଯଦି (used[i1]) {
                    ଜାରି;
                }

                ଅଚଳ auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                ଯଦି (!h_safe(node1.op())) {
                    ଭାଙ୍ଗ;
                }

                ଅଚଳ ତାର୍କିକ is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                ଯଦି ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = ସତ୍ୟ;
                } ଅନ୍ୟଥା {
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

    ଫେରନ୍ତୁ res;
}

ଶୂନ୍ୟ ggml_graph_optimize(ggml_cgraph * gf) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = gf->n_nodes;

    std::vector<ଗଣ୍ଠି_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ଗଣ୍ଠି_info ଗଣ୍ଠି = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            ଗଣ୍ଠି.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(ଗଣ୍ଠି));
    }

#ଯଦି 1
    // reorder to improve concurrency
    ଅଚଳ auto order = ggml_metal_graph_optimize_reorder(nodes);
#ଅନ୍ୟଥା
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> order(nodes.size());
    ପାଇଁ (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0;
        ପାଇଁ (ଅଚଳ auto i : order) {
            ଅଚଳ auto & ଗଣ୍ଠି = nodes[i];

            gf->nodes[j++] = ଗଣ୍ଠି.ଗଣ୍ଠି;

            ପାଇଁ (auto * fused : ଗଣ୍ଠି.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
