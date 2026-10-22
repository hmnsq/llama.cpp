#అన్తర్భావ "ggml-metal-common.h"
#అన్తర్భావ "ggml-metal-fusion.h"

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
స్థిర తార్కిక ggml_metal_fwht_supported_కొలత(int64_t n, size_t max_tg_mem) {
    ఒకవేళ (n == 64 || n == 128 || n == 256 || n == 512) {
        తిరిగి నిజం;
    }

    ఒకవేళ (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        తిరిగి (size_t) n * పరిమాణం(దశాంశం) <= max_tg_mem;
    }

    తిరిగి అబద్ధం;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
తార్కిక ggml_metal_op_mul_mat_use_fwht(స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t max_tg_mem) {
    తిరిగి ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_కొలత(op->src[1]->ne[0], max_tg_mem);
}

తార్కిక ggml_metal_op_mul_mat_use_mm(స్థిరరాశి నిర్మాణం ggml_tensor * op, తార్కిక has_simdgroup_mm) {
    స్థిరరాశి int64_t ne00 = op->src[0]->ne[0];
    స్థిరరాశి int64_t ne11 = op->src[1]->ne[1];

    తిరిగి !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

తార్కిక ggml_metal_op_mul_mat_id_use_mm(స్థిరరాశి నిర్మాణం ggml_tensor * op, తార్కిక has_simdgroup_mm) {
    స్థిరరాశి int64_t ne00 = op->src[0]->ne[0];
    స్థిరరాశి int64_t ne02 = op->src[0]->ne[2];
    స్థిరరాశి int64_t ne21 = op->src[2]->ne[1];

    తిరిగి has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
స్థిర constexpr int64_t GGML_METAL_MMA_అడ్డువరుసలు_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
స్థిర constexpr int64_t GGML_METAL_MMA_TILE_అడ్డువరుసలు = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
స్థిర constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

గణన ggml_metal_mma_రకం ggml_metal_mul_mv_mma_రకం(గణన ggml_type type, పూర్ణాంకం rt) {
    ఒకవేళ (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        తిరిగి GGML_METAL_MMA_రకం_BLK;
    }
    తిరిగి type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_రకం_Q5_K : GGML_METAL_MMA_రకం_GEN;
}

పూర్ణాంకం ggml_metal_mul_mv_mma_rt(స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    తిరిగి op->src[1]->ne[1] > GGML_METAL_MMA_TILE_అడ్డువరుసలు ? 2 : 1;
}

స్థిర తార్కిక ggml_metal_mul_mv_mma_type_supported(గణన ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_TQ2_0:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

int64_t ggml_metal_mul_mv_mma_k_అడుగు(గణన ggml_type type, పూర్ణాంకం rt) {
    ఒకవేళ (!ggml_metal_mul_mv_mma_type_supported(type)) {
        తిరిగి 0;
    }
    తిరిగి ggml_metal_mul_mv_mma_రకం(type, rt) == GGML_METAL_MMA_రకం_BLK ? ggml_blck_కొలత(type) : GGML_METAL_MMA_K_CHUNK;
}

స్థిర తార్కిక ggml_metal_mul_mat_mma_type_ok(స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    స్థిరరాశి int64_t అడుగు = ggml_metal_mul_mv_mma_k_అడుగు(src0->type, ggml_metal_mul_mv_mma_rt(op));

    తిరిగి అడుగు > 0 && src0->ne[0] % అడుగు == 0 && src0->nb[0] == ggml_type_కొలత(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
స్థిర int64_t ggml_metal_mul_mv_mma_అడ్డువరుసలు_min(గణన ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
            తిరిగి 6;
        స్థితి GGML_TYPE_TQ2_0:
            తిరిగి 5;
        స్థితి GGML_TYPE_BF16:
            తిరిగి 4;
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_MXFP4:
            తిరిగి 3;
        అప్రమేయ:
            తిరిగి 2;
    }
}

తార్కిక ggml_metal_op_mul_mat_use_mma(స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    స్థిరరాశి ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    స్థిరరాశి తార్కిక batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    తిరిగి ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_అడ్డువరుసలు_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_అడ్డువరుసలు_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == పరిమాణం(దశాంశం) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

తార్కిక ggml_metal_op_mul_mat_may_use_mma(స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    తిరిగి ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
నిర్మాణం ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; //చివర

    ggml_mem_range_type pt;
};

నిర్మాణం ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    పూర్ణాంకం debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(పూర్ణాంకం debug) {
    స్వయంచాలక * res = కొత్త ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    తిరిగి res;
}

శూన్య ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    తొలగించు mrs;
}

శూన్య ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

స్థిర తార్కిక ggml_mem_ranges_కూడు(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    తిరిగి నిజం;
}

స్థిర ggml_mem_range ggml_mem_range_from_tensor(స్థిరరాశి ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    ఒకవేళ (tensor->బఫరు) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->బఫరు,
            /*.p0 =*/ (uint64_t) tensor->దత్తాంశం,
            /*.p1 =*/ (uint64_t) tensor->దత్తాంశం + ggml_backend_buft_get_alloc_కొలత(tensor->బఫరు->buft, tensor),
            /*.pt =*/ pt,
        };
    } లేకపోతే {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    తిరిగి mr;
}

స్థిర ggml_mem_range ggml_mem_range_from_tensor_src(స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

స్థిర ggml_mem_range ggml_mem_range_from_tensor_dst(స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

స్థిర తార్కిక ggml_mem_ranges_కూడు_src(ggml_mem_ranges_t mrs, స్థిరరాశి ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ఒకవేళ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    తిరిగి ggml_mem_ranges_కూడు(mrs, mr);
}

స్థిర తార్కిక ggml_mem_ranges_కూడు_dst(ggml_mem_ranges_t mrs, స్థిరరాశి ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ఒకవేళ (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    తిరిగి ggml_mem_ranges_కూడు(mrs, mr);
}

తార్కిక ggml_mem_ranges_కూడు(ggml_mem_ranges_t mrs, స్థిరరాశి ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (tensor->src[i]) {
            ggml_mem_ranges_కూడు_src(mrs, tensor->src[i]);
        }
    }

    తిరిగి ggml_mem_ranges_కూడు_dst(mrs, tensor);
}

స్థిర తార్కిక ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    కొరకు (size_t i = 0; i < mrs->ranges.size(); i++) {
        స్థిరరాశి స్వయంచాలక & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        ఒకవేళ (mr.pb != cmp.pb) {
            కొనసాగు;
        }

        // intersecting source ranges are allowed
        ఒకవేళ (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            కొనసాగు;
        }

        ఒకవేళ (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            ఒకవేళ (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, స్థిరరాశి ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    స్థిరరాశి తార్కిక res = ggml_mem_ranges_check(mrs, mr);

    తిరిగి res;
}

స్థిర తార్కిక ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, స్థిరరాశి ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    స్థిరరాశి తార్కిక res = ggml_mem_ranges_check(mrs, mr);

    తిరిగి res;
}

తార్కిక ggml_mem_ranges_check(ggml_mem_ranges_t mrs, స్థిరరాశి ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (tensor->src[i]) {
            ఒకవేళ (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                తిరిగి అబద్ధం;
            }
        }
    }

    తిరిగి ggml_mem_ranges_check_dst(mrs, tensor);
}

నిర్మాణం కణుపు_info {
    ggml_tensor * కణుపు;

    std::vector<ggml_tensor *> fused;

    ggml_op op() స్థిరరాశి {
        తిరిగి కణుపు->op;
    }

    స్థిరరాశి ggml_tensor * dst() స్థిరరాశి {
        తిరిగి fused.ఖాళీలు() ? కణుపు : fused.back();
    }

    తార్కిక is_ఖాళీలు() స్థిరరాశి {
        తిరిగి ggml_op_is_ఖాళీలు(కణుపు->op);
    }

    శూన్య కూడు_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    తార్కిక is_output(స్థిరరాశి ggml_tensor * t) స్థిరరాశి {
        ఒకవేళ (t == కణుపు) {
            తిరిగి నిజం;
        }
        కొరకు (స్థిరరాశి స్వయంచాలక * f : fused) {
            ఒకవేళ (t == f) {
                తిరిగి నిజం;
            }
        }
        తిరిగి అబద్ధం;
    }
};

స్థిర std::vector<పూర్ణాంకం> ggml_metal_లేఖాచిత్రం_optimize_reorder(స్థిరరాశి std::vector<కణుపు_info> & nodes) {
    // helper to add node src and dst ranges
    స్థిరరాశి స్వయంచాలక & h_కూడు = [](ggml_mem_ranges_t mrs, స్థిరరాశి కణుపు_info & కణుపు) {
        // only external sources matter: sources produced by the fused group are internal
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్థిరరాశి ggml_tensor * src = కణుపు.కణుపు->src[i];
            ఒకవేళ (src && !కణుపు.is_output(src)) {
                ఒకవేళ (!ggml_mem_ranges_కూడు_src(mrs, src)) {
                    తిరిగి అబద్ధం;
                }
            }
        }

        కొరకు (స్థిరరాశి స్వయంచాలక * fused : కణుపు.fused) {
            కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
                స్థిరరాశి ggml_tensor * src = fused->src[i];
                ఒకవేళ (src && !కణుపు.is_output(src)) {
                    ఒకవేళ (!ggml_mem_ranges_కూడు_src(mrs, src)) {
                        తిరిగి అబద్ధం;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        ఒకవేళ (!ggml_mem_ranges_కూడు_dst(mrs, కణుపు.కణుపు)) {
            తిరిగి అబద్ధం;
        }
        కొరకు (స్థిరరాశి స్వయంచాలక * fused : కణుపు.fused) {
            ఒకవేళ (!ggml_mem_ranges_కూడు_dst(mrs, fused)) {
                తిరిగి అబద్ధం;
            }
        }

        తిరిగి నిజం;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    స్థిరరాశి స్వయంచాలక & h_check = [](ggml_mem_ranges_t mrs, స్థిరరాశి కణుపు_info & కణుపు) {
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్థిరరాశి ggml_tensor * src = కణుపు.కణుపు->src[i];
            ఒకవేళ (src && !కణుపు.is_output(src)) {
                ఒకవేళ (!ggml_mem_ranges_check_src(mrs, src)) {
                    తిరిగి అబద్ధం;
                }
            }
        }

        కొరకు (స్థిరరాశి స్వయంచాలక * fused : కణుపు.fused) {
            కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
                స్థిరరాశి ggml_tensor * src = fused->src[i];
                ఒకవేళ (src && !కణుపు.is_output(src)) {
                    ఒకవేళ (!ggml_mem_ranges_check_src(mrs, src)) {
                        తిరిగి అబద్ధం;
                    }
                }
            }
        }

        ఒకవేళ (!ggml_mem_ranges_check_dst(mrs, కణుపు.కణుపు)) {
            తిరిగి అబద్ధం;
        }
        కొరకు (స్థిరరాశి స్వయంచాలక * fused : కణుపు.fused) {
            ఒకవేళ (!ggml_mem_ranges_check_dst(mrs, fused)) {
                తిరిగి అబద్ధం;
            }
        }

        తిరిగి నిజం;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    స్థిరరాశి స్వయంచాలక & h_సురక్షిత = [](ggml_op op) {
        ఎంపిక (op) {
            స్థితి GGML_OP_MUL_MAT:
            స్థితి GGML_OP_MUL_MAT_ID:
            స్థితి GGML_OP_ROPE:
            స్థితి GGML_OP_NORM:
            స్థితి GGML_OP_RMS_NORM:
            స్థితి GGML_OP_సమూహం_NORM:
            స్థితి GGML_OP_L2_NORM:
            స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            స్థితి GGML_OP_SSM_CONV:
            స్థితి GGML_OP_SSM_పరిశీలన:
            స్థితి GGML_OP_CLAMP:
            స్థితి GGML_OP_TRI:
            స్థితి GGML_OP_DIAG:
            స్థితి GGML_OP_MUL:
            స్థితి GGML_OP_కూడు:
            స్థితి GGML_OP_SUB:
            స్థితి GGML_OP_DIV:
            స్థితి GGML_OP_GLU:
            స్థితి GGML_OP_SCALE:
            స్థితి GGML_OP_UNARY:
            స్థితి GGML_OP_GET_అడ్డువరుసలు:
            స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            స్థితి GGML_OP_సమితి:
            స్థితి GGML_OP_CPY:
            స్థితి GGML_OP_CONT:
            స్థితి GGML_OP_REPEAT:
                తిరిగి నిజం;
            అప్రమేయ:
                తిరిగి ggml_op_is_ఖాళీలు(op);
        }
    };

    స్థిరరాశి పూర్ణాంకం n = nodes.size();

    std::vector<పూర్ణాంకం> res;
    res.reserve(n);

    std::vector<తార్కిక> వాడిన(n, అబద్ధం);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    కొరకు (పూర్ణాంకం i0 = 0; i0 < n; i0++) {
        ఒకవేళ (వాడిన[i0]) {
            కొనసాగు;
        }

        స్థిరరాశి స్వయంచాలక & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        ఒకవేళ (!node0.is_ఖాళీలు() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_కూడు(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr పూర్ణాంకం N_FORWARD = 64;

            కొరకు (పూర్ణాంకం i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                ఒకవేళ (వాడిన[i1]) {
                    కొనసాగు;
                }

                స్థిరరాశి స్వయంచాలక & node1 = nodes[i1];

                // disallow reordering of certain ops
                ఒకవేళ (!h_సురక్షిత(node1.op())) {
                    ఆపు;
                }

                స్థిరరాశి తార్కిక is_ఖాళీలు = node1.is_ఖాళీలు();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                ఒకవేళ ((is_ఖాళీలు || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_కూడు(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    వాడిన[i1] = నిజం;
                } లేకపోతే {
                    // expand the set of nodes that haven't been processed yet
                    h_కూడు(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_కూడు(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    తిరిగి res;
}

శూన్య ggml_లేఖాచిత్రం_optimize(ggml_cgraph * gf) {
    స్థిరరాశి పూర్ణాంకం n = gf->n_nodes;

    std::vector<కణుపు_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        కణుపు_info కణుపు = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        స్థిరరాశి పూర్ణాంకం f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        కొరకు (పూర్ణాంకం k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            కణుపు.కూడు_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(కణుపు));
    }

#ఒకవేళ 1
    // reorder to improve concurrency
    స్థిరరాశి స్వయంచాలక క్రమం = ggml_metal_లేఖాచిత్రం_optimize_reorder(nodes);
#లేకపోతే
    std::vector<పూర్ణాంకం> క్రమం(nodes.size());
    కొరకు (size_t i = 0; i < nodes.size(); i++) {
        క్రమం[i] = i;
    }
#అయితే_ముగింపు

    // unfuse
    {
        పూర్ణాంకం j = 0;
        కొరకు (స్థిరరాశి స్వయంచాలక i : క్రమం) {
            స్థిరరాశి స్వయంచాలక & కణుపు = nodes[i];

            gf->nodes[j++] = కణుపు.కణుపు;

            కొరకు (స్వయంచాలక * fused : కణుపు.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
