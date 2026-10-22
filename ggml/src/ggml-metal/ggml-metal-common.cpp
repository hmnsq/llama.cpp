#અન્તર્ભાવ "ggml-metal-common.h"
#અન્તર્ભાવ "ggml-metal-fusion.h"

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
સ્થિર તાર્કિક ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    જો (n == 64 || n == 128 || n == 256 || n == 512) {
        પાછા સાચું;
    }

    જો (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        પાછા (size_t) n * કદ(દશાંશ) <= max_tg_mem;
    }

    પાછા ખોટું;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
તાર્કિક ggml_metal_op_mul_mat_use_fwht(અચલ સંરચના ggml_tensor * op, size_t max_tg_mem) {
    પાછા ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

તાર્કિક ggml_metal_op_mul_mat_use_mm(અચલ સંરચના ggml_tensor * op, તાર્કિક has_simdgroup_mm) {
    અચલ int64_t ne00 = op->src[0]->ne[0];
    અચલ int64_t ne11 = op->src[1]->ne[1];

    પાછા !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

તાર્કિક ggml_metal_op_mul_mat_id_use_mm(અચલ સંરચના ggml_tensor * op, તાર્કિક has_simdgroup_mm) {
    અચલ int64_t ne00 = op->src[0]->ne[0];
    અચલ int64_t ne02 = op->src[0]->ne[2];
    અચલ int64_t ne21 = op->src[2]->ne[1];

    પાછા has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
સ્થિર constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
સ્થિર constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
સ્થિર constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

ગણના ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(ગણના ggml_type type, પૂર્ણાંક rt) {
    જો (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        પાછા GGML_METAL_MMA_KIND_BLK;
    }
    પાછા type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

પૂર્ણાંક ggml_metal_mul_mv_mma_rt(અચલ સંરચના ggml_tensor * op) {
    પાછા op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

સ્થિર તાર્કિક ggml_metal_mul_mv_mma_type_supported(ગણના ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_F32:
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_BF16:
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ2_S:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_TQ2_0:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(ગણના ggml_type type, પૂર્ણાંક rt) {
    જો (!ggml_metal_mul_mv_mma_type_supported(type)) {
        પાછા 0;
    }
    પાછા ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

સ્થિર તાર્કિક ggml_metal_mul_mat_mma_type_ok(અચલ સંરચના ggml_tensor * op) {
    અચલ ggml_tensor * src0 = op->src[0];
    અચલ int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    પાછા step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
સ્થિર int64_t ggml_metal_mul_mv_mma_rows_min(ગણના ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_F32:
            પાછા 6;
        સ્થિતિ GGML_TYPE_TQ2_0:
            પાછા 5;
        સ્થિતિ GGML_TYPE_BF16:
            પાછા 4;
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_MXFP4:
            પાછા 3;
        મૂળભૂત:
            પાછા 2;
    }
}

તાર્કિક ggml_metal_op_mul_mat_use_mma(અચલ સંરચના ggml_tensor * op) {
    અચલ ggml_tensor * src0 = op->src[0];
    અચલ ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    અચલ તાર્કિક batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    પાછા ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == કદ(દશાંશ) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

તાર્કિક ggml_metal_op_mul_mat_may_use_mma(અચલ સંરચના ggml_tensor * op) {
    પાછા ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
સંરચના ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

સંરચના ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    પૂર્ણાંક debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(પૂર્ણાંક debug) {
    auto * res = નવું ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    પાછા res;
}

શૂન્ય ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

શૂન્ય ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

સ્થિર તાર્કિક ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    પાછા સાચું;
}

સ્થિર ggml_mem_range ggml_mem_range_from_tensor(અચલ ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    જો (tensor->buffer) {
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
    } નહીંતર {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    પાછા mr;
}

સ્થિર ggml_mem_range ggml_mem_range_from_tensor_src(અચલ ggml_tensor * tensor) {
    પાછા ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

સ્થિર ggml_mem_range ggml_mem_range_from_tensor_dst(અચલ ggml_tensor * tensor) {
    પાછા ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

સ્થિર તાર્કિક ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, અચલ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    જો (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    પાછા ggml_mem_ranges_add(mrs, mr);
}

સ્થિર તાર્કિક ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, અચલ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    જો (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    પાછા ggml_mem_ranges_add(mrs, mr);
}

તાર્કિક ggml_mem_ranges_add(ggml_mem_ranges_t mrs, અચલ ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        જો (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    પાછા ggml_mem_ranges_add_dst(mrs, tensor);
}

સ્થિર તાર્કિક ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    માટે (size_t i = 0; i < mrs->ranges.size(); i++) {
        અચલ auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        જો (mr.pb != cmp.pb) {
            ચાલુ;
        }

        // intersecting source ranges are allowed
        જો (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            ચાલુ;
        }

        જો (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            જો (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            પાછા ખોટું;
        }
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, અચલ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    અચલ તાર્કિક res = ggml_mem_ranges_check(mrs, mr);

    પાછા res;
}

સ્થિર તાર્કિક ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, અચલ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    અચલ તાર્કિક res = ggml_mem_ranges_check(mrs, mr);

    પાછા res;
}

તાર્કિક ggml_mem_ranges_check(ggml_mem_ranges_t mrs, અચલ ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        જો (tensor->src[i]) {
            જો (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                પાછા ખોટું;
            }
        }
    }

    પાછા ggml_mem_ranges_check_dst(mrs, tensor);
}

સંરચના ગાંઠ_info {
    ggml_tensor * ગાંઠ;

    std::vector<ggml_tensor *> fused;

    ggml_op op() અચલ {
        પાછા ગાંઠ->op;
    }

    અચલ ggml_tensor * dst() અચલ {
        પાછા fused.empty() ? ગાંઠ : fused.back();
    }

    તાર્કિક is_empty() અચલ {
        પાછા ggml_op_is_empty(ગાંઠ->op);
    }

    શૂન્ય add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    તાર્કિક is_output(અચલ ggml_tensor * t) અચલ {
        જો (t == ગાંઠ) {
            પાછા સાચું;
        }
        માટે (અચલ auto * f : fused) {
            જો (t == f) {
                પાછા સાચું;
            }
        }
        પાછા ખોટું;
    }
};

સ્થિર std::vector<પૂર્ણાંક> ggml_metal_graph_optimize_reorder(અચલ std::vector<ગાંઠ_info> & nodes) {
    // helper to add node src and dst ranges
    અચલ auto & h_add = [](ggml_mem_ranges_t mrs, અચલ ગાંઠ_info & ગાંઠ) {
        // only external sources matter: sources produced by the fused group are internal
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            અચલ ggml_tensor * src = ગાંઠ.ગાંઠ->src[i];
            જો (src && !ગાંઠ.is_output(src)) {
                જો (!ggml_mem_ranges_add_src(mrs, src)) {
                    પાછા ખોટું;
                }
            }
        }

        માટે (અચલ auto * fused : ગાંઠ.fused) {
            માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
                અચલ ggml_tensor * src = fused->src[i];
                જો (src && !ગાંઠ.is_output(src)) {
                    જો (!ggml_mem_ranges_add_src(mrs, src)) {
                        પાછા ખોટું;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        જો (!ggml_mem_ranges_add_dst(mrs, ગાંઠ.ગાંઠ)) {
            પાછા ખોટું;
        }
        માટે (અચલ auto * fused : ગાંઠ.fused) {
            જો (!ggml_mem_ranges_add_dst(mrs, fused)) {
                પાછા ખોટું;
            }
        }

        પાછા સાચું;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    અચલ auto & h_check = [](ggml_mem_ranges_t mrs, અચલ ગાંઠ_info & ગાંઠ) {
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            અચલ ggml_tensor * src = ગાંઠ.ગાંઠ->src[i];
            જો (src && !ગાંઠ.is_output(src)) {
                જો (!ggml_mem_ranges_check_src(mrs, src)) {
                    પાછા ખોટું;
                }
            }
        }

        માટે (અચલ auto * fused : ગાંઠ.fused) {
            માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
                અચલ ggml_tensor * src = fused->src[i];
                જો (src && !ગાંઠ.is_output(src)) {
                    જો (!ggml_mem_ranges_check_src(mrs, src)) {
                        પાછા ખોટું;
                    }
                }
            }
        }

        જો (!ggml_mem_ranges_check_dst(mrs, ગાંઠ.ગાંઠ)) {
            પાછા ખોટું;
        }
        માટે (અચલ auto * fused : ગાંઠ.fused) {
            જો (!ggml_mem_ranges_check_dst(mrs, fused)) {
                પાછા ખોટું;
            }
        }

        પાછા સાચું;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    અચલ auto & h_safe = [](ggml_op op) {
        પસંદગી (op) {
            સ્થિતિ GGML_OP_MUL_MAT:
            સ્થિતિ GGML_OP_MUL_MAT_ID:
            સ્થિતિ GGML_OP_ROPE:
            સ્થિતિ GGML_OP_NORM:
            સ્થિતિ GGML_OP_RMS_NORM:
            સ્થિતિ GGML_OP_GROUP_NORM:
            સ્થિતિ GGML_OP_L2_NORM:
            સ્થિતિ GGML_OP_SUM_ROWS:
            સ્થિતિ GGML_OP_SSM_CONV:
            સ્થિતિ GGML_OP_SSM_SCAN:
            સ્થિતિ GGML_OP_CLAMP:
            સ્થિતિ GGML_OP_TRI:
            સ્થિતિ GGML_OP_DIAG:
            સ્થિતિ GGML_OP_MUL:
            સ્થિતિ GGML_OP_ADD:
            સ્થિતિ GGML_OP_SUB:
            સ્થિતિ GGML_OP_DIV:
            સ્થિતિ GGML_OP_GLU:
            સ્થિતિ GGML_OP_SCALE:
            સ્થિતિ GGML_OP_UNARY:
            સ્થિતિ GGML_OP_GET_ROWS:
            સ્થિતિ GGML_OP_SET_ROWS:
            સ્થિતિ GGML_OP_SET:
            સ્થિતિ GGML_OP_CPY:
            સ્થિતિ GGML_OP_CONT:
            સ્થિતિ GGML_OP_REPEAT:
                પાછા સાચું;
            મૂળભૂત:
                પાછા ggml_op_is_empty(op);
        }
    };

    અચલ પૂર્ણાંક n = nodes.size();

    std::vector<પૂર્ણાંક> res;
    res.reserve(n);

    std::vector<તાર્કિક> used(n, ખોટું);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    માટે (પૂર્ણાંક i0 = 0; i0 < n; i0++) {
        જો (used[i0]) {
            ચાલુ;
        }

        અચલ auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        જો (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr પૂર્ણાંક N_FORWARD = 64;

            માટે (પૂર્ણાંક i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                જો (used[i1]) {
                    ચાલુ;
                }

                અચલ auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                જો (!h_safe(node1.op())) {
                    તોડો;
                }

                અચલ તાર્કિક is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                જો ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = સાચું;
                } નહીંતર {
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

    પાછા res;
}

શૂન્ય ggml_graph_optimize(ggml_cgraph * gf) {
    અચલ પૂર્ણાંક n = gf->n_nodes;

    std::vector<ગાંઠ_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        ગાંઠ_info ગાંઠ = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        અચલ પૂર્ણાંક f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        માટે (પૂર્ણાંક k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            ગાંઠ.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(ગાંઠ));
    }

#જો 1
    // reorder to improve concurrency
    અચલ auto order = ggml_metal_graph_optimize_reorder(nodes);
#નહીંતર
    std::vector<પૂર્ણાંક> order(nodes.size());
    માટે (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        પૂર્ણાંક j = 0;
        માટે (અચલ auto i : order) {
            અચલ auto & ગાંઠ = nodes[i];

            gf->nodes[j++] = ગાંઠ.ગાંઠ;

            માટે (auto * fused : ગાંઠ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
