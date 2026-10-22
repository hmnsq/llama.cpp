#அந்தர்பாவ "ggml-metal-common.h"
#அந்தர்பாவ "ggml-metal-fusion.h"

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-backend-impl.h"

#அந்தர்பாவ <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
நிலையான தருக்க ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    எனில் (n == 64 || n == 128 || n == 256 || n == 512) {
        திரும்பு உண்மை;
    }

    எனில் (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        திரும்பு (size_t) n * அளவு(தசம) <= max_tg_mem;
    }

    திரும்பு பொய்;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
தருக்க ggml_metal_op_mul_mat_use_fwht(மாறிலி கட்டமைப்பு ggml_tensor * op, size_t max_tg_mem) {
    திரும்பு ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

தருக்க ggml_metal_op_mul_mat_use_mm(மாறிலி கட்டமைப்பு ggml_tensor * op, தருக்க has_simdgroup_mm) {
    மாறிலி int64_t ne00 = op->src[0]->ne[0];
    மாறிலி int64_t ne11 = op->src[1]->ne[1];

    திரும்பு !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

தருக்க ggml_metal_op_mul_mat_id_use_mm(மாறிலி கட்டமைப்பு ggml_tensor * op, தருக்க has_simdgroup_mm) {
    மாறிலி int64_t ne00 = op->src[0]->ne[0];
    மாறிலி int64_t ne02 = op->src[0]->ne[2];
    மாறிலி int64_t ne21 = op->src[2]->ne[1];

    திரும்பு has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
நிலையான constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
நிலையான constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
நிலையான constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

எண்ணிக்கை ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(எண்ணிக்கை ggml_type type, முழு rt) {
    எனில் (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        திரும்பு GGML_METAL_MMA_KIND_BLK;
    }
    திரும்பு type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

முழு ggml_metal_mul_mv_mma_rt(மாறிலி கட்டமைப்பு ggml_tensor * op) {
    திரும்பு op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

நிலையான தருக்க ggml_metal_mul_mv_mma_type_supported(எண்ணிக்கை ggml_type type) {
    தேர்வு (type) {
        நிலை GGML_TYPE_F32:
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_BF16:
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ2_S:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_TQ2_0:
            திரும்பு உண்மை;
        இயல்பு:
            திரும்பு பொய்;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(எண்ணிக்கை ggml_type type, முழு rt) {
    எனில் (!ggml_metal_mul_mv_mma_type_supported(type)) {
        திரும்பு 0;
    }
    திரும்பு ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

நிலையான தருக்க ggml_metal_mul_mat_mma_type_ok(மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி ggml_tensor * src0 = op->src[0];
    மாறிலி int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    திரும்பு step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
நிலையான int64_t ggml_metal_mul_mv_mma_rows_min(எண்ணிக்கை ggml_type type) {
    தேர்வு (type) {
        நிலை GGML_TYPE_F32:
            திரும்பு 6;
        நிலை GGML_TYPE_TQ2_0:
            திரும்பு 5;
        நிலை GGML_TYPE_BF16:
            திரும்பு 4;
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_MXFP4:
            திரும்பு 3;
        இயல்பு:
            திரும்பு 2;
    }
}

தருக்க ggml_metal_op_mul_mat_use_mma(மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி ggml_tensor * src0 = op->src[0];
    மாறிலி ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    மாறிலி தருக்க batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    திரும்பு ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == அளவு(தசம) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

தருக்க ggml_metal_op_mul_mat_may_use_mma(மாறிலி கட்டமைப்பு ggml_tensor * op) {
    திரும்பு ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
கட்டமைப்பு ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

கட்டமைப்பு ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    முழு debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(முழு debug) {
    auto * res = புதிய ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    திரும்பு res;
}

வெறுமை ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

வெறுமை ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

நிலையான தருக்க ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    திரும்பு உண்மை;
}

நிலையான ggml_mem_range ggml_mem_range_from_tensor(மாறிலி ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    எனில் (tensor->buffer) {
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
    } இல்லையெனில் {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    திரும்பு mr;
}

நிலையான ggml_mem_range ggml_mem_range_from_tensor_src(மாறிலி ggml_tensor * tensor) {
    திரும்பு ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

நிலையான ggml_mem_range ggml_mem_range_from_tensor_dst(மாறிலி ggml_tensor * tensor) {
    திரும்பு ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

நிலையான தருக்க ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, மாறிலி ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    எனில் (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    திரும்பு ggml_mem_ranges_add(mrs, mr);
}

நிலையான தருக்க ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, மாறிலி ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    எனில் (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    திரும்பு ggml_mem_ranges_add(mrs, mr);
}

தருக்க ggml_mem_ranges_add(ggml_mem_ranges_t mrs, மாறிலி ggml_tensor * tensor) {
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        எனில் (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    திரும்பு ggml_mem_ranges_add_dst(mrs, tensor);
}

நிலையான தருக்க ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    க்கு (size_t i = 0; i < mrs->ranges.size(); i++) {
        மாறிலி auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        எனில் (mr.pb != cmp.pb) {
            தொடர்;
        }

        // intersecting source ranges are allowed
        எனில் (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            தொடர்;
        }

        எனில் (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            எனில் (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, மாறிலி ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    மாறிலி தருக்க res = ggml_mem_ranges_check(mrs, mr);

    திரும்பு res;
}

நிலையான தருக்க ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, மாறிலி ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    மாறிலி தருக்க res = ggml_mem_ranges_check(mrs, mr);

    திரும்பு res;
}

தருக்க ggml_mem_ranges_check(ggml_mem_ranges_t mrs, மாறிலி ggml_tensor * tensor) {
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        எனில் (tensor->src[i]) {
            எனில் (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                திரும்பு பொய்;
            }
        }
    }

    திரும்பு ggml_mem_ranges_check_dst(mrs, tensor);
}

கட்டமைப்பு முனை_info {
    ggml_tensor * முனை;

    std::vector<ggml_tensor *> fused;

    ggml_op op() மாறிலி {
        திரும்பு முனை->op;
    }

    மாறிலி ggml_tensor * dst() மாறிலி {
        திரும்பு fused.empty() ? முனை : fused.back();
    }

    தருக்க is_empty() மாறிலி {
        திரும்பு ggml_op_is_empty(முனை->op);
    }

    வெறுமை add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    தருக்க is_output(மாறிலி ggml_tensor * t) மாறிலி {
        எனில் (t == முனை) {
            திரும்பு உண்மை;
        }
        க்கு (மாறிலி auto * f : fused) {
            எனில் (t == f) {
                திரும்பு உண்மை;
            }
        }
        திரும்பு பொய்;
    }
};

நிலையான std::vector<முழு> ggml_metal_graph_optimize_reorder(மாறிலி std::vector<முனை_info> & nodes) {
    // helper to add node src and dst ranges
    மாறிலி auto & h_add = [](ggml_mem_ranges_t mrs, மாறிலி முனை_info & முனை) {
        // only external sources matter: sources produced by the fused group are internal
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            மாறிலி ggml_tensor * src = முனை.முனை->src[i];
            எனில் (src && !முனை.is_output(src)) {
                எனில் (!ggml_mem_ranges_add_src(mrs, src)) {
                    திரும்பு பொய்;
                }
            }
        }

        க்கு (மாறிலி auto * fused : முனை.fused) {
            க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
                மாறிலி ggml_tensor * src = fused->src[i];
                எனில் (src && !முனை.is_output(src)) {
                    எனில் (!ggml_mem_ranges_add_src(mrs, src)) {
                        திரும்பு பொய்;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        எனில் (!ggml_mem_ranges_add_dst(mrs, முனை.முனை)) {
            திரும்பு பொய்;
        }
        க்கு (மாறிலி auto * fused : முனை.fused) {
            எனில் (!ggml_mem_ranges_add_dst(mrs, fused)) {
                திரும்பு பொய்;
            }
        }

        திரும்பு உண்மை;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    மாறிலி auto & h_check = [](ggml_mem_ranges_t mrs, மாறிலி முனை_info & முனை) {
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            மாறிலி ggml_tensor * src = முனை.முனை->src[i];
            எனில் (src && !முனை.is_output(src)) {
                எனில் (!ggml_mem_ranges_check_src(mrs, src)) {
                    திரும்பு பொய்;
                }
            }
        }

        க்கு (மாறிலி auto * fused : முனை.fused) {
            க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
                மாறிலி ggml_tensor * src = fused->src[i];
                எனில் (src && !முனை.is_output(src)) {
                    எனில் (!ggml_mem_ranges_check_src(mrs, src)) {
                        திரும்பு பொய்;
                    }
                }
            }
        }

        எனில் (!ggml_mem_ranges_check_dst(mrs, முனை.முனை)) {
            திரும்பு பொய்;
        }
        க்கு (மாறிலி auto * fused : முனை.fused) {
            எனில் (!ggml_mem_ranges_check_dst(mrs, fused)) {
                திரும்பு பொய்;
            }
        }

        திரும்பு உண்மை;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    மாறிலி auto & h_safe = [](ggml_op op) {
        தேர்வு (op) {
            நிலை GGML_OP_MUL_MAT:
            நிலை GGML_OP_MUL_MAT_ID:
            நிலை GGML_OP_ROPE:
            நிலை GGML_OP_NORM:
            நிலை GGML_OP_RMS_NORM:
            நிலை GGML_OP_GROUP_NORM:
            நிலை GGML_OP_L2_NORM:
            நிலை GGML_OP_SUM_ROWS:
            நிலை GGML_OP_SSM_CONV:
            நிலை GGML_OP_SSM_SCAN:
            நிலை GGML_OP_CLAMP:
            நிலை GGML_OP_TRI:
            நிலை GGML_OP_DIAG:
            நிலை GGML_OP_MUL:
            நிலை GGML_OP_ADD:
            நிலை GGML_OP_SUB:
            நிலை GGML_OP_DIV:
            நிலை GGML_OP_GLU:
            நிலை GGML_OP_SCALE:
            நிலை GGML_OP_UNARY:
            நிலை GGML_OP_GET_ROWS:
            நிலை GGML_OP_SET_ROWS:
            நிலை GGML_OP_SET:
            நிலை GGML_OP_CPY:
            நிலை GGML_OP_CONT:
            நிலை GGML_OP_REPEAT:
                திரும்பு உண்மை;
            இயல்பு:
                திரும்பு ggml_op_is_empty(op);
        }
    };

    மாறிலி முழு n = nodes.size();

    std::vector<முழு> res;
    res.reserve(n);

    std::vector<தருக்க> used(n, பொய்);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    க்கு (முழு i0 = 0; i0 < n; i0++) {
        எனில் (used[i0]) {
            தொடர்;
        }

        மாறிலி auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        எனில் (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr முழு N_FORWARD = 64;

            க்கு (முழு i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                எனில் (used[i1]) {
                    தொடர்;
                }

                மாறிலி auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                எனில் (!h_safe(node1.op())) {
                    நிறுத்து;
                }

                மாறிலி தருக்க is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                எனில் ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = உண்மை;
                } இல்லையெனில் {
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

    திரும்பு res;
}

வெறுமை ggml_graph_optimize(ggml_cgraph * gf) {
    மாறிலி முழு n = gf->n_nodes;

    std::vector<முனை_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    க்கு (முழு i = 0; i < n; i++) {
        முனை_info முனை = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        மாறிலி முழு f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        க்கு (முழு k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            முனை.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(முனை));
    }

#எனில் 1
    // reorder to improve concurrency
    மாறிலி auto order = ggml_metal_graph_optimize_reorder(nodes);
#இல்லையெனில்
    std::vector<முழு> order(nodes.size());
    க்கு (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        முழு j = 0;
        க்கு (மாறிலி auto i : order) {
            மாறிலி auto & முனை = nodes[i];

            gf->nodes[j++] = முனை.முனை;

            க்கு (auto * fused : முனை.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
