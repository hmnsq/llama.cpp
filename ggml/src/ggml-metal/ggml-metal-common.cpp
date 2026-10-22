#ضمن "ggml-metal-common.h"
#ضمن "ggml-metal-fusion.h"

#ضمن "ggml.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"

#ضمن <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
ساكن منطقي ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    إذا (n == 64 || n == 128 || n == 256 || n == 512) {
        أرجع true;
    }

    إذا (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        أرجع (size_t) n * حجم(عائم) <= max_tg_mem;
    }

    أرجع خطأ;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
منطقي ggml_metal_op_mul_mat_use_fwht(ثابت بنية ggml_tensor * op, size_t max_tg_mem) {
    أرجع ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

منطقي ggml_metal_op_mul_mat_use_mm(ثابت بنية ggml_tensor * op, منطقي has_simdgroup_mm) {
    ثابت int64_t ne00 = op->src[0]->ne[0];
    ثابت int64_t ne11 = op->src[1]->ne[1];

    أرجع !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

منطقي ggml_metal_op_mul_mat_id_use_mm(ثابت بنية ggml_tensor * op, منطقي has_simdgroup_mm) {
    ثابت int64_t ne00 = op->src[0]->ne[0];
    ثابت int64_t ne02 = op->src[0]->ne[2];
    ثابت int64_t ne21 = op->src[2]->ne[1];

    أرجع has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
ساكن constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
ساكن constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
ساكن constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

تعداد ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(تعداد ggml_type type, صحيح rt) {
    إذا (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        أرجع GGML_METAL_MMA_KIND_BLK;
    }
    أرجع type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

صحيح ggml_metal_mul_mv_mma_rt(ثابت بنية ggml_tensor * op) {
    أرجع op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

ساكن منطقي ggml_metal_mul_mv_mma_type_supported(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_TQ2_0:
            أرجع true;
        افتراضي:
            أرجع خطأ;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(تعداد ggml_type type, صحيح rt) {
    إذا (!ggml_metal_mul_mv_mma_type_supported(type)) {
        أرجع 0;
    }
    أرجع ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

ساكن منطقي ggml_metal_mul_mat_mma_type_ok(ثابت بنية ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    أرجع step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
ساكن int64_t ggml_metal_mul_mv_mma_rows_min(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_F32:
            أرجع 6;
        حالة GGML_TYPE_TQ2_0:
            أرجع 5;
        حالة GGML_TYPE_BF16:
            أرجع 4;
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
            أرجع 3;
        افتراضي:
            أرجع 2;
    }
}

منطقي ggml_metal_op_mul_mat_use_mma(ثابت بنية ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ثابت منطقي batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    أرجع ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == حجم(عائم) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

منطقي ggml_metal_op_mul_mat_may_use_mma(ثابت بنية ggml_tensor * op) {
    أرجع ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
بنية ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

بنية ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    صحيح debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(صحيح debug) {
    auto * res = جديد ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    أرجع res;
}

فراغ ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

فراغ ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

ساكن منطقي ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    أرجع true;
}

ساكن ggml_mem_range ggml_mem_range_from_tensor(ثابت ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    إذا (tensor->buffer) {
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
    } وإلا {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    أرجع mr;
}

ساكن ggml_mem_range ggml_mem_range_from_tensor_src(ثابت ggml_tensor * tensor) {
    أرجع ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

ساكن ggml_mem_range ggml_mem_range_from_tensor_dst(ثابت ggml_tensor * tensor) {
    أرجع ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

ساكن منطقي ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    إذا (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    أرجع ggml_mem_ranges_add(mrs, mr);
}

ساكن منطقي ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    إذا (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    أرجع ggml_mem_ranges_add(mrs, mr);
}

منطقي ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    أرجع ggml_mem_ranges_add_dst(mrs, tensor);
}

ساكن منطقي ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    لكل (size_t i = 0; i < mrs->ranges.size(); i++) {
        ثابت auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        إذا (mr.pb != cmp.pb) {
            استمر;
        }

        // intersecting source ranges are allowed
        إذا (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            استمر;
        }

        إذا (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            إذا (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            أرجع خطأ;
        }
    }

    أرجع true;
}

ساكن منطقي ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ثابت منطقي res = ggml_mem_ranges_check(mrs, mr);

    أرجع res;
}

ساكن منطقي ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ثابت منطقي res = ggml_mem_ranges_check(mrs, mr);

    أرجع res;
}

منطقي ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i]) {
            إذا (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                أرجع خطأ;
            }
        }
    }

    أرجع ggml_mem_ranges_check_dst(mrs, tensor);
}

بنية عقدة_info {
    ggml_tensor * عقدة;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ثابت {
        أرجع عقدة->op;
    }

    ثابت ggml_tensor * dst() ثابت {
        أرجع fused.empty() ? عقدة : fused.back();
    }

    منطقي is_empty() ثابت {
        أرجع ggml_op_is_empty(عقدة->op);
    }

    فراغ add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    منطقي is_output(ثابت ggml_tensor * t) ثابت {
        إذا (t == عقدة) {
            أرجع true;
        }
        لكل (ثابت auto * f : fused) {
            إذا (t == f) {
                أرجع true;
            }
        }
        أرجع خطأ;
    }
};

ساكن std::vector<صحيح> ggml_metal_graph_optimize_reorder(ثابت std::vector<عقدة_info> & nodes) {
    // helper to add node src and dst ranges
    ثابت auto & h_add = [](ggml_mem_ranges_t mrs, ثابت عقدة_info & عقدة) {
        // only external sources matter: sources produced by the fused group are internal
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت ggml_tensor * src = عقدة.عقدة->src[i];
            إذا (src && !عقدة.is_output(src)) {
                إذا (!ggml_mem_ranges_add_src(mrs, src)) {
                    أرجع خطأ;
                }
            }
        }

        لكل (ثابت auto * fused : عقدة.fused) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                ثابت ggml_tensor * src = fused->src[i];
                إذا (src && !عقدة.is_output(src)) {
                    إذا (!ggml_mem_ranges_add_src(mrs, src)) {
                        أرجع خطأ;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        إذا (!ggml_mem_ranges_add_dst(mrs, عقدة.عقدة)) {
            أرجع خطأ;
        }
        لكل (ثابت auto * fused : عقدة.fused) {
            إذا (!ggml_mem_ranges_add_dst(mrs, fused)) {
                أرجع خطأ;
            }
        }

        أرجع true;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ثابت auto & h_check = [](ggml_mem_ranges_t mrs, ثابت عقدة_info & عقدة) {
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت ggml_tensor * src = عقدة.عقدة->src[i];
            إذا (src && !عقدة.is_output(src)) {
                إذا (!ggml_mem_ranges_check_src(mrs, src)) {
                    أرجع خطأ;
                }
            }
        }

        لكل (ثابت auto * fused : عقدة.fused) {
            لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
                ثابت ggml_tensor * src = fused->src[i];
                إذا (src && !عقدة.is_output(src)) {
                    إذا (!ggml_mem_ranges_check_src(mrs, src)) {
                        أرجع خطأ;
                    }
                }
            }
        }

        إذا (!ggml_mem_ranges_check_dst(mrs, عقدة.عقدة)) {
            أرجع خطأ;
        }
        لكل (ثابت auto * fused : عقدة.fused) {
            إذا (!ggml_mem_ranges_check_dst(mrs, fused)) {
                أرجع خطأ;
            }
        }

        أرجع true;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ثابت auto & h_safe = [](ggml_op op) {
        بدل (op) {
            حالة GGML_OP_MUL_MAT:
            حالة GGML_OP_MUL_MAT_ID:
            حالة GGML_OP_ROPE:
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_GROUP_NORM:
            حالة GGML_OP_L2_NORM:
            حالة GGML_OP_SUM_ROWS:
            حالة GGML_OP_SSM_CONV:
            حالة GGML_OP_SSM_SCAN:
            حالة GGML_OP_CLAMP:
            حالة GGML_OP_TRI:
            حالة GGML_OP_DIAG:
            حالة GGML_OP_MUL:
            حالة GGML_OP_ADD:
            حالة GGML_OP_SUB:
            حالة GGML_OP_DIV:
            حالة GGML_OP_GLU:
            حالة GGML_OP_SCALE:
            حالة GGML_OP_UNARY:
            حالة GGML_OP_GET_ROWS:
            حالة GGML_OP_SET_ROWS:
            حالة GGML_OP_SET:
            حالة GGML_OP_CPY:
            حالة GGML_OP_CONT:
            حالة GGML_OP_REPEAT:
                أرجع true;
            افتراضي:
                أرجع ggml_op_is_empty(op);
        }
    };

    ثابت صحيح n = nodes.size();

    std::vector<صحيح> res;
    res.reserve(n);

    std::vector<منطقي> used(n, خطأ);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    لكل (صحيح i0 = 0; i0 < n; i0++) {
        إذا (used[i0]) {
            استمر;
        }

        ثابت auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        إذا (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr صحيح N_FORWARD = 64;

            لكل (صحيح i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                إذا (used[i1]) {
                    استمر;
                }

                ثابت auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                إذا (!h_safe(node1.op())) {
                    اكسر;
                }

                ثابت منطقي is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                إذا ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } وإلا {
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

    أرجع res;
}

فراغ ggml_graph_optimize(ggml_cgraph * gf) {
    ثابت صحيح n = gf->n_nodes;

    std::vector<عقدة_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    لكل (صحيح i = 0; i < n; i++) {
        عقدة_info عقدة = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ثابت صحيح f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        لكل (صحيح k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            عقدة.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(عقدة));
    }

#إذا 1
    // reorder to improve concurrency
    ثابت auto order = ggml_metal_graph_optimize_reorder(nodes);
#وإلا
    std::vector<صحيح> order(nodes.size());
    لكل (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        صحيح j = 0;
        لكل (ثابت auto i : order) {
            ثابت auto & عقدة = nodes[i];

            gf->nodes[j++] = عقدة.عقدة;

            لكل (auto * fused : عقدة.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
