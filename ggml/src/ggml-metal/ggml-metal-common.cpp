#अन्तर्भाव "ggml-metal-common.h"
#अन्तर्भाव "ggml-metal-fusion.h"

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
स्थिर तार्किक ggml_metal_fwht_supported_size(int64_t n, माप_प्रकार max_tg_mem) {
    जर (n == 64 || n == 128 || n == 256 || n == 512) {
        परत सत्य;
    }

    जर (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        परत (माप_प्रकार) n * आकार(दशांश) <= max_tg_mem;
    }

    परत असत्य;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
तार्किक ggml_metal_op_mul_mat_use_fwht(स्थिरांक संरचना ggml_tensor * op, माप_प्रकार max_tg_mem) {
    परत ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

तार्किक ggml_metal_op_mul_mat_use_mm(स्थिरांक संरचना ggml_tensor * op, तार्किक has_simdgroup_mm) {
    स्थिरांक int64_t ne00 = op->src[0]->ne[0];
    स्थिरांक int64_t ne11 = op->src[1]->ne[1];

    परत !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

तार्किक ggml_metal_op_mul_mat_id_use_mm(स्थिरांक संरचना ggml_tensor * op, तार्किक has_simdgroup_mm) {
    स्थिरांक int64_t ne00 = op->src[0]->ne[0];
    स्थिरांक int64_t ne02 = op->src[0]->ne[2];
    स्थिरांक int64_t ne21 = op->src[2]->ne[1];

    परत has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
स्थिर constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
स्थिर constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
स्थिर constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

गणना ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(गणना ggml_type type, पूर्णांक rt) {
    जर (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        परत GGML_METAL_MMA_KIND_BLK;
    }
    परत type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

पूर्णांक ggml_metal_mul_mv_mma_rt(स्थिरांक संरचना ggml_tensor * op) {
    परत op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

स्थिर तार्किक ggml_metal_mul_mv_mma_type_supported(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_F32:
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_BF16:
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ2_S:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_TQ2_0:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(गणना ggml_type type, पूर्णांक rt) {
    जर (!ggml_metal_mul_mv_mma_type_supported(type)) {
        परत 0;
    }
    परत ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

स्थिर तार्किक ggml_metal_mul_mat_mma_type_ok(स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक ggml_tensor * src0 = op->src[0];
    स्थिरांक int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    परत step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
स्थिर int64_t ggml_metal_mul_mv_mma_rows_min(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_F32:
            परत 6;
        स्थिती GGML_TYPE_TQ2_0:
            परत 5;
        स्थिती GGML_TYPE_BF16:
            परत 4;
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_MXFP4:
            परत 3;
        पूर्वनिर्धारित:
            परत 2;
    }
}

तार्किक ggml_metal_op_mul_mat_use_mma(स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक ggml_tensor * src0 = op->src[0];
    स्थिरांक ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    स्थिरांक तार्किक batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    परत ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == आकार(दशांश) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

तार्किक ggml_metal_op_mul_mat_may_use_mma(स्थिरांक संरचना ggml_tensor * op) {
    परत ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
संरचना ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

संरचना ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    पूर्णांक debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(पूर्णांक debug) {
    स्वतः * res = नवीन ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    परत res;
}

शून्य ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

शून्य ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

स्थिर तार्किक ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    परत सत्य;
}

स्थिर ggml_mem_range ggml_mem_range_from_tensor(स्थिरांक ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    जर (tensor->buffer) {
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
    } अन्यथा {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    परत mr;
}

स्थिर ggml_mem_range ggml_mem_range_from_tensor_src(स्थिरांक ggml_tensor * tensor) {
    परत ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

स्थिर ggml_mem_range ggml_mem_range_from_tensor_dst(स्थिरांक ggml_tensor * tensor) {
    परत ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

स्थिर तार्किक ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, स्थिरांक ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    जर (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    परत ggml_mem_ranges_add(mrs, mr);
}

स्थिर तार्किक ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, स्थिरांक ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    जर (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    परत ggml_mem_ranges_add(mrs, mr);
}

तार्किक ggml_mem_ranges_add(ggml_mem_ranges_t mrs, स्थिरांक ggml_tensor * tensor) {
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        जर (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    परत ggml_mem_ranges_add_dst(mrs, tensor);
}

स्थिर तार्किक ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    साठी (माप_प्रकार i = 0; i < mrs->ranges.size(); i++) {
        स्थिरांक स्वतः & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        जर (mr.pb != cmp.pb) {
            सुरूठेवा;
        }

        // intersecting source ranges are allowed
        जर (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            सुरूठेवा;
        }

        जर (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            जर (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            परत असत्य;
        }
    }

    परत सत्य;
}

स्थिर तार्किक ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, स्थिरांक ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    स्थिरांक तार्किक res = ggml_mem_ranges_check(mrs, mr);

    परत res;
}

स्थिर तार्किक ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, स्थिरांक ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    स्थिरांक तार्किक res = ggml_mem_ranges_check(mrs, mr);

    परत res;
}

तार्किक ggml_mem_ranges_check(ggml_mem_ranges_t mrs, स्थिरांक ggml_tensor * tensor) {
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        जर (tensor->src[i]) {
            जर (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                परत असत्य;
            }
        }
    }

    परत ggml_mem_ranges_check_dst(mrs, tensor);
}

संरचना गाठ_info {
    ggml_tensor * गाठ;

    std::vector<ggml_tensor *> fused;

    ggml_op op() स्थिरांक {
        परत गाठ->op;
    }

    स्थिरांक ggml_tensor * dst() स्थिरांक {
        परत fused.empty() ? गाठ : fused.back();
    }

    तार्किक is_empty() स्थिरांक {
        परत ggml_op_is_empty(गाठ->op);
    }

    शून्य add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    तार्किक is_output(स्थिरांक ggml_tensor * t) स्थिरांक {
        जर (t == गाठ) {
            परत सत्य;
        }
        साठी (स्थिरांक स्वतः * f : fused) {
            जर (t == f) {
                परत सत्य;
            }
        }
        परत असत्य;
    }
};

स्थिर std::vector<पूर्णांक> ggml_metal_graph_optimize_reorder(स्थिरांक std::vector<गाठ_info> & nodes) {
    // helper to add node src and dst ranges
    स्थिरांक स्वतः & h_add = [](ggml_mem_ranges_t mrs, स्थिरांक गाठ_info & गाठ) {
        // only external sources matter: sources produced by the fused group are internal
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्थिरांक ggml_tensor * src = गाठ.गाठ->src[i];
            जर (src && !गाठ.is_output(src)) {
                जर (!ggml_mem_ranges_add_src(mrs, src)) {
                    परत असत्य;
                }
            }
        }

        साठी (स्थिरांक स्वतः * fused : गाठ.fused) {
            साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
                स्थिरांक ggml_tensor * src = fused->src[i];
                जर (src && !गाठ.is_output(src)) {
                    जर (!ggml_mem_ranges_add_src(mrs, src)) {
                        परत असत्य;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        जर (!ggml_mem_ranges_add_dst(mrs, गाठ.गाठ)) {
            परत असत्य;
        }
        साठी (स्थिरांक स्वतः * fused : गाठ.fused) {
            जर (!ggml_mem_ranges_add_dst(mrs, fused)) {
                परत असत्य;
            }
        }

        परत सत्य;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    स्थिरांक स्वतः & h_check = [](ggml_mem_ranges_t mrs, स्थिरांक गाठ_info & गाठ) {
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्थिरांक ggml_tensor * src = गाठ.गाठ->src[i];
            जर (src && !गाठ.is_output(src)) {
                जर (!ggml_mem_ranges_check_src(mrs, src)) {
                    परत असत्य;
                }
            }
        }

        साठी (स्थिरांक स्वतः * fused : गाठ.fused) {
            साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
                स्थिरांक ggml_tensor * src = fused->src[i];
                जर (src && !गाठ.is_output(src)) {
                    जर (!ggml_mem_ranges_check_src(mrs, src)) {
                        परत असत्य;
                    }
                }
            }
        }

        जर (!ggml_mem_ranges_check_dst(mrs, गाठ.गाठ)) {
            परत असत्य;
        }
        साठी (स्थिरांक स्वतः * fused : गाठ.fused) {
            जर (!ggml_mem_ranges_check_dst(mrs, fused)) {
                परत असत्य;
            }
        }

        परत सत्य;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    स्थिरांक स्वतः & h_safe = [](ggml_op op) {
        निवड (op) {
            स्थिती GGML_OP_MUL_MAT:
            स्थिती GGML_OP_MUL_MAT_ID:
            स्थिती GGML_OP_ROPE:
            स्थिती GGML_OP_NORM:
            स्थिती GGML_OP_RMS_NORM:
            स्थिती GGML_OP_GROUP_NORM:
            स्थिती GGML_OP_L2_NORM:
            स्थिती GGML_OP_SUM_ROWS:
            स्थिती GGML_OP_SSM_CONV:
            स्थिती GGML_OP_SSM_SCAN:
            स्थिती GGML_OP_CLAMP:
            स्थिती GGML_OP_TRI:
            स्थिती GGML_OP_DIAG:
            स्थिती GGML_OP_MUL:
            स्थिती GGML_OP_ADD:
            स्थिती GGML_OP_SUB:
            स्थिती GGML_OP_DIV:
            स्थिती GGML_OP_GLU:
            स्थिती GGML_OP_SCALE:
            स्थिती GGML_OP_UNARY:
            स्थिती GGML_OP_GET_ROWS:
            स्थिती GGML_OP_SET_ROWS:
            स्थिती GGML_OP_SET:
            स्थिती GGML_OP_CPY:
            स्थिती GGML_OP_CONT:
            स्थिती GGML_OP_REPEAT:
                परत सत्य;
            पूर्वनिर्धारित:
                परत ggml_op_is_empty(op);
        }
    };

    स्थिरांक पूर्णांक n = nodes.size();

    std::vector<पूर्णांक> res;
    res.reserve(n);

    std::vector<तार्किक> used(n, असत्य);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    साठी (पूर्णांक i0 = 0; i0 < n; i0++) {
        जर (used[i0]) {
            सुरूठेवा;
        }

        स्थिरांक स्वतः & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        जर (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr पूर्णांक N_FORWARD = 64;

            साठी (पूर्णांक i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                जर (used[i1]) {
                    सुरूठेवा;
                }

                स्थिरांक स्वतः & node1 = nodes[i1];

                // disallow reordering of certain ops
                जर (!h_safe(node1.op())) {
                    थांबा;
                }

                स्थिरांक तार्किक is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                जर ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = सत्य;
                } अन्यथा {
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

    परत res;
}

शून्य ggml_graph_optimize(ggml_cgraph * gf) {
    स्थिरांक पूर्णांक n = gf->n_nodes;

    std::vector<गाठ_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    साठी (पूर्णांक i = 0; i < n; i++) {
        गाठ_info गाठ = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        स्थिरांक पूर्णांक f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        साठी (पूर्णांक k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            गाठ.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(गाठ));
    }

#जर 1
    // reorder to improve concurrency
    स्थिरांक स्वतः order = ggml_metal_graph_optimize_reorder(nodes);
#अन्यथा
    std::vector<पूर्णांक> order(nodes.size());
    साठी (माप_प्रकार i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        पूर्णांक j = 0;
        साठी (स्थिरांक स्वतः i : order) {
            स्थिरांक स्वतः & गाठ = nodes[i];

            gf->nodes[j++] = गाठ.गाठ;

            साठी (स्वतः * fused : गाठ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
