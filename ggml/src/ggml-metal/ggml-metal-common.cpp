#समावेश "ggml-metal-common.h"
#समावेश "ggml-metal-fusion.h"

#समावेश "ggml.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-backend-impl.h"

#समावेश <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
अटल bool ggml_metal_fwht_supported_आकार(int64_t n, माप_प्रकार max_tg_mem) {
    अगर (n == 64 || n == 128 || n == 256 || n == 512) {
        वापस true;
    }

    अगर (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        वापस (माप_प्रकार) n * माप(भग्न) <= max_tg_mem;
    }

    वापस false;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
bool ggml_metal_op_mul_mat_use_fwht(स्थिर काष्ठा ggml_tensor * op, माप_प्रकार max_tg_mem) {
    वापस ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_आकार(op->src[1]->ne[0], max_tg_mem);
}

bool ggml_metal_op_mul_mat_use_mm(स्थिर काष्ठा ggml_tensor * op, bool has_simdgroup_mm) {
    स्थिर int64_t ne00 = op->src[0]->ne[0];
    स्थिर int64_t ne11 = op->src[1]->ne[1];

    वापस !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool ggml_metal_op_mul_mat_id_use_mm(स्थिर काष्ठा ggml_tensor * op, bool has_simdgroup_mm) {
    स्थिर int64_t ne00 = op->src[0]->ne[0];
    स्थिर int64_t ne02 = op->src[0]->ne[2];
    स्थिर int64_t ne21 = op->src[2]->ne[1];

    वापस has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
अटल constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
अटल constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
अटल constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

क्रमागत ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(क्रमागत ggml_type type, पूर्णांक rt) {
    अगर (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        वापस GGML_METAL_MMA_KIND_BLK;
    }
    वापस type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

पूर्णांक ggml_metal_mul_mv_mma_rt(स्थिर काष्ठा ggml_tensor * op) {
    वापस op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

अटल bool ggml_metal_mul_mv_mma_type_supported(क्रमागत ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_TQ2_0:
            वापस true;
        शेष:
            वापस false;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(क्रमागत ggml_type type, पूर्णांक rt) {
    अगर (!ggml_metal_mul_mv_mma_type_supported(type)) {
        वापस 0;
    }
    वापस ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_आकार(type) : GGML_METAL_MMA_K_CHUNK;
}

अटल bool ggml_metal_mul_mat_mma_type_ok(स्थिर काष्ठा ggml_tensor * op) {
    स्थिर ggml_tensor * src0 = op->src[0];
    स्थिर int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    वापस step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_आकार(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
अटल int64_t ggml_metal_mul_mv_mma_rows_min(क्रमागत ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_F32:
            वापस 6;
        हाल GGML_TYPE_TQ2_0:
            वापस 5;
        हाल GGML_TYPE_BF16:
            वापस 4;
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_MXFP4:
            वापस 3;
        शेष:
            वापस 2;
    }
}

bool ggml_metal_op_mul_mat_use_mma(स्थिर काष्ठा ggml_tensor * op) {
    स्थिर ggml_tensor * src0 = op->src[0];
    स्थिर ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    स्थिर bool batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    वापस ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == माप(भग्न) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

bool ggml_metal_op_mul_mat_may_use_mma(स्थिर काष्ठा ggml_tensor * op) {
    वापस ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
काष्ठा ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

काष्ठा ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    पूर्णांक debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(पूर्णांक debug) {
    स्वतः * res = नया ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    वापस res;
}

व्योम ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    मिटाओ mrs;
}

व्योम ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.सफा();
}

अटल bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    वापस true;
}

अटल ggml_mem_range ggml_mem_range_from_tensor(स्थिर ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    अगर (tensor->बफर) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->बफर,
            /*.p0 =*/ (uint64_t) tensor->तथ्य,
            /*.p1 =*/ (uint64_t) tensor->तथ्य + ggml_backend_buft_get_alloc_आकार(tensor->बफर->buft, tensor),
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

    वापस mr;
}

अटल ggml_mem_range ggml_mem_range_from_tensor_src(स्थिर ggml_tensor * tensor) {
    वापस ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

अटल ggml_mem_range ggml_mem_range_from_tensor_dst(स्थिर ggml_tensor * tensor) {
    वापस ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

अटल bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, स्थिर ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    अगर (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    वापस ggml_mem_ranges_add(mrs, mr);
}

अटल bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, स्थिर ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    अगर (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    वापस ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, स्थिर ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    वापस ggml_mem_ranges_add_dst(mrs, tensor);
}

अटल bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    क्रम (माप_प्रकार i = 0; i < mrs->ranges.size(); i++) {
        स्थिर स्वतः & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        अगर (mr.pb != cmp.pb) {
            जारी;
        }

        // intersecting source ranges are allowed
        अगर (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            जारी;
        }

        अगर (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            अगर (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            वापस false;
        }
    }

    वापस true;
}

अटल bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, स्थिर ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    स्थिर bool res = ggml_mem_ranges_check(mrs, mr);

    वापस res;
}

अटल bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, स्थिर ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    स्थिर bool res = ggml_mem_ranges_check(mrs, mr);

    वापस res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, स्थिर ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (tensor->src[i]) {
            अगर (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                वापस false;
            }
        }
    }

    वापस ggml_mem_ranges_check_dst(mrs, tensor);
}

काष्ठा गाँठ_info {
    ggml_tensor * गाँठ;

    std::vector<ggml_tensor *> fused;

    ggml_op op() स्थिर {
        वापस गाँठ->op;
    }

    स्थिर ggml_tensor * dst() स्थिर {
        वापस fused.empty() ? गाँठ : fused.back();
    }

    bool is_empty() स्थिर {
        वापस ggml_op_is_empty(गाँठ->op);
    }

    व्योम add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    bool is_output(स्थिर ggml_tensor * t) स्थिर {
        अगर (t == गाँठ) {
            वापस true;
        }
        क्रम (स्थिर स्वतः * f : fused) {
            अगर (t == f) {
                वापस true;
            }
        }
        वापस false;
    }
};

अटल std::vector<पूर्णांक> ggml_metal_graph_optimize_reorder(स्थिर std::vector<गाँठ_info> & nodes) {
    // helper to add node src and dst ranges
    स्थिर स्वतः & h_add = [](ggml_mem_ranges_t mrs, स्थिर गाँठ_info & गाँठ) {
        // only external sources matter: sources produced by the fused group are internal
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्थिर ggml_tensor * src = गाँठ.गाँठ->src[i];
            अगर (src && !गाँठ.is_output(src)) {
                अगर (!ggml_mem_ranges_add_src(mrs, src)) {
                    वापस false;
                }
            }
        }

        क्रम (स्थिर स्वतः * fused : गाँठ.fused) {
            क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
                स्थिर ggml_tensor * src = fused->src[i];
                अगर (src && !गाँठ.is_output(src)) {
                    अगर (!ggml_mem_ranges_add_src(mrs, src)) {
                        वापस false;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        अगर (!ggml_mem_ranges_add_dst(mrs, गाँठ.गाँठ)) {
            वापस false;
        }
        क्रम (स्थिर स्वतः * fused : गाँठ.fused) {
            अगर (!ggml_mem_ranges_add_dst(mrs, fused)) {
                वापस false;
            }
        }

        वापस true;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    स्थिर स्वतः & h_check = [](ggml_mem_ranges_t mrs, स्थिर गाँठ_info & गाँठ) {
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्थिर ggml_tensor * src = गाँठ.गाँठ->src[i];
            अगर (src && !गाँठ.is_output(src)) {
                अगर (!ggml_mem_ranges_check_src(mrs, src)) {
                    वापस false;
                }
            }
        }

        क्रम (स्थिर स्वतः * fused : गाँठ.fused) {
            क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
                स्थिर ggml_tensor * src = fused->src[i];
                अगर (src && !गाँठ.is_output(src)) {
                    अगर (!ggml_mem_ranges_check_src(mrs, src)) {
                        वापस false;
                    }
                }
            }
        }

        अगर (!ggml_mem_ranges_check_dst(mrs, गाँठ.गाँठ)) {
            वापस false;
        }
        क्रम (स्थिर स्वतः * fused : गाँठ.fused) {
            अगर (!ggml_mem_ranges_check_dst(mrs, fused)) {
                वापस false;
            }
        }

        वापस true;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    स्थिर स्वतः & h_safe = [](ggml_op op) {
        चयन (op) {
            हाल GGML_OP_MUL_MAT:
            हाल GGML_OP_MUL_MAT_ID:
            हाल GGML_OP_ROPE:
            हाल GGML_OP_NORM:
            हाल GGML_OP_RMS_NORM:
            हाल GGML_OP_GROUP_NORM:
            हाल GGML_OP_L2_NORM:
            हाल GGML_OP_योग_ROWS:
            हाल GGML_OP_SSM_CONV:
            हाल GGML_OP_SSM_SCAN:
            हाल GGML_OP_CLAMP:
            हाल GGML_OP_TRI:
            हाल GGML_OP_DIAG:
            हाल GGML_OP_MUL:
            हाल GGML_OP_ADD:
            हाल GGML_OP_SUB:
            हाल GGML_OP_DIV:
            हाल GGML_OP_GLU:
            हाल GGML_OP_SCALE:
            हाल GGML_OP_UNARY:
            हाल GGML_OP_GET_ROWS:
            हाल GGML_OP_SET_ROWS:
            हाल GGML_OP_SET:
            हाल GGML_OP_CPY:
            हाल GGML_OP_CONT:
            हाल GGML_OP_REPEAT:
                वापस true;
            शेष:
                वापस ggml_op_is_empty(op);
        }
    };

    स्थिर पूर्णांक n = nodes.size();

    std::vector<पूर्णांक> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    क्रम (पूर्णांक i0 = 0; i0 < n; i0++) {
        अगर (used[i0]) {
            जारी;
        }

        स्थिर स्वतः & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        अगर (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr पूर्णांक N_FORWARD = 64;

            क्रम (पूर्णांक i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                अगर (used[i1]) {
                    जारी;
                }

                स्थिर स्वतः & node1 = nodes[i1];

                // disallow reordering of certain ops
                अगर (!h_safe(node1.op())) {
                    अवरोध;
                }

                स्थिर bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                अगर ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
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

    वापस res;
}

व्योम ggml_graph_optimize(ggml_cgraph * gf) {
    स्थिर पूर्णांक n = gf->n_nodes;

    std::vector<गाँठ_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    क्रम (पूर्णांक i = 0; i < n; i++) {
        गाँठ_info गाँठ = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        स्थिर पूर्णांक f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        क्रम (पूर्णांक k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            गाँठ.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(गाँठ));
    }

#अगर 1
    // reorder to improve concurrency
    स्थिर स्वतः order = ggml_metal_graph_optimize_reorder(nodes);
#अन्यथा
    std::vector<पूर्णांक> order(nodes.size());
    क्रम (माप_प्रकार i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#पूर्ण_अगर

    // unfuse
    {
        पूर्णांक j = 0;
        क्रम (स्थिर स्वतः i : order) {
            स्थिर स्वतः & गाँठ = nodes[i];

            gf->nodes[j++] = गाँठ.गाँठ;

            क्रम (स्वतः * fused : गाँठ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
