#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
static bool ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    නම් (n == 64 || n == 128 || n == 256 || n == 512) {
        ආපසු true;
    }

    නම් (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        ආපසු (size_t) n * parimaa_na(float) <= max_tg_mem;
    }

    ආපසු false;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
bool ggml_metal_op_mul_mat_use_fwht(const sa_nracanaa ggml_tensor * op, size_t max_tg_mem) {
    ආපසු ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

bool ggml_metal_op_mul_mat_use_mm(const sa_nracanaa ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    ආපසු !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool ggml_metal_op_mul_mat_id_use_mm(const sa_nracanaa ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne21 = op->src[2]->ne[1];

    ආපසු has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

ga_nanaa ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(ga_nanaa ggml_type type, පූර්ණ_සංඛ්‍යා rt) {
    නම් (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        ආපසු GGML_METAL_MMA_KIND_BLK;
    }
    ආපසු type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

පූර්ණ_සංඛ්‍යා ggml_metal_mul_mv_mma_rt(const sa_nracanaa ggml_tensor * op) {
    ආපසු op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

static bool ggml_metal_mul_mv_mma_type_supported(ga_nanaa ggml_type type) {
    switch (type) {
        pak_sa GGML_TYPE_F32:
        pak_sa GGML_TYPE_F16:
        pak_sa GGML_TYPE_Q4_0:
        pak_sa GGML_TYPE_Q4_1:
        pak_sa GGML_TYPE_Q5_0:
        pak_sa GGML_TYPE_Q5_1:
        pak_sa GGML_TYPE_Q8_0:
        pak_sa GGML_TYPE_Q4_K:
        pak_sa GGML_TYPE_Q5_K:
        pak_sa GGML_TYPE_Q6_K:
        pak_sa GGML_TYPE_BF16:
        pak_sa GGML_TYPE_Q1_0:
        pak_sa GGML_TYPE_Q2_0:
        pak_sa GGML_TYPE_MXFP4:
        pak_sa GGML_TYPE_Q2_K:
        pak_sa GGML_TYPE_Q3_K:
        pak_sa GGML_TYPE_IQ2_XXS:
        pak_sa GGML_TYPE_IQ2_XS:
        pak_sa GGML_TYPE_IQ2_S:
        pak_sa GGML_TYPE_IQ3_XXS:
        pak_sa GGML_TYPE_IQ3_S:
        pak_sa GGML_TYPE_IQ1_S:
        pak_sa GGML_TYPE_IQ1_M:
        pak_sa GGML_TYPE_IQ4_NL:
        pak_sa GGML_TYPE_IQ4_XS:
        pak_sa GGML_TYPE_TQ2_0:
            ආපසු true;
        default:
            ආපසු false;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(ga_nanaa ggml_type type, පූර්ණ_සංඛ්‍යා rt) {
    නම් (!ggml_metal_mul_mv_mma_type_supported(type)) {
        ආපසු 0;
    }
    ආපසු ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

static bool ggml_metal_mul_mat_mma_type_ok(const sa_nracanaa ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    ආපසු step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
static int64_t ggml_metal_mul_mv_mma_rows_min(ga_nanaa ggml_type type) {
    switch (type) {
        pak_sa GGML_TYPE_F32:
            ආපසු 6;
        pak_sa GGML_TYPE_TQ2_0:
            ආපසු 5;
        pak_sa GGML_TYPE_BF16:
            ආපසු 4;
        pak_sa GGML_TYPE_F16:
        pak_sa GGML_TYPE_Q2_0:
        pak_sa GGML_TYPE_Q4_K:
        pak_sa GGML_TYPE_Q5_0:
        pak_sa GGML_TYPE_Q5_1:
        pak_sa GGML_TYPE_Q2_K:
        pak_sa GGML_TYPE_IQ4_NL:
        pak_sa GGML_TYPE_MXFP4:
            ආපසු 3;
        default:
            ආපසු 2;
    }
}

bool ggml_metal_op_mul_mat_use_mma(const sa_nracanaa ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    const bool batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    ආපසු ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == parimaa_na(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

bool ggml_metal_op_mul_mat_may_use_mma(const sa_nracanaa ggml_tensor * op) {
    ආපසු ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
sa_nracanaa ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

sa_nracanaa ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    පූර්ණ_සංඛ්‍යා debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(පූර්ණ_සංඛ්‍යා debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    ආපසු res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    ආපසු true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    නම් (tensor->buffer) {
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
    } නැතහොත් {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    ආපසු mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    ආපසු ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    ආපසු ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    නම් (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ආපසු ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    නම් (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    ආපසු ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < GGML_MAX_SRC; i++) {
        නම් (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    ආපසු ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    සඳහා (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        නම් (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        නම් (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        නම් (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            නම් (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            ආපසු false;
        }
    }

    ආපසු true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    ආපසු res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    ආපසු res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < GGML_MAX_SRC; i++) {
        නම් (tensor->src[i]) {
            නම් (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                ආපසු false;
            }
        }
    }

    ආපසු ggml_mem_ranges_check_dst(mrs, tensor);
}

sa_nracanaa node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        ආපසු node->op;
    }

    const ggml_tensor * dst() const {
        ආපසු fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        ආපසු ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    bool is_output(const ggml_tensor * t) const {
        නම් (t == node) {
            ආපසු true;
        }
        සඳහා (const auto * f : fused) {
            නම් (t == f) {
                ආපසු true;
            }
        }
        ආපසු false;
    }
};

static std::vector<පූර්ණ_සංඛ්‍යා> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        // only external sources matter: sources produced by the fused group are internal
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = node.node->src[i];
            නම් (src && !node.is_output(src)) {
                නම් (!ggml_mem_ranges_add_src(mrs, src)) {
                    ආපසු false;
                }
            }
        }

        සඳහා (const auto * fused : node.fused) {
            සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < GGML_MAX_SRC; i++) {
                const ggml_tensor * src = fused->src[i];
                නම් (src && !node.is_output(src)) {
                    නම් (!ggml_mem_ranges_add_src(mrs, src)) {
                        ආපසු false;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        නම් (!ggml_mem_ranges_add_dst(mrs, node.node)) {
            ආපසු false;
        }
        සඳහා (const auto * fused : node.fused) {
            නම් (!ggml_mem_ranges_add_dst(mrs, fused)) {
                ආපසු false;
            }
        }

        ආපසු true;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = node.node->src[i];
            නම් (src && !node.is_output(src)) {
                නම් (!ggml_mem_ranges_check_src(mrs, src)) {
                    ආපසු false;
                }
            }
        }

        සඳහා (const auto * fused : node.fused) {
            සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < GGML_MAX_SRC; i++) {
                const ggml_tensor * src = fused->src[i];
                නම් (src && !node.is_output(src)) {
                    නම් (!ggml_mem_ranges_check_src(mrs, src)) {
                        ආපසු false;
                    }
                }
            }
        }

        නම් (!ggml_mem_ranges_check_dst(mrs, node.node)) {
            ආපසු false;
        }
        සඳහා (const auto * fused : node.fused) {
            නම් (!ggml_mem_ranges_check_dst(mrs, fused)) {
                ආපසු false;
            }
        }

        ආපසු true;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            pak_sa GGML_OP_MUL_MAT:
            pak_sa GGML_OP_MUL_MAT_ID:
            pak_sa GGML_OP_ROPE:
            pak_sa GGML_OP_NORM:
            pak_sa GGML_OP_RMS_NORM:
            pak_sa GGML_OP_GROUP_NORM:
            pak_sa GGML_OP_L2_NORM:
            pak_sa GGML_OP_SUM_ROWS:
            pak_sa GGML_OP_SSM_CONV:
            pak_sa GGML_OP_SSM_SCAN:
            pak_sa GGML_OP_CLAMP:
            pak_sa GGML_OP_TRI:
            pak_sa GGML_OP_DIAG:
            pak_sa GGML_OP_MUL:
            pak_sa GGML_OP_ADD:
            pak_sa GGML_OP_SUB:
            pak_sa GGML_OP_DIV:
            pak_sa GGML_OP_GLU:
            pak_sa GGML_OP_SCALE:
            pak_sa GGML_OP_UNARY:
            pak_sa GGML_OP_GET_ROWS:
            pak_sa GGML_OP_SET_ROWS:
            pak_sa GGML_OP_SET:
            pak_sa GGML_OP_CPY:
            pak_sa GGML_OP_CONT:
            pak_sa GGML_OP_REPEAT:
                ආපසු true;
            default:
                ආපසු ggml_op_is_empty(op);
        }
    };

    const පූර්ණ_සංඛ්‍යා n = nodes.size();

    std::vector<පූර්ණ_සංඛ්‍යා> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    සඳහා (පූර්ණ_සංඛ්‍යා i0 = 0; i0 < n; i0++) {
        නම් (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        නම් (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr පූර්ණ_සංඛ්‍යා N_FORWARD = 64;

            සඳහා (පූර්ණ_සංඛ්‍යා i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                නම් (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                නම් (!h_safe(node1.op())) {
                    bha_nga;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                නම් ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } නැතහොත් {
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

    ආපසු res;
}

void ggml_graph_optimize(ggml_cgraph * gf) {
    const පූර්ණ_සංඛ්‍යා n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        const පූර්ණ_සංඛ්‍යා f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        සඳහා (පූර්ණ_සංඛ්‍යා k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }

#නම් 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#නැතහොත්
    std::vector<පූර්ණ_සංඛ්‍යා> order(nodes.size());
    සඳහා (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        පූර්ණ_සංඛ්‍යා j = 0;
        සඳහා (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            සඳහා (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
