#شامل "ggml-metal-common.h"
#شامل "ggml-metal-fusion.h"

#شامل "ggml.h"
#شامل "ggml-impl.h"
#شامل "ggml-backend-impl.h"

#شامل <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
ساکن منطقی ggml_metal_fwht_supported_سائز(int64_t n, size_t max_tg_mem) {
    اگر (n == 64 || n == 128 || n == 256 || n == 512) {
        واپس سچ;
    }

    اگر (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        واپس (size_t) n * حجم(اعشاری) <= max_tg_mem;
    }

    واپس جھوٹ;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
منطقی ggml_metal_op_mul_mat_use_fwht(ثابت ساخت ggml_tensor * op, size_t max_tg_mem) {
    واپس ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_سائز(op->src[1]->ne[0], max_tg_mem);
}

منطقی ggml_metal_op_mul_mat_use_mm(ثابت ساخت ggml_tensor * op, منطقی has_simdgroup_mm) {
    ثابت int64_t ne00 = op->src[0]->ne[0];
    ثابت int64_t ne11 = op->src[1]->ne[1];

    واپس !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

منطقی ggml_metal_op_mul_mat_id_use_mm(ثابت ساخت ggml_tensor * op, منطقی has_simdgroup_mm) {
    ثابت int64_t ne00 = op->src[0]->ne[0];
    ثابت int64_t ne02 = op->src[0]->ne[2];
    ثابت int64_t ne21 = op->src[2]->ne[1];

    واپس has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
ساکن constexpr int64_t GGML_METAL_MMA_صفیں_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
ساکن constexpr int64_t GGML_METAL_MMA_TILE_صفیں = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
ساکن constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

شمار ggml_metal_mma_قسم ggml_metal_mul_mv_mma_قسم(شمار ggml_type type, صحیح rt) {
    اگر (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        واپس GGML_METAL_MMA_قسم_BLK;
    }
    واپس type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_قسم_Q5_K : GGML_METAL_MMA_قسم_GEN;
}

صحیح ggml_metal_mul_mv_mma_rt(ثابت ساخت ggml_tensor * op) {
    واپس op->src[1]->ne[1] > GGML_METAL_MMA_TILE_صفیں ? 2 : 1;
}

ساکن منطقی ggml_metal_mul_mv_mma_type_supported(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_BF16:
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_TQ2_0:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

int64_t ggml_metal_mul_mv_mma_k_قدم(شمار ggml_type type, صحیح rt) {
    اگر (!ggml_metal_mul_mv_mma_type_supported(type)) {
        واپس 0;
    }
    واپس ggml_metal_mul_mv_mma_قسم(type, rt) == GGML_METAL_MMA_قسم_BLK ? ggml_blck_سائز(type) : GGML_METAL_MMA_K_CHUNK;
}

ساکن منطقی ggml_metal_mul_mat_mma_type_ok(ثابت ساخت ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت int64_t قدم = ggml_metal_mul_mv_mma_k_قدم(src0->type, ggml_metal_mul_mv_mma_rt(op));

    واپس قدم > 0 && src0->ne[0] % قدم == 0 && src0->nb[0] == ggml_type_سائز(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
ساکن int64_t ggml_metal_mul_mv_mma_صفیں_min(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_F32:
            واپس 6;
        حالت GGML_TYPE_TQ2_0:
            واپس 5;
        حالت GGML_TYPE_BF16:
            واپس 4;
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_MXFP4:
            واپس 3;
        طےشدہ:
            واپس 2;
    }
}

منطقی ggml_metal_op_mul_mat_use_mma(ثابت ساخت ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    ثابت منطقی batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    واپس ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_صفیں_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_صفیں_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == حجم(اعشاری) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

منطقی ggml_metal_op_mul_mat_may_use_mma(ثابت ساخت ggml_tensor * op) {
    واپس ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
ساخت ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; //انتہا

    ggml_mem_range_type pt;
};

ساخت ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    صحیح debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(صحیح debug) {
    خودکار * res = نیا ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    واپس res;
}

خالی ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    حذف mrs;
}

خالی ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

ساکن منطقی ggml_mem_ranges_جوڑو(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    واپس سچ;
}

ساکن ggml_mem_range ggml_mem_range_from_tensor(ثابت ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    اگر (tensor->بفر) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->بفر,
            /*.p0 =*/ (uint64_t) tensor->ڈیٹا,
            /*.p1 =*/ (uint64_t) tensor->ڈیٹا + ggml_backend_buft_get_alloc_سائز(tensor->بفر->buft, tensor),
            /*.pt =*/ pt,
        };
    } ورنہ {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    واپس mr;
}

ساکن ggml_mem_range ggml_mem_range_from_tensor_src(ثابت ggml_tensor * tensor) {
    واپس ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

ساکن ggml_mem_range ggml_mem_range_from_tensor_dst(ثابت ggml_tensor * tensor) {
    واپس ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

ساکن منطقی ggml_mem_ranges_جوڑو_src(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    اگر (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    واپس ggml_mem_ranges_جوڑو(mrs, mr);
}

ساکن منطقی ggml_mem_ranges_جوڑو_dst(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    اگر (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    واپس ggml_mem_ranges_جوڑو(mrs, mr);
}

منطقی ggml_mem_ranges_جوڑو(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (tensor->src[i]) {
            ggml_mem_ranges_جوڑو_src(mrs, tensor->src[i]);
        }
    }

    واپس ggml_mem_ranges_جوڑو_dst(mrs, tensor);
}

ساکن منطقی ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    برائے (size_t i = 0; i < mrs->ranges.size(); i++) {
        ثابت خودکار & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        اگر (mr.pb != cmp.pb) {
            جاری;
        }

        // intersecting source ranges are allowed
        اگر (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            جاری;
        }

        اگر (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            اگر (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            واپس جھوٹ;
        }
    }

    واپس سچ;
}

ساکن منطقی ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    ثابت منطقی res = ggml_mem_ranges_check(mrs, mr);

    واپس res;
}

ساکن منطقی ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    ثابت منطقی res = ggml_mem_ranges_check(mrs, mr);

    واپس res;
}

منطقی ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ثابت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (tensor->src[i]) {
            اگر (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                واپس جھوٹ;
            }
        }
    }

    واپس ggml_mem_ranges_check_dst(mrs, tensor);
}

ساخت گرہ_info {
    ggml_tensor * گرہ;

    std::vector<ggml_tensor *> fused;

    ggml_op op() ثابت {
        واپس گرہ->op;
    }

    ثابت ggml_tensor * dst() ثابت {
        واپس fused.خالی_خانے() ? گرہ : fused.back();
    }

    منطقی is_خالی_خانے() ثابت {
        واپس ggml_op_is_خالی_خانے(گرہ->op);
    }

    خالی جوڑو_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    منطقی is_output(ثابت ggml_tensor * t) ثابت {
        اگر (t == گرہ) {
            واپس سچ;
        }
        برائے (ثابت خودکار * f : fused) {
            اگر (t == f) {
                واپس سچ;
            }
        }
        واپس جھوٹ;
    }
};

ساکن std::vector<صحیح> ggml_metal_گراف_optimize_reorder(ثابت std::vector<گرہ_info> & nodes) {
    // helper to add node src and dst ranges
    ثابت خودکار & h_جوڑو = [](ggml_mem_ranges_t mrs, ثابت گرہ_info & گرہ) {
        // only external sources matter: sources produced by the fused group are internal
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت ggml_tensor * src = گرہ.گرہ->src[i];
            اگر (src && !گرہ.is_output(src)) {
                اگر (!ggml_mem_ranges_جوڑو_src(mrs, src)) {
                    واپس جھوٹ;
                }
            }
        }

        برائے (ثابت خودکار * fused : گرہ.fused) {
            برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
                ثابت ggml_tensor * src = fused->src[i];
                اگر (src && !گرہ.is_output(src)) {
                    اگر (!ggml_mem_ranges_جوڑو_src(mrs, src)) {
                        واپس جھوٹ;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        اگر (!ggml_mem_ranges_جوڑو_dst(mrs, گرہ.گرہ)) {
            واپس جھوٹ;
        }
        برائے (ثابت خودکار * fused : گرہ.fused) {
            اگر (!ggml_mem_ranges_جوڑو_dst(mrs, fused)) {
                واپس جھوٹ;
            }
        }

        واپس سچ;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    ثابت خودکار & h_check = [](ggml_mem_ranges_t mrs, ثابت گرہ_info & گرہ) {
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت ggml_tensor * src = گرہ.گرہ->src[i];
            اگر (src && !گرہ.is_output(src)) {
                اگر (!ggml_mem_ranges_check_src(mrs, src)) {
                    واپس جھوٹ;
                }
            }
        }

        برائے (ثابت خودکار * fused : گرہ.fused) {
            برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
                ثابت ggml_tensor * src = fused->src[i];
                اگر (src && !گرہ.is_output(src)) {
                    اگر (!ggml_mem_ranges_check_src(mrs, src)) {
                        واپس جھوٹ;
                    }
                }
            }
        }

        اگر (!ggml_mem_ranges_check_dst(mrs, گرہ.گرہ)) {
            واپس جھوٹ;
        }
        برائے (ثابت خودکار * fused : گرہ.fused) {
            اگر (!ggml_mem_ranges_check_dst(mrs, fused)) {
                واپس جھوٹ;
            }
        }

        واپس سچ;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    ثابت خودکار & h_محفوظ = [](ggml_op op) {
        انتخاب (op) {
            حالت GGML_OP_MUL_MAT:
            حالت GGML_OP_MUL_MAT_ID:
            حالت GGML_OP_ROPE:
            حالت GGML_OP_NORM:
            حالت GGML_OP_RMS_NORM:
            حالت GGML_OP_گروہ_NORM:
            حالت GGML_OP_L2_NORM:
            حالت GGML_OP_مجموع_صفیں:
            حالت GGML_OP_SSM_CONV:
            حالت GGML_OP_SSM_جائزہ:
            حالت GGML_OP_CLAMP:
            حالت GGML_OP_TRI:
            حالت GGML_OP_DIAG:
            حالت GGML_OP_MUL:
            حالت GGML_OP_جوڑو:
            حالت GGML_OP_SUB:
            حالت GGML_OP_DIV:
            حالت GGML_OP_GLU:
            حالت GGML_OP_SCALE:
            حالت GGML_OP_UNARY:
            حالت GGML_OP_GET_صفیں:
            حالت GGML_OP_سیٹ_صفیں:
            حالت GGML_OP_سیٹ:
            حالت GGML_OP_CPY:
            حالت GGML_OP_CONT:
            حالت GGML_OP_REPEAT:
                واپس سچ;
            طےشدہ:
                واپس ggml_op_is_خالی_خانے(op);
        }
    };

    ثابت صحیح n = nodes.size();

    std::vector<صحیح> res;
    res.reserve(n);

    std::vector<منطقی> مستعمل(n, جھوٹ);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    برائے (صحیح i0 = 0; i0 < n; i0++) {
        اگر (مستعمل[i0]) {
            جاری;
        }

        ثابت خودکار & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        اگر (!node0.is_خالی_خانے() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_جوڑو(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr صحیح N_FORWARD = 64;

            برائے (صحیح i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                اگر (مستعمل[i1]) {
                    جاری;
                }

                ثابت خودکار & node1 = nodes[i1];

                // disallow reordering of certain ops
                اگر (!h_محفوظ(node1.op())) {
                    توڑو;
                }

                ثابت منطقی is_خالی_خانے = node1.is_خالی_خانے();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                اگر ((is_خالی_خانے || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_جوڑو(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    مستعمل[i1] = سچ;
                } ورنہ {
                    // expand the set of nodes that haven't been processed yet
                    h_جوڑو(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_جوڑو(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    واپس res;
}

خالی ggml_گراف_optimize(ggml_cgraph * gf) {
    ثابت صحیح n = gf->n_nodes;

    std::vector<گرہ_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    برائے (صحیح i = 0; i < n; i++) {
        گرہ_info گرہ = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        ثابت صحیح f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        برائے (صحیح k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            گرہ.جوڑو_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(گرہ));
    }

#اگر 1
    // reorder to improve concurrency
    ثابت خودکار ترتیب = ggml_metal_گراف_optimize_reorder(nodes);
#ورنہ
    std::vector<صحیح> ترتیب(nodes.size());
    برائے (size_t i = 0; i < nodes.size(); i++) {
        ترتیب[i] = i;
    }
#اگر_ختم

    // unfuse
    {
        صحیح j = 0;
        برائے (ثابت خودکار i : ترتیب) {
            ثابت خودکار & گرہ = nodes[i];

            gf->nodes[j++] = گرہ.گرہ;

            برائے (خودکار * fused : گرہ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
