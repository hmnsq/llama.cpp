// Element-wise operations: dst[i] = src0[i] op src1[i]
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#define DEFINE_BLOCK_OP(name, op_insn)                                                                             \
    स्थिर ओळीत शून्य name(दशांश * dst_block, स्थिरांक दशांश * src0_block, स्थिरांक दशांश * src1_block, पूर्णांक elements) { \
        स्थिरांक int32_t vec_end = (elements / 8) * 8;                                                                \
        स्थिरांक int32_t tail    = elements - vec_end;                                                                \
                                                                                                                   \
        अचिन्हित दीर्घ temp_mask;                                                                                   \
        __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));                                                         \
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        साठी (int32_t i = 0; i < vec_end; i += 8) {                                                                 \
            __asm__ अस्थिर(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(दशांश (*)[8]) & dst_block[i])                                                         \
                : [s0] "m"(*(स्थिरांक दशांश (*)[8]) & src0_block[i]), [s1] "m"(*(स्थिरांक दशांश (*)[8]) & src1_block[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        जर (tail > 0) {                                                                                            \
            स्थिरांक अचिन्हित दीर्घ tail_m0 = (1ul << tail) - 1;                                                       \
            __asm__ अस्थिर(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_block[vec_end]), [s1] "r"(&src1_block[vec_end]), [d] "r"(&dst_block[vec_end]),    \
                  [पंचांग] "r"(tail_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));                                                          \
    }

DEFINE_BLOCK_OP(block_mul_cache_aligned, "fmul.ps")
DEFINE_BLOCK_OP(block_add_cache_aligned, "fadd.ps")
DEFINE_BLOCK_OP(block_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#define DEFINE_BLOCK_OP_BROADCAST(name, op_insn)                                                                     \
    स्थिर ओळीत शून्य name(दशांश * dst_block, स्थिरांक दशांश * src0_block, दशांश scalar, पूर्णांक elements) {               \
        स्थिरांक int32_t vec_end = (elements / 8) * 8;                                                                  \
        स्थिरांक int32_t tail    = elements - vec_end;                                                                  \
                                                                                                                     \
        अचिन्हित दीर्घ temp_mask;                                                                                     \
        __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));                                                           \
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        साठी (int32_t i = 0; i < vec_end; i += 8) {                                                                   \
            __asm__ अस्थिर(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(दशांश (*)[8]) & dst_block[i])                                                           \
                : [s0] "m"(*(स्थिरांक दशांश (*)[8]) & src0_block[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        जर (tail > 0) {                                                                                              \
            स्थिरांक अचिन्हित दीर्घ tail_m0 = (1ul << tail) - 1;                                                         \
            __asm__ अस्थिर(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_block[vec_end]), [ps] "r"(&scalar), [d] "r"(&dst_block[vec_end]), [पंचांग] "r"(tail_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));                                                            \
    }

DEFINE_BLOCK_OP_BROADCAST(block_mul_broadcast, "fmul.ps")
DEFINE_BLOCK_OP_BROADCAST(block_add_broadcast, "fadd.ps")
DEFINE_BLOCK_OP_BROADCAST(block_sub_broadcast, "fsub.ps")

स्थिर ओळीत दशांश scalar_el_map(दशांश src0, दशांश src1, गणना ggml_op operation) {
    निवड (operation) {
        स्थिती GGML_OP_MUL:
            परत src0 * src1;
        स्थिती GGML_OP_ADD:
            परत src0 + src1;
        स्थिती GGML_OP_SUB:
            परत src0 - src1;
        पूर्वनिर्धारित:
            परत 0.0f;
    }
}

पूर्णांक entry_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;  // Unsupported type combination
    }

    दशांश * src0_data = (दशांश *) src0->data;
    दशांश * src1_data = (दशांश *) src1->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !src1_data || !dst_data) {
        परत -1;  // Null data pointer
    }

#ifdef ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    स्थिरांक माप_प्रकार src0_bytes = (माप_प्रकार) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    स्थिरांक माप_प्रकार src1_bytes = (माप_प्रकार) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif

    गणना ggml_op operation = dst->op;

    जर (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        परत -1;  // Unsupported operation
    }

    स्थिरांक int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    स्थिरांक int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    स्थिरांक int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    स्थिरांक माप_प्रकार nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    स्थिरांक माप_प्रकार nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिरांक माप_प्रकार nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    स्थिरांक bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    स्थिरांक bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    स्थिरांक bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    जर (no_broadcast && all_contiguous) {
        स्थिरांक int64_t total_elements         = ne0 * ne1 * ne2 * ne3;
        स्थिरांक int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes
        स्थिरांक int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        स्थिरांक int64_t cl_per_thread = (total_cachelines + अंक_threads - 1) / अंक_threads;
        स्थिरांक int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        जर (cl_end > total_cachelines) {
            cl_end = total_cachelines;
        }

        जर (cl_start >= total_cachelines) {
            परत 0;
        }

        स्थिरांक int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end * elements_per_cacheline;
        जर (elem_end > total_elements) {
            elem_end = total_elements;
        }
        स्थिरांक int32_t मोजणी = (int32_t) (elem_end - elem_start);

        निवड (operation) {
            स्थिती GGML_OP_MUL:
                block_mul_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, मोजणी);
                थांबा;
            स्थिती GGML_OP_ADD:
                block_add_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, मोजणी);
                थांबा;
            स्थिती GGML_OP_SUB:
                block_sub_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, मोजणी);
                थांबा;
            पूर्वनिर्धारित:
                परत 1;
        }
#ifdef ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_data + elem_start, (माप_प्रकार) मोजणी * आकार(दशांश));
        WAIT_CACHEOPS;
        FENCE;
#endif
        परत 0;
    }

    // Slow path: broadcasting or non-contiguous
    स्थिरांक int64_t total_rows = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t end_row;

    जर (cache_aligned) {
        स्थिरांक int64_t rows_per_thread = (total_rows + अंक_threads - 1) / अंक_threads;
        start_row                     = thread_id * rows_per_thread;
        end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;
    } अन्यथा {
        स्थिरांक int64_t rows_per_group = et_rows_per_cacheline_group(ne0, आकार(दशांश));
        स्थिरांक int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        जर (thread_id >= total_groups) {
            परत 0;
        }

        स्थिरांक int64_t group_start = thread_id;
        साठी (int64_t grp = group_start; grp < total_groups; grp += अंक_threads) {
            स्थिरांक int64_t group_row_start = grp * rows_per_group;
            int64_t       group_row_end   = group_row_start + rows_per_group;
            जर (group_row_end > total_rows) {
                group_row_end = total_rows;
            }

#ifdef ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            स्थिरांक int64_t first_i03      = group_row_start / (ne2 * ne1);
            स्थिरांक int64_t first_i02      = (group_row_start - first_i03 * ne2 * ne1) / ne1;
            स्थिरांक int64_t first_i01      = (group_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            अक्षर *        group_dst_base = (अक्षर *) dst_data + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#endif

            साठी (int64_t ir = group_row_start; ir < group_row_end; ir++) {
                स्थिरांक int64_t i03 = ir / (ne2 * ne1);
                स्थिरांक int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                स्थिरांक int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                स्थिरांक int64_t i13 = i03 % ne13;
                स्थिरांक int64_t i12 = i02 % ne12;
                स्थिरांक int64_t i11 = i01 % ne11;

                दशांश *       dst_ptr = (दशांश *) ((अक्षर *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
                स्थिरांक दशांश * src0_ptr =
                    (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
                स्थिरांक दशांश * src1_ptr =
                    (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

                जर (ne10 == 1) {
                    स्थिरांक दशांश scalar = src1_ptr[0];
                    साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } अन्यथा {
                    साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ifdef ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            स्थिरांक int64_t nrows = group_row_end - group_row_start;
            जर (nrows > 0) {
                FENCE;
                evict_region_past_l2(group_dst_base, (माप_प्रकार) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#endif
        }

        परत 0;
    }

    जर (start_row >= total_rows) {
        परत 0;
    }

    साठी (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        स्थिरांक int64_t i03 = ir / (ne2 * ne1);
        स्थिरांक int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        स्थिरांक int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        स्थिरांक int64_t i13 = i03 % ne13;
        स्थिरांक int64_t i12 = i02 % ne12;
        स्थिरांक int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        दशांश *       dst_ptr  = (दशांश *) ((अक्षर *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        स्थिरांक दशांश * src0_ptr = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        स्थिरांक दशांश * src1_ptr = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

        जर (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            दशांश scalar = src1_ptr[0];
            निवड (operation) {
                स्थिती GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (पूर्णांक) ne0);
                    थांबा;
                स्थिती GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (पूर्णांक) ne0);
                    थांबा;
                स्थिती GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (पूर्णांक) ne0);
                    थांबा;
                पूर्वनिर्धारित:
                    परत 1;
            }
        } अन्यथा {
            // Broadcasting in dimension 0: src1 repeats across src0
            स्थिरांक int64_t nr0 = ne0 / ne10;

            साठी (int64_t r = 0; r < nr0; r++) {
                स्थिरांक दशांश * src0_block = src0_ptr + r * ne10;
                दशांश *       dst_block  = dst_ptr + r * ne10;

                निवड (operation) {
                    स्थिती GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (पूर्णांक) ne10);
                        थांबा;
                    स्थिती GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (पूर्णांक) ne10);
                        थांबा;
                    स्थिती GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (पूर्णांक) ne10);
                        थांबा;
                    पूर्वनिर्धारित:
                        परत 1;
                }
            }
        }
    }

#ifdef ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    जर (end_row > start_row) {
        FENCE;
        evict_region_past_l2((अक्षर *) dst_data + start_row * nb1, (माप_प्रकार) (end_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#endif
    परत 0;
}
