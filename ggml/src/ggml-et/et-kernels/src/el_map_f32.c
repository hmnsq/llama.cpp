// Element-wise operations: dst[i] = src0[i] op src1[i]
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#define DEFINE_BLOCK_OP(name, op_insn)                                                                             \
    স্থির অন্তঃরেখ শূন্য name(দশমিক * dst_block, ধ্রুবক দশমিক * src0_block, ধ্রুবক দশমিক * src1_block, পূর্ণসংখ্যা elements) { \
        ধ্রুবক int32_t vec_end = (elements / 8) * 8;                                                                \
        ধ্রুবক int32_t লেজ    = elements - vec_end;                                                                \
                                                                                                                   \
        unsigned long অস্থায়ী_mask;                                                                                   \
        __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));                                                         \
        __asm__ অস্থির("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        জন্য (int32_t i = 0; i < vec_end; i += 8) {                                                                 \
            __asm__ অস্থির(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(দশমিক (*)[8]) & dst_block[i])                                                         \
                : [s0] "m"(*(ধ্রুবক দশমিক (*)[8]) & src0_block[i]), [s1] "m"(*(ধ্রুবক দশমিক (*)[8]) & src1_block[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        যদি (লেজ > 0) {                                                                                            \
            ধ্রুবক unsigned long লেজ_m0 = (1ul << লেজ) - 1;                                                       \
            __asm__ অস্থির(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_block[vec_end]), [s1] "r"(&src1_block[vec_end]), [d] "r"(&dst_block[vec_end]),    \
                  [tm] "r"(লেজ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));                                                          \
    }

DEFINE_BLOCK_OP(block_mul_cache_aligned, "fmul.ps")
DEFINE_BLOCK_OP(block_add_cache_aligned, "fadd.ps")
DEFINE_BLOCK_OP(block_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#define DEFINE_BLOCK_OP_BROADCAST(name, op_insn)                                                                     \
    স্থির অন্তঃরেখ শূন্য name(দশমিক * dst_block, ধ্রুবক দশমিক * src0_block, দশমিক scalar, পূর্ণসংখ্যা elements) {               \
        ধ্রুবক int32_t vec_end = (elements / 8) * 8;                                                                  \
        ধ্রুবক int32_t লেজ    = elements - vec_end;                                                                  \
                                                                                                                     \
        unsigned long অস্থায়ী_mask;                                                                                     \
        __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));                                                           \
        __asm__ অস্থির("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        জন্য (int32_t i = 0; i < vec_end; i += 8) {                                                                   \
            __asm__ অস্থির(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(দশমিক (*)[8]) & dst_block[i])                                                           \
                : [s0] "m"(*(ধ্রুবক দশমিক (*)[8]) & src0_block[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        যদি (লেজ > 0) {                                                                                              \
            ধ্রুবক unsigned long লেজ_m0 = (1ul << লেজ) - 1;                                                         \
            __asm__ অস্থির(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_block[vec_end]), [ps] "r"(&scalar), [d] "r"(&dst_block[vec_end]), [tm] "r"(লেজ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));                                                            \
    }

DEFINE_BLOCK_OP_BROADCAST(block_mul_broadcast, "fmul.ps")
DEFINE_BLOCK_OP_BROADCAST(block_add_broadcast, "fadd.ps")
DEFINE_BLOCK_OP_BROADCAST(block_sub_broadcast, "fsub.ps")

স্থির অন্তঃরেখ দশমিক scalar_el_map(দশমিক src0, দশমিক src1, গণনা ggml_op operation) {
    নির্বাচন (operation) {
        ক্ষেত্র GGML_OP_MUL:
            ফেরত src0 * src1;
        ক্ষেত্র GGML_OP_ADD:
            ফেরত src0 + src1;
        ক্ষেত্র GGML_OP_SUB:
            ফেরত src0 - src1;
        পূর্বনির্ধারিত:
            ফেরত 0.0f;
    }
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;  // Invalid pointer
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * src1 = &params->src1;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;  // Unsupported type combination
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * src1_তথ্য = (দশমিক *) src1->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !src1_তথ্য || !dst_তথ্য) {
        ফেরত -1;  // Null data pointer
    }

#ifdef ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ধ্রুবক size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ধ্রুবক size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_তথ্য, src0_bytes);
    evict_region_past_l2(src1_তথ্য, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif

    গণনা ggml_op operation = dst->op;

    যদি (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        ফেরত -1;  // Unsupported operation
    }

    ধ্রুবক int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ধ্রুবক int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ধ্রুবক int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    ধ্রুবক size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ধ্রুবক size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুবক size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    ধ্রুবক bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    ধ্রুবক bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ধ্রুবক bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    যদি (no_broadcast && all_contiguous) {
        ধ্রুবক int64_t যোগ_elements         = ne0 * ne1 * ne2 * ne3;
        ধ্রুবক int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes
        ধ্রুবক int64_t যোগ_cachelines       = (যোগ_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        ধ্রুবক int64_t cl_per_thread = (যোগ_cachelines + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        যদি (cl_end > যোগ_cachelines) {
            cl_end = যোগ_cachelines;
        }

        যদি (cl_start >= যোগ_cachelines) {
            ফেরত 0;
        }

        ধ্রুবক int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end * elements_per_cacheline;
        যদি (elem_end > যোগ_elements) {
            elem_end = যোগ_elements;
        }
        ধ্রুবক int32_t গুনতি = (int32_t) (elem_end - elem_start);

        নির্বাচন (operation) {
            ক্ষেত্র GGML_OP_MUL:
                block_mul_cache_aligned(dst_তথ্য + elem_start, src0_তথ্য + elem_start, src1_তথ্য + elem_start, গুনতি);
                বিরতি;
            ক্ষেত্র GGML_OP_ADD:
                block_add_cache_aligned(dst_তথ্য + elem_start, src0_তথ্য + elem_start, src1_তথ্য + elem_start, গুনতি);
                বিরতি;
            ক্ষেত্র GGML_OP_SUB:
                block_sub_cache_aligned(dst_তথ্য + elem_start, src0_তথ্য + elem_start, src1_তথ্য + elem_start, গুনতি);
                বিরতি;
            পূর্বনির্ধারিত:
                ফেরত 1;
        }
#ifdef ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_তথ্য + elem_start, (size_t) গুনতি * আকার(দশমিক));
        WAIT_CACHEOPS;
        FENCE;
#endif
        ফেরত 0;
    }

    // Slow path: broadcasting or non-contiguous
    ধ্রুবক int64_t যোগ_rows = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t end_row;

    যদি (cache_aligned) {
        ধ্রুবক int64_t rows_per_thread = (যোগ_rows + অঙ্ক_threads - 1) / অঙ্ক_threads;
        start_row                     = thread_id * rows_per_thread;
        end_row = (start_row + rows_per_thread < যোগ_rows) ? (start_row + rows_per_thread) : যোগ_rows;
    } অন্যথা {
        ধ্রুবক int64_t rows_per_group = et_rows_per_cacheline_group(ne0, আকার(দশমিক));
        ধ্রুবক int64_t যোগ_groups   = (যোগ_rows + rows_per_group - 1) / rows_per_group;

        যদি (thread_id >= যোগ_groups) {
            ফেরত 0;
        }

        ধ্রুবক int64_t group_start = thread_id;
        জন্য (int64_t grp = group_start; grp < যোগ_groups; grp += অঙ্ক_threads) {
            ধ্রুবক int64_t group_row_start = grp * rows_per_group;
            int64_t       group_row_end   = group_row_start + rows_per_group;
            যদি (group_row_end > যোগ_rows) {
                group_row_end = যোগ_rows;
            }

#ifdef ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ধ্রুবক int64_t first_i03      = group_row_start / (ne2 * ne1);
            ধ্রুবক int64_t first_i02      = (group_row_start - first_i03 * ne2 * ne1) / ne1;
            ধ্রুবক int64_t first_i01      = (group_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            অক্ষর *        group_dst_base = (অক্ষর *) dst_তথ্য + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#endif

            জন্য (int64_t ir = group_row_start; ir < group_row_end; ir++) {
                ধ্রুবক int64_t i03 = ir / (ne2 * ne1);
                ধ্রুবক int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ধ্রুবক int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ধ্রুবক int64_t i13 = i03 % ne13;
                ধ্রুবক int64_t i12 = i02 % ne12;
                ধ্রুবক int64_t i11 = i01 % ne11;

                দশমিক *       dst_ptr = (দশমিক *) ((অক্ষর *) dst_তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ধ্রুবক দশমিক * src0_ptr =
                    (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ধ্রুবক দশমিক * src1_ptr =
                    (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + i13 * nb13 + i12 * nb12 + i11 * nb11);

                যদি (ne10 == 1) {
                    ধ্রুবক দশমিক scalar = src1_ptr[0];
                    জন্য (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } অন্যথা {
                    জন্য (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ifdef ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ধ্রুবক int64_t nrows = group_row_end - group_row_start;
            যদি (nrows > 0) {
                FENCE;
                evict_region_past_l2(group_dst_base, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#endif
        }

        ফেরত 0;
    }

    যদি (start_row >= যোগ_rows) {
        ফেরত 0;
    }

    জন্য (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        ধ্রুবক int64_t i03 = ir / (ne2 * ne1);
        ধ্রুবক int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ধ্রুবক int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ধ্রুবক int64_t i13 = i03 % ne13;
        ধ্রুবক int64_t i12 = i02 % ne12;
        ধ্রুবক int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        দশমিক *       dst_ptr  = (দশমিক *) ((অক্ষর *) dst_তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ধ্রুবক দশমিক * src0_ptr = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ধ্রুবক দশমিক * src1_ptr = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + i13 * nb13 + i12 * nb12 + i11 * nb11);

        যদি (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            দশমিক scalar = src1_ptr[0];
            নির্বাচন (operation) {
                ক্ষেত্র GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (পূর্ণসংখ্যা) ne0);
                    বিরতি;
                ক্ষেত্র GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (পূর্ণসংখ্যা) ne0);
                    বিরতি;
                ক্ষেত্র GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (পূর্ণসংখ্যা) ne0);
                    বিরতি;
                পূর্বনির্ধারিত:
                    ফেরত 1;
            }
        } অন্যথা {
            // Broadcasting in dimension 0: src1 repeats across src0
            ধ্রুবক int64_t nr0 = ne0 / ne10;

            জন্য (int64_t r = 0; r < nr0; r++) {
                ধ্রুবক দশমিক * src0_block = src0_ptr + r * ne10;
                দশমিক *       dst_block  = dst_ptr + r * ne10;

                নির্বাচন (operation) {
                    ক্ষেত্র GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (পূর্ণসংখ্যা) ne10);
                        বিরতি;
                    ক্ষেত্র GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (পূর্ণসংখ্যা) ne10);
                        বিরতি;
                    ক্ষেত্র GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (পূর্ণসংখ্যা) ne10);
                        বিরতি;
                    পূর্বনির্ধারিত:
                        ফেরত 1;
                }
            }
        }
    }

#ifdef ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    যদি (end_row > start_row) {
        FENCE;
        evict_region_past_l2((অক্ষর *) dst_তথ্য + start_row * nb1, (size_t) (end_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#endif
    ফেরত 0;
}
