// Element-wise operations: dst[i] = src0[i] op src1[i]
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#define DEFINE_BLOCK_OP(name, op_insn)                                                                             \
    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ name(ಭಿನ್ನ * dst_block, ಧ್ರುವ ಭಿನ್ನ * src0_block, ಧ್ರುವ ಭಿನ್ನ * src1_block, ಪೂರ್ಣಾಙ್ಕ elements) { \
        ಧ್ರುವ int32_t vec_end = (elements / 8) * 8;                                                                \
        ಧ್ರುವ int32_t tail    = elements - vec_end;                                                                \
                                                                                                                   \
        unsigned long temp_mask;                                                                                   \
        __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(temp_mask));                                                         \
        __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        ಪ್ರತಿ (int32_t i = 0; i < vec_end; i += 8) {                                                                 \
            __asm__ ಚಞ್ಚಲ(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_block[i])                                                         \
                : [s0] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src0_block[i]), [s1] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src1_block[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        ಯದಿ (tail > 0) {                                                                                            \
            ಧ್ರುವ unsigned long tail_m0 = (1ul << tail) - 1;                                                       \
            __asm__ ಚಞ್ಚಲ(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_block[vec_end]), [s1] "r"(&src1_block[vec_end]), [d] "r"(&dst_block[vec_end]),    \
                  [tm] "r"(tail_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(temp_mask));                                                          \
    }

DEFINE_BLOCK_OP(block_mul_cache_aligned, "fmul.ps")
DEFINE_BLOCK_OP(block_add_cache_aligned, "fadd.ps")
DEFINE_BLOCK_OP(block_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#define DEFINE_BLOCK_OP_BROADCAST(name, op_insn)                                                                     \
    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ name(ಭಿನ್ನ * dst_block, ಧ್ರುವ ಭಿನ್ನ * src0_block, ಭಿನ್ನ scalar, ಪೂರ್ಣಾಙ್ಕ elements) {               \
        ಧ್ರುವ int32_t vec_end = (elements / 8) * 8;                                                                  \
        ಧ್ರುವ int32_t tail    = elements - vec_end;                                                                  \
                                                                                                                     \
        unsigned long temp_mask;                                                                                     \
        __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(temp_mask));                                                           \
        __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        ಪ್ರತಿ (int32_t i = 0; i < vec_end; i += 8) {                                                                   \
            __asm__ ಚಞ್ಚಲ(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_block[i])                                                           \
                : [s0] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src0_block[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        ಯದಿ (tail > 0) {                                                                                              \
            ಧ್ರುವ unsigned long tail_m0 = (1ul << tail) - 1;                                                         \
            __asm__ ಚಞ್ಚಲ(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_block[vec_end]), [ps] "r"(&scalar), [d] "r"(&dst_block[vec_end]), [tm] "r"(tail_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(temp_mask));                                                            \
    }

DEFINE_BLOCK_OP_BROADCAST(block_mul_broadcast, "fmul.ps")
DEFINE_BLOCK_OP_BROADCAST(block_add_broadcast, "fadd.ps")
DEFINE_BLOCK_OP_BROADCAST(block_sub_broadcast, "fsub.ps")

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ scalar_el_map(ಭಿನ್ನ src0, ಭಿನ್ನ src1, ಗಣನಾ ggml_op operation) {
    ವಿಕಲ್ಪ (operation) {
        ಪಕ್ಷ GGML_OP_MUL:
            ಪ್ರತ್ಯಾಗಮ src0 * src1;
        ಪಕ್ಷ GGML_OP_ADD:
            ಪ್ರತ್ಯಾಗಮ src0 + src1;
        ಪಕ್ಷ GGML_OP_SUB:
            ಪ್ರತ್ಯಾಗಮ src0 - src1;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ 0.0f;
    }
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid pointer
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported type combination
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * src1_data = (ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !src1_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Null data pointer
    }

#ifdef ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ಧ್ರುವ size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ಧ್ರುವ size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif

    ಗಣನಾ ggml_op operation = dst->op;

    ಯದಿ (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported operation
    }

    ಧ್ರುವ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ಧ್ರುವ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ಧ್ರುವ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    ಧ್ರುವ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ಧ್ರುವ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಧ್ರುವ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    ಧ್ರುವ bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    ಧ್ರುವ bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ಧ್ರುವ bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    ಯದಿ (no_broadcast && all_contiguous) {
        ಧ್ರುವ int64_t total_elements         = ne0 * ne1 * ne2 * ne3;
        ಧ್ರುವ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes
        ಧ್ರುವ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        ಧ್ರುವ int64_t cl_per_thread = (total_cachelines + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
        ಧ್ರುವ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        ಯದಿ (cl_end > total_cachelines) {
            cl_end = total_cachelines;
        }

        ಯದಿ (cl_start >= total_cachelines) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        ಧ್ರುವ int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end * elements_per_cacheline;
        ಯದಿ (elem_end > total_elements) {
            elem_end = total_elements;
        }
        ಧ್ರುವ int32_t ಎಣಿಕೆ = (int32_t) (elem_end - elem_start);

        ವಿಕಲ್ಪ (operation) {
            ಪಕ್ಷ GGML_OP_MUL:
                block_mul_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, ಎಣಿಕೆ);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_ADD:
                block_add_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, ಎಣಿಕೆ);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_SUB:
                block_sub_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, ಎಣಿಕೆ);
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ 1;
        }
#ifdef ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_data + elem_start, (size_t) ಎಣಿಕೆ * ಪರಿಮಾಣ(ಭಿನ್ನ));
        WAIT_CACHEOPS;
        FENCE;
#endif
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Slow path: broadcasting or non-contiguous
    ಧ್ರುವ int64_t total_rows = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t end_row;

    ಯದಿ (cache_aligned) {
        ಧ್ರುವ int64_t rows_per_thread = (total_rows + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
        start_row                     = thread_id * rows_per_thread;
        end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, ಪರಿಮಾಣ(ಭಿನ್ನ));
        ಧ್ರುವ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        ಯದಿ (thread_id >= total_groups) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        ಧ್ರುವ int64_t group_start = thread_id;
        ಪ್ರತಿ (int64_t grp = group_start; grp < total_groups; grp += ಅಂಕೆ_threads) {
            ಧ್ರುವ int64_t group_row_start = grp * rows_per_group;
            int64_t       group_row_end   = group_row_start + rows_per_group;
            ಯದಿ (group_row_end > total_rows) {
                group_row_end = total_rows;
            }

#ifdef ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ಧ್ರುವ int64_t first_i03      = group_row_start / (ne2 * ne1);
            ಧ್ರುವ int64_t first_i02      = (group_row_start - first_i03 * ne2 * ne1) / ne1;
            ಧ್ರುವ int64_t first_i01      = (group_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            ವರ್ಣ *        group_dst_base = (ವರ್ಣ *) dst_data + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#endif

            ಪ್ರತಿ (int64_t ir = group_row_start; ir < group_row_end; ir++) {
                ಧ್ರುವ int64_t i03 = ir / (ne2 * ne1);
                ಧ್ರುವ int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ಧ್ರುವ int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ಧ್ರುವ int64_t i13 = i03 % ne13;
                ಧ್ರುವ int64_t i12 = i02 % ne12;
                ಧ್ರುವ int64_t i11 = i01 % ne11;

                ಭಿನ್ನ *       dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ಧ್ರುವ ಭಿನ್ನ * src0_ptr =
                    (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ಧ್ರುವ ಭಿನ್ನ * src1_ptr =
                    (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

                ಯದಿ (ne10 == 1) {
                    ಧ್ರುವ ಭಿನ್ನ scalar = src1_ptr[0];
                    ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } ಅನ್ಯಥಾ {
                    ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ifdef ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ಧ್ರುವ int64_t nrows = group_row_end - group_row_start;
            ಯದಿ (nrows > 0) {
                FENCE;
                evict_region_past_l2(group_dst_base, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#endif
        }

        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (start_row >= total_rows) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪ್ರತಿ (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        ಧ್ರುವ int64_t i03 = ir / (ne2 * ne1);
        ಧ್ರುವ int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ಧ್ರುವ int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ಧ್ರುವ int64_t i13 = i03 % ne13;
        ಧ್ರುವ int64_t i12 = i02 % ne12;
        ಧ್ರುವ int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        ಭಿನ್ನ *       dst_ptr  = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ಧ್ರುವ ಭಿನ್ನ * src0_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ಧ್ರುವ ಭಿನ್ನ * src1_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

        ಯದಿ (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            ಭಿನ್ನ scalar = src1_ptr[0];
            ವಿಕಲ್ಪ (operation) {
                ಪಕ್ಷ GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (ಪೂರ್ಣಾಙ್ಕ) ne0);
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (ಪೂರ್ಣಾಙ್ಕ) ne0);
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (ಪೂರ್ಣಾಙ್ಕ) ne0);
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    ಪ್ರತ್ಯಾಗಮ 1;
            }
        } ಅನ್ಯಥಾ {
            // Broadcasting in dimension 0: src1 repeats across src0
            ಧ್ರುವ int64_t nr0 = ne0 / ne10;

            ಪ್ರತಿ (int64_t r = 0; r < nr0; r++) {
                ಧ್ರುವ ಭಿನ್ನ * src0_block = src0_ptr + r * ne10;
                ಭಿನ್ನ *       dst_block  = dst_ptr + r * ne10;

                ವಿಕಲ್ಪ (operation) {
                    ಪಕ್ಷ GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (ಪೂರ್ಣಾಙ್ಕ) ne10);
                        ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (ಪೂರ್ಣಾಙ್ಕ) ne10);
                        ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (ಪೂರ್ಣಾಙ್ಕ) ne10);
                        ಭಙ್ಗ;
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ 1;
                }
            }
        }
    }

#ifdef ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    ಯದಿ (end_row > start_row) {
        FENCE;
        evict_region_past_l2((ವರ್ಣ *) dst_data + start_row * nb1, (size_t) (end_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#endif
    ಪ್ರತ್ಯಾಗಮ 0;
}
