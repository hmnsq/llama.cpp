// Element-wise operations: dst[i] = src0[i] op src1[i]
#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#define DEFINE_BLOCK_OP(name, op_insn)                                                                             \
    நிலையான உள்வரி வெறுமை name(தசம * dst_block, மாறிலி தசம * src0_block, மாறிலி தசம * src1_block, முழு elements) { \
        மாறிலி int32_t vec_end = (elements / 8) * 8;                                                                \
        மாறிலி int32_t tail    = elements - vec_end;                                                                \
                                                                                                                   \
        unsigned long temp_mask;                                                                                   \
        __asm__ நிலையற்ற("mova.x.m %0" : "=r"(temp_mask));                                                         \
        __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        க்கு (int32_t i = 0; i < vec_end; i += 8) {                                                                 \
            __asm__ நிலையற்ற(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(தசம (*)[8]) & dst_block[i])                                                         \
                : [s0] "m"(*(மாறிலி தசம (*)[8]) & src0_block[i]), [s1] "m"(*(மாறிலி தசம (*)[8]) & src1_block[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        எனில் (tail > 0) {                                                                                            \
            மாறிலி unsigned long tail_m0 = (1ul << tail) - 1;                                                       \
            __asm__ நிலையற்ற(                                                                                      \
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
        __asm__ நிலையற்ற("mova.m.x %0" ::"r"(temp_mask));                                                          \
    }

DEFINE_BLOCK_OP(block_mul_cache_aligned, "fmul.ps")
DEFINE_BLOCK_OP(block_add_cache_aligned, "fadd.ps")
DEFINE_BLOCK_OP(block_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#define DEFINE_BLOCK_OP_BROADCAST(name, op_insn)                                                                     \
    நிலையான உள்வரி வெறுமை name(தசம * dst_block, மாறிலி தசம * src0_block, தசம scalar, முழு elements) {               \
        மாறிலி int32_t vec_end = (elements / 8) * 8;                                                                  \
        மாறிலி int32_t tail    = elements - vec_end;                                                                  \
                                                                                                                     \
        unsigned long temp_mask;                                                                                     \
        __asm__ நிலையற்ற("mova.x.m %0" : "=r"(temp_mask));                                                           \
        __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        க்கு (int32_t i = 0; i < vec_end; i += 8) {                                                                   \
            __asm__ நிலையற்ற(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(தசம (*)[8]) & dst_block[i])                                                           \
                : [s0] "m"(*(மாறிலி தசம (*)[8]) & src0_block[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        எனில் (tail > 0) {                                                                                              \
            மாறிலி unsigned long tail_m0 = (1ul << tail) - 1;                                                         \
            __asm__ நிலையற்ற(                                                                                        \
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
        __asm__ நிலையற்ற("mova.m.x %0" ::"r"(temp_mask));                                                            \
    }

DEFINE_BLOCK_OP_BROADCAST(block_mul_broadcast, "fmul.ps")
DEFINE_BLOCK_OP_BROADCAST(block_add_broadcast, "fadd.ps")
DEFINE_BLOCK_OP_BROADCAST(block_sub_broadcast, "fsub.ps")

நிலையான உள்வரி தசம scalar_el_map(தசம src0, தசம src1, எண்ணிக்கை ggml_op operation) {
    தேர்வு (operation) {
        நிலை GGML_OP_MUL:
            திரும்பு src0 * src1;
        நிலை GGML_OP_ADD:
            திரும்பு src0 + src1;
        நிலை GGML_OP_SUB:
            திரும்பு src0 - src1;
        இயல்பு:
            திரும்பு 0.0f;
    }
}

முழு entry_point(கட்டமைப்பு ggml_et_இருமம்_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;  // Invalid pointer
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;  // Unsupported type combination
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * src1_data = (தசம *) src1->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !src1_data || !dst_data) {
        திரும்பு -1;  // Null data pointer
    }

#ifdef ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    மாறிலி size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    மாறிலி size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif

    எண்ணிக்கை ggml_op operation = dst->op;

    எனில் (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        திரும்பு -1;  // Unsupported operation
    }

    மாறிலி int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    மாறிலி int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    மாறிலி int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    மாறிலி size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    மாறிலி size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    மாறிலி size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    மாறிலி bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    மாறிலி bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    மாறிலி bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    எனில் (no_broadcast && all_contiguous) {
        மாறிலி int64_t total_elements         = ne0 * ne1 * ne2 * ne3;
        மாறிலி int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes
        மாறிலி int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        மாறிலி int64_t cl_per_thread = (total_cachelines + இலக்கம்_threads - 1) / இலக்கம்_threads;
        மாறிலி int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        எனில் (cl_end > total_cachelines) {
            cl_end = total_cachelines;
        }

        எனில் (cl_start >= total_cachelines) {
            திரும்பு 0;
        }

        மாறிலி int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end * elements_per_cacheline;
        எனில் (elem_end > total_elements) {
            elem_end = total_elements;
        }
        மாறிலி int32_t மொத்தம் = (int32_t) (elem_end - elem_start);

        தேர்வு (operation) {
            நிலை GGML_OP_MUL:
                block_mul_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, மொத்தம்);
                நிறுத்து;
            நிலை GGML_OP_ADD:
                block_add_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, மொத்தம்);
                நிறுத்து;
            நிலை GGML_OP_SUB:
                block_sub_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, மொத்தம்);
                நிறுத்து;
            இயல்பு:
                திரும்பு 1;
        }
#ifdef ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_data + elem_start, (size_t) மொத்தம் * அளவு(தசம));
        WAIT_CACHEOPS;
        FENCE;
#endif
        திரும்பு 0;
    }

    // Slow path: broadcasting or non-contiguous
    மாறிலி int64_t total_rows = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t end_row;

    எனில் (cache_aligned) {
        மாறிலி int64_t rows_per_thread = (total_rows + இலக்கம்_threads - 1) / இலக்கம்_threads;
        start_row                     = thread_id * rows_per_thread;
        end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;
    } இல்லையெனில் {
        மாறிலி int64_t rows_per_group = et_rows_per_cacheline_group(ne0, அளவு(தசம));
        மாறிலி int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        எனில் (thread_id >= total_groups) {
            திரும்பு 0;
        }

        மாறிலி int64_t group_start = thread_id;
        க்கு (int64_t grp = group_start; grp < total_groups; grp += இலக்கம்_threads) {
            மாறிலி int64_t group_row_start = grp * rows_per_group;
            int64_t       group_row_end   = group_row_start + rows_per_group;
            எனில் (group_row_end > total_rows) {
                group_row_end = total_rows;
            }

#ifdef ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            மாறிலி int64_t first_i03      = group_row_start / (ne2 * ne1);
            மாறிலி int64_t first_i02      = (group_row_start - first_i03 * ne2 * ne1) / ne1;
            மாறிலி int64_t first_i01      = (group_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            எழுத்து *        group_dst_base = (எழுத்து *) dst_data + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#endif

            க்கு (int64_t ir = group_row_start; ir < group_row_end; ir++) {
                மாறிலி int64_t i03 = ir / (ne2 * ne1);
                மாறிலி int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                மாறிலி int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                மாறிலி int64_t i13 = i03 % ne13;
                மாறிலி int64_t i12 = i02 % ne12;
                மாறிலி int64_t i11 = i01 % ne11;

                தசம *       dst_ptr = (தசம *) ((எழுத்து *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
                மாறிலி தசம * src0_ptr =
                    (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
                மாறிலி தசம * src1_ptr =
                    (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

                எனில் (ne10 == 1) {
                    மாறிலி தசம scalar = src1_ptr[0];
                    க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } இல்லையெனில் {
                    க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ifdef ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            மாறிலி int64_t nrows = group_row_end - group_row_start;
            எனில் (nrows > 0) {
                FENCE;
                evict_region_past_l2(group_dst_base, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#endif
        }

        திரும்பு 0;
    }

    எனில் (start_row >= total_rows) {
        திரும்பு 0;
    }

    க்கு (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        மாறிலி int64_t i03 = ir / (ne2 * ne1);
        மாறிலி int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        மாறிலி int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        மாறிலி int64_t i13 = i03 % ne13;
        மாறிலி int64_t i12 = i02 % ne12;
        மாறிலி int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        தசம *       dst_ptr  = (தசம *) ((எழுத்து *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        மாறிலி தசம * src0_ptr = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        மாறிலி தசம * src1_ptr = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

        எனில் (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            தசம scalar = src1_ptr[0];
            தேர்வு (operation) {
                நிலை GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (முழு) ne0);
                    நிறுத்து;
                நிலை GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (முழு) ne0);
                    நிறுத்து;
                நிலை GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (முழு) ne0);
                    நிறுத்து;
                இயல்பு:
                    திரும்பு 1;
            }
        } இல்லையெனில் {
            // Broadcasting in dimension 0: src1 repeats across src0
            மாறிலி int64_t nr0 = ne0 / ne10;

            க்கு (int64_t r = 0; r < nr0; r++) {
                மாறிலி தசம * src0_block = src0_ptr + r * ne10;
                தசம *       dst_block  = dst_ptr + r * ne10;

                தேர்வு (operation) {
                    நிலை GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (முழு) ne10);
                        நிறுத்து;
                    நிலை GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (முழு) ne10);
                        நிறுத்து;
                    நிலை GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (முழு) ne10);
                        நிறுத்து;
                    இயல்பு:
                        திரும்பு 1;
                }
            }
        }
    }

#ifdef ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    எனில் (end_row > start_row) {
        FENCE;
        evict_region_past_l2((எழுத்து *) dst_data + start_row * nb1, (size_t) (end_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#endif
    திரும்பு 0;
}
