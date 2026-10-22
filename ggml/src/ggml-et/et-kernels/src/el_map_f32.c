// Element-wise operations: dst[i] = src0[i] op src1[i]
#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#define DEFINE_BLOCK_OP(name, op_insn)                                                                             \
    സ്ഥിതം വരിയിൽ ശൂന്യം name(ദശാംശം * dst_block, സ്ഥിരം ദശാംശം * src0_block, സ്ഥിരം ദശാംശം * src1_block, പൂർണ്ണസംഖ്യ elements) { \
        സ്ഥിരം int32_t vec_end = (elements / 8) * 8;                                                                \
        സ്ഥിരം int32_t tail    = elements - vec_end;                                                                \
                                                                                                                   \
        unsigned long temp_mask;                                                                                   \
        __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(temp_mask));                                                         \
        __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        വേണ്ടി (int32_t i = 0; i < vec_end; i += 8) {                                                                 \
            __asm__ അസ്ഥിര(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(ദശാംശം (*)[8]) & dst_block[i])                                                         \
                : [s0] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src0_block[i]), [s1] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src1_block[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        എങ്കിൽ (tail > 0) {                                                                                            \
            സ്ഥിരം unsigned long tail_m0 = (1ul << tail) - 1;                                                       \
            __asm__ അസ്ഥിര(                                                                                      \
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
        __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(temp_mask));                                                          \
    }

DEFINE_BLOCK_OP(block_mul_cache_aligned, "fmul.ps")
DEFINE_BLOCK_OP(block_add_cache_aligned, "fadd.ps")
DEFINE_BLOCK_OP(block_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#define DEFINE_BLOCK_OP_BROADCAST(name, op_insn)                                                                     \
    സ്ഥിതം വരിയിൽ ശൂന്യം name(ദശാംശം * dst_block, സ്ഥിരം ദശാംശം * src0_block, ദശാംശം scalar, പൂർണ്ണസംഖ്യ elements) {               \
        സ്ഥിരം int32_t vec_end = (elements / 8) * 8;                                                                  \
        സ്ഥിരം int32_t tail    = elements - vec_end;                                                                  \
                                                                                                                     \
        unsigned long temp_mask;                                                                                     \
        __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(temp_mask));                                                           \
        __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        വേണ്ടി (int32_t i = 0; i < vec_end; i += 8) {                                                                   \
            __asm__ അസ്ഥിര(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(ദശാംശം (*)[8]) & dst_block[i])                                                           \
                : [s0] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src0_block[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        എങ്കിൽ (tail > 0) {                                                                                              \
            സ്ഥിരം unsigned long tail_m0 = (1ul << tail) - 1;                                                         \
            __asm__ അസ്ഥിര(                                                                                        \
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
        __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(temp_mask));                                                            \
    }

DEFINE_BLOCK_OP_BROADCAST(block_mul_broadcast, "fmul.ps")
DEFINE_BLOCK_OP_BROADCAST(block_add_broadcast, "fadd.ps")
DEFINE_BLOCK_OP_BROADCAST(block_sub_broadcast, "fsub.ps")

സ്ഥിതം വരിയിൽ ദശാംശം scalar_el_map(ദശാംശം src0, ദശാംശം src1, എണ്ണൽ ggml_op operation) {
    തിരഞ്ഞെടുപ്പ് (operation) {
        അവസ്ഥ GGML_OP_MUL:
            തിരികെ src0 * src1;
        അവസ്ഥ GGML_OP_ADD:
            തിരികെ src0 + src1;
        അവസ്ഥ GGML_OP_SUB:
            തിരികെ src0 - src1;
        സ്ഥിരസ്ഥിതി:
            തിരികെ 0.0f;
    }
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_ദ്വിമാന_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;  // Invalid pointer
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;  // Unsupported type combination
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * src1_data = (ദശാംശം *) src1->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !src1_data || !dst_data) {
        തിരികെ -1;  // Null data pointer
    }

#ifdef ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    സ്ഥിരം size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    സ്ഥിരം size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif

    എണ്ണൽ ggml_op operation = dst->op;

    എങ്കിൽ (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        തിരികെ -1;  // Unsupported operation
    }

    സ്ഥിരം int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    സ്ഥിരം int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    സ്ഥിരം int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    സ്ഥിരം size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    സ്ഥിരം size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    സ്ഥിരം size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    സ്ഥിരം bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    സ്ഥിരം bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    സ്ഥിരം bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    എങ്കിൽ (no_broadcast && all_contiguous) {
        സ്ഥിരം int64_t total_elements         = ne0 * ne1 * ne2 * ne3;
        സ്ഥിരം int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes
        സ്ഥിരം int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        സ്ഥിരം int64_t cl_per_thread = (total_cachelines + അക്കം_threads - 1) / അക്കം_threads;
        സ്ഥിരം int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        എങ്കിൽ (cl_end > total_cachelines) {
            cl_end = total_cachelines;
        }

        എങ്കിൽ (cl_start >= total_cachelines) {
            തിരികെ 0;
        }

        സ്ഥിരം int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end * elements_per_cacheline;
        എങ്കിൽ (elem_end > total_elements) {
            elem_end = total_elements;
        }
        സ്ഥിരം int32_t എണ്ണം = (int32_t) (elem_end - elem_start);

        തിരഞ്ഞെടുപ്പ് (operation) {
            അവസ്ഥ GGML_OP_MUL:
                block_mul_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, എണ്ണം);
                നിർത്തുക;
            അവസ്ഥ GGML_OP_ADD:
                block_add_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, എണ്ണം);
                നിർത്തുക;
            അവസ്ഥ GGML_OP_SUB:
                block_sub_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, എണ്ണം);
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                തിരികെ 1;
        }
#ifdef ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_data + elem_start, (size_t) എണ്ണം * വലുപ്പം(ദശാംശം));
        WAIT_CACHEOPS;
        FENCE;
#endif
        തിരികെ 0;
    }

    // Slow path: broadcasting or non-contiguous
    സ്ഥിരം int64_t total_rows = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t end_row;

    എങ്കിൽ (cache_aligned) {
        സ്ഥിരം int64_t rows_per_thread = (total_rows + അക്കം_threads - 1) / അക്കം_threads;
        start_row                     = thread_id * rows_per_thread;
        end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;
    } അല്ലെങ്കിൽ {
        സ്ഥിരം int64_t rows_per_group = et_rows_per_cacheline_group(ne0, വലുപ്പം(ദശാംശം));
        സ്ഥിരം int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        എങ്കിൽ (thread_id >= total_groups) {
            തിരികെ 0;
        }

        സ്ഥിരം int64_t group_start = thread_id;
        വേണ്ടി (int64_t grp = group_start; grp < total_groups; grp += അക്കം_threads) {
            സ്ഥിരം int64_t group_row_start = grp * rows_per_group;
            int64_t       group_row_end   = group_row_start + rows_per_group;
            എങ്കിൽ (group_row_end > total_rows) {
                group_row_end = total_rows;
            }

#ifdef ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            സ്ഥിരം int64_t first_i03      = group_row_start / (ne2 * ne1);
            സ്ഥിരം int64_t first_i02      = (group_row_start - first_i03 * ne2 * ne1) / ne1;
            സ്ഥിരം int64_t first_i01      = (group_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            അക്ഷരം *        group_dst_base = (അക്ഷരം *) dst_data + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#endif

            വേണ്ടി (int64_t ir = group_row_start; ir < group_row_end; ir++) {
                സ്ഥിരം int64_t i03 = ir / (ne2 * ne1);
                സ്ഥിരം int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                സ്ഥിരം int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                സ്ഥിരം int64_t i13 = i03 % ne13;
                സ്ഥിരം int64_t i12 = i02 % ne12;
                സ്ഥിരം int64_t i11 = i01 % ne11;

                ദശാംശം *       dst_ptr = (ദശാംശം *) ((അക്ഷരം *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
                സ്ഥിരം ദശാംശം * src0_ptr =
                    (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
                സ്ഥിരം ദശാംശം * src1_ptr =
                    (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

                എങ്കിൽ (ne10 == 1) {
                    സ്ഥിരം ദശാംശം scalar = src1_ptr[0];
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } അല്ലെങ്കിൽ {
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ifdef ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            സ്ഥിരം int64_t nrows = group_row_end - group_row_start;
            എങ്കിൽ (nrows > 0) {
                FENCE;
                evict_region_past_l2(group_dst_base, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#endif
        }

        തിരികെ 0;
    }

    എങ്കിൽ (start_row >= total_rows) {
        തിരികെ 0;
    }

    വേണ്ടി (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        സ്ഥിരം int64_t i03 = ir / (ne2 * ne1);
        സ്ഥിരം int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        സ്ഥിരം int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        സ്ഥിരം int64_t i13 = i03 % ne13;
        സ്ഥിരം int64_t i12 = i02 % ne12;
        സ്ഥിരം int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        ദശാംശം *       dst_ptr  = (ദശാംശം *) ((അക്ഷരം *) dst_data + i03 * nb3 + i02 * nb2 + i01 * nb1);
        സ്ഥിരം ദശാംശം * src0_ptr = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
        സ്ഥിരം ദശാംശം * src1_ptr = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

        എങ്കിൽ (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            ദശാംശം scalar = src1_ptr[0];
            തിരഞ്ഞെടുപ്പ് (operation) {
                അവസ്ഥ GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (പൂർണ്ണസംഖ്യ) ne0);
                    നിർത്തുക;
                അവസ്ഥ GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (പൂർണ്ണസംഖ്യ) ne0);
                    നിർത്തുക;
                അവസ്ഥ GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (പൂർണ്ണസംഖ്യ) ne0);
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ 1;
            }
        } അല്ലെങ്കിൽ {
            // Broadcasting in dimension 0: src1 repeats across src0
            സ്ഥിരം int64_t nr0 = ne0 / ne10;

            വേണ്ടി (int64_t r = 0; r < nr0; r++) {
                സ്ഥിരം ദശാംശം * src0_block = src0_ptr + r * ne10;
                ദശാംശം *       dst_block  = dst_ptr + r * ne10;

                തിരഞ്ഞെടുപ്പ് (operation) {
                    അവസ്ഥ GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (പൂർണ്ണസംഖ്യ) ne10);
                        നിർത്തുക;
                    അവസ്ഥ GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (പൂർണ്ണസംഖ്യ) ne10);
                        നിർത്തുക;
                    അവസ്ഥ GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (പൂർണ്ണസംഖ്യ) ne10);
                        നിർത്തുക;
                    സ്ഥിരസ്ഥിതി:
                        തിരികെ 1;
                }
            }
        }
    }

#ifdef ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    എങ്കിൽ (end_row > start_row) {
        FENCE;
        evict_region_past_l2((അക്ഷരം *) dst_data + start_row * nb1, (size_t) (end_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#endif
    തിരികെ 0;
}
