// Element-wise operations: dst[i] = src0[i] op src1[i]
#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#घोषणा DEFINE_BLOCK_OP(name, op_insn)                                                                             \
    अटल अंतरभूत व्योम name(भग्न * dst_block, स्थिर भग्न * src0_block, स्थिर भग्न * src1_block, पूर्णांक elements) { \
        स्थिर int32_t vec_end = (elements / 8) * 8;                                                                \
        स्थिर int32_t पुच्छ    = elements - vec_end;                                                                \
                                                                                                                   \
        अचिन्हित दीर्घ अस्थायी_mask;                                                                                   \
        __asm__ अस्थिर("mova.x.m %0" : "=r"(अस्थायी_mask));                                                         \
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        क्रम (int32_t i = 0; i < vec_end; i += 8) {                                                                 \
            __asm__ अस्थिर(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(भग्न (*)[8]) & dst_block[i])                                                         \
                : [s0] "m"(*(स्थिर भग्न (*)[8]) & src0_block[i]), [s1] "m"(*(स्थिर भग्न (*)[8]) & src1_block[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        अगर (पुच्छ > 0) {                                                                                            \
            स्थिर अचिन्हित दीर्घ पुच्छ_m0 = (1ul << पुच्छ) - 1;                                                       \
            __asm__ अस्थिर(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_block[vec_end]), [s1] "r"(&src1_block[vec_end]), [d] "r"(&dst_block[vec_end]),    \
                  [पंचांग] "r"(पुच्छ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ अस्थिर("mova.m.x %0" ::"r"(अस्थायी_mask));                                                          \
    }

DEFINE_BLOCK_OP(block_mul_cache_aligned, "fmul.ps")
DEFINE_BLOCK_OP(block_add_cache_aligned, "fadd.ps")
DEFINE_BLOCK_OP(block_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#घोषणा DEFINE_BLOCK_OP_BROADCAST(name, op_insn)                                                                     \
    अटल अंतरभूत व्योम name(भग्न * dst_block, स्थिर भग्न * src0_block, भग्न scalar, पूर्णांक elements) {               \
        स्थिर int32_t vec_end = (elements / 8) * 8;                                                                  \
        स्थिर int32_t पुच्छ    = elements - vec_end;                                                                  \
                                                                                                                     \
        अचिन्हित दीर्घ अस्थायी_mask;                                                                                     \
        __asm__ अस्थिर("mova.x.m %0" : "=r"(अस्थायी_mask));                                                           \
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        क्रम (int32_t i = 0; i < vec_end; i += 8) {                                                                   \
            __asm__ अस्थिर(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(भग्न (*)[8]) & dst_block[i])                                                           \
                : [s0] "m"(*(स्थिर भग्न (*)[8]) & src0_block[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        अगर (पुच्छ > 0) {                                                                                              \
            स्थिर अचिन्हित दीर्घ पुच्छ_m0 = (1ul << पुच्छ) - 1;                                                         \
            __asm__ अस्थिर(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_block[vec_end]), [ps] "r"(&scalar), [d] "r"(&dst_block[vec_end]), [पंचांग] "r"(पुच्छ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ अस्थिर("mova.m.x %0" ::"r"(अस्थायी_mask));                                                            \
    }

DEFINE_BLOCK_OP_BROADCAST(block_mul_broadcast, "fmul.ps")
DEFINE_BLOCK_OP_BROADCAST(block_add_broadcast, "fadd.ps")
DEFINE_BLOCK_OP_BROADCAST(block_sub_broadcast, "fsub.ps")

अटल अंतरभूत भग्न scalar_el_map(भग्न src0, भग्न src1, क्रमागत ggml_op operation) {
    चयन (operation) {
        हाल GGML_OP_MUL:
            वापस src0 * src1;
        हाल GGML_OP_ADD:
            वापस src0 + src1;
        हाल GGML_OP_SUB:
            वापस src0 - src1;
        शेष:
            वापस 0.0f;
    }
}

पूर्णांक entry_point(काष्ठा ggml_et_द्विआधारी_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;  // Invalid pointer
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;  // Unsupported type combination
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * src1_तथ्य = (भग्न *) src1->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !src1_तथ्य || !dst_तथ्य) {
        वापस -1;  // Null data pointer
    }

#अगर_घोषित ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    स्थिर माप_प्रकार src0_bytes = (माप_प्रकार) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    स्थिर माप_प्रकार src1_bytes = (माप_प्रकार) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_तथ्य, src0_bytes);
    evict_region_past_l2(src1_तथ्य, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#पूर्ण_अगर

    क्रमागत ggml_op operation = dst->op;

    अगर (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        वापस -1;  // Unsupported operation
    }

    स्थिर int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    स्थिर int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    स्थिर int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    स्थिर माप_प्रकार nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    स्थिर माप_प्रकार nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिर माप_प्रकार nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    स्थिर bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    स्थिर bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    स्थिर bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    अगर (no_broadcast && all_contiguous) {
        स्थिर int64_t योग_elements         = ne0 * ne1 * ne2 * ne3;
        स्थिर int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes
        स्थिर int64_t योग_cachelines       = (योग_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        स्थिर int64_t cl_per_thread = (योग_cachelines + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        अगर (cl_end > योग_cachelines) {
            cl_end = योग_cachelines;
        }

        अगर (cl_start >= योग_cachelines) {
            वापस 0;
        }

        स्थिर int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end * elements_per_cacheline;
        अगर (elem_end > योग_elements) {
            elem_end = योग_elements;
        }
        स्थिर int32_t गिनती = (int32_t) (elem_end - elem_start);

        चयन (operation) {
            हाल GGML_OP_MUL:
                block_mul_cache_aligned(dst_तथ्य + elem_start, src0_तथ्य + elem_start, src1_तथ्य + elem_start, गिनती);
                अवरोध;
            हाल GGML_OP_ADD:
                block_add_cache_aligned(dst_तथ्य + elem_start, src0_तथ्य + elem_start, src1_तथ्य + elem_start, गिनती);
                अवरोध;
            हाल GGML_OP_SUB:
                block_sub_cache_aligned(dst_तथ्य + elem_start, src0_तथ्य + elem_start, src1_तथ्य + elem_start, गिनती);
                अवरोध;
            शेष:
                वापस 1;
        }
#अगर_घोषित ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_तथ्य + elem_start, (माप_प्रकार) गिनती * माप(भग्न));
        WAIT_CACHEOPS;
        FENCE;
#पूर्ण_अगर
        वापस 0;
    }

    // Slow path: broadcasting or non-contiguous
    स्थिर int64_t योग_rows = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t end_row;

    अगर (cache_aligned) {
        स्थिर int64_t rows_per_thread = (योग_rows + अंक_threads - 1) / अंक_threads;
        start_row                     = thread_id * rows_per_thread;
        end_row = (start_row + rows_per_thread < योग_rows) ? (start_row + rows_per_thread) : योग_rows;
    } अन्यथा {
        स्थिर int64_t rows_per_group = et_rows_per_cacheline_group(ne0, माप(भग्न));
        स्थिर int64_t योग_groups   = (योग_rows + rows_per_group - 1) / rows_per_group;

        अगर (thread_id >= योग_groups) {
            वापस 0;
        }

        स्थिर int64_t group_start = thread_id;
        क्रम (int64_t grp = group_start; grp < योग_groups; grp += अंक_threads) {
            स्थिर int64_t group_row_start = grp * rows_per_group;
            int64_t       group_row_end   = group_row_start + rows_per_group;
            अगर (group_row_end > योग_rows) {
                group_row_end = योग_rows;
            }

#अगर_घोषित ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            स्थिर int64_t first_i03      = group_row_start / (ne2 * ne1);
            स्थिर int64_t first_i02      = (group_row_start - first_i03 * ne2 * ne1) / ne1;
            स्थिर int64_t first_i01      = (group_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            अक्षर *        group_dst_base = (अक्षर *) dst_तथ्य + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#पूर्ण_अगर

            क्रम (int64_t ir = group_row_start; ir < group_row_end; ir++) {
                स्थिर int64_t i03 = ir / (ne2 * ne1);
                स्थिर int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                स्थिर int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                स्थिर int64_t i13 = i03 % ne13;
                स्थिर int64_t i12 = i02 % ne12;
                स्थिर int64_t i11 = i01 % ne11;

                भग्न *       dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1);
                स्थिर भग्न * src0_ptr =
                    (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01);
                स्थिर भग्न * src1_ptr =
                    (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + i13 * nb13 + i12 * nb12 + i11 * nb11);

                अगर (ne10 == 1) {
                    स्थिर भग्न scalar = src1_ptr[0];
                    क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } अन्यथा {
                    क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#अगर_घोषित ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            स्थिर int64_t nrows = group_row_end - group_row_start;
            अगर (nrows > 0) {
                FENCE;
                evict_region_past_l2(group_dst_base, (माप_प्रकार) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#पूर्ण_अगर
        }

        वापस 0;
    }

    अगर (start_row >= योग_rows) {
        वापस 0;
    }

    क्रम (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        स्थिर int64_t i03 = ir / (ne2 * ne1);
        स्थिर int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        स्थिर int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        स्थिर int64_t i13 = i03 % ne13;
        स्थिर int64_t i12 = i02 % ne12;
        स्थिर int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        भग्न *       dst_ptr  = (भग्न *) ((अक्षर *) dst_तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1);
        स्थिर भग्न * src0_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01);
        स्थिर भग्न * src1_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + i13 * nb13 + i12 * nb12 + i11 * nb11);

        अगर (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            भग्न scalar = src1_ptr[0];
            चयन (operation) {
                हाल GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (पूर्णांक) ne0);
                    अवरोध;
                हाल GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (पूर्णांक) ne0);
                    अवरोध;
                हाल GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (पूर्णांक) ne0);
                    अवरोध;
                शेष:
                    वापस 1;
            }
        } अन्यथा {
            // Broadcasting in dimension 0: src1 repeats across src0
            स्थिर int64_t nr0 = ne0 / ne10;

            क्रम (int64_t r = 0; r < nr0; r++) {
                स्थिर भग्न * src0_block = src0_ptr + r * ne10;
                भग्न *       dst_block  = dst_ptr + r * ne10;

                चयन (operation) {
                    हाल GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (पूर्णांक) ne10);
                        अवरोध;
                    हाल GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (पूर्णांक) ne10);
                        अवरोध;
                    हाल GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (पूर्णांक) ne10);
                        अवरोध;
                    शेष:
                        वापस 1;
                }
            }
        }
    }

#अगर_घोषित ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    अगर (end_row > start_row) {
        FENCE;
        evict_region_past_l2((अक्षर *) dst_तथ्य + start_row * nb1, (माप_प्रकार) (end_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#पूर्ण_अगर
    वापस 0;
}
