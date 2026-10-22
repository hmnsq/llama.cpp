//******************************************************************************
// Bare Metal SET_ROWS F32 Kernel
// Writes source data rows to specific indices in destination tensor
//
// Algorithm:
// 1. Read row indices from src1 (int64 tensor)
// 2. For each source row, write it to destination at the specified index
// 3. Handle type conversion: F32 source -> F32/F16 destination
// 4. Support multi-dimensional tensor operations
//
// Operation: dst[indices[i]] = src[i] for i = 0..num_source_rows
// This is the inverse of GET_ROWS operation
//
// As ET is not a cache coherent processor yet SET_ROWS often are setting
// small mount of large rows (KV cache). There's several strategies to
// optimize this operation, including cacheline-based parallelization.
//
// - distribute work at cacheline granularity
// - if previous does not work, find the LCM of cacheline size
//
// Features supported:
// - F32 source data (always F32 input)
// - F32 and F16 destination data (with transcoding)
// - Int64 row indices (vs Int32 in GET_ROWS)
// - Multi-dimensional tensor support
// - Sequential source reads, scattered destination writes
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <assert.h>
#समावेश <stdbool.h>
#समावेश <stdint.h>

#घोषणा CACHE_पंक्ति_आकार_BYTES 64
#घोषणा CACHE_पंक्ति_F32_ELEMS  16  // 64 / 4
#घोषणा CACHE_पंक्ति_F16_ELEMS  32  // 64 / 2

अटल int64_t gcd64(int64_t a, int64_t b) {
    जबतक (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    वापस a;
}

काष्ठा ggml_et_set_rows_params {
    काष्ठा ggml_tensor src0;  // F32 source data tensor
    काष्ठा ggml_tensor src1;  // I64 row indices tensor
    काष्ठा ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
अटल व्योम copy_cache_aligned_f32(भग्न * dst, स्थिर भग्न * src) {
    __asm__ अस्थिर(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
अटल व्योम copy_cache_aligned_f16(uint16_t * dst, स्थिर भग्न * src) {
    अचिन्हित दीर्घ mask_अस्थायी;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    भग्न      offset_vec_storage[8];
    uint32_t * offsets = (uint32_t *) offset_vec_storage;
    क्रम (पूर्णांक j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ अस्थिर(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [mask_अस्थायी] "=&r"(mask_अस्थायी)
        : [offsets] "r"(offset_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    क्रम (पूर्णांक i = 0; i < 32; i += 8) {
        __asm__ अस्थिर(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ अस्थिर("mova.m.x  %[mask_temp]         \n\t" : : [mask_अस्थायी] "r"(mask_अस्थायी));
}

अटल अंतरभूत माप_प्रकार tensor_bytes(स्थिर काष्ठा ggml_tensor * t) {
    वापस (माप_प्रकार) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णांक entry_point(काष्ठा ggml_et_set_rows_params * params, व्योम * env) {
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

    काष्ठा ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    काष्ठा ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    काष्ठा ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        वापस -1;  // Invalid source types
    }

    अगर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        वापस -1;  // Unsupported destination type
    }

    भग्न *   src0_तथ्य = (भग्न *) src0->तथ्य;
    int64_t * src1_तथ्य = (int64_t *) src1->तथ्य;
    व्योम *    dst_तथ्य  = dst->तथ्य;

    अगर (!src0_तथ्य || !src1_तथ्य || !dst_तथ्य) {
        वापस -1;  // Null data pointer
    }

    स्थिर int64_t ne00 = src0->ne[0];  // Source columns (row width)
    स्थिर int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    स्थिर int64_t ne02 = src0->ne[2];  // Source batch dimension
    स्थिर int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    स्थिर int64_t nb01 = src0->nb[1];
    स्थिर int64_t nb02 = src0->nb[2];
    स्थिर int64_t nb03 = src0->nb[3];

    स्थिर int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    स्थिर int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    स्थिर int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    स्थिर int64_t nb10 = src1->nb[0];
    स्थिर int64_t nb11 = src1->nb[1];
    स्थिर int64_t nb12 = src1->nb[2];

    स्थिर int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    स्थिर int64_t nb1 = dst->nb[1];
    स्थिर int64_t nb2 = dst->nb[2];
    स्थिर int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    अगर (ne10 != ne01) {
        वापस -1;  // Number of indices must match number of source rows
    }
#अगर_घोषित ET_UBERKERNEL
    evict_region_past_l2(params->src0.तथ्य, tensor_bytes(&params->src0));
    evict_region_past_l2(params->src1.तथ्य, tensor_bytes(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#पूर्ण_अगर
    स्थिर int64_t योग_rows = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    स्थिर int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_पंक्ति_F16_ELEMS : CACHE_पंक्ति_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    स्थिर bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    अगर (row_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        स्थिर int64_t cls_per_row    = ne00 / dst_cl_elems;
        स्थिर int64_t योग_cls      = योग_rows * cls_per_row;
        स्थिर int64_t cls_per_thread = (योग_cls + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        अगर (my_end > योग_cls) {
            my_end = योग_cls;
        }
        अगर (my_start >= योग_cls) {
            वापस 0;
        }

        क्रम (int64_t cl = my_start; cl < my_end; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            स्थिर int64_t row_flat  = cl / cls_per_row;
            स्थिर int64_t cl_in_row = cl % cls_per_row;

            // Decompose flat row -> (i03, i02, i01)
            स्थिर int64_t i01 = row_flat % ne01;
            स्थिर int64_t tmp = row_flat / ne01;
            स्थिर int64_t i02 = tmp % ne02;
            स्थिर int64_t i03 = tmp / ne02;

            // Look up destination row index
            स्थिर int64_t i12               = i03 % ne12;
            स्थिर int64_t i11               = i02 % ne11;
            स्थिर int64_t i10               = i01;
            स्थिर int64_t सूचकांक_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            स्थिर int64_t dst_row_सूचकांक     = *(int64_t *) ((अक्षर *) src1_तथ्य + सूचकांक_byte_offset);

            अगर (dst_row_सूचकांक < 0 || dst_row_सूचकांक >= ne_dst1) {
                वापस -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            स्थिर int64_t elem_offset = cl_in_row * dst_cl_elems;
            स्थिर भग्न * src_ptr =
                (स्थिर भग्न *) ((अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_offset;

            // Destination pointer: scattered row base + cache-line offset
            अक्षर * dst_row_base = (अक्षर *) dst_तथ्य + dst_row_सूचकांक * nb1 + i02 * nb2 + i03 * nb3;

            अगर (dst->type == GGML_TYPE_F32) {
                भग्न * dst_ptr = (भग्न *) dst_row_base + elem_offset;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } अन्यथा {
                uint16_t * dst_ptr = (uint16_t *) dst_row_base + elem_offset;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } अन्यथा अगर (nb1 % CACHE_पंक्ति_आकार_BYTES == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        स्थिर int64_t g              = gcd64(ne00, dst_cl_elems);
        स्थिर int64_t rows_per_group = dst_cl_elems / g;  // lcm / ne00
        स्थिर int64_t cls_per_group  = ne00 / g;          // lcm / dst_cl_elems

        स्थिर int64_t योग_groups   = (योग_rows + rows_per_group - 1) / rows_per_group;
        स्थिर int64_t योग_cls      = योग_groups * cls_per_group;
        स्थिर int64_t cls_per_thread = (योग_cls + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        अगर (my_end > योग_cls) {
            my_end = योग_cls;
        }
        अगर (my_start >= योग_cls) {
            वापस 0;
        }

#अगर_घोषित BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#पूर्ण_अगर


        क्रम (int64_t cl = my_start; cl < my_end; cl++) {
            स्थिर int64_t group_idx   = cl / cls_per_group;
            स्थिर int64_t cl_in_group = cl % cls_per_group;

            // Element range [elem_start, elem_end) within the flattened group
            स्थिर int64_t elem_start = cl_in_group * dst_cl_elems;
            स्थिर int64_t elem_end   = elem_start + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            स्थिर int64_t r_first = elem_start / ne00;
            स्थिर int64_t r_last  = (elem_end - 1) / ne00;

            क्रम (int64_t r = r_first; r <= r_last; r++) {
                स्थिर int64_t row_flat = group_idx * rows_per_group + r;
                अगर (row_flat >= योग_rows) {
                    अवरोध;
                }

                // Column range within this row
                int64_t col_begin = (r == r_first) ? (elem_start - r * ne00) : 0;
                int64_t col_end   = (r == r_last) ? (elem_end - r * ne00) : ne00;
                अगर (col_end > ne00) {
                    col_end = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                स्थिर int64_t i01 = row_flat % ne01;
                स्थिर int64_t tmp = row_flat / ne01;
                स्थिर int64_t i02 = tmp % ne02;
                स्थिर int64_t i03 = tmp / ne02;

                // Look up destination row index
                स्थिर int64_t i12               = i03 % ne12;
                स्थिर int64_t i11               = i02 % ne11;
                स्थिर int64_t i10               = i01;
                स्थिर int64_t सूचकांक_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
                स्थिर int64_t dst_row_सूचकांक     = *(int64_t *) ((अक्षर *) src1_तथ्य + सूचकांक_byte_offset);

                अगर (dst_row_सूचकांक < 0 || dst_row_सूचकांक >= ne_dst1) {
                    वापस -1;
                }

                स्थिर भग्न * src_row = (स्थिर भग्न *) ((अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);
                अक्षर *        dst_row_base = (अक्षर *) dst_तथ्य + dst_row_सूचकांक * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                स्थिर bool full_cl = (col_begin % dst_cl_elems == 0) && (col_end - col_begin == dst_cl_elems);

                अगर (dst->type == GGML_TYPE_F32) {
                    भग्न * dp = (भग्न *) dst_row_base;
                    अगर (full_cl) {
                        copy_cache_aligned_f32(dp + col_begin, src_row + col_begin);
                    } अन्यथा {
                        क्रम (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = src_row[i];
                        }
                    }
                } अन्यथा {
                    uint16_t * dp = (uint16_t *) dst_row_base;
                    अगर (full_cl) {
                        copy_cache_aligned_f16(dp + col_begin, src_row + col_begin);
                    } अन्यथा {
                        क्रम (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = fp32_to_fp16(src_row[i]);
                        }
                    }
                }
            }
        }

#अगर_घोषित BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#पूर्ण_अगर


    } अन्यथा {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        क्रम (int64_t row_flat = thread_id; row_flat < योग_rows; row_flat += अंक_threads) {
            स्थिर int64_t i01 = row_flat % ne01;
            स्थिर int64_t tmp = row_flat / ne01;
            स्थिर int64_t i02 = tmp % ne02;
            स्थिर int64_t i03 = tmp / ne02;

            // Look up destination row index
            स्थिर int64_t i12               = i03 % ne12;
            स्थिर int64_t i11               = i02 % ne11;
            स्थिर int64_t i10               = i01;
            स्थिर int64_t सूचकांक_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            स्थिर int64_t dst_row_सूचकांक     = *(int64_t *) ((अक्षर *) src1_तथ्य + सूचकांक_byte_offset);

            अगर (dst_row_सूचकांक < 0 || dst_row_सूचकांक >= ne_dst1) {
                वापस -1;
            }

            स्थिर भग्न * src_row      = (स्थिर भग्न *) ((अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);
            अक्षर *        dst_row_base = (अक्षर *) dst_तथ्य + dst_row_सूचकांक * nb1 + i02 * nb2 + i03 * nb3;

            अगर (dst->type == GGML_TYPE_F32) {
                अस्थिर भग्न * dst_row = (अस्थिर भग्न *) dst_row_base;
                क्रम (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_row + i, src_row[i]);
                }
            } अन्यथा {
                अस्थिर uint16_t * dst_row = (अस्थिर uint16_t *) dst_row_base;
                क्रम (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_row + i, fp32_to_fp16(src_row[i]));
                }
            }
        }
    }

#अगर_घोषित BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#पूर्ण_अगर
    वापस 0;
}
