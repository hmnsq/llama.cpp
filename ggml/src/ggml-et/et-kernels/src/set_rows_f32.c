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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#define CACHE_ಸಾಲು_SIZE_BYTES 64
#define CACHE_ಸಾಲು_F32_ELEMS  16  // 64 / 4
#define CACHE_ಸಾಲು_F16_ELEMS  32  // 64 / 2

ಸ್ಥಿರ int64_t gcd64(int64_t a, int64_t b) {
    ಯಾವತ್ (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    ಪ್ರತ್ಯಾಗಮ a;
}

ಸಂರಚನಾ ggml_et_set_rows_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 source data tensor
    ಸಂರಚನಾ ggml_tensor src1;  // I64 row indices tensor
    ಸಂರಚನಾ ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
ಸ್ಥಿರ ಶೂನ್ಯ copy_cache_aligned_f32(ಭಿನ್ನ * dst, ಧ್ರುವ ಭಿನ್ನ * src) {
    __asm__ ಚಞ್ಚಲ(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
ಸ್ಥಿರ ಶೂನ್ಯ copy_cache_aligned_f16(uint16_t * dst, ಧ್ರುವ ಭಿನ್ನ * src) {
    unsigned long mask_temp;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    ಭಿನ್ನ      offset_vec_storage[8];
    uint32_t * offsets = (uint32_t *) offset_vec_storage;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ ಚಞ್ಚಲ(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [mask_temp] "=&r"(mask_temp)
        : [offsets] "r"(offset_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 32; i += 8) {
        __asm__ ಚಞ್ಚಲ(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ ಚಞ್ಚಲ("mova.m.x  %[mask_temp]         \n\t" : : [mask_temp] "r"(mask_temp));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t tensor_bytes(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_set_rows_params * params, ಶೂನ್ಯ * env) {
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

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid source types
    }

    ಯದಿ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported destination type
    }

    ಭಿನ್ನ *   src0_data = (ಭಿನ್ನ *) src0->data;
    int64_t * src1_data = (int64_t *) src1->data;
    ಶೂನ್ಯ *    dst_data  = dst->data;

    ಯದಿ (!src0_data || !src1_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Null data pointer
    }

    ಧ್ರುವ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ಧ್ರುವ int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    ಧ್ರುವ int64_t ne02 = src0->ne[2];  // Source batch dimension
    ಧ್ರುವ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ಧ್ರುವ int64_t nb01 = src0->nb[1];
    ಧ್ರುವ int64_t nb02 = src0->nb[2];
    ಧ್ರುವ int64_t nb03 = src0->nb[3];

    ಧ್ರುವ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ಧ್ರುವ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ಧ್ರುವ int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    ಧ್ರುವ int64_t nb10 = src1->nb[0];
    ಧ್ರುವ int64_t nb11 = src1->nb[1];
    ಧ್ರುವ int64_t nb12 = src1->nb[2];

    ಧ್ರುವ int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    ಧ್ರುವ int64_t nb1 = dst->nb[1];
    ಧ್ರುವ int64_t nb2 = dst->nb[2];
    ಧ್ರುವ int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    ಯದಿ (ne10 != ne01) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Number of indices must match number of source rows
    }
#ifdef ET_UBERKERNEL
    evict_region_past_l2(params->src0.data, tensor_bytes(&params->src0));
    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    ಧ್ರುವ int64_t total_rows = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ಧ್ರುವ int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_ಸಾಲು_F16_ELEMS : CACHE_ಸಾಲು_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ಧ್ರುವ bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    ಯದಿ (row_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ಧ್ರುವ int64_t cls_per_row    = ne00 / dst_cl_elems;
        ಧ್ರುವ int64_t total_cls      = total_rows * cls_per_row;
        ಧ್ರುವ int64_t cls_per_thread = (total_cls + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
        ಧ್ರುವ int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        ಯದಿ (my_end > total_cls) {
            my_end = total_cls;
        }
        ಯದಿ (my_start >= total_cls) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        ಪ್ರತಿ (int64_t cl = my_start; cl < my_end; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ಧ್ರುವ int64_t row_flat  = cl / cls_per_row;
            ಧ್ರುವ int64_t cl_in_row = cl % cls_per_row;

            // Decompose flat row -> (i03, i02, i01)
            ಧ್ರುವ int64_t i01 = row_flat % ne01;
            ಧ್ರುವ int64_t tmp = row_flat / ne01;
            ಧ್ರುವ int64_t i02 = tmp % ne02;
            ಧ್ರುವ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ಧ್ರುವ int64_t i12               = i03 % ne12;
            ಧ್ರುವ int64_t i11               = i02 % ne11;
            ಧ್ರುವ int64_t i10               = i01;
            ಧ್ರುವ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ಧ್ರುವ int64_t dst_row_index     = *(int64_t *) ((ವರ್ಣ *) src1_data + index_byte_offset);

            ಯದಿ (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                ಪ್ರತ್ಯಾಗಮ -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ಧ್ರುವ int64_t elem_offset = cl_in_row * dst_cl_elems;
            ಧ್ರುವ ಭಿನ್ನ * src_ptr =
                (ಧ್ರುವ ಭಿನ್ನ *) ((ವರ್ಣ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_offset;

            // Destination pointer: scattered row base + cache-line offset
            ವರ್ಣ * dst_row_base = (ವರ್ಣ *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            ಯದಿ (dst->type == GGML_TYPE_F32) {
                ಭಿನ್ನ * dst_ptr = (ಭಿನ್ನ *) dst_row_base + elem_offset;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } ಅನ್ಯಥಾ {
                uint16_t * dst_ptr = (uint16_t *) dst_row_base + elem_offset;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } ಅನ್ಯಥಾ ಯದಿ (nb1 % CACHE_ಸಾಲು_SIZE_BYTES == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ಧ್ರುವ int64_t g              = gcd64(ne00, dst_cl_elems);
        ಧ್ರುವ int64_t rows_per_group = dst_cl_elems / g;  // lcm / ne00
        ಧ್ರುವ int64_t cls_per_group  = ne00 / g;          // lcm / dst_cl_elems

        ಧ್ರುವ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;
        ಧ್ರುವ int64_t total_cls      = total_groups * cls_per_group;
        ಧ್ರುವ int64_t cls_per_thread = (total_cls + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
        ಧ್ರುವ int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        ಯದಿ (my_end > total_cls) {
            my_end = total_cls;
        }
        ಯದಿ (my_start >= total_cls) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

#ifdef BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#endif


        ಪ್ರತಿ (int64_t cl = my_start; cl < my_end; cl++) {
            ಧ್ರುವ int64_t group_idx   = cl / cls_per_group;
            ಧ್ರುವ int64_t cl_in_group = cl % cls_per_group;

            // Element range [elem_start, elem_end) within the flattened group
            ಧ್ರುವ int64_t elem_start = cl_in_group * dst_cl_elems;
            ಧ್ರುವ int64_t elem_end   = elem_start + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ಧ್ರುವ int64_t r_first = elem_start / ne00;
            ಧ್ರುವ int64_t r_last  = (elem_end - 1) / ne00;

            ಪ್ರತಿ (int64_t r = r_first; r <= r_last; r++) {
                ಧ್ರುವ int64_t row_flat = group_idx * rows_per_group + r;
                ಯದಿ (row_flat >= total_rows) {
                    ಭಙ್ಗ;
                }

                // Column range within this row
                int64_t col_begin = (r == r_first) ? (elem_start - r * ne00) : 0;
                int64_t col_end   = (r == r_last) ? (elem_end - r * ne00) : ne00;
                ಯದಿ (col_end > ne00) {
                    col_end = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ಧ್ರುವ int64_t i01 = row_flat % ne01;
                ಧ್ರುವ int64_t tmp = row_flat / ne01;
                ಧ್ರುವ int64_t i02 = tmp % ne02;
                ಧ್ರುವ int64_t i03 = tmp / ne02;

                // Look up destination row index
                ಧ್ರುವ int64_t i12               = i03 % ne12;
                ಧ್ರುವ int64_t i11               = i02 % ne11;
                ಧ್ರುವ int64_t i10               = i01;
                ಧ್ರುವ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ಧ್ರುವ int64_t dst_row_index     = *(int64_t *) ((ವರ್ಣ *) src1_data + index_byte_offset);

                ಯದಿ (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                    ಪ್ರತ್ಯಾಗಮ -1;
                }

                ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ವರ್ಣ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                ವರ್ಣ *        dst_row_base = (ವರ್ಣ *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ಧ್ರುವ bool full_cl = (col_begin % dst_cl_elems == 0) && (col_end - col_begin == dst_cl_elems);

                ಯದಿ (dst->type == GGML_TYPE_F32) {
                    ಭಿನ್ನ * dp = (ಭಿನ್ನ *) dst_row_base;
                    ಯದಿ (full_cl) {
                        copy_cache_aligned_f32(dp + col_begin, src_row + col_begin);
                    } ಅನ್ಯಥಾ {
                        ಪ್ರತಿ (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = src_row[i];
                        }
                    }
                } ಅನ್ಯಥಾ {
                    uint16_t * dp = (uint16_t *) dst_row_base;
                    ಯದಿ (full_cl) {
                        copy_cache_aligned_f16(dp + col_begin, src_row + col_begin);
                    } ಅನ್ಯಥಾ {
                        ಪ್ರತಿ (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = fp32_to_fp16(src_row[i]);
                        }
                    }
                }
            }
        }

#ifdef BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#endif


    } ಅನ್ಯಥಾ {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        ಪ್ರತಿ (int64_t row_flat = thread_id; row_flat < total_rows; row_flat += ಅಂಕೆ_threads) {
            ಧ್ರುವ int64_t i01 = row_flat % ne01;
            ಧ್ರುವ int64_t tmp = row_flat / ne01;
            ಧ್ರುವ int64_t i02 = tmp % ne02;
            ಧ್ರುವ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ಧ್ರುವ int64_t i12               = i03 % ne12;
            ಧ್ರುವ int64_t i11               = i02 % ne11;
            ಧ್ರುವ int64_t i10               = i01;
            ಧ್ರುವ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ಧ್ರುವ int64_t dst_row_index     = *(int64_t *) ((ವರ್ಣ *) src1_data + index_byte_offset);

            ಯದಿ (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                ಪ್ರತ್ಯಾಗಮ -1;
            }

            ಧ್ರುವ ಭಿನ್ನ * src_row      = (ಧ್ರುವ ಭಿನ್ನ *) ((ವರ್ಣ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ವರ್ಣ *        dst_row_base = (ವರ್ಣ *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            ಯದಿ (dst->type == GGML_TYPE_F32) {
                ಚಞ್ಚಲ ಭಿನ್ನ * dst_row = (ಚಞ್ಚಲ ಭಿನ್ನ *) dst_row_base;
                ಪ್ರತಿ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_row + i, src_row[i]);
                }
            } ಅನ್ಯಥಾ {
                ಚಞ್ಚಲ uint16_t * dst_row = (ಚಞ್ಚಲ uint16_t *) dst_row_base;
                ಪ್ರತಿ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_row + i, fp32_to_fp16(src_row[i]));
                }
            }
        }
    }

#ifdef BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#endif
    ಪ್ರತ್ಯಾಗಮ 0;
}
