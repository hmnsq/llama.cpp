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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#define CACHE_ଧାଡ଼ି_SIZE_BYTES 64
#define CACHE_ଧାଡ଼ି_F32_ELEMS  16  // 64 / 4
#define CACHE_ଧାଡ଼ି_F16_ELEMS  32  // 64 / 2

ସ୍ଥିର int64_t gcd64(int64_t a, int64_t b) {
    ଯେତେବେଳେ (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    ଫେରନ୍ତୁ a;
}

ସଂରଚନା ggml_et_set_rows_params {
    ସଂରଚନା ggml_tensor src0;  // F32 source data tensor
    ସଂରଚନା ggml_tensor src1;  // I64 row indices tensor
    ସଂରଚନା ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
ସ୍ଥିର ଶୂନ୍ୟ copy_cache_aligned_f32(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src) {
    __asm__ ଅସ୍ଥିର(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
ସ୍ଥିର ଶୂନ୍ୟ copy_cache_aligned_f16(uint16_t * dst, ଅଚଳ ଦଶମିକ * src) {
    unsigned long mask_temp;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    ଦଶମିକ      offset_vec_storage[8];
    uint32_t * offsets = (uint32_t *) offset_vec_storage;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [mask_temp] "=&r"(mask_temp)
        : [offsets] "r"(offset_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 32; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x  %[mask_temp]         \n\t" : : [mask_temp] "r"(mask_temp));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t tensor_bytes(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_set_rows_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;  // Invalid pointer
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    ସଂରଚନା ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    ସଂରଚନା ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        ଫେରନ୍ତୁ -1;  // Invalid source types
    }

    ଯଦି (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ -1;  // Unsupported destination type
    }

    ଦଶମିକ *   src0_data = (ଦଶମିକ *) src0->data;
    int64_t * src1_data = (int64_t *) src1->data;
    ଶୂନ୍ୟ *    dst_data  = dst->data;

    ଯଦି (!src0_data || !src1_data || !dst_data) {
        ଫେରନ୍ତୁ -1;  // Null data pointer
    }

    ଅଚଳ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ଅଚଳ int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    ଅଚଳ int64_t ne02 = src0->ne[2];  // Source batch dimension
    ଅଚଳ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ଅଚଳ int64_t nb01 = src0->nb[1];
    ଅଚଳ int64_t nb02 = src0->nb[2];
    ଅଚଳ int64_t nb03 = src0->nb[3];

    ଅଚଳ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ଅଚଳ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ଅଚଳ int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    ଅଚଳ int64_t nb10 = src1->nb[0];
    ଅଚଳ int64_t nb11 = src1->nb[1];
    ଅଚଳ int64_t nb12 = src1->nb[2];

    ଅଚଳ int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    ଅଚଳ int64_t nb1 = dst->nb[1];
    ଅଚଳ int64_t nb2 = dst->nb[2];
    ଅଚଳ int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    ଯଦି (ne10 != ne01) {
        ଫେରନ୍ତୁ -1;  // Number of indices must match number of source rows
    }
#ifdef ET_UBERKERNEL
    evict_region_past_l2(params->src0.data, tensor_bytes(&params->src0));
    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    ଅଚଳ int64_t total_rows = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ଅଚଳ int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_ଧାଡ଼ି_F16_ELEMS : CACHE_ଧାଡ଼ି_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ଅଚଳ bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    ଯଦି (row_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ଅଚଳ int64_t cls_per_row    = ne00 / dst_cl_elems;
        ଅଚଳ int64_t total_cls      = total_rows * cls_per_row;
        ଅଚଳ int64_t cls_per_thread = (total_cls + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
        ଅଚଳ int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        ଯଦି (my_end > total_cls) {
            my_end = total_cls;
        }
        ଯଦି (my_start >= total_cls) {
            ଫେରନ୍ତୁ 0;
        }

        ପାଇଁ (int64_t cl = my_start; cl < my_end; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ଅଚଳ int64_t row_flat  = cl / cls_per_row;
            ଅଚଳ int64_t cl_in_row = cl % cls_per_row;

            // Decompose flat row -> (i03, i02, i01)
            ଅଚଳ int64_t i01 = row_flat % ne01;
            ଅଚଳ int64_t tmp = row_flat / ne01;
            ଅଚଳ int64_t i02 = tmp % ne02;
            ଅଚଳ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ଅଚଳ int64_t i12               = i03 % ne12;
            ଅଚଳ int64_t i11               = i02 % ne11;
            ଅଚଳ int64_t i10               = i01;
            ଅଚଳ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ଅଚଳ int64_t dst_row_index     = *(int64_t *) ((ଅକ୍ଷର *) src1_data + index_byte_offset);

            ଯଦି (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                ଫେରନ୍ତୁ -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ଅଚଳ int64_t elem_offset = cl_in_row * dst_cl_elems;
            ଅଚଳ ଦଶମିକ * src_ptr =
                (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_offset;

            // Destination pointer: scattered row base + cache-line offset
            ଅକ୍ଷର * dst_row_base = (ଅକ୍ଷର *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            ଯଦି (dst->type == GGML_TYPE_F32) {
                ଦଶମିକ * dst_ptr = (ଦଶମିକ *) dst_row_base + elem_offset;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } ଅନ୍ୟଥା {
                uint16_t * dst_ptr = (uint16_t *) dst_row_base + elem_offset;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } ଅନ୍ୟଥା ଯଦି (nb1 % CACHE_ଧାଡ଼ି_SIZE_BYTES == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ଅଚଳ int64_t g              = gcd64(ne00, dst_cl_elems);
        ଅଚଳ int64_t rows_per_group = dst_cl_elems / g;  // lcm / ne00
        ଅଚଳ int64_t cls_per_group  = ne00 / g;          // lcm / dst_cl_elems

        ଅଚଳ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;
        ଅଚଳ int64_t total_cls      = total_groups * cls_per_group;
        ଅଚଳ int64_t cls_per_thread = (total_cls + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
        ଅଚଳ int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        ଯଦି (my_end > total_cls) {
            my_end = total_cls;
        }
        ଯଦି (my_start >= total_cls) {
            ଫେରନ୍ତୁ 0;
        }

#ifdef BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#endif


        ପାଇଁ (int64_t cl = my_start; cl < my_end; cl++) {
            ଅଚଳ int64_t group_idx   = cl / cls_per_group;
            ଅଚଳ int64_t cl_in_group = cl % cls_per_group;

            // Element range [elem_start, elem_end) within the flattened group
            ଅଚଳ int64_t elem_start = cl_in_group * dst_cl_elems;
            ଅଚଳ int64_t elem_end   = elem_start + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ଅଚଳ int64_t r_first = elem_start / ne00;
            ଅଚଳ int64_t r_last  = (elem_end - 1) / ne00;

            ପାଇଁ (int64_t r = r_first; r <= r_last; r++) {
                ଅଚଳ int64_t row_flat = group_idx * rows_per_group + r;
                ଯଦି (row_flat >= total_rows) {
                    ଭାଙ୍ଗ;
                }

                // Column range within this row
                int64_t col_begin = (r == r_first) ? (elem_start - r * ne00) : 0;
                int64_t col_end   = (r == r_last) ? (elem_end - r * ne00) : ne00;
                ଯଦି (col_end > ne00) {
                    col_end = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ଅଚଳ int64_t i01 = row_flat % ne01;
                ଅଚଳ int64_t tmp = row_flat / ne01;
                ଅଚଳ int64_t i02 = tmp % ne02;
                ଅଚଳ int64_t i03 = tmp / ne02;

                // Look up destination row index
                ଅଚଳ int64_t i12               = i03 % ne12;
                ଅଚଳ int64_t i11               = i02 % ne11;
                ଅଚଳ int64_t i10               = i01;
                ଅଚଳ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ଅଚଳ int64_t dst_row_index     = *(int64_t *) ((ଅକ୍ଷର *) src1_data + index_byte_offset);

                ଯଦି (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                    ଫେରନ୍ତୁ -1;
                }

                ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                ଅକ୍ଷର *        dst_row_base = (ଅକ୍ଷର *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ଅଚଳ bool full_cl = (col_begin % dst_cl_elems == 0) && (col_end - col_begin == dst_cl_elems);

                ଯଦି (dst->type == GGML_TYPE_F32) {
                    ଦଶମିକ * dp = (ଦଶମିକ *) dst_row_base;
                    ଯଦି (full_cl) {
                        copy_cache_aligned_f32(dp + col_begin, src_row + col_begin);
                    } ଅନ୍ୟଥା {
                        ପାଇଁ (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = src_row[i];
                        }
                    }
                } ଅନ୍ୟଥା {
                    uint16_t * dp = (uint16_t *) dst_row_base;
                    ଯଦି (full_cl) {
                        copy_cache_aligned_f16(dp + col_begin, src_row + col_begin);
                    } ଅନ୍ୟଥା {
                        ପାଇଁ (int64_t i = col_begin; i < col_end; i++) {
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


    } ଅନ୍ୟଥା {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        ପାଇଁ (int64_t row_flat = thread_id; row_flat < total_rows; row_flat += ଅଙ୍କ_threads) {
            ଅଚଳ int64_t i01 = row_flat % ne01;
            ଅଚଳ int64_t tmp = row_flat / ne01;
            ଅଚଳ int64_t i02 = tmp % ne02;
            ଅଚଳ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ଅଚଳ int64_t i12               = i03 % ne12;
            ଅଚଳ int64_t i11               = i02 % ne11;
            ଅଚଳ int64_t i10               = i01;
            ଅଚଳ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ଅଚଳ int64_t dst_row_index     = *(int64_t *) ((ଅକ୍ଷର *) src1_data + index_byte_offset);

            ଯଦି (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                ଫେରନ୍ତୁ -1;
            }

            ଅଚଳ ଦଶମିକ * src_row      = (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ଅକ୍ଷର *        dst_row_base = (ଅକ୍ଷର *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            ଯଦି (dst->type == GGML_TYPE_F32) {
                ଅସ୍ଥିର ଦଶମିକ * dst_row = (ଅସ୍ଥିର ଦଶମିକ *) dst_row_base;
                ପାଇଁ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_row + i, src_row[i]);
                }
            } ଅନ୍ୟଥା {
                ଅସ୍ଥିର uint16_t * dst_row = (ଅସ୍ଥିର uint16_t *) dst_row_base;
                ପାଇଁ (int64_t i = 0; i < ne00; i++) {
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
    ଫେରନ୍ତୁ 0;
}
