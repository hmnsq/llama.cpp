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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

#define CACHE_पंक्ति_SIZE_BYTES 64
#define CACHE_पंक्ति_F32_ELEMS  16  // 64 / 4
#define CACHE_पंक्ति_F16_ELEMS  32  // 64 / 2

स्थिर int64_t gcd64(int64_t a, int64_t b) {
    यावत् (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    प्रत्यागम a;
}

संरचना ggml_et_set_rows_params {
    संरचना ggml_tensor src0;  // F32 source data tensor
    संरचना ggml_tensor src1;  // I64 row indices tensor
    संरचना ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
स्थिर शून्य copy_cache_aligned_f32(भिन्न * dst, ध्रुव भिन्न * src) {
    __asm__ चञ्चल(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
स्थिर शून्य copy_cache_aligned_f16(uint16_t * dst, ध्रुव भिन्न * src) {
    unsigned long mask_temp;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    भिन्न      offset_vec_storage[8];
    uint32_t * offsets = (uint32_t *) offset_vec_storage;
    प्रति (पूर्णाङ्क j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ चञ्चल(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [mask_temp] "=&r"(mask_temp)
        : [offsets] "r"(offset_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    प्रति (पूर्णाङ्क i = 0; i < 32; i += 8) {
        __asm__ चञ्चल(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ चञ्चल("mova.m.x  %[mask_temp]         \n\t" : : [mask_temp] "r"(mask_temp));
}

स्थिर अन्तःस्थ size_t tensor_bytes(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णाङ्क entry_point(संरचना ggml_et_set_rows_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    संरचना ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    संरचना ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        प्रत्यागम -1;  // Invalid source types
    }

    यदि (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        प्रत्यागम -1;  // Unsupported destination type
    }

    भिन्न *   src0_data = (भिन्न *) src0->data;
    int64_t * src1_data = (int64_t *) src1->data;
    शून्य *    dst_data  = dst->data;

    यदि (!src0_data || !src1_data || !dst_data) {
        प्रत्यागम -1;  // Null data pointer
    }

    ध्रुव int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ध्रुव int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    ध्रुव int64_t ne02 = src0->ne[2];  // Source batch dimension
    ध्रुव int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];

    ध्रुव int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ध्रुव int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ध्रुव int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    ध्रुव int64_t nb10 = src1->nb[0];
    ध्रुव int64_t nb11 = src1->nb[1];
    ध्रुव int64_t nb12 = src1->nb[2];

    ध्रुव int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    ध्रुव int64_t nb1 = dst->nb[1];
    ध्रुव int64_t nb2 = dst->nb[2];
    ध्रुव int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    यदि (ne10 != ne01) {
        प्रत्यागम -1;  // Number of indices must match number of source rows
    }
#ifdef ET_UBERKERNEL
    evict_region_past_l2(params->src0.data, tensor_bytes(&params->src0));
    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    ध्रुव int64_t total_rows = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ध्रुव int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_पंक्ति_F16_ELEMS : CACHE_पंक्ति_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ध्रुव bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    यदि (row_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ध्रुव int64_t cls_per_row    = ne00 / dst_cl_elems;
        ध्रुव int64_t total_cls      = total_rows * cls_per_row;
        ध्रुव int64_t cls_per_thread = (total_cls + अंक_threads - 1) / अंक_threads;
        ध्रुव int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        यदि (my_end > total_cls) {
            my_end = total_cls;
        }
        यदि (my_start >= total_cls) {
            प्रत्यागम 0;
        }

        प्रति (int64_t cl = my_start; cl < my_end; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ध्रुव int64_t row_flat  = cl / cls_per_row;
            ध्रुव int64_t cl_in_row = cl % cls_per_row;

            // Decompose flat row -> (i03, i02, i01)
            ध्रुव int64_t i01 = row_flat % ne01;
            ध्रुव int64_t tmp = row_flat / ne01;
            ध्रुव int64_t i02 = tmp % ne02;
            ध्रुव int64_t i03 = tmp / ne02;

            // Look up destination row index
            ध्रुव int64_t i12               = i03 % ne12;
            ध्रुव int64_t i11               = i02 % ne11;
            ध्रुव int64_t i10               = i01;
            ध्रुव int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ध्रुव int64_t dst_row_index     = *(int64_t *) ((वर्ण *) src1_data + index_byte_offset);

            यदि (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                प्रत्यागम -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ध्रुव int64_t elem_offset = cl_in_row * dst_cl_elems;
            ध्रुव भिन्न * src_ptr =
                (ध्रुव भिन्न *) ((वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_offset;

            // Destination pointer: scattered row base + cache-line offset
            वर्ण * dst_row_base = (वर्ण *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            यदि (dst->type == GGML_TYPE_F32) {
                भिन्न * dst_ptr = (भिन्न *) dst_row_base + elem_offset;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } अन्यथा {
                uint16_t * dst_ptr = (uint16_t *) dst_row_base + elem_offset;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } अन्यथा यदि (nb1 % CACHE_पंक्ति_SIZE_BYTES == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ध्रुव int64_t g              = gcd64(ne00, dst_cl_elems);
        ध्रुव int64_t rows_per_group = dst_cl_elems / g;  // lcm / ne00
        ध्रुव int64_t cls_per_group  = ne00 / g;          // lcm / dst_cl_elems

        ध्रुव int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;
        ध्रुव int64_t total_cls      = total_groups * cls_per_group;
        ध्रुव int64_t cls_per_thread = (total_cls + अंक_threads - 1) / अंक_threads;
        ध्रुव int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        यदि (my_end > total_cls) {
            my_end = total_cls;
        }
        यदि (my_start >= total_cls) {
            प्रत्यागम 0;
        }

#ifdef BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#endif


        प्रति (int64_t cl = my_start; cl < my_end; cl++) {
            ध्रुव int64_t group_idx   = cl / cls_per_group;
            ध्रुव int64_t cl_in_group = cl % cls_per_group;

            // Element range [elem_start, elem_end) within the flattened group
            ध्रुव int64_t elem_start = cl_in_group * dst_cl_elems;
            ध्रुव int64_t elem_end   = elem_start + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ध्रुव int64_t r_first = elem_start / ne00;
            ध्रुव int64_t r_last  = (elem_end - 1) / ne00;

            प्रति (int64_t r = r_first; r <= r_last; r++) {
                ध्रुव int64_t row_flat = group_idx * rows_per_group + r;
                यदि (row_flat >= total_rows) {
                    भङ्ग;
                }

                // Column range within this row
                int64_t col_begin = (r == r_first) ? (elem_start - r * ne00) : 0;
                int64_t col_end   = (r == r_last) ? (elem_end - r * ne00) : ne00;
                यदि (col_end > ne00) {
                    col_end = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ध्रुव int64_t i01 = row_flat % ne01;
                ध्रुव int64_t tmp = row_flat / ne01;
                ध्रुव int64_t i02 = tmp % ne02;
                ध्रुव int64_t i03 = tmp / ne02;

                // Look up destination row index
                ध्रुव int64_t i12               = i03 % ne12;
                ध्रुव int64_t i11               = i02 % ne11;
                ध्रुव int64_t i10               = i01;
                ध्रुव int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ध्रुव int64_t dst_row_index     = *(int64_t *) ((वर्ण *) src1_data + index_byte_offset);

                यदि (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                    प्रत्यागम -1;
                }

                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                वर्ण *        dst_row_base = (वर्ण *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ध्रुव bool full_cl = (col_begin % dst_cl_elems == 0) && (col_end - col_begin == dst_cl_elems);

                यदि (dst->type == GGML_TYPE_F32) {
                    भिन्न * dp = (भिन्न *) dst_row_base;
                    यदि (full_cl) {
                        copy_cache_aligned_f32(dp + col_begin, src_row + col_begin);
                    } अन्यथा {
                        प्रति (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = src_row[i];
                        }
                    }
                } अन्यथा {
                    uint16_t * dp = (uint16_t *) dst_row_base;
                    यदि (full_cl) {
                        copy_cache_aligned_f16(dp + col_begin, src_row + col_begin);
                    } अन्यथा {
                        प्रति (int64_t i = col_begin; i < col_end; i++) {
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


    } अन्यथा {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        प्रति (int64_t row_flat = thread_id; row_flat < total_rows; row_flat += अंक_threads) {
            ध्रुव int64_t i01 = row_flat % ne01;
            ध्रुव int64_t tmp = row_flat / ne01;
            ध्रुव int64_t i02 = tmp % ne02;
            ध्रुव int64_t i03 = tmp / ne02;

            // Look up destination row index
            ध्रुव int64_t i12               = i03 % ne12;
            ध्रुव int64_t i11               = i02 % ne11;
            ध्रुव int64_t i10               = i01;
            ध्रुव int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ध्रुव int64_t dst_row_index     = *(int64_t *) ((वर्ण *) src1_data + index_byte_offset);

            यदि (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                प्रत्यागम -1;
            }

            ध्रुव भिन्न * src_row      = (ध्रुव भिन्न *) ((वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            वर्ण *        dst_row_base = (वर्ण *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            यदि (dst->type == GGML_TYPE_F32) {
                चञ्चल भिन्न * dst_row = (चञ्चल भिन्न *) dst_row_base;
                प्रति (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_row + i, src_row[i]);
                }
            } अन्यथा {
                चञ्चल uint16_t * dst_row = (चञ्चल uint16_t *) dst_row_base;
                प्रति (int64_t i = 0; i < ne00; i++) {
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
    प्रत्यागम 0;
}
