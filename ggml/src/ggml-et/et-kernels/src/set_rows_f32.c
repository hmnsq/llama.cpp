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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

#define CACHE_લીટી_SIZE_BYTES 64
#define CACHE_લીટી_F32_ELEMS  16  // 64 / 4
#define CACHE_લીટી_F16_ELEMS  32  // 64 / 2

સ્થિર int64_t gcd64(int64_t a, int64_t b) {
    જ્યારે (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    પાછા a;
}

સંરચના ggml_et_set_rows_params {
    સંરચના ggml_tensor src0;  // F32 source data tensor
    સંરચના ggml_tensor src1;  // I64 row indices tensor
    સંરચના ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
સ્થિર શૂન્ય copy_cache_aligned_f32(દશાંશ * dst, અચલ દશાંશ * src) {
    __asm__ અસ્થિર(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
સ્થિર શૂન્ય copy_cache_aligned_f16(uint16_t * dst, અચલ દશાંશ * src) {
    unsigned long mask_temp;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    દશાંશ      offset_vec_storage[8];
    uint32_t * offsets = (uint32_t *) offset_vec_storage;
    માટે (પૂર્ણાંક j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ અસ્થિર(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [mask_temp] "=&r"(mask_temp)
        : [offsets] "r"(offset_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    માટે (પૂર્ણાંક i = 0; i < 32; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ અસ્થિર("mova.m.x  %[mask_temp]         \n\t" : : [mask_temp] "r"(mask_temp));
}

સ્થિર અંતઃરેખ size_t tensor_bytes(અચલ સંરચના ggml_tensor * t) {
    પાછા (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

પૂર્ણાંક entry_point(સંરચના ggml_et_set_rows_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;  // Invalid pointer
    }

    સંરચના ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    સંરચના ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    સંરચના ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        પાછા -1;  // Invalid source types
    }

    જો (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        પાછા -1;  // Unsupported destination type
    }

    દશાંશ *   src0_data = (દશાંશ *) src0->data;
    int64_t * src1_data = (int64_t *) src1->data;
    શૂન્ય *    dst_data  = dst->data;

    જો (!src0_data || !src1_data || !dst_data) {
        પાછા -1;  // Null data pointer
    }

    અચલ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    અચલ int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    અચલ int64_t ne02 = src0->ne[2];  // Source batch dimension
    અચલ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    અચલ int64_t nb01 = src0->nb[1];
    અચલ int64_t nb02 = src0->nb[2];
    અચલ int64_t nb03 = src0->nb[3];

    અચલ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    અચલ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    અચલ int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    અચલ int64_t nb10 = src1->nb[0];
    અચલ int64_t nb11 = src1->nb[1];
    અચલ int64_t nb12 = src1->nb[2];

    અચલ int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    અચલ int64_t nb1 = dst->nb[1];
    અચલ int64_t nb2 = dst->nb[2];
    અચલ int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    જો (ne10 != ne01) {
        પાછા -1;  // Number of indices must match number of source rows
    }
#ifdef ET_UBERKERNEL
    evict_region_past_l2(params->src0.data, tensor_bytes(&params->src0));
    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    અચલ int64_t total_rows = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    અચલ int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_લીટી_F16_ELEMS : CACHE_લીટી_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    અચલ bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    જો (row_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        અચલ int64_t cls_per_row    = ne00 / dst_cl_elems;
        અચલ int64_t total_cls      = total_rows * cls_per_row;
        અચલ int64_t cls_per_thread = (total_cls + અંક_threads - 1) / અંક_threads;
        અચલ int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        જો (my_end > total_cls) {
            my_end = total_cls;
        }
        જો (my_start >= total_cls) {
            પાછા 0;
        }

        માટે (int64_t cl = my_start; cl < my_end; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            અચલ int64_t row_flat  = cl / cls_per_row;
            અચલ int64_t cl_in_row = cl % cls_per_row;

            // Decompose flat row -> (i03, i02, i01)
            અચલ int64_t i01 = row_flat % ne01;
            અચલ int64_t tmp = row_flat / ne01;
            અચલ int64_t i02 = tmp % ne02;
            અચલ int64_t i03 = tmp / ne02;

            // Look up destination row index
            અચલ int64_t i12               = i03 % ne12;
            અચલ int64_t i11               = i02 % ne11;
            અચલ int64_t i10               = i01;
            અચલ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            અચલ int64_t dst_row_index     = *(int64_t *) ((અક્ષર *) src1_data + index_byte_offset);

            જો (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                પાછા -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            અચલ int64_t elem_offset = cl_in_row * dst_cl_elems;
            અચલ દશાંશ * src_ptr =
                (અચલ દશાંશ *) ((અક્ષર *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_offset;

            // Destination pointer: scattered row base + cache-line offset
            અક્ષર * dst_row_base = (અક્ષર *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            જો (dst->type == GGML_TYPE_F32) {
                દશાંશ * dst_ptr = (દશાંશ *) dst_row_base + elem_offset;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } નહીંતર {
                uint16_t * dst_ptr = (uint16_t *) dst_row_base + elem_offset;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } નહીંતર જો (nb1 % CACHE_લીટી_SIZE_BYTES == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        અચલ int64_t g              = gcd64(ne00, dst_cl_elems);
        અચલ int64_t rows_per_group = dst_cl_elems / g;  // lcm / ne00
        અચલ int64_t cls_per_group  = ne00 / g;          // lcm / dst_cl_elems

        અચલ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;
        અચલ int64_t total_cls      = total_groups * cls_per_group;
        અચલ int64_t cls_per_thread = (total_cls + અંક_threads - 1) / અંક_threads;
        અચલ int64_t my_start       = thread_id * cls_per_thread;
        int64_t       my_end         = my_start + cls_per_thread;
        જો (my_end > total_cls) {
            my_end = total_cls;
        }
        જો (my_start >= total_cls) {
            પાછા 0;
        }

#ifdef BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#endif


        માટે (int64_t cl = my_start; cl < my_end; cl++) {
            અચલ int64_t group_idx   = cl / cls_per_group;
            અચલ int64_t cl_in_group = cl % cls_per_group;

            // Element range [elem_start, elem_end) within the flattened group
            અચલ int64_t elem_start = cl_in_group * dst_cl_elems;
            અચલ int64_t elem_end   = elem_start + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            અચલ int64_t r_first = elem_start / ne00;
            અચલ int64_t r_last  = (elem_end - 1) / ne00;

            માટે (int64_t r = r_first; r <= r_last; r++) {
                અચલ int64_t row_flat = group_idx * rows_per_group + r;
                જો (row_flat >= total_rows) {
                    તોડો;
                }

                // Column range within this row
                int64_t col_begin = (r == r_first) ? (elem_start - r * ne00) : 0;
                int64_t col_end   = (r == r_last) ? (elem_end - r * ne00) : ne00;
                જો (col_end > ne00) {
                    col_end = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                અચલ int64_t i01 = row_flat % ne01;
                અચલ int64_t tmp = row_flat / ne01;
                અચલ int64_t i02 = tmp % ne02;
                અચલ int64_t i03 = tmp / ne02;

                // Look up destination row index
                અચલ int64_t i12               = i03 % ne12;
                અચલ int64_t i11               = i02 % ne11;
                અચલ int64_t i10               = i01;
                અચલ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
                અચલ int64_t dst_row_index     = *(int64_t *) ((અક્ષર *) src1_data + index_byte_offset);

                જો (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                    પાછા -1;
                }

                અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અક્ષર *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                અક્ષર *        dst_row_base = (અક્ષર *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                અચલ bool full_cl = (col_begin % dst_cl_elems == 0) && (col_end - col_begin == dst_cl_elems);

                જો (dst->type == GGML_TYPE_F32) {
                    દશાંશ * dp = (દશાંશ *) dst_row_base;
                    જો (full_cl) {
                        copy_cache_aligned_f32(dp + col_begin, src_row + col_begin);
                    } નહીંતર {
                        માટે (int64_t i = col_begin; i < col_end; i++) {
                            dp[i] = src_row[i];
                        }
                    }
                } નહીંતર {
                    uint16_t * dp = (uint16_t *) dst_row_base;
                    જો (full_cl) {
                        copy_cache_aligned_f16(dp + col_begin, src_row + col_begin);
                    } નહીંતર {
                        માટે (int64_t i = col_begin; i < col_end; i++) {
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


    } નહીંતર {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        માટે (int64_t row_flat = thread_id; row_flat < total_rows; row_flat += અંક_threads) {
            અચલ int64_t i01 = row_flat % ne01;
            અચલ int64_t tmp = row_flat / ne01;
            અચલ int64_t i02 = tmp % ne02;
            અચલ int64_t i03 = tmp / ne02;

            // Look up destination row index
            અચલ int64_t i12               = i03 % ne12;
            અચલ int64_t i11               = i02 % ne11;
            અચલ int64_t i10               = i01;
            અચલ int64_t index_byte_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;
            અચલ int64_t dst_row_index     = *(int64_t *) ((અક્ષર *) src1_data + index_byte_offset);

            જો (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                પાછા -1;
            }

            અચલ દશાંશ * src_row      = (અચલ દશાંશ *) ((અક્ષર *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            અક્ષર *        dst_row_base = (અક્ષર *) dst_data + dst_row_index * nb1 + i02 * nb2 + i03 * nb3;

            જો (dst->type == GGML_TYPE_F32) {
                અસ્થિર દશાંશ * dst_row = (અસ્થિર દશાંશ *) dst_row_base;
                માટે (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_row + i, src_row[i]);
                }
            } નહીંતર {
                અસ્થિર uint16_t * dst_row = (અસ્થિર uint16_t *) dst_row_base;
                માટે (int64_t i = 0; i < ne00; i++) {
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
    પાછા 0;
}
