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

#परिभाषय CACHE_पङ्क्तिः_आकार_अष्टकानि 64
#परिभाषय CACHE_पङ्क्तिः_F32_ELEMS  16  // 64 / 4
#परिभाषय CACHE_पङ्क्तिः_F16_ELEMS  32  // 64 / 2

स्थिर int64_t gcd64(int64_t a, int64_t b) {
    यावत् (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    प्रत्यागम a;
}

संरचना ggml_et_समुच्चयः_पङ्क्तिकाः_params {
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
    अचिह्नित दीर्घ आवरणम्_अस्थायी;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    भिन्न      विस्थापनम्_vec_storage[8];
    uint32_t * offsets = (uint32_t *) विस्थापनम्_vec_storage;
    प्रति (पूर्णाङ्क j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ चञ्चल(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [आवरणम्_अस्थायी] "=&r"(आवरणम्_अस्थायी)
        : [offsets] "r"(विस्थापनम्_vec_storage)
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

    __asm__ चञ्चल("mova.m.x  %[mask_temp]         \n\t" : : [आवरणम्_अस्थायी] "r"(आवरणम्_अस्थायी));
}

स्थिर अन्तःस्थ size_t tensor_अष्टकानि(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_समुच्चयः_पङ्क्तिकाः_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

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

    भिन्न *   src0_दत्त = (भिन्न *) src0->दत्त;
    int64_t * src1_दत्त = (int64_t *) src1->दत्त;
    शून्य *    dst_दत्त  = dst->दत्त;

    यदि (!src0_दत्त || !src1_दत्त || !dst_दत्त) {
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
#यदि_परिभाषितम् ET_UBERKERNEL
    evict_region_past_l2(params->src0.दत्त, tensor_अष्टकानि(&params->src0));
    evict_region_past_l2(params->src1.दत्त, tensor_अष्टकानि(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#यदि_अन्तः
    ध्रुव int64_t योग_पङ्क्तिकाः = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ध्रुव int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_पङ्क्तिः_F16_ELEMS : CACHE_पङ्क्तिः_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ध्रुव bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    यदि (row_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ध्रुव int64_t cls_प्रत्येकम्_row    = ne00 / dst_cl_elems;
        ध्रुव int64_t योग_cls      = योग_पङ्क्तिकाः * cls_प्रत्येकम्_row;
        ध्रुव int64_t cls_प्रत्येकम्_thread = (योग_cls + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
        ध्रुव int64_t my_start       = thread_id * cls_प्रत्येकम्_thread;
        int64_t       my_अन्तः         = my_start + cls_प्रत्येकम्_thread;
        यदि (my_अन्तः > योग_cls) {
            my_अन्तः = योग_cls;
        }
        यदि (my_start >= योग_cls) {
            प्रत्यागम 0;
        }

        प्रति (int64_t cl = my_start; cl < my_अन्तः; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ध्रुव int64_t row_flat  = cl / cls_प्रत्येकम्_row;
            ध्रुव int64_t cl_अन्तर्_row = cl % cls_प्रत्येकम्_row;

            // Decompose flat row -> (i03, i02, i01)
            ध्रुव int64_t i01 = row_flat % ne01;
            ध्रुव int64_t tmp = row_flat / ne01;
            ध्रुव int64_t i02 = tmp % ne02;
            ध्रुव int64_t i03 = tmp / ne02;

            // Look up destination row index
            ध्रुव int64_t i12               = i03 % ne12;
            ध्रुव int64_t i11               = i02 % ne11;
            ध्रुव int64_t i10               = i01;
            ध्रुव int64_t सूचकांक_byte_विस्थापनम् = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ध्रुव int64_t dst_row_सूचकांक     = *(int64_t *) ((वर्ण *) src1_दत्त + सूचकांक_byte_विस्थापनम्);

            यदि (dst_row_सूचकांक < 0 || dst_row_सूचकांक >= ne_dst1) {
                प्रत्यागम -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ध्रुव int64_t elem_विस्थापनम् = cl_अन्तर्_row * dst_cl_elems;
            ध्रुव भिन्न * src_ptr =
                (ध्रुव भिन्न *) ((वर्ण *) src0_दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_विस्थापनम्;

            // Destination pointer: scattered row base + cache-line offset
            वर्ण * dst_row_आधारः = (वर्ण *) dst_दत्त + dst_row_सूचकांक * nb1 + i02 * nb2 + i03 * nb3;

            यदि (dst->type == GGML_TYPE_F32) {
                भिन्न * dst_ptr = (भिन्न *) dst_row_आधारः + elem_विस्थापनम्;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } अन्यथा {
                uint16_t * dst_ptr = (uint16_t *) dst_row_आधारः + elem_विस्थापनम्;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } अन्यथा यदि (nb1 % CACHE_पङ्क्तिः_आकार_अष्टकानि == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ध्रुव int64_t g              = gcd64(ne00, dst_cl_elems);
        ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_गणः = dst_cl_elems / g;  // lcm / ne00
        ध्रुव int64_t cls_प्रत्येकम्_गणः  = ne00 / g;          // lcm / dst_cl_elems

        ध्रुव int64_t योग_groups   = (योग_पङ्क्तिकाः + पङ्क्तिकाः_प्रत्येकम्_गणः - 1) / पङ्क्तिकाः_प्रत्येकम्_गणः;
        ध्रुव int64_t योग_cls      = योग_groups * cls_प्रत्येकम्_गणः;
        ध्रुव int64_t cls_प्रत्येकम्_thread = (योग_cls + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
        ध्रुव int64_t my_start       = thread_id * cls_प्रत्येकम्_thread;
        int64_t       my_अन्तः         = my_start + cls_प्रत्येकम्_thread;
        यदि (my_अन्तः > योग_cls) {
            my_अन्तः = योग_cls;
        }
        यदि (my_start >= योग_cls) {
            प्रत्यागम 0;
        }

#यदि_परिभाषितम् BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#यदि_अन्तः


        प्रति (int64_t cl = my_start; cl < my_अन्तः; cl++) {
            ध्रुव int64_t गणः_idx   = cl / cls_प्रत्येकम्_गणः;
            ध्रुव int64_t cl_अन्तर्_गणः = cl % cls_प्रत्येकम्_गणः;

            // Element range [elem_start, elem_end) within the flattened group
            ध्रुव int64_t elem_start = cl_अन्तर्_गणः * dst_cl_elems;
            ध्रुव int64_t elem_अन्तः   = elem_start + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ध्रुव int64_t r_first = elem_start / ne00;
            ध्रुव int64_t r_चरमः  = (elem_अन्तः - 1) / ne00;

            प्रति (int64_t r = r_first; r <= r_चरमः; r++) {
                ध्रुव int64_t row_flat = गणः_idx * पङ्क्तिकाः_प्रत्येकम्_गणः + r;
                यदि (row_flat >= योग_पङ्क्तिकाः) {
                    भङ्ग;
                }

                // Column range within this row
                int64_t col_begin = (r == r_first) ? (elem_start - r * ne00) : 0;
                int64_t col_अन्तः   = (r == r_चरमः) ? (elem_अन्तः - r * ne00) : ne00;
                यदि (col_अन्तः > ne00) {
                    col_अन्तः = ne00;
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
                ध्रुव int64_t सूचकांक_byte_विस्थापनम् = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ध्रुव int64_t dst_row_सूचकांक     = *(int64_t *) ((वर्ण *) src1_दत्त + सूचकांक_byte_विस्थापनम्);

                यदि (dst_row_सूचकांक < 0 || dst_row_सूचकांक >= ne_dst1) {
                    प्रत्यागम -1;
                }

                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((वर्ण *) src0_दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03);
                वर्ण *        dst_row_आधारः = (वर्ण *) dst_दत्त + dst_row_सूचकांक * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ध्रुव bool पूर्णम्_cl = (col_begin % dst_cl_elems == 0) && (col_अन्तः - col_begin == dst_cl_elems);

                यदि (dst->type == GGML_TYPE_F32) {
                    भिन्न * dp = (भिन्न *) dst_row_आधारः;
                    यदि (पूर्णम्_cl) {
                        copy_cache_aligned_f32(dp + col_begin, src_row + col_begin);
                    } अन्यथा {
                        प्रति (int64_t i = col_begin; i < col_अन्तः; i++) {
                            dp[i] = src_row[i];
                        }
                    }
                } अन्यथा {
                    uint16_t * dp = (uint16_t *) dst_row_आधारः;
                    यदि (पूर्णम्_cl) {
                        copy_cache_aligned_f16(dp + col_begin, src_row + col_begin);
                    } अन्यथा {
                        प्रति (int64_t i = col_begin; i < col_अन्तः; i++) {
                            dp[i] = fp32_गन्तव्यम्_fp16(src_row[i]);
                        }
                    }
                }
            }
        }

#यदि_परिभाषितम् BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#यदि_अन्तः


    } अन्यथा {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        प्रति (int64_t row_flat = thread_id; row_flat < योग_पङ्क्तिकाः; row_flat += अङ्कः_तन्तवः) {
            ध्रुव int64_t i01 = row_flat % ne01;
            ध्रुव int64_t tmp = row_flat / ne01;
            ध्रुव int64_t i02 = tmp % ne02;
            ध्रुव int64_t i03 = tmp / ne02;

            // Look up destination row index
            ध्रुव int64_t i12               = i03 % ne12;
            ध्रुव int64_t i11               = i02 % ne11;
            ध्रुव int64_t i10               = i01;
            ध्रुव int64_t सूचकांक_byte_विस्थापनम् = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ध्रुव int64_t dst_row_सूचकांक     = *(int64_t *) ((वर्ण *) src1_दत्त + सूचकांक_byte_विस्थापनम्);

            यदि (dst_row_सूचकांक < 0 || dst_row_सूचकांक >= ne_dst1) {
                प्रत्यागम -1;
            }

            ध्रुव भिन्न * src_row      = (ध्रुव भिन्न *) ((वर्ण *) src0_दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03);
            वर्ण *        dst_row_आधारः = (वर्ण *) dst_दत्त + dst_row_सूचकांक * nb1 + i02 * nb2 + i03 * nb3;

            यदि (dst->type == GGML_TYPE_F32) {
                चञ्चल भिन्न * dst_row = (चञ्चल भिन्न *) dst_row_आधारः;
                प्रति (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_row + i, src_row[i]);
                }
            } अन्यथा {
                चञ्चल uint16_t * dst_row = (चञ्चल uint16_t *) dst_row_आधारः;
                प्रति (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_row + i, fp32_गन्तव्यम्_fp16(src_row[i]));
                }
            }
        }
    }

#यदि_परिभाषितम् BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#यदि_अन्तः
    प्रत्यागम 0;
}
