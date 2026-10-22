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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#నిర్వచించు CACHE_పంక్తి_కొలత_బైట్లు 64
#నిర్వచించు CACHE_పంక్తి_F32_ELEMS  16  // 64 / 4
#నిర్వచించు CACHE_పంక్తి_F16_ELEMS  32  // 64 / 2

స్థిర int64_t gcd64(int64_t a, int64_t b) {
    అయితే (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    తిరిగి a;
}

నిర్మాణం ggml_et_సమితి_అడ్డువరుసలు_params {
    నిర్మాణం ggml_tensor src0;  // F32 source data tensor
    నిర్మాణం ggml_tensor src1;  // I64 row indices tensor
    నిర్మాణం ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
స్థిర శూన్య copy_cache_aligned_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src) {
    __asm__ అస్థిర(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
స్థిర శూన్య copy_cache_aligned_f16(uint16_t * dst, స్థిరరాశి దశాంశం * src) {
    గుర్తులేని దీర్ఘ ముసుగు_తాత్కాలిక;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    దశాంశం      స్థానభ్రంశం_vec_storage[8];
    uint32_t * offsets = (uint32_t *) స్థానభ్రంశం_vec_storage;
    కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ అస్థిర(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [ముసుగు_తాత్కాలిక] "=&r"(ముసుగు_తాత్కాలిక)
        : [offsets] "r"(స్థానభ్రంశం_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    కొరకు (పూర్ణాంకం i = 0; i < 32; i += 8) {
        __asm__ అస్థిర(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ అస్థిర("mova.m.x  %[mask_temp]         \n\t" : : [ముసుగు_తాత్కాలిక] "r"(ముసుగు_తాత్కాలిక));
}

స్థిర అంతఃవరుస size_t tensor_బైట్లు(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_సమితి_అడ్డువరుసలు_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;  // Invalid pointer
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    నిర్మాణం ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        తిరిగి -1;  // Invalid source types
    }

    ఒకవేళ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        తిరిగి -1;  // Unsupported destination type
    }

    దశాంశం *   src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    int64_t * src1_దత్తాంశం = (int64_t *) src1->దత్తాంశం;
    శూన్య *    dst_దత్తాంశం  = dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

    స్థిరరాశి int64_t ne00 = src0->ne[0];  // Source columns (row width)
    స్థిరరాశి int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    స్థిరరాశి int64_t ne02 = src0->ne[2];  // Source batch dimension
    స్థిరరాశి int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    స్థిరరాశి int64_t nb01 = src0->nb[1];
    స్థిరరాశి int64_t nb02 = src0->nb[2];
    స్థిరరాశి int64_t nb03 = src0->nb[3];

    స్థిరరాశి int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    స్థిరరాశి int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    స్థిరరాశి int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    స్థిరరాశి int64_t nb10 = src1->nb[0];
    స్థిరరాశి int64_t nb11 = src1->nb[1];
    స్థిరరాశి int64_t nb12 = src1->nb[2];

    స్థిరరాశి int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    స్థిరరాశి int64_t nb1 = dst->nb[1];
    స్థిరరాశి int64_t nb2 = dst->nb[2];
    స్థిరరాశి int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    ఒకవేళ (ne10 != ne01) {
        తిరిగి -1;  // Number of indices must match number of source rows
    }
#నిర్వచితమైతే ET_UBERKERNEL
    evict_region_past_l2(params->src0.దత్తాంశం, tensor_బైట్లు(&params->src0));
    evict_region_past_l2(params->src1.దత్తాంశం, tensor_బైట్లు(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#అయితే_ముగింపు
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    స్థిరరాశి int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_పంక్తి_F16_ELEMS : CACHE_పంక్తి_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    స్థిరరాశి bool అడ్డువరుస_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    ఒకవేళ (అడ్డువరుస_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        స్థిరరాశి int64_t cls_చొప్పున_అడ్డువరుస    = ne00 / dst_cl_elems;
        స్థిరరాశి int64_t మొత్తం_cls      = మొత్తం_అడ్డువరుసలు * cls_చొప్పున_అడ్డువరుస;
        స్థిరరాశి int64_t cls_చొప్పున_thread = (మొత్తం_cls + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t my_ప్రారంభం       = thread_id * cls_చొప్పున_thread;
        int64_t       my_చివర         = my_ప్రారంభం + cls_చొప్పున_thread;
        ఒకవేళ (my_చివర > మొత్తం_cls) {
            my_చివర = మొత్తం_cls;
        }
        ఒకవేళ (my_ప్రారంభం >= మొత్తం_cls) {
            తిరిగి 0;
        }

        కొరకు (int64_t cl = my_ప్రారంభం; cl < my_చివర; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            స్థిరరాశి int64_t అడ్డువరుస_flat  = cl / cls_చొప్పున_అడ్డువరుస;
            స్థిరరాశి int64_t cl_లోపల_అడ్డువరుస = cl % cls_చొప్పున_అడ్డువరుస;

            // Decompose flat row -> (i03, i02, i01)
            స్థిరరాశి int64_t i01 = అడ్డువరుస_flat % ne01;
            స్థిరరాశి int64_t tmp = అడ్డువరుస_flat / ne01;
            స్థిరరాశి int64_t i02 = tmp % ne02;
            స్థిరరాశి int64_t i03 = tmp / ne02;

            // Look up destination row index
            స్థిరరాశి int64_t i12               = i03 % ne12;
            స్థిరరాశి int64_t i11               = i02 % ne11;
            స్థిరరాశి int64_t i10               = i01;
            స్థిరరాశి int64_t సూచీ_byte_స్థానభ్రంశం = i10 * nb10 + i11 * nb11 + i12 * nb12;
            స్థిరరాశి int64_t dst_అడ్డువరుస_సూచీ     = *(int64_t *) ((అక్షరం *) src1_దత్తాంశం + సూచీ_byte_స్థానభ్రంశం);

            ఒకవేళ (dst_అడ్డువరుస_సూచీ < 0 || dst_అడ్డువరుస_సూచీ >= ne_dst1) {
                తిరిగి -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            స్థిరరాశి int64_t elem_స్థానభ్రంశం = cl_లోపల_అడ్డువరుస * dst_cl_elems;
            స్థిరరాశి దశాంశం * src_ptr =
                (స్థిరరాశి దశాంశం *) ((అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_స్థానభ్రంశం;

            // Destination pointer: scattered row base + cache-line offset
            అక్షరం * dst_అడ్డువరుస_పునాది = (అక్షరం *) dst_దత్తాంశం + dst_అడ్డువరుస_సూచీ * nb1 + i02 * nb2 + i03 * nb3;

            ఒకవేళ (dst->type == GGML_TYPE_F32) {
                దశాంశం * dst_ptr = (దశాంశం *) dst_అడ్డువరుస_పునాది + elem_స్థానభ్రంశం;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } లేకపోతే {
                uint16_t * dst_ptr = (uint16_t *) dst_అడ్డువరుస_పునాది + elem_స్థానభ్రంశం;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } లేకపోతే ఒకవేళ (nb1 % CACHE_పంక్తి_కొలత_బైట్లు == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        స్థిరరాశి int64_t g              = gcd64(ne00, dst_cl_elems);
        స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_సమూహం = dst_cl_elems / g;  // lcm / ne00
        స్థిరరాశి int64_t cls_చొప్పున_సమూహం  = ne00 / g;          // lcm / dst_cl_elems

        స్థిరరాశి int64_t మొత్తం_groups   = (మొత్తం_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_సమూహం - 1) / అడ్డువరుసలు_చొప్పున_సమూహం;
        స్థిరరాశి int64_t మొత్తం_cls      = మొత్తం_groups * cls_చొప్పున_సమూహం;
        స్థిరరాశి int64_t cls_చొప్పున_thread = (మొత్తం_cls + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t my_ప్రారంభం       = thread_id * cls_చొప్పున_thread;
        int64_t       my_చివర         = my_ప్రారంభం + cls_చొప్పున_thread;
        ఒకవేళ (my_చివర > మొత్తం_cls) {
            my_చివర = మొత్తం_cls;
        }
        ఒకవేళ (my_ప్రారంభం >= మొత్తం_cls) {
            తిరిగి 0;
        }

#నిర్వచితమైతే BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#అయితే_ముగింపు


        కొరకు (int64_t cl = my_ప్రారంభం; cl < my_చివర; cl++) {
            స్థిరరాశి int64_t సమూహం_idx   = cl / cls_చొప్పున_సమూహం;
            స్థిరరాశి int64_t cl_లోపల_సమూహం = cl % cls_చొప్పున_సమూహం;

            // Element range [elem_start, elem_end) within the flattened group
            స్థిరరాశి int64_t elem_ప్రారంభం = cl_లోపల_సమూహం * dst_cl_elems;
            స్థిరరాశి int64_t elem_చివర   = elem_ప్రారంభం + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            స్థిరరాశి int64_t r_మొదటి = elem_ప్రారంభం / ne00;
            స్థిరరాశి int64_t r_చివరి  = (elem_చివర - 1) / ne00;

            కొరకు (int64_t r = r_మొదటి; r <= r_చివరి; r++) {
                స్థిరరాశి int64_t అడ్డువరుస_flat = సమూహం_idx * అడ్డువరుసలు_చొప్పున_సమూహం + r;
                ఒకవేళ (అడ్డువరుస_flat >= మొత్తం_అడ్డువరుసలు) {
                    ఆపు;
                }

                // Column range within this row
                int64_t నిలువు_begin = (r == r_మొదటి) ? (elem_ప్రారంభం - r * ne00) : 0;
                int64_t నిలువు_చివర   = (r == r_చివరి) ? (elem_చివర - r * ne00) : ne00;
                ఒకవేళ (నిలువు_చివర > ne00) {
                    నిలువు_చివర = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                స్థిరరాశి int64_t i01 = అడ్డువరుస_flat % ne01;
                స్థిరరాశి int64_t tmp = అడ్డువరుస_flat / ne01;
                స్థిరరాశి int64_t i02 = tmp % ne02;
                స్థిరరాశి int64_t i03 = tmp / ne02;

                // Look up destination row index
                స్థిరరాశి int64_t i12               = i03 % ne12;
                స్థిరరాశి int64_t i11               = i02 % ne11;
                స్థిరరాశి int64_t i10               = i01;
                స్థిరరాశి int64_t సూచీ_byte_స్థానభ్రంశం = i10 * nb10 + i11 * nb11 + i12 * nb12;
                స్థిరరాశి int64_t dst_అడ్డువరుస_సూచీ     = *(int64_t *) ((అక్షరం *) src1_దత్తాంశం + సూచీ_byte_స్థానభ్రంశం);

                ఒకవేళ (dst_అడ్డువరుస_సూచీ < 0 || dst_అడ్డువరుస_సూచీ >= ne_dst1) {
                    తిరిగి -1;
                }

                స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);
                అక్షరం *        dst_అడ్డువరుస_పునాది = (అక్షరం *) dst_దత్తాంశం + dst_అడ్డువరుస_సూచీ * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                స్థిరరాశి bool నిండిన_cl = (నిలువు_begin % dst_cl_elems == 0) && (నిలువు_చివర - నిలువు_begin == dst_cl_elems);

                ఒకవేళ (dst->type == GGML_TYPE_F32) {
                    దశాంశం * dp = (దశాంశం *) dst_అడ్డువరుస_పునాది;
                    ఒకవేళ (నిండిన_cl) {
                        copy_cache_aligned_f32(dp + నిలువు_begin, src_అడ్డువరుస + నిలువు_begin);
                    } లేకపోతే {
                        కొరకు (int64_t i = నిలువు_begin; i < నిలువు_చివర; i++) {
                            dp[i] = src_అడ్డువరుస[i];
                        }
                    }
                } లేకపోతే {
                    uint16_t * dp = (uint16_t *) dst_అడ్డువరుస_పునాది;
                    ఒకవేళ (నిండిన_cl) {
                        copy_cache_aligned_f16(dp + నిలువు_begin, src_అడ్డువరుస + నిలువు_begin);
                    } లేకపోతే {
                        కొరకు (int64_t i = నిలువు_begin; i < నిలువు_చివర; i++) {
                            dp[i] = fp32_గమ్యం_fp16(src_అడ్డువరుస[i]);
                        }
                    }
                }
            }
        }

#నిర్వచితమైతే BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#అయితే_ముగింపు


    } లేకపోతే {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        కొరకు (int64_t అడ్డువరుస_flat = thread_id; అడ్డువరుస_flat < మొత్తం_అడ్డువరుసలు; అడ్డువరుస_flat += అంకె_దారాలు) {
            స్థిరరాశి int64_t i01 = అడ్డువరుస_flat % ne01;
            స్థిరరాశి int64_t tmp = అడ్డువరుస_flat / ne01;
            స్థిరరాశి int64_t i02 = tmp % ne02;
            స్థిరరాశి int64_t i03 = tmp / ne02;

            // Look up destination row index
            స్థిరరాశి int64_t i12               = i03 % ne12;
            స్థిరరాశి int64_t i11               = i02 % ne11;
            స్థిరరాశి int64_t i10               = i01;
            స్థిరరాశి int64_t సూచీ_byte_స్థానభ్రంశం = i10 * nb10 + i11 * nb11 + i12 * nb12;
            స్థిరరాశి int64_t dst_అడ్డువరుస_సూచీ     = *(int64_t *) ((అక్షరం *) src1_దత్తాంశం + సూచీ_byte_స్థానభ్రంశం);

            ఒకవేళ (dst_అడ్డువరుస_సూచీ < 0 || dst_అడ్డువరుస_సూచీ >= ne_dst1) {
                తిరిగి -1;
            }

            స్థిరరాశి దశాంశం * src_అడ్డువరుస      = (స్థిరరాశి దశాంశం *) ((అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);
            అక్షరం *        dst_అడ్డువరుస_పునాది = (అక్షరం *) dst_దత్తాంశం + dst_అడ్డువరుస_సూచీ * nb1 + i02 * nb2 + i03 * nb3;

            ఒకవేళ (dst->type == GGML_TYPE_F32) {
                అస్థిర దశాంశం * dst_అడ్డువరుస = (అస్థిర దశాంశం *) dst_అడ్డువరుస_పునాది;
                కొరకు (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_అడ్డువరుస + i, src_అడ్డువరుస[i]);
                }
            } లేకపోతే {
                అస్థిర uint16_t * dst_అడ్డువరుస = (అస్థిర uint16_t *) dst_అడ్డువరుస_పునాది;
                కొరకు (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_అడ్డువరుస + i, fp32_గమ్యం_fp16(src_అడ్డువరుస[i]));
                }
            }
        }
    }

#నిర్వచితమైతే BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#అయితే_ముగింపు
    తిరిగి 0;
}
