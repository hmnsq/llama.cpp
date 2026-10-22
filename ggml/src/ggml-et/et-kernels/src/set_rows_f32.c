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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>

#وصف_ڪريو CACHE_سٽ_سائيز_بائيٽون 64
#وصف_ڪريو CACHE_سٽ_F32_ELEMS  16  // 64 / 4
#وصف_ڪريو CACHE_سٽ_F16_ELEMS  32  // 64 / 2

ساكن int64_t gcd64(int64_t a, int64_t b) {
    طالما (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    أرجع a;
}

بنية ggml_et_سيٽ_صفون_params {
    بنية ggml_tensor src0;  // F32 source data tensor
    بنية ggml_tensor src1;  // I64 row indices tensor
    بنية ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
ساكن فراغ copy_cache_aligned_f32(عائم * dst, ثابت عائم * src) {
    __asm__ متطاير(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
ساكن فراغ copy_cache_aligned_f16(uint16_t * dst, ثابت عائم * src) {
    بنانشان ڊگهو نقاب_عارضي;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    عائم      وٿي_vec_storage[8];
    uint32_t * offsets = (uint32_t *) وٿي_vec_storage;
    لكل (صحيح j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ متطاير(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [نقاب_عارضي] "=&r"(نقاب_عارضي)
        : [offsets] "r"(وٿي_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    لكل (صحيح i = 0; i < 32; i += 8) {
        __asm__ متطاير(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ متطاير("mova.m.x  %[mask_temp]         \n\t" : : [نقاب_عارضي] "r"(نقاب_عارضي));
}

ساكن مضمن size_t tensor_بائيٽون(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحيح داخلا_نقطو(بنية ggml_et_سيٽ_صفون_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    بنية ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    بنية ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        أرجع -1;  // Invalid source types
    }

    إذا (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        أرجع -1;  // Unsupported destination type
    }

    عائم *   src0_معلومات = (عائم *) src0->معلومات;
    int64_t * src1_معلومات = (int64_t *) src1->معلومات;
    فراغ *    dst_معلومات  = dst->معلومات;

    إذا (!src0_معلومات || !src1_معلومات || !dst_معلومات) {
        أرجع -1;  // Null data pointer
    }

    ثابت int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ثابت int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    ثابت int64_t ne02 = src0->ne[2];  // Source batch dimension
    ثابت int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ثابت int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ثابت int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    ثابت int64_t nb10 = src1->nb[0];
    ثابت int64_t nb11 = src1->nb[1];
    ثابت int64_t nb12 = src1->nb[2];

    ثابت int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    ثابت int64_t nb1 = dst->nb[1];
    ثابت int64_t nb2 = dst->nb[2];
    ثابت int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    إذا (ne10 != ne01) {
        أرجع -1;  // Number of indices must match number of source rows
    }
#جيڪڏهن_وصف_ٿيل ET_UBERKERNEL
    evict_region_past_l2(params->src0.معلومات, tensor_بائيٽون(&params->src0));
    evict_region_past_l2(params->src1.معلومات, tensor_بائيٽون(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#جيڪڏهن_پورو
    ثابت int64_t ڪل_صفون = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ثابت int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_سٽ_F16_ELEMS : CACHE_سٽ_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ثابت bool صف_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    إذا (صف_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ثابت int64_t cls_في_صف    = ne00 / dst_cl_elems;
        ثابت int64_t ڪل_cls      = ڪل_صفون * cls_في_صف;
        ثابت int64_t cls_في_thread = (ڪل_cls + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t my_شروعات       = thread_id * cls_في_thread;
        int64_t       my_پڇاڙي         = my_شروعات + cls_في_thread;
        إذا (my_پڇاڙي > ڪل_cls) {
            my_پڇاڙي = ڪل_cls;
        }
        إذا (my_شروعات >= ڪل_cls) {
            أرجع 0;
        }

        لكل (int64_t cl = my_شروعات; cl < my_پڇاڙي; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ثابت int64_t صف_flat  = cl / cls_في_صف;
            ثابت int64_t cl_اندريون_صف = cl % cls_في_صف;

            // Decompose flat row -> (i03, i02, i01)
            ثابت int64_t i01 = صف_flat % ne01;
            ثابت int64_t tmp = صف_flat / ne01;
            ثابت int64_t i02 = tmp % ne02;
            ثابت int64_t i03 = tmp / ne02;

            // Look up destination row index
            ثابت int64_t i12               = i03 % ne12;
            ثابت int64_t i11               = i02 % ne11;
            ثابت int64_t i10               = i01;
            ثابت int64_t اشاريو_byte_وٿي = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ثابت int64_t dst_صف_اشاريو     = *(int64_t *) ((حرف *) src1_معلومات + اشاريو_byte_وٿي);

            إذا (dst_صف_اشاريو < 0 || dst_صف_اشاريو >= ne_dst1) {
                أرجع -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ثابت int64_t elem_وٿي = cl_اندريون_صف * dst_cl_elems;
            ثابت عائم * src_ptr =
                (ثابت عائم *) ((حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_وٿي;

            // Destination pointer: scattered row base + cache-line offset
            حرف * dst_صف_بنياد = (حرف *) dst_معلومات + dst_صف_اشاريو * nb1 + i02 * nb2 + i03 * nb3;

            إذا (dst->type == GGML_TYPE_F32) {
                عائم * dst_ptr = (عائم *) dst_صف_بنياد + elem_وٿي;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } وإلا {
                uint16_t * dst_ptr = (uint16_t *) dst_صف_بنياد + elem_وٿي;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } وإلا إذا (nb1 % CACHE_سٽ_سائيز_بائيٽون == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ثابت int64_t g              = gcd64(ne00, dst_cl_elems);
        ثابت int64_t صفون_في_گروهه = dst_cl_elems / g;  // lcm / ne00
        ثابت int64_t cls_في_گروهه  = ne00 / g;          // lcm / dst_cl_elems

        ثابت int64_t ڪل_groups   = (ڪل_صفون + صفون_في_گروهه - 1) / صفون_في_گروهه;
        ثابت int64_t ڪل_cls      = ڪل_groups * cls_في_گروهه;
        ثابت int64_t cls_في_thread = (ڪل_cls + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t my_شروعات       = thread_id * cls_في_thread;
        int64_t       my_پڇاڙي         = my_شروعات + cls_في_thread;
        إذا (my_پڇاڙي > ڪل_cls) {
            my_پڇاڙي = ڪل_cls;
        }
        إذا (my_شروعات >= ڪل_cls) {
            أرجع 0;
        }

#جيڪڏهن_وصف_ٿيل BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#جيڪڏهن_پورو


        لكل (int64_t cl = my_شروعات; cl < my_پڇاڙي; cl++) {
            ثابت int64_t گروهه_idx   = cl / cls_في_گروهه;
            ثابت int64_t cl_اندريون_گروهه = cl % cls_في_گروهه;

            // Element range [elem_start, elem_end) within the flattened group
            ثابت int64_t elem_شروعات = cl_اندريون_گروهه * dst_cl_elems;
            ثابت int64_t elem_پڇاڙي   = elem_شروعات + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ثابت int64_t r_پهريون = elem_شروعات / ne00;
            ثابت int64_t r_آخري  = (elem_پڇاڙي - 1) / ne00;

            لكل (int64_t r = r_پهريون; r <= r_آخري; r++) {
                ثابت int64_t صف_flat = گروهه_idx * صفون_في_گروهه + r;
                إذا (صف_flat >= ڪل_صفون) {
                    اكسر;
                }

                // Column range within this row
                int64_t کانو_begin = (r == r_پهريون) ? (elem_شروعات - r * ne00) : 0;
                int64_t کانو_پڇاڙي   = (r == r_آخري) ? (elem_پڇاڙي - r * ne00) : ne00;
                إذا (کانو_پڇاڙي > ne00) {
                    کانو_پڇاڙي = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ثابت int64_t i01 = صف_flat % ne01;
                ثابت int64_t tmp = صف_flat / ne01;
                ثابت int64_t i02 = tmp % ne02;
                ثابت int64_t i03 = tmp / ne02;

                // Look up destination row index
                ثابت int64_t i12               = i03 % ne12;
                ثابت int64_t i11               = i02 % ne11;
                ثابت int64_t i10               = i01;
                ثابت int64_t اشاريو_byte_وٿي = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ثابت int64_t dst_صف_اشاريو     = *(int64_t *) ((حرف *) src1_معلومات + اشاريو_byte_وٿي);

                إذا (dst_صف_اشاريو < 0 || dst_صف_اشاريو >= ne_dst1) {
                    أرجع -1;
                }

                ثابت عائم * src_صف = (ثابت عائم *) ((حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);
                حرف *        dst_صف_بنياد = (حرف *) dst_معلومات + dst_صف_اشاريو * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ثابت bool ڀريل_cl = (کانو_begin % dst_cl_elems == 0) && (کانو_پڇاڙي - کانو_begin == dst_cl_elems);

                إذا (dst->type == GGML_TYPE_F32) {
                    عائم * dp = (عائم *) dst_صف_بنياد;
                    إذا (ڀريل_cl) {
                        copy_cache_aligned_f32(dp + کانو_begin, src_صف + کانو_begin);
                    } وإلا {
                        لكل (int64_t i = کانو_begin; i < کانو_پڇاڙي; i++) {
                            dp[i] = src_صف[i];
                        }
                    }
                } وإلا {
                    uint16_t * dp = (uint16_t *) dst_صف_بنياد;
                    إذا (ڀريل_cl) {
                        copy_cache_aligned_f16(dp + کانو_begin, src_صف + کانو_begin);
                    } وإلا {
                        لكل (int64_t i = کانو_begin; i < کانو_پڇاڙي; i++) {
                            dp[i] = fp32_منزل_fp16(src_صف[i]);
                        }
                    }
                }
            }
        }

#جيڪڏهن_وصف_ٿيل BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#جيڪڏهن_پورو


    } وإلا {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        لكل (int64_t صف_flat = thread_id; صف_flat < ڪل_صفون; صف_flat += انگ_ڌاڳا) {
            ثابت int64_t i01 = صف_flat % ne01;
            ثابت int64_t tmp = صف_flat / ne01;
            ثابت int64_t i02 = tmp % ne02;
            ثابت int64_t i03 = tmp / ne02;

            // Look up destination row index
            ثابت int64_t i12               = i03 % ne12;
            ثابت int64_t i11               = i02 % ne11;
            ثابت int64_t i10               = i01;
            ثابت int64_t اشاريو_byte_وٿي = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ثابت int64_t dst_صف_اشاريو     = *(int64_t *) ((حرف *) src1_معلومات + اشاريو_byte_وٿي);

            إذا (dst_صف_اشاريو < 0 || dst_صف_اشاريو >= ne_dst1) {
                أرجع -1;
            }

            ثابت عائم * src_صف      = (ثابت عائم *) ((حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);
            حرف *        dst_صف_بنياد = (حرف *) dst_معلومات + dst_صف_اشاريو * nb1 + i02 * nb2 + i03 * nb3;

            إذا (dst->type == GGML_TYPE_F32) {
                متطاير عائم * dst_صف = (متطاير عائم *) dst_صف_بنياد;
                لكل (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_صف + i, src_صف[i]);
                }
            } وإلا {
                متطاير uint16_t * dst_صف = (متطاير uint16_t *) dst_صف_بنياد;
                لكل (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_صف + i, fp32_منزل_fp16(src_صف[i]));
                }
            }
        }
    }

#جيڪڏهن_وصف_ٿيل BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#جيڪڏهن_پورو
    أرجع 0;
}
