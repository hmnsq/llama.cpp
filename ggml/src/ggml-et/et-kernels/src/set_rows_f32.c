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

#تعریف_کن CACHE_سطر_اندازه_بایتها 64
#تعریف_کن CACHE_سطر_F32_ELEMS  16  // 64 / 4
#تعریف_کن CACHE_سطر_F16_ELEMS  32  // 64 / 2

ساكن int64_t gcd64(int64_t a, int64_t b) {
    طالما (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    أرجع a;
}

بنية ggml_et_مجموعه_ردیفها_params {
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
    بیعلامت دراز ماسک_موقت;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    عائم      جابه_جایی_vec_storage[8];
    uint32_t * offsets = (uint32_t *) جابه_جایی_vec_storage;
    لكل (صحيح j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ متطاير(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [ماسک_موقت] "=&r"(ماسک_موقت)
        : [offsets] "r"(جابه_جایی_vec_storage)
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

    __asm__ متطاير("mova.m.x  %[mask_temp]         \n\t" : : [ماسک_موقت] "r"(ماسک_موقت));
}

ساكن مضمن size_t tensor_بایتها(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحيح مدخل_نقطه(بنية ggml_et_مجموعه_ردیفها_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

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

    عائم *   src0_داده = (عائم *) src0->داده;
    int64_t * src1_داده = (int64_t *) src1->داده;
    فراغ *    dst_داده  = dst->داده;

    إذا (!src0_داده || !src1_داده || !dst_داده) {
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
#اگر_تعریف_شده ET_UBERKERNEL
    evict_region_past_l2(params->src0.داده, tensor_بایتها(&params->src0));
    evict_region_past_l2(params->src1.داده, tensor_بایتها(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#پایان_اگر
    ثابت int64_t مجموع_ردیفها = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ثابت int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_سطر_F16_ELEMS : CACHE_سطر_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ثابت bool ردیف_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    إذا (ردیف_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ثابت int64_t cls_در_هر_ردیف    = ne00 / dst_cl_elems;
        ثابت int64_t مجموع_cls      = مجموع_ردیفها * cls_در_هر_ردیف;
        ثابت int64_t cls_در_هر_thread = (مجموع_cls + رقم_نخها - 1) / رقم_نخها;
        ثابت int64_t my_شروع       = thread_id * cls_در_هر_thread;
        int64_t       my_انتها         = my_شروع + cls_در_هر_thread;
        إذا (my_انتها > مجموع_cls) {
            my_انتها = مجموع_cls;
        }
        إذا (my_شروع >= مجموع_cls) {
            أرجع 0;
        }

        لكل (int64_t cl = my_شروع; cl < my_انتها; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ثابت int64_t ردیف_flat  = cl / cls_در_هر_ردیف;
            ثابت int64_t cl_درون_ردیف = cl % cls_در_هر_ردیف;

            // Decompose flat row -> (i03, i02, i01)
            ثابت int64_t i01 = ردیف_flat % ne01;
            ثابت int64_t tmp = ردیف_flat / ne01;
            ثابت int64_t i02 = tmp % ne02;
            ثابت int64_t i03 = tmp / ne02;

            // Look up destination row index
            ثابت int64_t i12               = i03 % ne12;
            ثابت int64_t i11               = i02 % ne11;
            ثابت int64_t i10               = i01;
            ثابت int64_t شاخص_byte_جابه_جایی = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ثابت int64_t dst_ردیف_شاخص     = *(int64_t *) ((حرف *) src1_داده + شاخص_byte_جابه_جایی);

            إذا (dst_ردیف_شاخص < 0 || dst_ردیف_شاخص >= ne_dst1) {
                أرجع -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ثابت int64_t elem_جابه_جایی = cl_درون_ردیف * dst_cl_elems;
            ثابت عائم * src_ptr =
                (ثابت عائم *) ((حرف *) src0_داده + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_جابه_جایی;

            // Destination pointer: scattered row base + cache-line offset
            حرف * dst_ردیف_پایه = (حرف *) dst_داده + dst_ردیف_شاخص * nb1 + i02 * nb2 + i03 * nb3;

            إذا (dst->type == GGML_TYPE_F32) {
                عائم * dst_ptr = (عائم *) dst_ردیف_پایه + elem_جابه_جایی;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } وإلا {
                uint16_t * dst_ptr = (uint16_t *) dst_ردیف_پایه + elem_جابه_جایی;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } وإلا إذا (nb1 % CACHE_سطر_اندازه_بایتها == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ثابت int64_t g              = gcd64(ne00, dst_cl_elems);
        ثابت int64_t ردیفها_در_هر_گروه = dst_cl_elems / g;  // lcm / ne00
        ثابت int64_t cls_در_هر_گروه  = ne00 / g;          // lcm / dst_cl_elems

        ثابت int64_t مجموع_groups   = (مجموع_ردیفها + ردیفها_در_هر_گروه - 1) / ردیفها_در_هر_گروه;
        ثابت int64_t مجموع_cls      = مجموع_groups * cls_در_هر_گروه;
        ثابت int64_t cls_در_هر_thread = (مجموع_cls + رقم_نخها - 1) / رقم_نخها;
        ثابت int64_t my_شروع       = thread_id * cls_در_هر_thread;
        int64_t       my_انتها         = my_شروع + cls_در_هر_thread;
        إذا (my_انتها > مجموع_cls) {
            my_انتها = مجموع_cls;
        }
        إذا (my_شروع >= مجموع_cls) {
            أرجع 0;
        }

#اگر_تعریف_شده BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#پایان_اگر


        لكل (int64_t cl = my_شروع; cl < my_انتها; cl++) {
            ثابت int64_t گروه_idx   = cl / cls_در_هر_گروه;
            ثابت int64_t cl_درون_گروه = cl % cls_در_هر_گروه;

            // Element range [elem_start, elem_end) within the flattened group
            ثابت int64_t elem_شروع = cl_درون_گروه * dst_cl_elems;
            ثابت int64_t elem_انتها   = elem_شروع + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ثابت int64_t r_اولی = elem_شروع / ne00;
            ثابت int64_t r_آخرین  = (elem_انتها - 1) / ne00;

            لكل (int64_t r = r_اولی; r <= r_آخرین; r++) {
                ثابت int64_t ردیف_flat = گروه_idx * ردیفها_در_هر_گروه + r;
                إذا (ردیف_flat >= مجموع_ردیفها) {
                    اكسر;
                }

                // Column range within this row
                int64_t ستون_begin = (r == r_اولی) ? (elem_شروع - r * ne00) : 0;
                int64_t ستون_انتها   = (r == r_آخرین) ? (elem_انتها - r * ne00) : ne00;
                إذا (ستون_انتها > ne00) {
                    ستون_انتها = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ثابت int64_t i01 = ردیف_flat % ne01;
                ثابت int64_t tmp = ردیف_flat / ne01;
                ثابت int64_t i02 = tmp % ne02;
                ثابت int64_t i03 = tmp / ne02;

                // Look up destination row index
                ثابت int64_t i12               = i03 % ne12;
                ثابت int64_t i11               = i02 % ne11;
                ثابت int64_t i10               = i01;
                ثابت int64_t شاخص_byte_جابه_جایی = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ثابت int64_t dst_ردیف_شاخص     = *(int64_t *) ((حرف *) src1_داده + شاخص_byte_جابه_جایی);

                إذا (dst_ردیف_شاخص < 0 || dst_ردیف_شاخص >= ne_dst1) {
                    أرجع -1;
                }

                ثابت عائم * src_ردیف = (ثابت عائم *) ((حرف *) src0_داده + i01 * nb01 + i02 * nb02 + i03 * nb03);
                حرف *        dst_ردیف_پایه = (حرف *) dst_داده + dst_ردیف_شاخص * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ثابت bool پر_cl = (ستون_begin % dst_cl_elems == 0) && (ستون_انتها - ستون_begin == dst_cl_elems);

                إذا (dst->type == GGML_TYPE_F32) {
                    عائم * dp = (عائم *) dst_ردیف_پایه;
                    إذا (پر_cl) {
                        copy_cache_aligned_f32(dp + ستون_begin, src_ردیف + ستون_begin);
                    } وإلا {
                        لكل (int64_t i = ستون_begin; i < ستون_انتها; i++) {
                            dp[i] = src_ردیف[i];
                        }
                    }
                } وإلا {
                    uint16_t * dp = (uint16_t *) dst_ردیف_پایه;
                    إذا (پر_cl) {
                        copy_cache_aligned_f16(dp + ستون_begin, src_ردیف + ستون_begin);
                    } وإلا {
                        لكل (int64_t i = ستون_begin; i < ستون_انتها; i++) {
                            dp[i] = fp32_مقصد_fp16(src_ردیف[i]);
                        }
                    }
                }
            }
        }

#اگر_تعریف_شده BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#پایان_اگر


    } وإلا {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        لكل (int64_t ردیف_flat = thread_id; ردیف_flat < مجموع_ردیفها; ردیف_flat += رقم_نخها) {
            ثابت int64_t i01 = ردیف_flat % ne01;
            ثابت int64_t tmp = ردیف_flat / ne01;
            ثابت int64_t i02 = tmp % ne02;
            ثابت int64_t i03 = tmp / ne02;

            // Look up destination row index
            ثابت int64_t i12               = i03 % ne12;
            ثابت int64_t i11               = i02 % ne11;
            ثابت int64_t i10               = i01;
            ثابت int64_t شاخص_byte_جابه_جایی = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ثابت int64_t dst_ردیف_شاخص     = *(int64_t *) ((حرف *) src1_داده + شاخص_byte_جابه_جایی);

            إذا (dst_ردیف_شاخص < 0 || dst_ردیف_شاخص >= ne_dst1) {
                أرجع -1;
            }

            ثابت عائم * src_ردیف      = (ثابت عائم *) ((حرف *) src0_داده + i01 * nb01 + i02 * nb02 + i03 * nb03);
            حرف *        dst_ردیف_پایه = (حرف *) dst_داده + dst_ردیف_شاخص * nb1 + i02 * nb2 + i03 * nb3;

            إذا (dst->type == GGML_TYPE_F32) {
                متطاير عائم * dst_ردیف = (متطاير عائم *) dst_ردیف_پایه;
                لكل (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_ردیف + i, src_ردیف[i]);
                }
            } وإلا {
                متطاير uint16_t * dst_ردیف = (متطاير uint16_t *) dst_ردیف_پایه;
                لكل (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_ردیف + i, fp32_مقصد_fp16(src_ردیف[i]));
                }
            }
        }
    }

#اگر_تعریف_شده BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#پایان_اگر
    أرجع 0;
}
