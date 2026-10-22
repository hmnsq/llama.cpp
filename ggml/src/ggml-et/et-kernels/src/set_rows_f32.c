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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_F32_ELEMS  16  // 64 / 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_F16_ELEMS  32  // 64 / 2

ਸਥਿਰ int64_t gcd64(int64_t a, int64_t b) {
    ਜਦੋਂ (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    ਵਾਪਸ a;
}

ਬਣਤਰ ggml_et_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 source data tensor
    ਬਣਤਰ ggml_tensor src1;  // I64 row indices tensor
    ਬਣਤਰ ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
ਸਥਿਰ ਖਾਲੀ copy_cache_aligned_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src) {
    __asm__ ਅਸਥਿਰ(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
ਸਥਿਰ ਖਾਲੀ copy_cache_aligned_f16(uint16_t * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਨਕਾਬ_ਆਰਜ਼ੀ;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    ਦਸ਼ਮਲਵ      ਵਿੱਥ_vec_storage[8];
    uint32_t * offsets = (uint32_t *) ਵਿੱਥ_vec_storage;
    ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [ਨਕਾਬ_ਆਰਜ਼ੀ] "=&r"(ਨਕਾਬ_ਆਰਜ਼ੀ)
        : [offsets] "r"(ਵਿੱਥ_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x  %[mask_temp]         \n\t" : : [ਨਕਾਬ_ਆਰਜ਼ੀ] "r"(ਨਕਾਬ_ਆਰਜ਼ੀ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t tensor_ਬਾਈਟਾਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        ਵਾਪਸ -1;  // Invalid source types
    }

    ਜੇ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ਵਾਪਸ -1;  // Unsupported destination type
    }

    ਦਸ਼ਮਲਵ *   src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    int64_t * src1_ਅੰਕੜੇ = (int64_t *) src1->ਅੰਕੜੇ;
    ਖਾਲੀ *    dst_ਅੰਕੜੇ  = dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

    ਅਚਲ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ਅਚਲ int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    ਅਚਲ int64_t ne02 = src0->ne[2];  // Source batch dimension
    ਅਚਲ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ਅਚਲ int64_t nb01 = src0->nb[1];
    ਅਚਲ int64_t nb02 = src0->nb[2];
    ਅਚਲ int64_t nb03 = src0->nb[3];

    ਅਚਲ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ਅਚਲ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ਅਚਲ int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    ਅਚਲ int64_t nb10 = src1->nb[0];
    ਅਚਲ int64_t nb11 = src1->nb[1];
    ਅਚਲ int64_t nb12 = src1->nb[2];

    ਅਚਲ int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    ਅਚਲ int64_t nb1 = dst->nb[1];
    ਅਚਲ int64_t nb2 = dst->nb[2];
    ਅਚਲ int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    ਜੇ (ne10 != ne01) {
        ਵਾਪਸ -1;  // Number of indices must match number of source rows
    }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    evict_region_past_l2(params->src0.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&params->src0));
    evict_region_past_l2(params->src1.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#ਜੇ_ਅੰਤ
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ਅਚਲ int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_ਸਤਰ_F16_ELEMS : CACHE_ਸਤਰ_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ਅਚਲ bool ਪੰਕਤੀ_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    ਜੇ (ਪੰਕਤੀ_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ਅਚਲ int64_t cls_ਪ੍ਰਤੀ_ਪੰਕਤੀ    = ne00 / dst_cl_elems;
        ਅਚਲ int64_t ਕੁੱਲ_cls      = ਕੁੱਲ_ਪੰਕਤੀਆਂ * cls_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ int64_t cls_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cls + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t my_ਸ਼ੁਰੂ       = thread_id * cls_ਪ੍ਰਤੀ_thread;
        int64_t       my_ਸਿਰਾ         = my_ਸ਼ੁਰੂ + cls_ਪ੍ਰਤੀ_thread;
        ਜੇ (my_ਸਿਰਾ > ਕੁੱਲ_cls) {
            my_ਸਿਰਾ = ਕੁੱਲ_cls;
        }
        ਜੇ (my_ਸ਼ੁਰੂ >= ਕੁੱਲ_cls) {
            ਵਾਪਸ 0;
        }

        ਲਈ (int64_t cl = my_ਸ਼ੁਰੂ; cl < my_ਸਿਰਾ; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ਅਚਲ int64_t ਪੰਕਤੀ_flat  = cl / cls_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ int64_t cl_ਵਿੱਚ_ਪੰਕਤੀ = cl % cls_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

            // Decompose flat row -> (i03, i02, i01)
            ਅਚਲ int64_t i01 = ਪੰਕਤੀ_flat % ne01;
            ਅਚਲ int64_t tmp = ਪੰਕਤੀ_flat / ne01;
            ਅਚਲ int64_t i02 = tmp % ne02;
            ਅਚਲ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ਅਚਲ int64_t i12               = i03 % ne12;
            ਅਚਲ int64_t i11               = i02 % ne11;
            ਅਚਲ int64_t i10               = i01;
            ਅਚਲ int64_t ਸੂਚਕਾਂਕ_byte_ਵਿੱਥ = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ਅਚਲ int64_t dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ     = *(int64_t *) ((ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਸੂਚਕਾਂਕ_byte_ਵਿੱਥ);

            ਜੇ (dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ < 0 || dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ >= ne_dst1) {
                ਵਾਪਸ -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ਅਚਲ int64_t elem_ਵਿੱਥ = cl_ਵਿੱਚ_ਪੰਕਤੀ * dst_cl_elems;
            ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr =
                (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_ਵਿੱਥ;

            // Destination pointer: scattered row base + cache-line offset
            ਅੱਖਰ * dst_ਪੰਕਤੀ_ਅਧਾਰ = (ਅੱਖਰ *) dst_ਅੰਕੜੇ + dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * nb1 + i02 * nb2 + i03 * nb3;

            ਜੇ (dst->type == GGML_TYPE_F32) {
                ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *) dst_ਪੰਕਤੀ_ਅਧਾਰ + elem_ਵਿੱਥ;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } ਨਹੀਂਤਾਂ {
                uint16_t * dst_ptr = (uint16_t *) dst_ਪੰਕਤੀ_ਅਧਾਰ + elem_ਵਿੱਥ;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (nb1 % CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ਅਚਲ int64_t g              = gcd64(ne00, dst_cl_elems);
        ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ = dst_cl_elems / g;  // lcm / ne00
        ਅਚਲ int64_t cls_ਪ੍ਰਤੀ_ਸਮੂਹ  = ne00 / g;          // lcm / dst_cl_elems

        ਅਚਲ int64_t ਕੁੱਲ_groups   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਅਚਲ int64_t ਕੁੱਲ_cls      = ਕੁੱਲ_groups * cls_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਅਚਲ int64_t cls_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cls + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t my_ਸ਼ੁਰੂ       = thread_id * cls_ਪ੍ਰਤੀ_thread;
        int64_t       my_ਸਿਰਾ         = my_ਸ਼ੁਰੂ + cls_ਪ੍ਰਤੀ_thread;
        ਜੇ (my_ਸਿਰਾ > ਕੁੱਲ_cls) {
            my_ਸਿਰਾ = ਕੁੱਲ_cls;
        }
        ਜੇ (my_ਸ਼ੁਰੂ >= ਕੁੱਲ_cls) {
            ਵਾਪਸ 0;
        }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#ਜੇ_ਅੰਤ


        ਲਈ (int64_t cl = my_ਸ਼ੁਰੂ; cl < my_ਸਿਰਾ; cl++) {
            ਅਚਲ int64_t ਸਮੂਹ_idx   = cl / cls_ਪ੍ਰਤੀ_ਸਮੂਹ;
            ਅਚਲ int64_t cl_ਵਿੱਚ_ਸਮੂਹ = cl % cls_ਪ੍ਰਤੀ_ਸਮੂਹ;

            // Element range [elem_start, elem_end) within the flattened group
            ਅਚਲ int64_t elem_ਸ਼ੁਰੂ = cl_ਵਿੱਚ_ਸਮੂਹ * dst_cl_elems;
            ਅਚਲ int64_t elem_ਸਿਰਾ   = elem_ਸ਼ੁਰੂ + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ਅਚਲ int64_t r_ਪਹਿਲਾ = elem_ਸ਼ੁਰੂ / ne00;
            ਅਚਲ int64_t r_ਆਖਰੀ  = (elem_ਸਿਰਾ - 1) / ne00;

            ਲਈ (int64_t r = r_ਪਹਿਲਾ; r <= r_ਆਖਰੀ; r++) {
                ਅਚਲ int64_t ਪੰਕਤੀ_flat = ਸਮੂਹ_idx * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ + r;
                ਜੇ (ਪੰਕਤੀ_flat >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
                    ਰੋਕੋ;
                }

                // Column range within this row
                int64_t ਖਾਨਾ_begin = (r == r_ਪਹਿਲਾ) ? (elem_ਸ਼ੁਰੂ - r * ne00) : 0;
                int64_t ਖਾਨਾ_ਸਿਰਾ   = (r == r_ਆਖਰੀ) ? (elem_ਸਿਰਾ - r * ne00) : ne00;
                ਜੇ (ਖਾਨਾ_ਸਿਰਾ > ne00) {
                    ਖਾਨਾ_ਸਿਰਾ = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ਅਚਲ int64_t i01 = ਪੰਕਤੀ_flat % ne01;
                ਅਚਲ int64_t tmp = ਪੰਕਤੀ_flat / ne01;
                ਅਚਲ int64_t i02 = tmp % ne02;
                ਅਚਲ int64_t i03 = tmp / ne02;

                // Look up destination row index
                ਅਚਲ int64_t i12               = i03 % ne12;
                ਅਚਲ int64_t i11               = i02 % ne11;
                ਅਚਲ int64_t i10               = i01;
                ਅਚਲ int64_t ਸੂਚਕਾਂਕ_byte_ਵਿੱਥ = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ਅਚਲ int64_t dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ     = *(int64_t *) ((ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਸੂਚਕਾਂਕ_byte_ਵਿੱਥ);

                ਜੇ (dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ < 0 || dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ >= ne_dst1) {
                    ਵਾਪਸ -1;
                }

                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);
                ਅੱਖਰ *        dst_ਪੰਕਤੀ_ਅਧਾਰ = (ਅੱਖਰ *) dst_ਅੰਕੜੇ + dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ਅਚਲ bool ਭਰਿਆ_cl = (ਖਾਨਾ_begin % dst_cl_elems == 0) && (ਖਾਨਾ_ਸਿਰਾ - ਖਾਨਾ_begin == dst_cl_elems);

                ਜੇ (dst->type == GGML_TYPE_F32) {
                    ਦਸ਼ਮਲਵ * dp = (ਦਸ਼ਮਲਵ *) dst_ਪੰਕਤੀ_ਅਧਾਰ;
                    ਜੇ (ਭਰਿਆ_cl) {
                        copy_cache_aligned_f32(dp + ਖਾਨਾ_begin, src_ਪੰਕਤੀ + ਖਾਨਾ_begin);
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (int64_t i = ਖਾਨਾ_begin; i < ਖਾਨਾ_ਸਿਰਾ; i++) {
                            dp[i] = src_ਪੰਕਤੀ[i];
                        }
                    }
                } ਨਹੀਂਤਾਂ {
                    uint16_t * dp = (uint16_t *) dst_ਪੰਕਤੀ_ਅਧਾਰ;
                    ਜੇ (ਭਰਿਆ_cl) {
                        copy_cache_aligned_f16(dp + ਖਾਨਾ_begin, src_ਪੰਕਤੀ + ਖਾਨਾ_begin);
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (int64_t i = ਖਾਨਾ_begin; i < ਖਾਨਾ_ਸਿਰਾ; i++) {
                            dp[i] = fp32_ਮੰਜ਼ਿਲ_fp16(src_ਪੰਕਤੀ[i]);
                        }
                    }
                }
            }
        }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#ਜੇ_ਅੰਤ


    } ਨਹੀਂਤਾਂ {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        ਲਈ (int64_t ਪੰਕਤੀ_flat = thread_id; ਪੰਕਤੀ_flat < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ_flat += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t i01 = ਪੰਕਤੀ_flat % ne01;
            ਅਚਲ int64_t tmp = ਪੰਕਤੀ_flat / ne01;
            ਅਚਲ int64_t i02 = tmp % ne02;
            ਅਚਲ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ਅਚਲ int64_t i12               = i03 % ne12;
            ਅਚਲ int64_t i11               = i02 % ne11;
            ਅਚਲ int64_t i10               = i01;
            ਅਚਲ int64_t ਸੂਚਕਾਂਕ_byte_ਵਿੱਥ = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ਅਚਲ int64_t dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ     = *(int64_t *) ((ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਸੂਚਕਾਂਕ_byte_ਵਿੱਥ);

            ਜੇ (dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ < 0 || dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ >= ne_dst1) {
                ਵਾਪਸ -1;
            }

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ      = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ਅੱਖਰ *        dst_ਪੰਕਤੀ_ਅਧਾਰ = (ਅੱਖਰ *) dst_ਅੰਕੜੇ + dst_ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * nb1 + i02 * nb2 + i03 * nb3;

            ਜੇ (dst->type == GGML_TYPE_F32) {
                ਅਸਥਿਰ ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) dst_ਪੰਕਤੀ_ਅਧਾਰ;
                ਲਈ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_ਪੰਕਤੀ + i, src_ਪੰਕਤੀ[i]);
                }
            } ਨਹੀਂਤਾਂ {
                ਅਸਥਿਰ uint16_t * dst_ਪੰਕਤੀ = (ਅਸਥਿਰ uint16_t *) dst_ਪੰਕਤੀ_ਅਧਾਰ;
                ਲਈ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_ਪੰਕਤੀ + i, fp32_ਮੰਜ਼ਿਲ_fp16(src_ਪੰਕਤੀ[i]));
                }
            }
        }
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#ਜੇ_ਅੰਤ
    ਵਾਪਸ 0;
}
