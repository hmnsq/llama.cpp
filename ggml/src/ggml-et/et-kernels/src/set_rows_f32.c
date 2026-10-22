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

#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು 64
#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_F32_ELEMS  16  // 64 / 4
#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_F16_ELEMS  32  // 64 / 2

ಸ್ಥಿರ int64_t gcd64(int64_t a, int64_t b) {
    ಆಗ (b) {
        int64_t t = b;
        b         = a % b;
        a         = t;
    }
    ಹಿಂತಿರುಗಿ a;
}

ರಚನೆ ggml_et_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params {
    ರಚನೆ ggml_tensor src0;  // F32 source data tensor
    ರಚನೆ ggml_tensor src1;  // I64 row indices tensor
    ರಚನೆ ggml_tensor dst;   // F32/F16 destination tensor
};

// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
ಸ್ಥಿರ ಶೂನ್ಯ copy_cache_aligned_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src) {
    __asm__ ಅಸ್ಥಿರ(
        "flq2 f0, 0(%[src]) \n\t"   // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"  // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"   // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"  // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
ಸ್ಥಿರ ಶೂನ್ಯ copy_cache_aligned_f16(uint16_t * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಮುಖವಾಡ_ತಾತ್ಕಾಲಿಕ;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    ದಶಮಾಂಶ      ಸ್ಥಾನಾಂತರ_vec_storage[8];
    uint32_t * offsets = (uint32_t *) ಸ್ಥಾನಾಂತರ_vec_storage;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [ಮುಖವಾಡ_ತಾತ್ಕಾಲಿಕ] "=&r"(ಮುಖವಾಡ_ತಾತ್ಕಾಲಿಕ)
        : [offsets] "r"(ಸ್ಥಾನಾಂತರ_vec_storage)
        : "f1");

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x  %[mask_temp]         \n\t" : : [ಮುಖವಾಡ_ತಾತ್ಕಾಲಿಕ] "r"(ಮುಖವಾಡ_ತಾತ್ಕಾಲಿಕ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t tensor_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;  // Source data tensor (F32)
    ರಚನೆ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I64)
    ರಚನೆ ggml_tensor * dst  = &params->dst;   // Destination tensor (F32/F16)

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid source types
    }

    ಆದರೆ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported destination type
    }

    ದಶಮಾಂಶ *   src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    int64_t * src1_ದತ್ತಾಂಶ = (int64_t *) src1->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ *    dst_ದತ್ತಾಂಶ  = dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];  // Source batch dimension
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ಸ್ಥಿರಾಂಕ int64_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ಸ್ಥಿರಾಂಕ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    ಸ್ಥಿರಾಂಕ int64_t nb10 = src1->nb[0];
    ಸ್ಥಿರಾಂಕ int64_t nb11 = src1->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb12 = src1->nb[2];

    ಸ್ಥಿರಾಂಕ int64_t ne_dst1 = dst->ne[1];  // Number of rows in destination (for bounds checking)

    ಸ್ಥಿರಾಂಕ int64_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    ಆದರೆ (ne10 != ne01) {
        ಹಿಂತಿರುಗಿ -1;  // Number of indices must match number of source rows
    }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    evict_region_past_l2(params->src0.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&params->src0));
    evict_region_past_l2(params->src1.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&params->src1));
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne01 * ne02 * ne03;

    // Determine cache-line element count based on destination type
    ಸ್ಥಿರಾಂಕ int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_ಸಾಲು_F16_ELEMS : CACHE_ಸಾಲು_F32_ELEMS;

    // Check if rows are cache-line aligned in the destination
    ಸ್ಥಿರಾಂಕ bool ಅಡ್ಡಸಾಲು_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    ಆದರೆ (ಅಡ್ಡಸಾಲು_cache_aligned) {
        // Cache-aligned path: distribute dst cache lines across threads
        // Each thread owns complete cache lines -> no coherence conflicts
        ಸ್ಥಿರಾಂಕ int64_t cls_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು    = ne00 / dst_cl_elems;
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cls      = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು * cls_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ int64_t cls_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cls + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t my_ಆರಂಭ       = thread_id * cls_ಪ್ರತಿ_thread;
        int64_t       my_ಕೊನೆ         = my_ಆರಂಭ + cls_ಪ್ರತಿ_thread;
        ಆದರೆ (my_ಕೊನೆ > ಒಟ್ಟು_cls) {
            my_ಕೊನೆ = ಒಟ್ಟು_cls;
        }
        ಆದರೆ (my_ಆರಂಭ >= ಒಟ್ಟು_cls) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಗಾಗಿ (int64_t cl = my_ಆರಂಭ; cl < my_ಕೊನೆ; cl++) {
            // Map flat cache-line index -> (row, offset within row)
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_flat  = cl / cls_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ int64_t cl_ಒಳ_ಅಡ್ಡಸಾಲು = cl % cls_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

            // Decompose flat row -> (i03, i02, i01)
            ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಡ್ಡಸಾಲು_flat % ne01;
            ಸ್ಥಿರಾಂಕ int64_t tmp = ಅಡ್ಡಸಾಲು_flat / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i02 = tmp % ne02;
            ಸ್ಥಿರಾಂಕ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ಸ್ಥಿರಾಂಕ int64_t i12               = i03 % ne12;
            ಸ್ಥಿರಾಂಕ int64_t i11               = i02 % ne11;
            ಸ್ಥಿರಾಂಕ int64_t i10               = i01;
            ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ_byte_ಸ್ಥಾನಾಂತರ = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ     = *(int64_t *) ((ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸೂಚ್ಯಂಕ_byte_ಸ್ಥಾನಾಂತರ);

            ಆದರೆ (dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ < 0 || dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ >= ne_dst1) {
                ಹಿಂತಿರುಗಿ -1;
            }

            // Source pointer: row base + cache-line offset (always F32 source)
            ಸ್ಥಿರಾಂಕ int64_t elem_ಸ್ಥಾನಾಂತರ = cl_ಒಳ_ಅಡ್ಡಸಾಲು * dst_cl_elems;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr =
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03) + elem_ಸ್ಥಾನಾಂತರ;

            // Destination pointer: scattered row base + cache-line offset
            ಅಕ್ಷರ * dst_ಅಡ್ಡಸಾಲು_ತಳ = (ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * nb1 + i02 * nb2 + i03 * nb3;

            ಆದರೆ (dst->type == GGML_TYPE_F32) {
                ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *) dst_ಅಡ್ಡಸಾಲು_ತಳ + elem_ಸ್ಥಾನಾಂತರ;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } ಇಲ್ಲದಿದ್ದರೆ {
                uint16_t * dst_ptr = (uint16_t *) dst_ಅಡ್ಡಸಾಲು_ತಳ + elem_ಸ್ಥಾನಾಂತರ;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (nb1 % CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು == 0) {
        // LCM-aligned path: destination row stride is cache-line-aligned, so
        // scattered rows never share a cache line even though ne00 doesn't
        // fill complete cache lines.  Group rows via lcm(ne00, dst_cl_elems)
        // and distribute cache lines across threads — each thread exclusively
        // owns its cache lines, so normal stores are safe (no atomics needed).
        ಸ್ಥಿರಾಂಕ int64_t g              = gcd64(ne00, dst_cl_elems);
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು = dst_cl_elems / g;  // lcm / ne00
        ಸ್ಥಿರಾಂಕ int64_t cls_ಪ್ರತಿ_ಗುಂಪು  = ne00 / g;          // lcm / dst_cl_elems

        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_groups   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cls      = ಒಟ್ಟು_groups * cls_ಪ್ರತಿ_ಗುಂಪು;
        ಸ್ಥಿರಾಂಕ int64_t cls_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cls + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t my_ಆರಂಭ       = thread_id * cls_ಪ್ರತಿ_thread;
        int64_t       my_ಕೊನೆ         = my_ಆರಂಭ + cls_ಪ್ರತಿ_thread;
        ಆದರೆ (my_ಕೊನೆ > ಒಟ್ಟು_cls) {
            my_ಕೊನೆ = ಒಟ್ಟು_cls;
        }
        ಆದರೆ (my_ಆರಂಭ >= ಒಟ್ಟು_cls) {
            ಹಿಂತಿರುಗಿ 0;
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#ಆದರೆ_ಅಂತ್ಯ


        ಗಾಗಿ (int64_t cl = my_ಆರಂಭ; cl < my_ಕೊನೆ; cl++) {
            ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು_idx   = cl / cls_ಪ್ರತಿ_ಗುಂಪು;
            ಸ್ಥಿರಾಂಕ int64_t cl_ಒಳ_ಗುಂಪು = cl % cls_ಪ್ರತಿ_ಗುಂಪು;

            // Element range [elem_start, elem_end) within the flattened group
            ಸ್ಥಿರಾಂಕ int64_t elem_ಆರಂಭ = cl_ಒಳ_ಗುಂಪು * dst_cl_elems;
            ಸ್ಥಿರಾಂಕ int64_t elem_ಕೊನೆ   = elem_ಆರಂಭ + dst_cl_elems;

            // Which row(s) inside this group does the cache line touch?
            ಸ್ಥಿರಾಂಕ int64_t r_ಮೊದಲ = elem_ಆರಂಭ / ne00;
            ಸ್ಥಿರಾಂಕ int64_t r_ಕೊನೆಯ  = (elem_ಕೊನೆ - 1) / ne00;

            ಗಾಗಿ (int64_t r = r_ಮೊದಲ; r <= r_ಕೊನೆಯ; r++) {
                ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_flat = ಗುಂಪು_idx * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು + r;
                ಆದರೆ (ಅಡ್ಡಸಾಲು_flat >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
                    ನಿಲ್ಲಿಸು;
                }

                // Column range within this row
                int64_t ಲಂಬಸಾಲು_begin = (r == r_ಮೊದಲ) ? (elem_ಆರಂಭ - r * ne00) : 0;
                int64_t ಲಂಬಸಾಲು_ಕೊನೆ   = (r == r_ಕೊನೆಯ) ? (elem_ಕೊನೆ - r * ne00) : ne00;
                ಆದರೆ (ಲಂಬಸಾಲು_ಕೊನೆ > ne00) {
                    ಲಂಬಸಾಲು_ಕೊನೆ = ne00;
                }

                // Decompose flat row -> (i03, i02, i01)
                ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಡ್ಡಸಾಲು_flat % ne01;
                ಸ್ಥಿರಾಂಕ int64_t tmp = ಅಡ್ಡಸಾಲು_flat / ne01;
                ಸ್ಥಿರಾಂಕ int64_t i02 = tmp % ne02;
                ಸ್ಥಿರಾಂಕ int64_t i03 = tmp / ne02;

                // Look up destination row index
                ಸ್ಥಿರಾಂಕ int64_t i12               = i03 % ne12;
                ಸ್ಥಿರಾಂಕ int64_t i11               = i02 % ne11;
                ಸ್ಥಿರಾಂಕ int64_t i10               = i01;
                ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ_byte_ಸ್ಥಾನಾಂತರ = i10 * nb10 + i11 * nb11 + i12 * nb12;
                ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ     = *(int64_t *) ((ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸೂಚ್ಯಂಕ_byte_ಸ್ಥಾನಾಂತರ);

                ಆದರೆ (dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ < 0 || dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ >= ne_dst1) {
                    ಹಿಂತಿರುಗಿ -1;
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
                ಅಕ್ಷರ *        dst_ಅಡ್ಡಸಾಲು_ತಳ = (ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * nb1 + i02 * nb2 + i03 * nb3;

                // nb1 is cache-line-aligned, so dst_row_base is too.
                // Use aligned copy when the column range fills a complete
                // cache line at a cache-line-aligned offset within the row.
                ಸ್ಥಿರಾಂಕ bool ತುಂಬಿದ_cl = (ಲಂಬಸಾಲು_begin % dst_cl_elems == 0) && (ಲಂಬಸಾಲು_ಕೊನೆ - ಲಂಬಸಾಲು_begin == dst_cl_elems);

                ಆದರೆ (dst->type == GGML_TYPE_F32) {
                    ದಶಮಾಂಶ * dp = (ದಶಮಾಂಶ *) dst_ಅಡ್ಡಸಾಲು_ತಳ;
                    ಆದರೆ (ತುಂಬಿದ_cl) {
                        copy_cache_aligned_f32(dp + ಲಂಬಸಾಲು_begin, src_ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು_begin);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (int64_t i = ಲಂಬಸಾಲು_begin; i < ಲಂಬಸಾಲು_ಕೊನೆ; i++) {
                            dp[i] = src_ಅಡ್ಡಸಾಲು[i];
                        }
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    uint16_t * dp = (uint16_t *) dst_ಅಡ್ಡಸಾಲು_ತಳ;
                    ಆದರೆ (ತುಂಬಿದ_cl) {
                        copy_cache_aligned_f16(dp + ಲಂಬಸಾಲು_begin, src_ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು_begin);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (int64_t i = ಲಂಬಸಾಲು_begin; i < ಲಂಬಸಾಲು_ಕೊನೆ; i++) {
                            dp[i] = fp32_ಗಮ್ಯ_fp16(src_ಅಡ್ಡಸಾಲು[i]);
                        }
                    }
                }
            }
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#ಆದರೆ_ಅಂತ್ಯ


    } ಇಲ್ಲದಿದ್ದರೆ {
        // Fallback: nb1 not cache-line-aligned, so scattered destination rows
        // may share a cache line.  Use atomic global stores to bypass L1D.
        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು_flat = thread_id; ಅಡ್ಡಸಾಲು_flat < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು_flat += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಡ್ಡಸಾಲು_flat % ne01;
            ಸ್ಥಿರಾಂಕ int64_t tmp = ಅಡ್ಡಸಾಲು_flat / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i02 = tmp % ne02;
            ಸ್ಥಿರಾಂಕ int64_t i03 = tmp / ne02;

            // Look up destination row index
            ಸ್ಥಿರಾಂಕ int64_t i12               = i03 % ne12;
            ಸ್ಥಿರಾಂಕ int64_t i11               = i02 % ne11;
            ಸ್ಥಿರಾಂಕ int64_t i10               = i01;
            ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ_byte_ಸ್ಥಾನಾಂತರ = i10 * nb10 + i11 * nb11 + i12 * nb12;
            ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ     = *(int64_t *) ((ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸೂಚ್ಯಂಕ_byte_ಸ್ಥಾನಾಂತರ);

            ಆದರೆ (dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ < 0 || dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ >= ne_dst1) {
                ಹಿಂತಿರುಗಿ -1;
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು      = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ಅಕ್ಷರ *        dst_ಅಡ್ಡಸಾಲು_ತಳ = (ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + dst_ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * nb1 + i02 * nb2 + i03 * nb3;

            ಆದರೆ (dst->type == GGML_TYPE_F32) {
                ಅಸ್ಥಿರ ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) dst_ಅಡ್ಡಸಾಲು_ತಳ;
                ಗಾಗಿ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f32(dst_ಅಡ್ಡಸಾಲು + i, src_ಅಡ್ಡಸಾಲು[i]);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಅಸ್ಥಿರ uint16_t * dst_ಅಡ್ಡಸಾಲು = (ಅಸ್ಥಿರ uint16_t *) dst_ಅಡ್ಡಸಾಲು_ತಳ;
                ಗಾಗಿ (int64_t i = 0; i < ne00; i++) {
                    atomic_store_f16(dst_ಅಡ್ಡಸಾಲು + i, fp32_ಗಮ್ಯ_fp16(src_ಅಡ್ಡಸಾಲು[i]));
                }
            }
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ BUILD_FOR_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(src1_data, tensor_bytes(src1));
    // // et_barrier(ET_BARRIER_GLOBAL);
    // FENCE;
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ 0;
}
