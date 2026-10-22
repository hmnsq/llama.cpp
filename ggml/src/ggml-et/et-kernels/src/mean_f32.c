//******************************************************************************
// MEAN F32 Kernel
// Row-wise mean reduction: dst[0, i1, i2, i3] = mean(src0[0..ne00-1, i1, i2, i3])
//
// Modes:
//   - total_rows >= shire_threads: row-parallel, each thread handles whole rows.
//   - total_rows <  shire_threads: intra-row reduction within a shire. Threads
//     within a shire cooperate via shire-local L2 SCP slots. All shires
//     duplicate the work because L2 SCP is per-shire (no cross-shire coherency).
//
// ne00 may be any positive size and rows may have any 4-byte alignment. We
// take the 8-wide vector path only when the row pointer is 32B-aligned and
// fall back to scalar for the leftover tail (or for the entire row when the
// row start is not 32B-aligned).
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_ಸರಾಸರಿ_params {
    ರಚನೆ ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    ರಚನೆ ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ partial_ಮೊತ್ತ_ತುಂಡು(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ತಳ, int32_t i_lo, int32_t i_hi) {
    ಆದರೆ (i_lo >= i_hi) {
        ಹಿಂತಿರುಗಿ 0.0f;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * p   = ತಳ + i_lo;
    int32_t       n   = i_hi - i_lo;
    ದಶಮಾಂಶ         acc = 0.0f;
    int32_t       i   = 0;

    ಆದರೆ (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        ದಶಮಾಂಶ zero = 0.0f;
        __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ಗಾಗಿ (; i + 8 <= n; i += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & p[i])
                : "f10", "f11");
        }

        ದಶಮಾಂಶ vec_ಮೊತ್ತ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_ಮೊತ್ತ)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_ಮೊತ್ತ;
    }

    ಗಾಗಿ (; i < n; i++) {
        acc += p[i];
    }
    ಹಿಂತಿರುಗಿ acc;
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ಸರಾಸರಿ_params * params, ಶೂನ್ಯ * env) {
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
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ size_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ size_t nb3 = dst->nb[3];

    ಆದರೆ (ne00 <= 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ int32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು    = (int32_t) (ne01 * ne02 * ne03);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     shire_ಎಳೆಗಳು = SOC_MINIONS_ಪ್ರತಿ_SHIRE * ಅಂಕೆ_HARTS_ಪ್ರತಿ_MINION;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   inv_ne00      = et_fdiv(1.0f, (ದಶಮಾಂಶ) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು >= shire_ಎಳೆಗಳು) {
        ಗಾಗಿ (int64_t ir = thread_id; ir < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ir += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne02 * ne01);
            ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ದಶಮಾಂಶ *       dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i01 * nb1 + i02 * nb2 + i03 * nb3);

            ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = partial_ಮೊತ್ತ_ತುಂಡು(src_ಅಡ್ಡಸಾಲು, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, ಅಡ್ಡಸಾಲು_ಮೊತ್ತ * inv_ne00);
        }
        // Shire co-work
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಪೂರ್ಣಾಂಕ shire_tid       = thread_id % shire_ಎಳೆಗಳು;
        ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = shire_ಎಳೆಗಳು / ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        ಪೂರ್ಣಾಂಕ my_ಅಡ್ಡಸಾಲು          = shire_tid / ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಪೂರ್ಣಾಂಕ local_tid       = shire_tid % ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಪೂರ್ಣಾಂಕ ಗುಂಪು_ತಳ      = my_ಅಡ್ಡಸಾಲು * ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (my_ಅಡ್ಡಸಾಲು >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ಹಿಂತಿರುಗಿ 0;
        }

        int64_t i1 = my_ಅಡ್ಡಸಾಲು % ne01;
        int64_t i2 = (my_ಅಡ್ಡಸಾಲು / ne01) % ne02;
        int64_t i3 = my_ಅಡ್ಡಸಾಲು / (ne01 * ne02);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ದಶಮಾಂಶ *       dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು - 1) / ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        chunk         = (chunk + 7) & ~7;
        ಆದರೆ (chunk < 8) {
            chunk = 8;
        }

        int32_t my_ಆರಂಭ = local_tid * chunk;
        int32_t my_ಕೊನೆ   = my_ಆರಂಭ + chunk;
        ಆದರೆ (my_ಕೊನೆ > (int32_t) ne00) {
            my_ಕೊನೆ = (int32_t) ne00;
        }
        ಆದರೆ (my_ಆರಂಭ > (int32_t) ne00) {
            my_ಆರಂಭ = my_ಕೊನೆ = (int32_t) ne00;
        }

        ಪೂರ್ಣಾಂಕ workers = ((int32_t) ne00 + chunk - 1) / chunk;
        ಆದರೆ (workers > ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
            workers = ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        }

        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ saved_ಮುಖವಾಡ;
        __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(saved_ಮುಖವಾಡ));
        __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

        ದಶಮಾಂಶ partial_ಮೊತ್ತ = partial_ಮೊತ್ತ_ತುಂಡು(src_ptr, my_ಆರಂಭ, my_ಕೊನೆ);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        ಅಸ್ಥಿರ ದಶಮಾಂಶ * my_ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_ಸ್ಥಾನ                 = partial_ಮೊತ್ತ;
        FENCE;
        evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) my_ಸ್ಥಾನ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        ಆದರೆ (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < workers; t++) {
                ಅಸ್ಥಿರ ದಶಮಾಂಶ * ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) (ಗುಂಪು_ತಳ + t) * 64);
                evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ಸ್ಥಾನ, 1, 64);
            }
            WAIT_CACHEOPS;

            ದಶಮಾಂಶ ಒಟ್ಟು_ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < workers; t++) {
                ಅಸ್ಥಿರ ದಶಮಾಂಶ * ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) (ಗುಂಪು_ತಳ + t) * 64);
                ಒಟ್ಟು_ಮೊತ್ತ += *ಸ್ಥಾನ;
            }
            atomic_store_f32(dst_ptr, ಒಟ್ಟು_ಮೊತ್ತ * inv_ne00);
        }

        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));
    }

    ಹಿಂತಿರುಗಿ 0;
}
