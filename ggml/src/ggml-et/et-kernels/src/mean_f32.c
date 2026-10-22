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

ಸಂರಚನಾ ggml_et_mean_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ partial_sum_slice(ಧ್ರುವ ಭಿನ್ನ * base, int32_t i_lo, int32_t i_hi) {
    ಯದಿ (i_lo >= i_hi) {
        ಪ್ರತ್ಯಾಗಮ 0.0f;
    }

    ಧ್ರುವ ಭಿನ್ನ * p   = base + i_lo;
    int32_t       n   = i_hi - i_lo;
    ಭಿನ್ನ         acc = 0.0f;
    int32_t       i   = 0;

    ಯದಿ (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        ಭಿನ್ನ zero = 0.0f;
        __asm__ ಚಞ್ಚಲ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ಪ್ರತಿ (; i + 8 <= n; i += 8) {
            __asm__ ಚಞ್ಚಲ(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & p[i])
                : "f10", "f11");
        }

        ಭಿನ್ನ vec_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_sum)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_sum;
    }

    ಪ್ರತಿ (; i < n; i++) {
        acc += p[i];
    }
    ಪ್ರತ್ಯಾಗಮ acc;
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_mean_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne00 = src0->ne[0];
    ಧ್ರುವ int64_t ne01 = src0->ne[1];
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    ಧ್ರುವ size_t nb01 = src0->nb[1];
    ಧ್ರುವ size_t nb02 = src0->nb[2];
    ಧ್ರುವ size_t nb03 = src0->nb[3];

    ಧ್ರುವ size_t nb1 = dst->nb[1];
    ಧ್ರುವ size_t nb2 = dst->nb[2];
    ಧ್ರುವ size_t nb3 = dst->nb[3];

    ಯದಿ (ne00 <= 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ int32_t total_rows    = (int32_t) (ne01 * ne02 * ne03);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shire_threads = SOC_MINIONS_PER_SHIRE * ಅಂಕೆ_HARTS_PER_MINION;
    ಧ್ರುವ ಭಿನ್ನ   inv_ne00      = et_fdiv(1.0f, (ಭಿನ್ನ) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    ಯದಿ (total_rows >= shire_threads) {
        ಪ್ರತಿ (int64_t ir = thread_id; ir < total_rows; ir += ಅಂಕೆ_threads) {
            ಧ್ರುವ int64_t i03 = ir / (ne02 * ne01);
            ಧ್ರುವ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ಧ್ರುವ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ಭಿನ್ನ *       dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

            ಭಿನ್ನ row_sum = partial_sum_slice(src_row, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, row_sum * inv_ne00);
        }
        // Shire co-work
    } ಅನ್ಯಥಾ {
        ಪೂರ್ಣಾಙ್ಕ shire_tid       = thread_id % shire_threads;
        ಪೂರ್ಣಾಙ್ಕ threads_per_row = shire_threads / total_rows;
        ಪೂರ್ಣಾಙ್ಕ my_row          = shire_tid / threads_per_row;
        ಪೂರ್ಣಾಙ್ಕ local_tid       = shire_tid % threads_per_row;
        ಪೂರ್ಣಾಙ್ಕ group_base      = my_row * threads_per_row;

        ಯದಿ (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        int64_t i1 = my_row % ne01;
        int64_t i2 = (my_row / ne01) % ne02;
        int64_t i3 = my_row / (ne01 * ne02);

        ಧ್ರುವ ಭಿನ್ನ * src_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ಭಿನ್ನ *       dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + threads_per_row - 1) / threads_per_row;
        chunk         = (chunk + 7) & ~7;
        ಯದಿ (chunk < 8) {
            chunk = 8;
        }

        int32_t my_start = local_tid * chunk;
        int32_t my_end   = my_start + chunk;
        ಯದಿ (my_end > (int32_t) ne00) {
            my_end = (int32_t) ne00;
        }
        ಯದಿ (my_start > (int32_t) ne00) {
            my_start = my_end = (int32_t) ne00;
        }

        ಪೂರ್ಣಾಙ್ಕ workers = ((int32_t) ne00 + chunk - 1) / chunk;
        ಯದಿ (workers > threads_per_row) {
            workers = threads_per_row;
        }

        unsigned long saved_mask;
        __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(saved_mask));
        __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");

        ಭಿನ್ನ partial_sum = partial_sum_slice(src_ptr, my_start, my_end);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        ಚಞ್ಚಲ ಭಿನ್ನ * my_slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        ಯದಿ (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < workers; t++) {
                ಚಞ್ಚಲ ಭಿನ್ನ * slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) slot, 1, 64);
            }
            WAIT_CACHEOPS;

            ಭಿನ್ನ total_sum = 0.0f;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < workers; t++) {
                ಚಞ್ಚಲ ಭಿನ್ನ * slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                total_sum += *slot;
            }
            atomic_store_f32(dst_ptr, total_sum * inv_ne00);
        }

        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
