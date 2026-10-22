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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_ਔਸਤ_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    ਬਣਤਰ ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ partial_ਜੋੜ_ਟੁਕੜਾ(ਅਚਲ ਦਸ਼ਮਲਵ * ਅਧਾਰ, int32_t i_lo, int32_t i_hi) {
    ਜੇ (i_lo >= i_hi) {
        ਵਾਪਸ 0.0f;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * p   = ਅਧਾਰ + i_lo;
    int32_t       n   = i_hi - i_lo;
    ਦਸ਼ਮਲਵ         acc = 0.0f;
    int32_t       i   = 0;

    ਜੇ (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        ਦਸ਼ਮਲਵ zero = 0.0f;
        __asm__ ਅਸਥਿਰ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ਲਈ (; i + 8 <= n; i += 8) {
            __asm__ ਅਸਥਿਰ(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & p[i])
                : "f10", "f11");
        }

        ਦਸ਼ਮਲਵ vec_ਜੋੜ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_ਜੋੜ)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_ਜੋੜ;
    }

    ਲਈ (; i < n; i++) {
        acc += p[i];
    }
    ਵਾਪਸ acc;
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਔਸਤ_params * params, ਖਾਲੀ * env) {
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
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ size_t nb01 = src0->nb[1];
    ਅਚਲ size_t nb02 = src0->nb[2];
    ਅਚਲ size_t nb03 = src0->nb[3];

    ਅਚਲ size_t nb1 = dst->nb[1];
    ਅਚਲ size_t nb2 = dst->nb[2];
    ਅਚਲ size_t nb3 = dst->nb[3];

    ਜੇ (ne00 <= 0) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ int32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ    = (int32_t) (ne01 * ne02 * ne03);
    ਅਚਲ ਪੂਰਨ     shire_ਧਾਗੇ = SOC_MINIONS_ਪ੍ਰਤੀ_SHIRE * ਅੰਕ_HARTS_ਪ੍ਰਤੀ_MINION;
    ਅਚਲ ਦਸ਼ਮਲਵ   inv_ne00      = et_fdiv(1.0f, (ਦਸ਼ਮਲਵ) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ >= shire_ਧਾਗੇ) {
        ਲਈ (int64_t ir = thread_id; ir < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ir += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t i03 = ir / (ne02 * ne01);
            ਅਚਲ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ਅਚਲ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ਦਸ਼ਮਲਵ *       dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i01 * nb1 + i02 * nb2 + i03 * nb3);

            ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ = partial_ਜੋੜ_ਟੁਕੜਾ(src_ਪੰਕਤੀ, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, ਪੰਕਤੀ_ਜੋੜ * inv_ne00);
        }
        // Shire co-work
    } ਨਹੀਂਤਾਂ {
        ਪੂਰਨ shire_tid       = thread_id % shire_ਧਾਗੇ;
        ਪੂਰਨ ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = shire_ਧਾਗੇ / ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        ਪੂਰਨ my_ਪੰਕਤੀ          = shire_tid / ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਪੂਰਨ local_tid       = shire_tid % ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਪੂਰਨ ਸਮੂਹ_ਅਧਾਰ      = my_ਪੰਕਤੀ * ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

        ਜੇ (my_ਪੰਕਤੀ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ਵਾਪਸ 0;
        }

        int64_t i1 = my_ਪੰਕਤੀ % ne01;
        int64_t i2 = (my_ਪੰਕਤੀ / ne01) % ne02;
        int64_t i3 = my_ਪੰਕਤੀ / (ne01 * ne02);

        ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ਦਸ਼ਮਲਵ *       dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ - 1) / ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        chunk         = (chunk + 7) & ~7;
        ਜੇ (chunk < 8) {
            chunk = 8;
        }

        int32_t my_ਸ਼ੁਰੂ = local_tid * chunk;
        int32_t my_ਸਿਰਾ   = my_ਸ਼ੁਰੂ + chunk;
        ਜੇ (my_ਸਿਰਾ > (int32_t) ne00) {
            my_ਸਿਰਾ = (int32_t) ne00;
        }
        ਜੇ (my_ਸ਼ੁਰੂ > (int32_t) ne00) {
            my_ਸ਼ੁਰੂ = my_ਸਿਰਾ = (int32_t) ne00;
        }

        ਪੂਰਨ workers = ((int32_t) ne00 + chunk - 1) / chunk;
        ਜੇ (workers > ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
            workers = ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        }

        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

        ਦਸ਼ਮਲਵ partial_ਜੋੜ = partial_ਜੋੜ_ਟੁਕੜਾ(src_ptr, my_ਸ਼ੁਰੂ, my_ਸਿਰਾ);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        ਅਸਥਿਰ ਦਸ਼ਮਲਵ * my_ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_ਥਾਂ                 = partial_ਜੋੜ;
        FENCE;
        evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) my_ਥਾਂ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        ਜੇ (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            ਲਈ (ਪੂਰਨ t = 0; t < workers; t++) {
                ਅਸਥਿਰ ਦਸ਼ਮਲਵ * ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local((uint64_t) (ਸਮੂਹ_ਅਧਾਰ + t) * 64);
                evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) ਥਾਂ, 1, 64);
            }
            WAIT_CACHEOPS;

            ਦਸ਼ਮਲਵ ਕੁੱਲ_ਜੋੜ = 0.0f;
            ਲਈ (ਪੂਰਨ t = 0; t < workers; t++) {
                ਅਸਥਿਰ ਦਸ਼ਮਲਵ * ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local((uint64_t) (ਸਮੂਹ_ਅਧਾਰ + t) * 64);
                ਕੁੱਲ_ਜੋੜ += *ਥਾਂ;
            }
            atomic_store_f32(dst_ptr, ਕੁੱਲ_ਜੋੜ * inv_ne00);
        }

        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
    }

    ਵਾਪਸ 0;
}
