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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_سراسري_params {
    بنية ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    بنية ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
ساكن مضمن عائم partial_جوڙ_ٽڪرو(ثابت عائم * بنياد, int32_t i_lo, int32_t i_hi) {
    إذا (i_lo >= i_hi) {
        أرجع 0.0f;
    }

    ثابت عائم * p   = بنياد + i_lo;
    int32_t       n   = i_hi - i_lo;
    عائم         acc = 0.0f;
    int32_t       i   = 0;

    إذا (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        عائم zero = 0.0f;
        __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        لكل (; i + 8 <= n; i += 8) {
            __asm__ متطاير(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ثابت عائم (*)[8]) & p[i])
                : "f10", "f11");
        }

        عائم vec_جوڙ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_جوڙ)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_جوڙ;
    }

    لكل (; i < n; i++) {
        acc += p[i];
    }
    أرجع acc;
}

صحيح داخلا_نقطو(بنية ggml_et_سراسري_params * params, فراغ * env) {
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
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت size_t nb01 = src0->nb[1];
    ثابت size_t nb02 = src0->nb[2];
    ثابت size_t nb03 = src0->nb[3];

    ثابت size_t nb1 = dst->nb[1];
    ثابت size_t nb2 = dst->nb[2];
    ثابت size_t nb3 = dst->nb[3];

    إذا (ne00 <= 0) {
        أرجع 0;
    }

    ثابت int32_t ڪل_صفون    = (int32_t) (ne01 * ne02 * ne03);
    ثابت صحيح     shire_ڌاڳا = SOC_MINIONS_في_SHIRE * انگ_HARTS_في_MINION;
    ثابت عائم   inv_ne00      = et_fdiv(1.0f, (عائم) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    إذا (ڪل_صفون >= shire_ڌاڳا) {
        لكل (int64_t ir = thread_id; ir < ڪل_صفون; ir += انگ_ڌاڳا) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);
            عائم *       dst_ptr = (عائم *) ((حرف *) dst_معلومات + i01 * nb1 + i02 * nb2 + i03 * nb3);

            عائم صف_جوڙ = partial_جوڙ_ٽڪرو(src_صف, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, صف_جوڙ * inv_ne00);
        }
        // Shire co-work
    } وإلا {
        صحيح shire_tid       = thread_id % shire_ڌاڳا;
        صحيح ڌاڳا_في_صف = shire_ڌاڳا / ڪل_صفون;
        صحيح my_صف          = shire_tid / ڌاڳا_في_صف;
        صحيح local_tid       = shire_tid % ڌاڳا_في_صف;
        صحيح گروهه_بنياد      = my_صف * ڌاڳا_في_صف;

        إذا (my_صف >= ڪل_صفون) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            أرجع 0;
        }

        int64_t i1 = my_صف % ne01;
        int64_t i2 = (my_صف / ne01) % ne02;
        int64_t i3 = my_صف / (ne01 * ne02);

        ثابت عائم * src_ptr = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
        عائم *       dst_ptr = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + ڌاڳا_في_صف - 1) / ڌاڳا_في_صف;
        chunk         = (chunk + 7) & ~7;
        إذا (chunk < 8) {
            chunk = 8;
        }

        int32_t my_شروعات = local_tid * chunk;
        int32_t my_پڇاڙي   = my_شروعات + chunk;
        إذا (my_پڇاڙي > (int32_t) ne00) {
            my_پڇاڙي = (int32_t) ne00;
        }
        إذا (my_شروعات > (int32_t) ne00) {
            my_شروعات = my_پڇاڙي = (int32_t) ne00;
        }

        صحيح workers = ((int32_t) ne00 + chunk - 1) / chunk;
        إذا (workers > ڌاڳا_في_صف) {
            workers = ڌاڳا_في_صف;
        }

        بنانشان ڊگهو saved_نقاب;
        __asm__ متطاير("mova.x.m %0" : "=r"(saved_نقاب));
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");

        عائم partial_جوڙ = partial_جوڙ_ٽڪرو(src_ptr, my_شروعات, my_پڇاڙي);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        متطاير عائم * my_جڳهه = (متطاير عائم *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_جڳهه                 = partial_جوڙ;
        FENCE;
        evict_منزل_l2((ثابت فراغ *) my_جڳهه, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        إذا (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            لكل (صحيح t = 0; t < workers; t++) {
                متطاير عائم * جڳهه = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (گروهه_بنياد + t) * 64);
                evict_منزل_l2((ثابت فراغ *) جڳهه, 1, 64);
            }
            WAIT_CACHEOPS;

            عائم ڪل_جوڙ = 0.0f;
            لكل (صحيح t = 0; t < workers; t++) {
                متطاير عائم * جڳهه = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (گروهه_بنياد + t) * 64);
                ڪل_جوڙ += *جڳهه;
            }
            atomic_store_f32(dst_ptr, ڪل_جوڙ * inv_ne00);
        }

        __asm__ متطاير("mova.m.x %0" ::"r"(saved_نقاب));
    }

    أرجع 0;
}
