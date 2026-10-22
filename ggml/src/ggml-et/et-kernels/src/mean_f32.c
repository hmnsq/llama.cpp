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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_اوسط_params {
    ساخت ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    ساخت ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
ساکن اندرونی اعشاری partial_مجموع_قاش(ثابت اعشاری * بنیاد, int32_t i_lo, int32_t i_hi) {
    اگر (i_lo >= i_hi) {
        واپس 0.0f;
    }

    ثابت اعشاری * p   = بنیاد + i_lo;
    int32_t       n   = i_hi - i_lo;
    اعشاری         acc = 0.0f;
    int32_t       i   = 0;

    اگر (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        اعشاری zero = 0.0f;
        __asm__ متلون("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        برائے (; i + 8 <= n; i += 8) {
            __asm__ متلون(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ثابت اعشاری (*)[8]) & p[i])
                : "f10", "f11");
        }

        اعشاری vec_مجموع;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_مجموع)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_مجموع;
    }

    برائے (; i < n; i++) {
        acc += p[i];
    }
    واپس acc;
}

صحیح اندراج_نقطہ(ساخت ggml_et_اوسط_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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

    اگر (ne00 <= 0) {
        واپس 0;
    }

    ثابت int32_t کل_صفیں    = (int32_t) (ne01 * ne02 * ne03);
    ثابت صحیح     shire_دھاگے = SOC_MINIONS_فی_SHIRE * عدد_HARTS_فی_MINION;
    ثابت اعشاری   inv_ne00      = et_fdiv(1.0f, (اعشاری) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    اگر (کل_صفیں >= shire_دھاگے) {
        برائے (int64_t ir = thread_id; ir < کل_صفیں; ir += عدد_دھاگے) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03);
            اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i01 * nb1 + i02 * nb2 + i03 * nb3);

            اعشاری صف_مجموع = partial_مجموع_قاش(src_صف, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, صف_مجموع * inv_ne00);
        }
        // Shire co-work
    } ورنہ {
        صحیح shire_tid       = thread_id % shire_دھاگے;
        صحیح دھاگے_فی_صف = shire_دھاگے / کل_صفیں;
        صحیح my_صف          = shire_tid / دھاگے_فی_صف;
        صحیح local_tid       = shire_tid % دھاگے_فی_صف;
        صحیح گروہ_بنیاد      = my_صف * دھاگے_فی_صف;

        اگر (my_صف >= کل_صفیں) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            واپس 0;
        }

        int64_t i1 = my_صف % ne01;
        int64_t i2 = (my_صف / ne01) % ne02;
        int64_t i3 = my_صف / (ne01 * ne02);

        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
        اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + دھاگے_فی_صف - 1) / دھاگے_فی_صف;
        chunk         = (chunk + 7) & ~7;
        اگر (chunk < 8) {
            chunk = 8;
        }

        int32_t my_آغاز = local_tid * chunk;
        int32_t my_انتہا   = my_آغاز + chunk;
        اگر (my_انتہا > (int32_t) ne00) {
            my_انتہا = (int32_t) ne00;
        }
        اگر (my_آغاز > (int32_t) ne00) {
            my_آغاز = my_انتہا = (int32_t) ne00;
        }

        صحیح workers = ((int32_t) ne00 + chunk - 1) / chunk;
        اگر (workers > دھاگے_فی_صف) {
            workers = دھاگے_فی_صف;
        }

        غیرعلامتی طویل saved_نقاب;
        __asm__ متلون("mova.x.m %0" : "=r"(saved_نقاب));
        __asm__ متلون("mov.m.x m0, x0, 0xFF");

        اعشاری partial_مجموع = partial_مجموع_قاش(src_ptr, my_آغاز, my_انتہا);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        متلون اعشاری * my_خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_خانہ                 = partial_مجموع;
        FENCE;
        evict_مقصد_l2((ثابت خالی *) my_خانہ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        اگر (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            برائے (صحیح t = 0; t < workers; t++) {
                متلون اعشاری * خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) (گروہ_بنیاد + t) * 64);
                evict_مقصد_l2((ثابت خالی *) خانہ, 1, 64);
            }
            WAIT_CACHEOPS;

            اعشاری کل_مجموع = 0.0f;
            برائے (صحیح t = 0; t < workers; t++) {
                متلون اعشاری * خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) (گروہ_بنیاد + t) * 64);
                کل_مجموع += *خانہ;
            }
            atomic_store_f32(dst_ptr, کل_مجموع * inv_ne00);
        }

        __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
    }

    واپس 0;
}
