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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_సగటు_params {
    నిర్మాణం ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    నిర్మాణం ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
స్థిర అంతఃవరుస దశాంశం partial_కూడిక_ముక్క(స్థిరరాశి దశాంశం * పునాది, int32_t i_lo, int32_t i_hi) {
    ఒకవేళ (i_lo >= i_hi) {
        తిరిగి 0.0f;
    }

    స్థిరరాశి దశాంశం * p   = పునాది + i_lo;
    int32_t       n   = i_hi - i_lo;
    దశాంశం         acc = 0.0f;
    int32_t       i   = 0;

    ఒకవేళ (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        దశాంశం zero = 0.0f;
        __asm__ అస్థిర("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        కొరకు (; i + 8 <= n; i += 8) {
            __asm__ అస్థిర(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & p[i])
                : "f10", "f11");
        }

        దశాంశం vec_కూడిక;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_కూడిక)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_కూడిక;
    }

    కొరకు (; i < n; i++) {
        acc += p[i];
    }
    తిరిగి acc;
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_సగటు_params * params, శూన్య * env) {
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
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి size_t nb01 = src0->nb[1];
    స్థిరరాశి size_t nb02 = src0->nb[2];
    స్థిరరాశి size_t nb03 = src0->nb[3];

    స్థిరరాశి size_t nb1 = dst->nb[1];
    స్థిరరాశి size_t nb2 = dst->nb[2];
    స్థిరరాశి size_t nb3 = dst->nb[3];

    ఒకవేళ (ne00 <= 0) {
        తిరిగి 0;
    }

    స్థిరరాశి int32_t మొత్తం_అడ్డువరుసలు    = (int32_t) (ne01 * ne02 * ne03);
    స్థిరరాశి పూర్ణాంకం     shire_దారాలు = SOC_MINIONS_చొప్పున_SHIRE * అంకె_HARTS_చొప్పున_MINION;
    స్థిరరాశి దశాంశం   inv_ne00      = et_fdiv(1.0f, (దశాంశం) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    ఒకవేళ (మొత్తం_అడ్డువరుసలు >= shire_దారాలు) {
        కొరకు (int64_t ir = thread_id; ir < మొత్తం_అడ్డువరుసలు; ir += అంకె_దారాలు) {
            స్థిరరాశి int64_t i03 = ir / (ne02 * ne01);
            స్థిరరాశి int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            స్థిరరాశి int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);
            దశాంశం *       dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i01 * nb1 + i02 * nb2 + i03 * nb3);

            దశాంశం అడ్డువరుస_కూడిక = partial_కూడిక_ముక్క(src_అడ్డువరుస, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, అడ్డువరుస_కూడిక * inv_ne00);
        }
        // Shire co-work
    } లేకపోతే {
        పూర్ణాంకం shire_tid       = thread_id % shire_దారాలు;
        పూర్ణాంకం దారాలు_చొప్పున_అడ్డువరుస = shire_దారాలు / మొత్తం_అడ్డువరుసలు;
        పూర్ణాంకం my_అడ్డువరుస          = shire_tid / దారాలు_చొప్పున_అడ్డువరుస;
        పూర్ణాంకం local_tid       = shire_tid % దారాలు_చొప్పున_అడ్డువరుస;
        పూర్ణాంకం సమూహం_పునాది      = my_అడ్డువరుస * దారాలు_చొప్పున_అడ్డువరుస;

        ఒకవేళ (my_అడ్డువరుస >= మొత్తం_అడ్డువరుసలు) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            తిరిగి 0;
        }

        int64_t i1 = my_అడ్డువరుస % ne01;
        int64_t i2 = (my_అడ్డువరుస / ne01) % ne02;
        int64_t i3 = my_అడ్డువరుస / (ne01 * ne02);

        స్థిరరాశి దశాంశం * src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
        దశాంశం *       dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + దారాలు_చొప్పున_అడ్డువరుస - 1) / దారాలు_చొప్పున_అడ్డువరుస;
        chunk         = (chunk + 7) & ~7;
        ఒకవేళ (chunk < 8) {
            chunk = 8;
        }

        int32_t my_ప్రారంభం = local_tid * chunk;
        int32_t my_చివర   = my_ప్రారంభం + chunk;
        ఒకవేళ (my_చివర > (int32_t) ne00) {
            my_చివర = (int32_t) ne00;
        }
        ఒకవేళ (my_ప్రారంభం > (int32_t) ne00) {
            my_ప్రారంభం = my_చివర = (int32_t) ne00;
        }

        పూర్ణాంకం workers = ((int32_t) ne00 + chunk - 1) / chunk;
        ఒకవేళ (workers > దారాలు_చొప్పున_అడ్డువరుస) {
            workers = దారాలు_చొప్పున_అడ్డువరుస;
        }

        గుర్తులేని దీర్ఘ saved_ముసుగు;
        __asm__ అస్థిర("mova.x.m %0" : "=r"(saved_ముసుగు));
        __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

        దశాంశం partial_కూడిక = partial_కూడిక_ముక్క(src_ptr, my_ప్రారంభం, my_చివర);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        అస్థిర దశాంశం * my_స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_స్థానం                 = partial_కూడిక;
        FENCE;
        evict_గమ్యం_l2((స్థిరరాశి శూన్య *) my_స్థానం, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        ఒకవేళ (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            కొరకు (పూర్ణాంకం t = 0; t < workers; t++) {
                అస్థిర దశాంశం * స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local((uint64_t) (సమూహం_పునాది + t) * 64);
                evict_గమ్యం_l2((స్థిరరాశి శూన్య *) స్థానం, 1, 64);
            }
            WAIT_CACHEOPS;

            దశాంశం మొత్తం_కూడిక = 0.0f;
            కొరకు (పూర్ణాంకం t = 0; t < workers; t++) {
                అస్థిర దశాంశం * స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local((uint64_t) (సమూహం_పునాది + t) * 64);
                మొత్తం_కూడిక += *స్థానం;
            }
            atomic_store_f32(dst_ptr, మొత్తం_కూడిక * inv_ne00);
        }

        __asm__ అస్థిర("mova.m.x %0" ::"r"(saved_ముసుగు));
    }

    తిరిగి 0;
}
