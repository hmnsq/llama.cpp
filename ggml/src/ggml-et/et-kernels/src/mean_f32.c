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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_mean_params {
    સંરચના ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    સંરચના ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
સ્થિર અંતઃરેખ દશાંશ partial_sum_slice(અચલ દશાંશ * base, int32_t i_lo, int32_t i_hi) {
    જો (i_lo >= i_hi) {
        પાછા 0.0f;
    }

    અચલ દશાંશ * p   = base + i_lo;
    int32_t       n   = i_hi - i_lo;
    દશાંશ         acc = 0.0f;
    int32_t       i   = 0;

    જો (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        દશાંશ zero = 0.0f;
        __asm__ અસ્થિર("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        માટે (; i + 8 <= n; i += 8) {
            __asm__ અસ્થિર(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(અચલ દશાંશ (*)[8]) & p[i])
                : "f10", "f11");
        }

        દશાંશ vec_sum;
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

    માટે (; i < n; i++) {
        acc += p[i];
    }
    પાછા acc;
}

પૂર્ણાંક entry_point(સંરચના ggml_et_mean_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];
    અચલ int64_t ne03 = src0->ne[3];

    અચલ size_t nb01 = src0->nb[1];
    અચલ size_t nb02 = src0->nb[2];
    અચલ size_t nb03 = src0->nb[3];

    અચલ size_t nb1 = dst->nb[1];
    અચલ size_t nb2 = dst->nb[2];
    અચલ size_t nb3 = dst->nb[3];

    જો (ne00 <= 0) {
        પાછા 0;
    }

    અચલ int32_t total_rows    = (int32_t) (ne01 * ne02 * ne03);
    અચલ પૂર્ણાંક     shire_threads = SOC_MINIONS_PER_SHIRE * અંક_HARTS_PER_MINION;
    અચલ દશાંશ   inv_ne00      = et_fdiv(1.0f, (દશાંશ) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    જો (total_rows >= shire_threads) {
        માટે (int64_t ir = thread_id; ir < total_rows; ir += અંક_threads) {
            અચલ int64_t i03 = ir / (ne02 * ne01);
            અચલ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            અચલ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            દશાંશ *       dst_ptr = (દશાંશ *) ((અક્ષર *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

            દશાંશ row_sum = partial_sum_slice(src_row, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, row_sum * inv_ne00);
        }
        // Shire co-work
    } નહીંતર {
        પૂર્ણાંક shire_tid       = thread_id % shire_threads;
        પૂર્ણાંક threads_per_row = shire_threads / total_rows;
        પૂર્ણાંક my_row          = shire_tid / threads_per_row;
        પૂર્ણાંક local_tid       = shire_tid % threads_per_row;
        પૂર્ણાંક group_base      = my_row * threads_per_row;

        જો (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            પાછા 0;
        }

        int64_t i1 = my_row % ne01;
        int64_t i2 = (my_row / ne01) % ne02;
        int64_t i3 = my_row / (ne01 * ne02);

        અચલ દશાંશ * src_ptr = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        દશાંશ *       dst_ptr = (દશાંશ *) ((અક્ષર *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + threads_per_row - 1) / threads_per_row;
        chunk         = (chunk + 7) & ~7;
        જો (chunk < 8) {
            chunk = 8;
        }

        int32_t my_start = local_tid * chunk;
        int32_t my_end   = my_start + chunk;
        જો (my_end > (int32_t) ne00) {
            my_end = (int32_t) ne00;
        }
        જો (my_start > (int32_t) ne00) {
            my_start = my_end = (int32_t) ne00;
        }

        પૂર્ણાંક workers = ((int32_t) ne00 + chunk - 1) / chunk;
        જો (workers > threads_per_row) {
            workers = threads_per_row;
        }

        unsigned long saved_mask;
        __asm__ અસ્થિર("mova.x.m %0" : "=r"(saved_mask));
        __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

        દશાંશ partial_sum = partial_sum_slice(src_ptr, my_start, my_end);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        અસ્થિર દશાંશ * my_slot = (અસ્થિર દશાંશ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((અચલ શૂન્ય *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        જો (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            માટે (પૂર્ણાંક t = 0; t < workers; t++) {
                અસ્થિર દશાંશ * slot = (અસ્થિર દશાંશ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                evict_to_l2((અચલ શૂન્ય *) slot, 1, 64);
            }
            WAIT_CACHEOPS;

            દશાંશ total_sum = 0.0f;
            માટે (પૂર્ણાંક t = 0; t < workers; t++) {
                અસ્થિર દશાંશ * slot = (અસ્થિર દશાંશ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                total_sum += *slot;
            }
            atomic_store_f32(dst_ptr, total_sum * inv_ne00);
        }

        __asm__ અસ્થિર("mova.m.x %0" ::"r"(saved_mask));
    }

    પાછા 0;
}
