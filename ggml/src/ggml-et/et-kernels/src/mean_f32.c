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

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "math_fp.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

সংরচনা ggml_et_mean_params {
    সংরচনা ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    সংরচনা ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
স্থির অন্তঃস্থ ভিন্ন partial_sum_slice(ধ্রুৱ ভিন্ন * base, int32_t i_lo, int32_t i_hi) {
    যদি (i_lo >= i_hi) {
        ঘূৰাই 0.0f;
    }

    ধ্রুৱ ভিন্ন * p   = base + i_lo;
    int32_t       n   = i_hi - i_lo;
    ভিন্ন         acc = 0.0f;
    int32_t       i   = 0;

    যদি (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        ভিন্ন zero = 0.0f;
        __asm__ চঞ্চল("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        বাবে (; i + 8 <= n; i += 8) {
            __asm__ চঞ্চল(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & p[i])
                : "f10", "f11");
        }

        ভিন্ন vec_sum;
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

    বাবে (; i < n; i++) {
        acc += p[i];
    }
    ঘূৰাই acc;
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_mean_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t ne00 = src0->ne[0];
    ধ্রুৱ int64_t ne01 = src0->ne[1];
    ধ্রুৱ int64_t ne02 = src0->ne[2];
    ধ্রুৱ int64_t ne03 = src0->ne[3];

    ধ্রুৱ size_t nb01 = src0->nb[1];
    ধ্রুৱ size_t nb02 = src0->nb[2];
    ধ্রুৱ size_t nb03 = src0->nb[3];

    ধ্রুৱ size_t nb1 = dst->nb[1];
    ধ্রুৱ size_t nb2 = dst->nb[2];
    ধ্রুৱ size_t nb3 = dst->nb[3];

    যদি (ne00 <= 0) {
        ঘূৰাই 0;
    }

    ধ্রুৱ int32_t total_rows    = (int32_t) (ne01 * ne02 * ne03);
    ধ্রুৱ পূর্ণাঙ্ক     shire_threads = SOC_MINIONS_PER_SHIRE * অঙ্ক_HARTS_PER_MINION;
    ধ্রুৱ ভিন্ন   inv_ne00      = et_fdiv(1.0f, (ভিন্ন) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    যদি (total_rows >= shire_threads) {
        বাবে (int64_t ir = thread_id; ir < total_rows; ir += অঙ্ক_threads) {
            ধ্রুৱ int64_t i03 = ir / (ne02 * ne01);
            ধ্রুৱ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ধ্রুৱ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ধ্রুৱ ভিন্ন * src_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ভিন্ন *       dst_ptr = (ভিন্ন *) ((ৱর্ণ *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

            ভিন্ন row_sum = partial_sum_slice(src_row, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, row_sum * inv_ne00);
        }
        // Shire co-work
    } অন্যথা {
        পূর্ণাঙ্ক shire_tid       = thread_id % shire_threads;
        পূর্ণাঙ্ক threads_per_row = shire_threads / total_rows;
        পূর্ণাঙ্ক my_row          = shire_tid / threads_per_row;
        পূর্ণাঙ্ক local_tid       = shire_tid % threads_per_row;
        পূর্ণাঙ্ক group_base      = my_row * threads_per_row;

        যদি (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ঘূৰাই 0;
        }

        int64_t i1 = my_row % ne01;
        int64_t i2 = (my_row / ne01) % ne02;
        int64_t i3 = my_row / (ne01 * ne02);

        ধ্রুৱ ভিন্ন * src_ptr = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ভিন্ন *       dst_ptr = (ভিন্ন *) ((ৱর্ণ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + threads_per_row - 1) / threads_per_row;
        chunk         = (chunk + 7) & ~7;
        যদি (chunk < 8) {
            chunk = 8;
        }

        int32_t my_start = local_tid * chunk;
        int32_t my_end   = my_start + chunk;
        যদি (my_end > (int32_t) ne00) {
            my_end = (int32_t) ne00;
        }
        যদি (my_start > (int32_t) ne00) {
            my_start = my_end = (int32_t) ne00;
        }

        পূর্ণাঙ্ক workers = ((int32_t) ne00 + chunk - 1) / chunk;
        যদি (workers > threads_per_row) {
            workers = threads_per_row;
        }

        unsigned long saved_mask;
        __asm__ চঞ্চল("mova.x.m %0" : "=r"(saved_mask));
        __asm__ চঞ্চল("mov.m.x m0, x0, 0xFF");

        ভিন্ন partial_sum = partial_sum_slice(src_ptr, my_start, my_end);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        চঞ্চল ভিন্ন * my_slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ধ্রুৱ শূন্য *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        যদি (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            বাবে (পূর্ণাঙ্ক t = 0; t < workers; t++) {
                চঞ্চল ভিন্ন * slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                evict_to_l2((ধ্রুৱ শূন্য *) slot, 1, 64);
            }
            WAIT_CACHEOPS;

            ভিন্ন total_sum = 0.0f;
            বাবে (পূর্ণাঙ্ক t = 0; t < workers; t++) {
                চঞ্চল ভিন্ন * slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                total_sum += *slot;
            }
            atomic_store_f32(dst_ptr, total_sum * inv_ne00);
        }

        __asm__ চঞ্চল("mova.m.x %0" ::"r"(saved_mask));
    }

    ঘূৰাই 0;
}
