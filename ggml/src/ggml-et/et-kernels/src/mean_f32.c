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

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_mean_params {
    ഘടന ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    ഘടന ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
സ്ഥിതം വരിയിൽ ദശാംശം partial_sum_slice(സ്ഥിരം ദശാംശം * base, int32_t i_lo, int32_t i_hi) {
    എങ്കിൽ (i_lo >= i_hi) {
        തിരികെ 0.0f;
    }

    സ്ഥിരം ദശാംശം * p   = base + i_lo;
    int32_t       n   = i_hi - i_lo;
    ദശാംശം         acc = 0.0f;
    int32_t       i   = 0;

    എങ്കിൽ (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        ദശാംശം zero = 0.0f;
        __asm__ അസ്ഥിര("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        വേണ്ടി (; i + 8 <= n; i += 8) {
            __asm__ അസ്ഥിര(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & p[i])
                : "f10", "f11");
        }

        ദശാംശം vec_sum;
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

    വേണ്ടി (; i < n; i++) {
        acc += p[i];
    }
    തിരികെ acc;
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_mean_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];
    സ്ഥിരം int64_t ne03 = src0->ne[3];

    സ്ഥിരം size_t nb01 = src0->nb[1];
    സ്ഥിരം size_t nb02 = src0->nb[2];
    സ്ഥിരം size_t nb03 = src0->nb[3];

    സ്ഥിരം size_t nb1 = dst->nb[1];
    സ്ഥിരം size_t nb2 = dst->nb[2];
    സ്ഥിരം size_t nb3 = dst->nb[3];

    എങ്കിൽ (ne00 <= 0) {
        തിരികെ 0;
    }

    സ്ഥിരം int32_t total_rows    = (int32_t) (ne01 * ne02 * ne03);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     shire_threads = SOC_MINIONS_PER_SHIRE * അക്കം_HARTS_PER_MINION;
    സ്ഥിരം ദശാംശം   inv_ne00      = et_fdiv(1.0f, (ദശാംശം) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    എങ്കിൽ (total_rows >= shire_threads) {
        വേണ്ടി (int64_t ir = thread_id; ir < total_rows; ir += അക്കം_threads) {
            സ്ഥിരം int64_t i03 = ir / (ne02 * ne01);
            സ്ഥിരം int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            സ്ഥിരം int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ദശാംശം *       dst_ptr = (ദശാംശം *) ((അക്ഷരം *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

            ദശാംശം row_sum = partial_sum_slice(src_row, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, row_sum * inv_ne00);
        }
        // Shire co-work
    } അല്ലെങ്കിൽ {
        പൂർണ്ണസംഖ്യ shire_tid       = thread_id % shire_threads;
        പൂർണ്ണസംഖ്യ threads_per_row = shire_threads / total_rows;
        പൂർണ്ണസംഖ്യ my_row          = shire_tid / threads_per_row;
        പൂർണ്ണസംഖ്യ local_tid       = shire_tid % threads_per_row;
        പൂർണ്ണസംഖ്യ group_base      = my_row * threads_per_row;

        എങ്കിൽ (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            തിരികെ 0;
        }

        int64_t i1 = my_row % ne01;
        int64_t i2 = (my_row / ne01) % ne02;
        int64_t i3 = my_row / (ne01 * ne02);

        സ്ഥിരം ദശാംശം * src_ptr = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ദശാംശം *       dst_ptr = (ദശാംശം *) ((അക്ഷരം *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + threads_per_row - 1) / threads_per_row;
        chunk         = (chunk + 7) & ~7;
        എങ്കിൽ (chunk < 8) {
            chunk = 8;
        }

        int32_t my_start = local_tid * chunk;
        int32_t my_end   = my_start + chunk;
        എങ്കിൽ (my_end > (int32_t) ne00) {
            my_end = (int32_t) ne00;
        }
        എങ്കിൽ (my_start > (int32_t) ne00) {
            my_start = my_end = (int32_t) ne00;
        }

        പൂർണ്ണസംഖ്യ workers = ((int32_t) ne00 + chunk - 1) / chunk;
        എങ്കിൽ (workers > threads_per_row) {
            workers = threads_per_row;
        }

        unsigned long saved_mask;
        __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(saved_mask));
        __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");

        ദശാംശം partial_sum = partial_sum_slice(src_ptr, my_start, my_end);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        അസ്ഥിര ദശാംശം * my_slot = (അസ്ഥിര ദശാംശം *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((സ്ഥിരം ശൂന്യം *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        എങ്കിൽ (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < workers; t++) {
                അസ്ഥിര ദശാംശം * slot = (അസ്ഥിര ദശാംശം *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                evict_to_l2((സ്ഥിരം ശൂന്യം *) slot, 1, 64);
            }
            WAIT_CACHEOPS;

            ദശാംശം total_sum = 0.0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < workers; t++) {
                അസ്ഥിര ദശാംശം * slot = (അസ്ഥിര ദശാംശം *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                total_sum += *slot;
            }
            atomic_store_f32(dst_ptr, total_sum * inv_ne00);
        }

        __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(saved_mask));
    }

    തിരികെ 0;
}
