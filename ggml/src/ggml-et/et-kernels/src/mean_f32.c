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

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_mean_params {
    काष्ठा ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    काष्ठा ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
अटल अंतरभूत भग्न partial_योग_slice(स्थिर भग्न * base, int32_t i_lo, int32_t i_hi) {
    अगर (i_lo >= i_hi) {
        वापस 0.0f;
    }

    स्थिर भग्न * p   = base + i_lo;
    int32_t       n   = i_hi - i_lo;
    भग्न         acc = 0.0f;
    int32_t       i   = 0;

    अगर (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        भग्न zero = 0.0f;
        __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        क्रम (; i + 8 <= n; i += 8) {
            __asm__ अस्थिर(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(स्थिर भग्न (*)[8]) & p[i])
                : "f10", "f11");
        }

        भग्न vec_योग;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(vec_योग)::"t0", "f1", "f2", "f3", "f4", "f5");
        acc = vec_योग;
    }

    क्रम (; i < n; i++) {
        acc += p[i];
    }
    वापस acc;
}

पूर्णांक entry_point(काष्ठा ggml_et_mean_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    स्थिर माप_प्रकार nb01 = src0->nb[1];
    स्थिर माप_प्रकार nb02 = src0->nb[2];
    स्थिर माप_प्रकार nb03 = src0->nb[3];

    स्थिर माप_प्रकार nb1 = dst->nb[1];
    स्थिर माप_प्रकार nb2 = dst->nb[2];
    स्थिर माप_प्रकार nb3 = dst->nb[3];

    अगर (ne00 <= 0) {
        वापस 0;
    }

    स्थिर int32_t योग_rows    = (int32_t) (ne01 * ne02 * ne03);
    स्थिर पूर्णांक     shire_threads = SOC_MINIONS_PER_SHIRE * अंक_HARTS_PER_MINION;
    स्थिर भग्न   inv_ne00      = et_fdiv(1.0f, (भग्न) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    अगर (योग_rows >= shire_threads) {
        क्रम (int64_t ir = thread_id; ir < योग_rows; ir += अंक_threads) {
            स्थिर int64_t i03 = ir / (ne02 * ne01);
            स्थिर int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            स्थिर int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);
            भग्न *       dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i01 * nb1 + i02 * nb2 + i03 * nb3);

            भग्न row_योग = partial_योग_slice(src_row, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, row_योग * inv_ne00);
        }
        // Shire co-work
    } अन्यथा {
        पूर्णांक shire_tid       = thread_id % shire_threads;
        पूर्णांक threads_per_row = shire_threads / योग_rows;
        पूर्णांक my_row          = shire_tid / threads_per_row;
        पूर्णांक local_tid       = shire_tid % threads_per_row;
        पूर्णांक group_base      = my_row * threads_per_row;

        अगर (my_row >= योग_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            वापस 0;
        }

        int64_t i1 = my_row % ne01;
        int64_t i2 = (my_row / ne01) % ne02;
        int64_t i3 = my_row / (ne01 * ne02);

        स्थिर भग्न * src_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
        भग्न *       dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + threads_per_row - 1) / threads_per_row;
        chunk         = (chunk + 7) & ~7;
        अगर (chunk < 8) {
            chunk = 8;
        }

        int32_t my_start = local_tid * chunk;
        int32_t my_end   = my_start + chunk;
        अगर (my_end > (int32_t) ne00) {
            my_end = (int32_t) ne00;
        }
        अगर (my_start > (int32_t) ne00) {
            my_start = my_end = (int32_t) ne00;
        }

        पूर्णांक workers = ((int32_t) ne00 + chunk - 1) / chunk;
        अगर (workers > threads_per_row) {
            workers = threads_per_row;
        }

        अचिन्हित दीर्घ saved_mask;
        __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

        भग्न partial_योग = partial_योग_slice(src_ptr, my_start, my_end);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        अस्थिर भग्न * my_slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_योग;
        FENCE;
        evict_to_l2((स्थिर व्योम *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        अगर (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            क्रम (पूर्णांक t = 0; t < workers; t++) {
                अस्थिर भग्न * slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                evict_to_l2((स्थिर व्योम *) slot, 1, 64);
            }
            WAIT_CACHEOPS;

            भग्न योग_योग = 0.0f;
            क्रम (पूर्णांक t = 0; t < workers; t++) {
                अस्थिर भग्न * slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
                योग_योग += *slot;
            }
            atomic_store_f32(dst_ptr, योग_योग * inv_ne00);
        }

        __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
    }

    वापस 0;
}
