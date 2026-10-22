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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_माध्यम्_params {
    संरचना ggml_tensor src0;  // F32 input  [ne00, ne01, ne02, ne03]
    संरचना ggml_tensor dst;   // F32 output [1,    ne01, ne02, ne03]
};

// Sum a contiguous F32 slice [base+i_lo, base+i_hi). Uses the 8-wide vector
// path only when `base + i_lo` is 32B-aligned; the tail (and the whole slice
// when misaligned) is summed with scalar fadd.s.
स्थिर अन्तःस्थ भिन्न partial_योग_खण्डिका(ध्रुव भिन्न * आधारः, int32_t i_lo, int32_t i_hi) {
    यदि (i_lo >= i_hi) {
        प्रत्यागम 0.0f;
    }

    ध्रुव भिन्न * p   = आधारः + i_lo;
    int32_t       n   = i_hi - i_lo;
    भिन्न         acc = 0.0f;
    int32_t       i   = 0;

    यदि (n >= 8 && (((uintptr_t) p) & 31) == 0) {
        भिन्न zero = 0.0f;
        __asm__ चञ्चल("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        प्रति (; i + 8 <= n; i += 8) {
            __asm__ चञ्चल(
                "flw.ps  f11, %[x]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x] "m"(*(ध्रुव भिन्न (*)[8]) & p[i])
                : "f10", "f11");
        }

        भिन्न vec_योग;
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

    प्रति (; i < n; i++) {
        acc += p[i];
    }
    प्रत्यागम acc;
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_माध्यम्_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];
    ध्रुव int64_t ne03 = src0->ne[3];

    ध्रुव size_t nb01 = src0->nb[1];
    ध्रुव size_t nb02 = src0->nb[2];
    ध्रुव size_t nb03 = src0->nb[3];

    ध्रुव size_t nb1 = dst->nb[1];
    ध्रुव size_t nb2 = dst->nb[2];
    ध्रुव size_t nb3 = dst->nb[3];

    यदि (ne00 <= 0) {
        प्रत्यागम 0;
    }

    ध्रुव int32_t योग_पङ्क्तिकाः    = (int32_t) (ne01 * ne02 * ne03);
    ध्रुव पूर्णाङ्क     shire_तन्तवः = SOC_MINIONS_प्रत्येकम्_SHIRE * अङ्कः_HARTS_प्रत्येकम्_MINION;
    ध्रुव भिन्न   inv_ne00      = et_fdiv(1.0f, (भिन्न) (int32_t) ne00);

    // Row-parallel: each thread owns whole rows.
    यदि (योग_पङ्क्तिकाः >= shire_तन्तवः) {
        प्रति (int64_t ir = thread_id; ir < योग_पङ्क्तिकाः; ir += अङ्कः_तन्तवः) {
            ध्रुव int64_t i03 = ir / (ne02 * ne01);
            ध्रुव int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ध्रुव int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03);
            भिन्न *       dst_ptr = (भिन्न *) ((वर्ण *) dst_दत्त + i01 * nb1 + i02 * nb2 + i03 * nb3);

            भिन्न row_योग = partial_योग_खण्डिका(src_row, 0, (int32_t) ne00);
            atomic_store_f32(dst_ptr, row_योग * inv_ne00);
        }
        // Shire co-work
    } अन्यथा {
        पूर्णाङ्क shire_tid       = thread_id % shire_तन्तवः;
        पूर्णाङ्क तन्तवः_प्रत्येकम्_row = shire_तन्तवः / योग_पङ्क्तिकाः;
        पूर्णाङ्क my_row          = shire_tid / तन्तवः_प्रत्येकम्_row;
        पूर्णाङ्क local_tid       = shire_tid % तन्तवः_प्रत्येकम्_row;
        पूर्णाङ्क गणः_आधारः      = my_row * तन्तवः_प्रत्येकम्_row;

        यदि (my_row >= योग_पङ्क्तिकाः) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            प्रत्यागम 0;
        }

        int64_t i1 = my_row % ne01;
        int64_t i2 = (my_row / ne01) % ne02;
        int64_t i3 = my_row / (ne01 * ne02);

        ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
        भिन्न *       dst_ptr = (भिन्न *) ((वर्ण *) dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk size in elements, rounded up to a multiple of 8 so that every
        // thread's slice start stays 32B-aligned relative to src_ptr (which
        // matters for the vector path inside partial_sum_slice).
        int32_t chunk = ((int32_t) ne00 + तन्तवः_प्रत्येकम्_row - 1) / तन्तवः_प्रत्येकम्_row;
        chunk         = (chunk + 7) & ~7;
        यदि (chunk < 8) {
            chunk = 8;
        }

        int32_t my_start = local_tid * chunk;
        int32_t my_अन्तः   = my_start + chunk;
        यदि (my_अन्तः > (int32_t) ne00) {
            my_अन्तः = (int32_t) ne00;
        }
        यदि (my_start > (int32_t) ne00) {
            my_start = my_अन्तः = (int32_t) ne00;
        }

        पूर्णाङ्क workers = ((int32_t) ne00 + chunk - 1) / chunk;
        यदि (workers > तन्तवः_प्रत्येकम्_row) {
            workers = तन्तवः_प्रत्येकम्_row;
        }

        अचिह्नित दीर्घ saved_आवरणम्;
        __asm__ चञ्चल("mova.x.m %0" : "=r"(saved_आवरणम्));
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

        भिन्न partial_योग = partial_योग_खण्डिका(src_ptr, my_start, my_अन्तः);

        // Publish partial to shire-local L2 SCP slot (64B per slot, one per
        // hart). evict_to_l2 is required on the WRITER because scalar stores
        // land in L1D first; readers must also evict before reading.
        चञ्चल भिन्न * my_slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_योग;
        FENCE;
        evict_गन्तव्यम्_l2((ध्रुव शून्य *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        यदि (local_tid == 0) {
            // Reader-side evictions for every contributing peer slot.
            प्रति (पूर्णाङ्क t = 0; t < workers; t++) {
                चञ्चल भिन्न * slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) (गणः_आधारः + t) * 64);
                evict_गन्तव्यम्_l2((ध्रुव शून्य *) slot, 1, 64);
            }
            WAIT_CACHEOPS;

            भिन्न योग_योग = 0.0f;
            प्रति (पूर्णाङ्क t = 0; t < workers; t++) {
                चञ्चल भिन्न * slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) (गणः_आधारः + t) * 64);
                योग_योग += *slot;
            }
            atomic_store_f32(dst_ptr, योग_योग * inv_ne00);
        }

        __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_आवरणम्));
    }

    प्रत्यागम 0;
}
