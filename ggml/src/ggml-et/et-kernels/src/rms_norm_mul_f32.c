
// Fused RMS Norm + MUL F32 Kernel

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <string.h>

// Fused RMS norm + MUL kernel parameters structure
સંરચના ggml_et_rms_norm_mul_params {
    સંરચના ggml_tensor src0;  // F32 input tensor (to be normalized)
    સંરચના ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    સંરચના ggml_tensor dst;   // F32 output tensor
    દશાંશ              eps;   // Epsilon for numerical stability
};

સ્થિર અંતઃરેખ size_t tensor_bytes(અચલ સંરચના ggml_tensor * t) {
    પાછા (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

પૂર્ણાંક entry_point(સંરચના ggml_et_rms_norm_mul_params * params, શૂન્ય * env) {
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
        પાછા -1;  // Invalid pointer
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * src1 = &params->src1;
    સંરચના ggml_tensor * dst  = &params->dst;

    દશાંશ eps = params->eps;

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;  // Unsupported type combination
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * src1_data = (દશાંશ *) src1->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    જો (!src0_data || !src1_data || !dst_data) {
        પાછા -1;  // Null data pointer
    }

    જો (eps < 0.0f) {
        પાછા -1;  // Invalid epsilon
    }

    અચલ int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    અચલ int64_t ne1 = dst->ne[1];  // Dimension 1
    અચલ int64_t ne2 = dst->ne[2];  // Dimension 2
    અચલ int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    અચલ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    અચલ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    અચલ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    જો (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        પાછા -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    અચલ દશાંશ   inv_ne0       = et_fdiv(1.0f, (દશાંશ) (int32_t) ne0);
    અચલ int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    અચલ પૂર્ણાંક     shire_threads = SOC_MINIONS_PER_SHIRE * અંક_HARTS_PER_MINION;

    જો (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        માટે (int64_t i3 = 0; i3 < ne3; i3++) {
            માટે (int64_t i2 = 0; i2 < ne2; i2++) {
                માટે (int64_t i1 = thread_id; i1 < ne1; i1 += અંક_threads) {
                    અચલ દશાંશ * src_ptr =
                        (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    દશાંશ * dst_ptr = (દશાંશ *) ((અક્ષર *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    અચલ દશાંશ * wgt_ptr = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    unsigned long saved_mask;
                    __asm__ અસ્થિર("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ અસ્થિર("fbci.pi f10, 0" ::: "f10");
                    માટે (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ અસ્થિર(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    દશાંશ sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    અચલ દશાંશ scale = et_powf(sum * inv_ne0 + eps, -0.5f);
                    જો (!(scale > 0.0f)) {
                        __asm__ અસ્થિર("mova.m.x %0" ::"r"(saved_mask));
                        પાછા -1;
                    }

                    uint32_t scale_bits;
                    __asm__ અસ્થિર("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                    __asm__ અસ્થિર("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                    માટે (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ અસ્થિર(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(દશાંશ (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(અચલ દશાંશ (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ અસ્થિર("mova.m.x %0" ::"r"(saved_mask));
                }
            }
        }
    } નહીંતર {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        પૂર્ણાંક shire_tid       = thread_id % shire_threads;
        પૂર્ણાંક threads_per_row = shire_threads / total_rows;
        પૂર્ણાંક my_row          = shire_tid / threads_per_row;
        પૂર્ણાંક local_tid       = shire_tid % threads_per_row;
        પૂર્ણાંક group_base      = my_row * threads_per_row;

        // Excess threads within this shire
        જો (my_row >= total_rows) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            પાછા 0;
        }

        // Unflatten row index
        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        અચલ દશાંશ * src_ptr = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        દશાંશ *       dst_ptr = (દશાંશ *) ((અક્ષર *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        અચલ દશાંશ * wgt_ptr = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        અચલ int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        જો (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        જો (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ અસ્થિર("mova.x.m %0" : "=r"(saved_mask));
        __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ અસ્થિર("fbci.pi f10, 0" ::: "f10");
        માટે (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ અસ્થિર(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        દશાંશ partial_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        અસ્થિર દશાંશ * my_slot = (અસ્થિર દશાંશ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_to_l2((અચલ શૂન્ય *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        પૂર્ણાંક workers = threads_per_row < total_cls ? threads_per_row : total_cls;

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

        અચલ દશાંશ scale = et_powf(total_sum * inv_ne0 + eps, -0.5f);
        જો (!(scale > 0.0f)) {
            __asm__ અસ્થિર("mova.m.x %0" ::"r"(saved_mask));
            પાછા -1;
        }

        // Apply scale * weights to own chunk
        જો (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ અસ્થિર("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ અસ્થિર("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            માટે (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ અસ્થિર(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(દશાંશ (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(અચલ દશાંશ (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ અસ્થિર("mova.m.x %0" ::"r"(saved_mask));
    }

    પાછા 0;
}
