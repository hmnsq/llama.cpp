//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

// L2 Norm kernel parameters structure
কাঠামো ggml_et_l2_norm_params {
    কাঠামো ggml_tensor src0;  // F32 input tensor
    কাঠামো ggml_tensor dst;   // F32 output tensor
    দশমিক              eps;   // Epsilon parameter for numerical stability
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_l2_norm_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;  // Invalid pointer
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;
    দশমিক                eps  = params->eps;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;  // Unsupported type combination
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;  // Null data pointer
    }

    যদি (eps < 0.0f) {
        ফেরত -1;  // Invalid epsilon
    }

    ধ্রুবক int64_t ne0 = dst->ne[0];
    ধ্রুবক int64_t ne1 = dst->ne[1];
    ধ্রুবক int64_t ne2 = dst->ne[2];
    ধ্রুবক int64_t ne3 = dst->ne[3];

    ধ্রুবক size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ধ্রুবক size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    (শূন্য) nb0;
    (শূন্য) nb00;

    যদি (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ফেরত -1;  // Shape mismatch
    }

    ধ্রুবক int32_t যোগ_rows    = (int32_t) (ne1 * ne2 * ne3);
    ধ্রুবক পূর্ণসংখ্যা     shire_threads = SOC_MINIONS_PER_SHIRE * অঙ্ক_HARTS_PER_MINION;

    যদি (যোগ_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        জন্য (int64_t i3 = 0; i3 < ne3; i3++) {
            জন্য (int64_t i2 = 0; i2 < ne2; i2++) {
                জন্য (int64_t i1 = thread_id; i1 < ne1; i1 += অঙ্ক_threads) {
                    ধ্রুবক দশমিক * src_ptr =
                        (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    দশমিক * dst_ptr = (দশমিক *) ((অক্ষর *) dst_তথ্য + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    দশমিক zero = 0.0f;
                    __asm__ অস্থির("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    জন্য (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ অস্থির(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    দশমিক যোগ_sq;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(যোগ_sq)::"t0", "f1", "f2", "f3", "f4", "f5");

                    দশমিক l2_norm = et_powf(যোগ_sq, 0.5f);
                    যদি (l2_norm < eps) {
                        l2_norm = eps;
                    }
                    ধ্রুবক দশমিক scale = et_fdiv(1.0f, l2_norm);

                    জন্য (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ অস্থির(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[scale_ptr]\n"
                            "fmul.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            : [ফলাফল] "=m"(*(দশমিক (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f11", "f12", "f13");
                    }
                }
            }
        }
    } অন্যথা {
        // Intra-row: threads within each shire cooperate via L2 SCP
        পূর্ণসংখ্যা shire_tid       = thread_id % shire_threads;
        পূর্ণসংখ্যা threads_per_row = shire_threads / যোগ_rows;
        পূর্ণসংখ্যা my_row          = shire_tid / threads_per_row;
        পূর্ণসংখ্যা local_tid       = shire_tid % threads_per_row;
        পূর্ণসংখ্যা group_base      = my_row * threads_per_row;

        যদি (my_row >= যোগ_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ফেরত 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ধ্রুবক দশমিক * src_ptr = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i3 * nb03 + i2 * nb02 + i1 * nb01);
        দশমিক *       dst_ptr = (দশমিক *) ((অক্ষর *) dst_তথ্য + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ধ্রুবক int32_t elems_per_cl   = 16;
        int32_t       যোগ_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (যোগ_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        যদি (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        যদি (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ অস্থির("mova.x.m %0" : "=r"(saved_mask));
        __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares
        __asm__ অস্থির("fbci.pi f10, 0" ::: "f10");
        জন্য (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ অস্থির(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        দশমিক partial_যোগ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_যোগ)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: L2SCP exchange
        অস্থির দশমিক * my_slot = (অস্থির দশমিক *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_যোগ;
        FENCE;
        evict_to_l2((ধ্রুবক শূন্য *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads reduce + apply scale to own chunk
        পূর্ণসংখ্যা workers = threads_per_row < যোগ_cls ? threads_per_row : যোগ_cls;

        জন্য (পূর্ণসংখ্যা t = 0; t < workers; t++) {
            অস্থির দশমিক * slot = (অস্থির দশমিক *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ধ্রুবক শূন্য *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        দশমিক যোগ_যোগ_sq = 0.0f;
        জন্য (পূর্ণসংখ্যা t = 0; t < workers; t++) {
            অস্থির দশমিক * slot = (অস্থির দশমিক *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            যোগ_যোগ_sq += *slot;
        }

        দশমিক l2_norm = et_powf(যোগ_যোগ_sq, 0.5f);
        যদি (l2_norm < eps) {
            l2_norm = eps;
        }
        ধ্রুবক দশমিক scale = et_fdiv(1.0f, l2_norm);

        যদি (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ অস্থির("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ অস্থির("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            জন্য (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ অস্থির(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [ফলাফল] "=m"(*(দশমিক (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ অস্থির("mova.m.x %0" ::"r"(saved_mask));
    }

    ফেরত 0;
}
