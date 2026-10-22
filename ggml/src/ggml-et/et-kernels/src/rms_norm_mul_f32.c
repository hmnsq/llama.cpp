
// Fused RMS Norm + MUL F32 Kernel

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <assert.h>
#شامل <stdint.h>
#شامل <string.h>

// Fused RMS norm + MUL kernel parameters structure
ساخت ggml_et_rms_norm_mul_params {
    ساخت ggml_tensor src0;  // F32 input tensor (to be normalized)
    ساخت ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    ساخت ggml_tensor dst;   // F32 output tensor
    اعشاری              eps;   // Epsilon for numerical stability
};

ساکن اندرونی size_t tensor_بائٹیں(ثابت ساخت ggml_tensor * t) {
    واپس (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحیح اندراج_نقطہ(ساخت ggml_et_rms_norm_mul_params * params, خالی * env) {
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
        واپس -1;  // Invalid pointer
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * dst  = &params->dst;

    اعشاری eps = params->eps;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported type combination
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * src1_ڈیٹا = (اعشاری *) src1->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

    اگر (eps < 0.0f) {
        واپس -1;  // Invalid epsilon
    }

    ثابت int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    ثابت int64_t ne1 = dst->ne[1];  // Dimension 1
    ثابت int64_t ne2 = dst->ne[2];  // Dimension 2
    ثابت int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    ثابت size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    ثابت size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        واپس -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    ثابت اعشاری   inv_ne0       = et_fdiv(1.0f, (اعشاری) (int32_t) ne0);
    ثابت int32_t کل_صفیں    = (int32_t) (ne1 * ne2 * ne3);
    ثابت صحیح     shire_دھاگے = SOC_MINIONS_فی_SHIRE * عدد_HARTS_فی_MINION;

    اگر (کل_صفیں >= shire_دھاگے) {
        // Row-parallel: each thread processes whole rows
        برائے (int64_t i3 = 0; i3 < ne3; i3++) {
            برائے (int64_t i2 = 0; i2 < ne2; i2++) {
                برائے (int64_t i1 = thread_id; i1 < ne1; i1 += عدد_دھاگے) {
                    ثابت اعشاری * src_ptr =
                        (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    ثابت اعشاری * wgt_ptr = (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    غیرعلامتی طویل saved_نقاب;
                    __asm__ متلون("mova.x.m %0" : "=r"(saved_نقاب));
                    __asm__ متلون("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ متلون("fbci.pi f10, 0" ::: "f10");
                    برائے (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    اعشاری مجموع;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(مجموع)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ثابت اعشاری scale = et_powf(مجموع * inv_ne0 + eps, -0.5f);
                    اگر (!(scale > 0.0f)) {
                        __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
                        واپس -1;
                    }

                    uint32_t scale_بٹ;
                    __asm__ متلون("fmv.x.s %0, %1" : "=r"(scale_بٹ) : "f"(scale));
                    __asm__ متلون("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_بٹ) : "f13");

                    برائے (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متلون(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [نتیجہ] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ثابت اعشاری (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
                }
            }
        }
    } ورنہ {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        صحیح shire_tid       = thread_id % shire_دھاگے;
        صحیح دھاگے_فی_صف = shire_دھاگے / کل_صفیں;
        صحیح my_صف          = shire_tid / دھاگے_فی_صف;
        صحیح local_tid       = shire_tid % دھاگے_فی_صف;
        صحیح گروہ_بنیاد      = my_صف * دھاگے_فی_صف;

        // Excess threads within this shire
        اگر (my_صف >= کل_صفیں) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            واپس 0;
        }

        // Unflatten row index
        int64_t i1 = my_صف % ne1;
        int64_t i2 = (my_صف / ne1) % ne2;
        int64_t i3 = my_صف / (ne1 * ne2);

        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
        اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ثابت اعشاری * wgt_ptr = (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        ثابت int32_t elems_فی_cl   = 16;
        int32_t       کل_cls      = ((int32_t) ne0 + elems_فی_cl - 1) / elems_فی_cl;
        int32_t       cls_فی_thread = (کل_cls + دھاگے_فی_صف - 1) / دھاگے_فی_صف;
        int32_t       my_آغاز       = local_tid * cls_فی_thread * elems_فی_cl;
        int32_t       my_انتہا         = my_آغاز + cls_فی_thread * elems_فی_cl;
        اگر (my_انتہا > (int32_t) ne0) {
            my_انتہا = (int32_t) ne0;
        }
        اگر (my_آغاز >= (int32_t) ne0) {
            my_آغاز = 0;
            my_انتہا   = 0;
        }

        غیرعلامتی طویل saved_نقاب;
        __asm__ متلون("mova.x.m %0" : "=r"(saved_نقاب));
        __asm__ متلون("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ متلون("fbci.pi f10, 0" ::: "f10");
        برائے (int32_t i0 = my_آغاز; i0 < my_انتہا; i0 += 8) {
            __asm__ متلون(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        اعشاری partial_مجموع;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_مجموع)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        متلون اعشاری * my_خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_خانہ                 = partial_مجموع;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_مقصد_l2((ثابت خالی *) my_خانہ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        صحیح workers = دھاگے_فی_صف < کل_cls ? دھاگے_فی_صف : کل_cls;

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

        ثابت اعشاری scale = et_powf(کل_مجموع * inv_ne0 + eps, -0.5f);
        اگر (!(scale > 0.0f)) {
            __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
            واپس -1;
        }

        // Apply scale * weights to own chunk
        اگر (my_آغاز < my_انتہا) {
            uint32_t scale_بٹ;
            __asm__ متلون("fmv.x.s %0, %1" : "=r"(scale_بٹ) : "f"(scale));
            __asm__ متلون("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_بٹ) : "f13");

            برائے (int32_t i0 = my_آغاز; i0 < my_انتہا; i0 += 8) {
                __asm__ متلون(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [نتیجہ] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ثابت اعشاری (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
    }

    واپس 0;
}
