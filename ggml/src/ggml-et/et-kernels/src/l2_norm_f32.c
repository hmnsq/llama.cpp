//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <assert.h>
#شامل <stdint.h>
#شامل <string.h>

// L2 Norm kernel parameters structure
ساخت ggml_et_l2_norm_params {
    ساخت ggml_tensor src0;  // F32 input tensor
    ساخت ggml_tensor dst;   // F32 output tensor
    اعشاری              eps;   // Epsilon parameter for numerical stability
};

صحیح اندراج_نقطہ(ساخت ggml_et_l2_norm_params * params, خالی * env) {
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
    ساخت ggml_tensor * dst  = &params->dst;
    اعشاری                eps  = params->eps;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported type combination
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

    اگر (eps < 0.0f) {
        واپس -1;  // Invalid epsilon
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    (خالی) nb0;
    (خالی) nb00;

    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        واپس -1;  // Shape mismatch
    }

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

                    اعشاری zero = 0.0f;
                    __asm__ متلون("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    برائے (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    اعشاری مجموع_sq;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(مجموع_sq)::"t0", "f1", "f2", "f3", "f4", "f5");

                    اعشاری l2_norm = et_powf(مجموع_sq, 0.5f);
                    اگر (l2_norm < eps) {
                        l2_norm = eps;
                    }
                    ثابت اعشاری scale = et_fdiv(1.0f, l2_norm);

                    برائے (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[scale_ptr]\n"
                            "fmul.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            : [نتیجہ] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f11", "f12", "f13");
                    }
                }
            }
        }
    } ورنہ {
        // Intra-row: threads within each shire cooperate via L2 SCP
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

        int64_t i1 = my_صف % ne1;
        int64_t i2 = (my_صف / ne1) % ne2;
        int64_t i3 = my_صف / (ne1 * ne2);

        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
        اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);

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

        // Phase 1: partial sum of squares
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

        // Phase 2: L2SCP exchange
        متلون اعشاری * my_خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_خانہ                 = partial_مجموع;
        FENCE;
        evict_مقصد_l2((ثابت خالی *) my_خانہ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads reduce + apply scale to own chunk
        صحیح workers = دھاگے_فی_صف < کل_cls ? دھاگے_فی_صف : کل_cls;

        برائے (صحیح t = 0; t < workers; t++) {
            متلون اعشاری * خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) (گروہ_بنیاد + t) * 64);
            evict_مقصد_l2((ثابت خالی *) خانہ, 1, 64);
        }
        WAIT_CACHEOPS;

        اعشاری کل_مجموع_sq = 0.0f;
        برائے (صحیح t = 0; t < workers; t++) {
            متلون اعشاری * خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) (گروہ_بنیاد + t) * 64);
            کل_مجموع_sq += *خانہ;
        }

        اعشاری l2_norm = et_powf(کل_مجموع_sq, 0.5f);
        اگر (l2_norm < eps) {
            l2_norm = eps;
        }
        ثابت اعشاری scale = et_fdiv(1.0f, l2_norm);

        اگر (my_آغاز < my_انتہا) {
            uint32_t scale_بٹ;
            __asm__ متلون("fmv.x.s %0, %1" : "=r"(scale_بٹ) : "f"(scale));
            __asm__ متلون("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_بٹ) : "f13");

            برائے (int32_t i0 = my_آغاز; i0 < my_انتہا; i0 += 8) {
                __asm__ متلون(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [نتیجہ] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
    }

    واپس 0;
}
