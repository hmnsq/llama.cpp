//******************************************************************************
// RMS Norm F32 Kernel
// Root Mean Square normalization: y[i] = x[i] / sqrt(mean(x^2) + eps)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <assert.h>
#شامل <stdint.h>
#شامل <string.h>

// RMS norm kernel parameters structure
ساخت ggml_et_rms_norm_params {
    ساخت ggml_tensor src0;  // F32 input tensor
    ساخت ggml_tensor dst;   // F32 output tensor
    اعشاری              eps;   // Epsilon parameter for numerical stability
};

صحیح اندراج_نقطہ(ساخت ggml_et_rms_norm_params * params, خالی * env) {
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

    ثابت int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    ثابت int64_t ne1 = dst->ne[1];  // Dimension 1
    ثابت int64_t ne2 = dst->ne[2];  // Dimension 2
    ثابت int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    ثابت size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        واپس -1;  // Shape mismatch
    }

    // RMS norm processes rows independently
    // Parallelize across rows using simple striding
    // TODO: ensure lines don't cross cache lines
    // Precompute reciprocal of row length (constant across all rows)
    ثابت اعشاری   inv_ne0    = et_fdiv(1.0f, (اعشاری) (int32_t) ne0);
    ثابت int32_t کل_صفیں = (int32_t) (ne1 * ne2 * ne3);

    // Intra-row cooperation only works within a single shire (barrier + L2SCP
    // are shire-local). Use per-shire thread count for the threshold.
    ثابت صحیح shire_دھاگے = SOC_MINIONS_فی_SHIRE * عدد_HARTS_فی_MINION;  // 64

    اگر (کل_صفیں >= shire_دھاگے) {
        // Row-parallel: each thread processes whole rows
        برائے (int64_t i3 = 0; i3 < ne3; i3++) {
            برائے (int64_t i2 = 0; i2 < ne2; i2++) {
                برائے (int64_t i1 = thread_id; i1 < ne1; i1 += عدد_دھاگے) {
                    ثابت اعشاری * src_ptr =
                        (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Set mask to enable all 8 vector lanes
                    غیرعلامتی طویل saved_نقاب;
                    __asm__ متلون("mova.x.m %0" : "=r"(saved_نقاب));
                    __asm__ متلون("mov.m.x m0, x0, 0xFF");

                    // Step 1: Compute sum of squares using 8-wide vectors
                    __asm__ متلون("fbci.pi f10, 0" ::: "f10");

                    برائے (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    // Horizontal reduce
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

                    // Step 2: scale = rsqrt(mean + eps)
                    ثابت اعشاری scale = et_powf(مجموع * inv_ne0 + eps, -0.5f);

                    اگر (!(scale > 0.0f)) {
                        __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
                        واپس -1;
                    }

                    // Step 3: Apply scaling: broadcast scale once, reuse across loop
                    uint32_t scale_بٹ;
                    __asm__ متلون("fmv.x.s %0, %1" : "=r"(scale_بٹ) : "f"(scale));
                    __asm__ متلون("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_بٹ) : "f13");

                    برائے (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متلون(
                            "flw.ps f12, %[x_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [نتیجہ] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                            : "f12", "f14");
                    }

                    __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
                }
            }
        }
    } ورنہ {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        صحیح shire_tid       = thread_id % shire_دھاگے;  // 0..63 within this shire
        صحیح دھاگے_فی_صف = shire_دھاگے / کل_صفیں;
        صحیح my_صف          = shire_tid / دھاگے_فی_صف;
        صحیح local_tid       = shire_tid % دھاگے_فی_صف;
        صحیح گروہ_بنیاد      = my_صف * دھاگے_فی_صف;  // shire-local group base

        // Excess threads within this shire, barrier and leave
        اگر (my_صف >= کل_صفیں) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            واپس 0;
        }

        // Unflatten row index
        int64_t i1 = my_صف % ne1;
        int64_t i2 = (my_صف / ne1) % ne2;
        int64_t i3 = my_صف / (ne1 * ne2);

        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
        اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);

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

        // Phase 1: each thread computes partial sum of squares on its chunk
        __asm__ متلون("fbci.pi f10, 0" ::: "f10");
        برائے (int32_t i0 = my_آغاز; i0 < my_انتہا; i0 += 8) {
            __asm__ متلون(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        // Horizontal reduce to scalar
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
        FENCE;
        evict_مقصد_l2((ثابت خالی *) my_خانہ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: ALL threads read partial sums, compute scale, apply to own chunk.
        // Each thread independently reduces to avoid a second barrier.
        صحیح workers = دھاگے_فی_صف < کل_cls ? دھاگے_فی_صف : کل_cls;

        // Evict stale L1D entries for worker slots
        برائے (صحیح t = 0; t < workers; t++) {
            متلون اعشاری * خانہ = (متلون اعشاری *) et_shire_l2scp_local((uint64_t) (گروہ_بنیاد + t) * 64);
            evict_مقصد_l2((ثابت خالی *) خانہ, 1, 64);
        }
        WAIT_CACHEOPS;

        // Every thread reduces the same partial sums -> same scale
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

        // Each thread applies scale to its own chunk only
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
