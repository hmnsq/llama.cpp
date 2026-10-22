
// Fused RMS Norm + MUL F32 Kernel

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <assert.h>
#ضمن <stdint.h>
#ضمن <string.h>

// Fused RMS norm + MUL kernel parameters structure
بنية ggml_et_rms_norm_mul_params {
    بنية ggml_tensor src0;  // F32 input tensor (to be normalized)
    بنية ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    بنية ggml_tensor dst;   // F32 output tensor
    عائم              eps;   // Epsilon for numerical stability
};

ساكن مضمن size_t tensor_بایټونه(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحيح ثبت_ټکی(بنية ggml_et_rms_norm_mul_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;

    عائم eps = params->eps;

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * src1_معلومات = (عائم *) src1->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    إذا (!src0_معلومات || !src1_معلومات || !dst_معلومات) {
        أرجع -1;  // Null data pointer
    }

    إذا (eps < 0.0f) {
        أرجع -1;  // Invalid epsilon
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
    إذا (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        أرجع -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    ثابت عائم   inv_ne0       = et_fdiv(1.0f, (عائم) (int32_t) ne0);
    ثابت int32_t ټول_لیکې    = (int32_t) (ne1 * ne2 * ne3);
    ثابت صحيح     shire_تارونه = SOC_MINIONS_فی_SHIRE * عدد_HARTS_فی_MINION;

    إذا (ټول_لیکې >= shire_تارونه) {
        // Row-parallel: each thread processes whole rows
        لكل (int64_t i3 = 0; i3 < ne3; i3++) {
            لكل (int64_t i2 = 0; i2 < ne2; i2++) {
                لكل (int64_t i1 = thread_id; i1 < ne1; i1 += عدد_تارونه) {
                    ثابت عائم * src_ptr =
                        (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_ptr = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    ثابت عائم * wgt_ptr = (ثابت عائم *) ((ثابت حرف *) src1_معلومات + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    بینښه اوږد saved_ماسک;
                    __asm__ متطاير("mova.x.m %0" : "=r"(saved_ماسک));
                    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ متطاير("fbci.pi f10, 0" ::: "f10");
                    لكل (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    عائم مجموعه;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(مجموعه)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ثابت عائم scale = et_powf(مجموعه * inv_ne0 + eps, -0.5f);
                    إذا (!(scale > 0.0f)) {
                        __asm__ متطاير("mova.m.x %0" ::"r"(saved_ماسک));
                        أرجع -1;
                    }

                    uint32_t scale_بټونه;
                    __asm__ متطاير("fmv.x.s %0, %1" : "=r"(scale_بټونه) : "f"(scale));
                    __asm__ متطاير("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_بټونه) : "f13");

                    لكل (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متطاير(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [پایله] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ثابت عائم (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ متطاير("mova.m.x %0" ::"r"(saved_ماسک));
                }
            }
        }
    } وإلا {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        صحيح shire_tid       = thread_id % shire_تارونه;
        صحيح تارونه_فی_لیکه = shire_تارونه / ټول_لیکې;
        صحيح my_لیکه          = shire_tid / تارونه_فی_لیکه;
        صحيح local_tid       = shire_tid % تارونه_فی_لیکه;
        صحيح ډله_بنسټ      = my_لیکه * تارونه_فی_لیکه;

        // Excess threads within this shire
        إذا (my_لیکه >= ټول_لیکې) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            أرجع 0;
        }

        // Unflatten row index
        int64_t i1 = my_لیکه % ne1;
        int64_t i2 = (my_لیکه / ne1) % ne2;
        int64_t i3 = my_لیکه / (ne1 * ne2);

        ثابت عائم * src_ptr = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
        عائم *       dst_ptr = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ثابت عائم * wgt_ptr = (ثابت عائم *) ((ثابت حرف *) src1_معلومات + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        ثابت int32_t elems_فی_cl   = 16;
        int32_t       ټول_cls      = ((int32_t) ne0 + elems_فی_cl - 1) / elems_فی_cl;
        int32_t       cls_فی_thread = (ټول_cls + تارونه_فی_لیکه - 1) / تارونه_فی_لیکه;
        int32_t       my_پیل       = local_tid * cls_فی_thread * elems_فی_cl;
        int32_t       my_پای_ټکی         = my_پیل + cls_فی_thread * elems_فی_cl;
        إذا (my_پای_ټکی > (int32_t) ne0) {
            my_پای_ټکی = (int32_t) ne0;
        }
        إذا (my_پیل >= (int32_t) ne0) {
            my_پیل = 0;
            my_پای_ټکی   = 0;
        }

        بینښه اوږد saved_ماسک;
        __asm__ متطاير("mova.x.m %0" : "=r"(saved_ماسک));
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ متطاير("fbci.pi f10, 0" ::: "f10");
        لكل (int32_t i0 = my_پیل; i0 < my_پای_ټکی; i0 += 8) {
            __asm__ متطاير(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        عائم partial_مجموعه;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_مجموعه)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        متطاير عائم * my_خونه = (متطاير عائم *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_خونه                 = partial_مجموعه;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_منزل_l2((ثابت فراغ *) my_خونه, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        صحيح workers = تارونه_فی_لیکه < ټول_cls ? تارونه_فی_لیکه : ټول_cls;

        لكل (صحيح t = 0; t < workers; t++) {
            متطاير عائم * خونه = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (ډله_بنسټ + t) * 64);
            evict_منزل_l2((ثابت فراغ *) خونه, 1, 64);
        }
        WAIT_CACHEOPS;

        عائم ټول_مجموعه = 0.0f;
        لكل (صحيح t = 0; t < workers; t++) {
            متطاير عائم * خونه = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (ډله_بنسټ + t) * 64);
            ټول_مجموعه += *خونه;
        }

        ثابت عائم scale = et_powf(ټول_مجموعه * inv_ne0 + eps, -0.5f);
        إذا (!(scale > 0.0f)) {
            __asm__ متطاير("mova.m.x %0" ::"r"(saved_ماسک));
            أرجع -1;
        }

        // Apply scale * weights to own chunk
        إذا (my_پیل < my_پای_ټکی) {
            uint32_t scale_بټونه;
            __asm__ متطاير("fmv.x.s %0, %1" : "=r"(scale_بټونه) : "f"(scale));
            __asm__ متطاير("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_بټونه) : "f13");

            لكل (int32_t i0 = my_پیل; i0 < my_پای_ټکی; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [پایله] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ثابت عائم (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ متطاير("mova.m.x %0" ::"r"(saved_ماسک));
    }

    أرجع 0;
}
