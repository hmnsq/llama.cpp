
// Fused RMS Norm + MUL F32 Kernel

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

// Fused RMS norm + MUL kernel parameters structure
ਬਣਤਰ ggml_et_rms_norm_mul_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor (to be normalized)
    ਬਣਤਰ ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor
    ਦਸ਼ਮਲਵ              eps;   // Epsilon for numerical stability
};

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t tensor_ਬਾਈਟਾਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_rms_norm_mul_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਦਸ਼ਮਲਵ eps = params->eps;

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;  // Unsupported type combination
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

    ਜੇ (eps < 0.0f) {
        ਵਾਪਸ -1;  // Invalid epsilon
    }

    ਅਚਲ int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    ਅਚਲ int64_t ne1 = dst->ne[1];  // Dimension 1
    ਅਚਲ int64_t ne2 = dst->ne[2];  // Dimension 2
    ਅਚਲ int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    ਅਚਲ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    ਅਚਲ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    ਅਚਲ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    ਜੇ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ਵਾਪਸ -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    ਅਚਲ ਦਸ਼ਮਲਵ   inv_ne0       = et_fdiv(1.0f, (ਦਸ਼ਮਲਵ) (int32_t) ne0);
    ਅਚਲ int32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ    = (int32_t) (ne1 * ne2 * ne3);
    ਅਚਲ ਪੂਰਨ     shire_ਧਾਗੇ = SOC_MINIONS_ਪ੍ਰਤੀ_SHIRE * ਅੰਕ_HARTS_ਪ੍ਰਤੀ_MINION;

    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ >= shire_ਧਾਗੇ) {
        // Row-parallel: each thread processes whole rows
        ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
            ਲਈ (int64_t i2 = 0; i2 < ne2; i2++) {
                ਲਈ (int64_t i1 = thread_id; i1 < ne1; i1 += ਅੰਕ_ਧਾਗੇ) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    ਅਚਲ ਦਸ਼ਮਲਵ * wgt_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
                    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
                    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ ਅਸਥਿਰ("fbci.pi f10, 0" ::: "f10");
                    ਲਈ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    ਦਸ਼ਮਲਵ ਜੋੜ;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(ਜੋੜ)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ਅਚਲ ਦਸ਼ਮਲਵ scale = et_powf(ਜੋੜ * inv_ne0 + eps, -0.5f);
                    ਜੇ (!(scale > 0.0f)) {
                        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
                        ਵਾਪਸ -1;
                    }

                    uint32_t scale_ਬਿੱਟ;
                    __asm__ ਅਸਥਿਰ("fmv.x.s %0, %1" : "=r"(scale_ਬਿੱਟ) : "f"(scale));
                    __asm__ ਅਸਥਿਰ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_ਬਿੱਟ) : "f13");

                    ਲਈ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [ਨਤੀਜਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        ਪੂਰਨ shire_tid       = thread_id % shire_ਧਾਗੇ;
        ਪੂਰਨ ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = shire_ਧਾਗੇ / ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        ਪੂਰਨ my_ਪੰਕਤੀ          = shire_tid / ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਪੂਰਨ local_tid       = shire_tid % ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਪੂਰਨ ਸਮੂਹ_ਅਧਾਰ      = my_ਪੰਕਤੀ * ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

        // Excess threads within this shire
        ਜੇ (my_ਪੰਕਤੀ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            ਵਾਪਸ 0;
        }

        // Unflatten row index
        int64_t i1 = my_ਪੰਕਤੀ % ne1;
        int64_t i2 = (my_ਪੰਕਤੀ / ne1) % ne2;
        int64_t i3 = my_ਪੰਕਤੀ / (ne1 * ne2);

        ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ਦਸ਼ਮਲਵ *       dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ਅਚਲ ਦਸ਼ਮਲਵ * wgt_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        ਅਚਲ int32_t elems_ਪ੍ਰਤੀ_cl   = 16;
        int32_t       ਕੁੱਲ_cls      = ((int32_t) ne0 + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;
        int32_t       cls_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cls + ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ - 1) / ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        int32_t       my_ਸ਼ੁਰੂ       = local_tid * cls_ਪ੍ਰਤੀ_thread * elems_ਪ੍ਰਤੀ_cl;
        int32_t       my_ਸਿਰਾ         = my_ਸ਼ੁਰੂ + cls_ਪ੍ਰਤੀ_thread * elems_ਪ੍ਰਤੀ_cl;
        ਜੇ (my_ਸਿਰਾ > (int32_t) ne0) {
            my_ਸਿਰਾ = (int32_t) ne0;
        }
        ਜੇ (my_ਸ਼ੁਰੂ >= (int32_t) ne0) {
            my_ਸ਼ੁਰੂ = 0;
            my_ਸਿਰਾ   = 0;
        }

        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ ਅਸਥਿਰ("fbci.pi f10, 0" ::: "f10");
        ਲਈ (int32_t i0 = my_ਸ਼ੁਰੂ; i0 < my_ਸਿਰਾ; i0 += 8) {
            __asm__ ਅਸਥਿਰ(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        ਦਸ਼ਮਲਵ partial_ਜੋੜ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_ਜੋੜ)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        ਅਸਥਿਰ ਦਸ਼ਮਲਵ * my_ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_ਥਾਂ                 = partial_ਜੋੜ;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) my_ਥਾਂ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        ਪੂਰਨ workers = ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ < ਕੁੱਲ_cls ? ਧਾਗੇ_ਪ੍ਰਤੀ_ਪੰਕਤੀ : ਕੁੱਲ_cls;

        ਲਈ (ਪੂਰਨ t = 0; t < workers; t++) {
            ਅਸਥਿਰ ਦਸ਼ਮਲਵ * ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local((uint64_t) (ਸਮੂਹ_ਅਧਾਰ + t) * 64);
            evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) ਥਾਂ, 1, 64);
        }
        WAIT_CACHEOPS;

        ਦਸ਼ਮਲਵ ਕੁੱਲ_ਜੋੜ = 0.0f;
        ਲਈ (ਪੂਰਨ t = 0; t < workers; t++) {
            ਅਸਥਿਰ ਦਸ਼ਮਲਵ * ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local((uint64_t) (ਸਮੂਹ_ਅਧਾਰ + t) * 64);
            ਕੁੱਲ_ਜੋੜ += *ਥਾਂ;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ scale = et_powf(ਕੁੱਲ_ਜੋੜ * inv_ne0 + eps, -0.5f);
        ਜੇ (!(scale > 0.0f)) {
            __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
            ਵਾਪਸ -1;
        }

        // Apply scale * weights to own chunk
        ਜੇ (my_ਸ਼ੁਰੂ < my_ਸਿਰਾ) {
            uint32_t scale_ਬਿੱਟ;
            __asm__ ਅਸਥਿਰ("fmv.x.s %0, %1" : "=r"(scale_ਬਿੱਟ) : "f"(scale));
            __asm__ ਅਸਥਿਰ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_ਬਿੱਟ) : "f13");

            ਲਈ (int32_t i0 = my_ਸ਼ੁਰੂ; i0 < my_ਸਿਰਾ; i0 += 8) {
                __asm__ ਅਸਥਿਰ(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [ਨਤੀਜਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
    }

    ਵਾਪਸ 0;
}
