//******************************************************************************
// RMS Norm F32 Kernel
// Root Mean Square normalization: y[i] = x[i] / sqrt(mean(x^2) + eps)
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <string.h>

// RMS norm kernel parameters structure
கட்டமைப்பு ggml_et_rms_norm_params {
    கட்டமைப்பு ggml_tensor src0;  // F32 input tensor
    கட்டமைப்பு ggml_tensor dst;   // F32 output tensor
    தசம              eps;   // Epsilon parameter for numerical stability
};

முழு entry_point(கட்டமைப்பு ggml_et_rms_norm_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;  // Invalid pointer
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;
    தசம                eps  = params->eps;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;  // Unsupported type combination
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;  // Null data pointer
    }


    எனில் (eps < 0.0f) {
        திரும்பு -1;  // Invalid epsilon
    }

    மாறிலி int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    மாறிலி int64_t ne1 = dst->ne[1];  // Dimension 1
    மாறிலி int64_t ne2 = dst->ne[2];  // Dimension 2
    மாறிலி int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    மாறிலி size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    மாறிலி size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    எனில் (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        திரும்பு -1;  // Shape mismatch
    }

    // RMS norm processes rows independently
    // Parallelize across rows using simple striding
    // TODO: ensure lines don't cross cache lines
    // Precompute reciprocal of row length (constant across all rows)
    மாறிலி தசம   inv_ne0    = et_fdiv(1.0f, (தசம) (int32_t) ne0);
    மாறிலி int32_t total_rows = (int32_t) (ne1 * ne2 * ne3);

    // Intra-row cooperation only works within a single shire (barrier + L2SCP
    // are shire-local). Use per-shire thread count for the threshold.
    மாறிலி முழு shire_threads = SOC_MINIONS_PER_SHIRE * இலக்கம்_HARTS_PER_MINION;  // 64

    எனில் (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        க்கு (int64_t i3 = 0; i3 < ne3; i3++) {
            க்கு (int64_t i2 = 0; i2 < ne2; i2++) {
                க்கு (int64_t i1 = thread_id; i1 < ne1; i1 += இலக்கம்_threads) {
                    மாறிலி தசம * src_ptr =
                        (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    தசம * dst_ptr = (தசம *) ((எழுத்து *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Set mask to enable all 8 vector lanes
                    unsigned long saved_mask;
                    __asm__ நிலையற்ற("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");

                    // Step 1: Compute sum of squares using 8-wide vectors
                    __asm__ நிலையற்ற("fbci.pi f10, 0" ::: "f10");

                    க்கு (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ நிலையற்ற(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(மாறிலி தசம (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    // Horizontal reduce
                    தசம sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    // Step 2: scale = rsqrt(mean + eps)
                    மாறிலி தசம scale = et_powf(sum * inv_ne0 + eps, -0.5f);

                    எனில் (!(scale > 0.0f)) {
                        __asm__ நிலையற்ற("mova.m.x %0" ::"r"(saved_mask));
                        திரும்பு -1;
                    }

                    // Step 3: Apply scaling: broadcast scale once, reuse across loop
                    uint32_t scale_bits;
                    __asm__ நிலையற்ற("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                    __asm__ நிலையற்ற("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                    க்கு (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ நிலையற்ற(
                            "flw.ps f12, %[x_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(தசம (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(மாறிலி தசம (*)[8]) & src_ptr[i0])
                            : "f12", "f14");
                    }

                    __asm__ நிலையற்ற("mova.m.x %0" ::"r"(saved_mask));
                }
            }
        }
    } இல்லையெனில் {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        முழு shire_tid       = thread_id % shire_threads;  // 0..63 within this shire
        முழு threads_per_row = shire_threads / total_rows;
        முழு my_row          = shire_tid / threads_per_row;
        முழு local_tid       = shire_tid % threads_per_row;
        முழு group_base      = my_row * threads_per_row;  // shire-local group base

        // Excess threads within this shire, barrier and leave
        எனில் (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            திரும்பு 0;
        }

        // Unflatten row index
        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        மாறிலி தசம * src_ptr = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        தசம *       dst_ptr = (தசம *) ((எழுத்து *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        மாறிலி int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        எனில் (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        எனில் (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ நிலையற்ற("mova.x.m %0" : "=r"(saved_mask));
        __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");

        // Phase 1: each thread computes partial sum of squares on its chunk
        __asm__ நிலையற்ற("fbci.pi f10, 0" ::: "f10");
        க்கு (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ நிலையற்ற(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(மாறிலி தசம (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        // Horizontal reduce to scalar
        தசம partial_sum;
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
        நிலையற்ற தசம * my_slot = (நிலையற்ற தசம *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((மாறிலி வெறுமை *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: ALL threads read partial sums, compute scale, apply to own chunk.
        // Each thread independently reduces to avoid a second barrier.
        முழு workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        // Evict stale L1D entries for worker slots
        க்கு (முழு t = 0; t < workers; t++) {
            நிலையற்ற தசம * slot = (நிலையற்ற தசம *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((மாறிலி வெறுமை *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        // Every thread reduces the same partial sums -> same scale
        தசம total_sum = 0.0f;
        க்கு (முழு t = 0; t < workers; t++) {
            நிலையற்ற தசம * slot = (நிலையற்ற தசம *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        மாறிலி தசம scale = et_powf(total_sum * inv_ne0 + eps, -0.5f);
        எனில் (!(scale > 0.0f)) {
            __asm__ நிலையற்ற("mova.m.x %0" ::"r"(saved_mask));
            திரும்பு -1;
        }

        // Each thread applies scale to its own chunk only
        எனில் (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ நிலையற்ற("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ நிலையற்ற("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            க்கு (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ நிலையற்ற(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(தசம (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(மாறிலி தசம (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ நிலையற்ற("mova.m.x %0" ::"r"(saved_mask));
    }

    திரும்பு 0;
}
