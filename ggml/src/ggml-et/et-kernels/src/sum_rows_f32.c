//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_sum_rows_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_sum_rows_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }


    ಧ್ರುವ int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    ಧ್ರುವ int64_t ne01 = src0->ne[1];
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    ಧ್ರುವ size_t nb01 = src0->nb[1];
    ಧ್ರುವ size_t nb02 = src0->nb[2];
    ಧ್ರುವ size_t nb03 = src0->nb[3];

    ಧ್ರುವ size_t nb1 = dst->nb[1];
    ಧ್ರುವ size_t nb2 = dst->nb[2];
    ಧ್ರುವ size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    ಧ್ರುವ int64_t total_rows = ne01 * ne02 * ne03;

    ಪ್ರತಿ (int64_t ir = thread_id; ir < total_rows; ir += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t i03 = ir / (ne02 * ne01);
        ಧ್ರುವ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ಧ್ರುವ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ಭಿನ್ನ *       dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        ಭಿನ್ನ zero = 0.0f;
        __asm__ ಚಞ್ಚಲ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ಪ್ರತಿ (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ ಚಞ್ಚಲ(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        ಭಿನ್ನ row_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(row_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, row_sum);
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
