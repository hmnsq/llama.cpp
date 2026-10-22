//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_sum_rows_params {
    संरचना ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    संरचना ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

पूर्णाङ्क entry_point(संरचना ggml_et_sum_rows_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }


    अचर int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    अचर int64_t ne01 = src0->ne[1];
    अचर int64_t ne02 = src0->ne[2];
    अचर int64_t ne03 = src0->ne[3];

    अचर माप_प्रकार nb01 = src0->nb[1];
    अचर माप_प्रकार nb02 = src0->nb[2];
    अचर माप_प्रकार nb03 = src0->nb[3];

    अचर माप_प्रकार nb1 = dst->nb[1];
    अचर माप_प्रकार nb2 = dst->nb[2];
    अचर माप_प्रकार nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    अचर int64_t total_rows = ne01 * ne02 * ne03;

    लागि (int64_t ir = thread_id; ir < total_rows; ir += अङ्क_threads) {
        अचर int64_t i03 = ir / (ne02 * ne01);
        अचर int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        अचर int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        भिन्न *       dst_ptr = (भिन्न *) ((वर्ण *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        भिन्न zero = 0.0f;
        __asm__ चञ्चल("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        लागि (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ चञ्चल(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(अचर भिन्न (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        भिन्न row_sum;
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

    फर्काऊ 0;
}
