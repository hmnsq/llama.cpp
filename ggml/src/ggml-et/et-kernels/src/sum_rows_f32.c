//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_योग_rows_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    काष्ठा ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

पूर्णांक entry_point(काष्ठा ggml_et_योग_rows_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }


    स्थिर int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    स्थिर माप_प्रकार nb01 = src0->nb[1];
    स्थिर माप_प्रकार nb02 = src0->nb[2];
    स्थिर माप_प्रकार nb03 = src0->nb[3];

    स्थिर माप_प्रकार nb1 = dst->nb[1];
    स्थिर माप_प्रकार nb2 = dst->nb[2];
    स्थिर माप_प्रकार nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    स्थिर int64_t योग_rows = ne01 * ne02 * ne03;

    क्रम (int64_t ir = thread_id; ir < योग_rows; ir += अंक_threads) {
        स्थिर int64_t i03 = ir / (ne02 * ne01);
        स्थिर int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        स्थिर int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);
        भग्न *       dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        भग्न zero = 0.0f;
        __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        क्रम (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ अस्थिर(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        भग्न row_योग;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(row_योग)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, row_योग);
    }

    वापस 0;
}
