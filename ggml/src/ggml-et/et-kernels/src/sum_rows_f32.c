//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_sum_rows_params {
    بنية ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    بنية ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

صحيح entry_point(بنية ggml_et_sum_rows_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    𐤀𐤌 (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    𐤀𐤌 (thread_id < 0) {
        أرجع 0;
    }

    𐤀𐤌 (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    𐤀𐤌 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    𐤀𐤌 (!src0_data || !dst_data) {
        أرجع -1;
    }


    ثابت int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت size_t nb01 = src0->nb[1];
    ثابت size_t nb02 = src0->nb[2];
    ثابت size_t nb03 = src0->nb[3];

    ثابت size_t nb1 = dst->nb[1];
    ثابت size_t nb2 = dst->nb[2];
    ثابت size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    ثابت int64_t total_rows = ne01 * ne02 * ne03;

    لكل (int64_t ir = thread_id; ir < total_rows; ir += num_threads) {
        ثابت int64_t i03 = ir / (ne02 * ne01);
        ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        عائم *       dst_ptr = (عائم *) ((حرف *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        عائم zero = 0.0f;
        __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        لكل (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ متطاير(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        عائم row_sum;
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

    أرجع 0;
}
