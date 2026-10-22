//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_مجموع_صفیں_params {
    ساخت ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ساخت ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

صحیح اندراج_نقطہ(ساخت ggml_et_مجموع_صفیں_params * params, خالی * env) {
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
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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
    ثابت int64_t کل_صفیں = ne01 * ne02 * ne03;

    برائے (int64_t ir = thread_id; ir < کل_صفیں; ir += عدد_دھاگے) {
        ثابت int64_t i03 = ir / (ne02 * ne01);
        ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03);
        اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        اعشاری zero = 0.0f;
        __asm__ متلون("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        برائے (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ متلون(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_صف[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        اعشاری صف_مجموع;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(صف_مجموع)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, صف_مجموع);
    }

    واپس 0;
}
