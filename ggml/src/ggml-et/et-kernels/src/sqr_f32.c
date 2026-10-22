//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ساخت ggml_et_sqr_params {
    ساخت ggml_tensor src0;  // F32 input tensor
    ساخت ggml_tensor dst;   // F32 output tensor
};

صحیح اندراج_نقطہ(ساخت ggml_et_sqr_params * params, خالی * env) {
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

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported type combination
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ثابت int64_t کل_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    ثابت int64_t cl_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
    int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
    اگر (cl_انتہا > کل_cachelines) {
        cl_انتہا = کل_cachelines;
    }

    اگر (cl_آغاز >= کل_cachelines) {
        واپس 0;
    }

    ثابت int64_t elem_آغاز = cl_آغاز * elements_فی_cacheline;
    int64_t       elem_انتہا   = cl_انتہا * elements_فی_cacheline;
    اگر (elem_انتہا > کل_elements) {
        elem_انتہا = کل_elements;
    }

    ثابت اعشاری * src_ptr = src0_ڈیٹا + elem_آغاز;
    اعشاری *       dst_ptr = dst_ڈیٹا + elem_آغاز;
    ثابت int32_t گنتی   = (int32_t) (elem_انتہا - elem_آغاز);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    برائے (int32_t i0 = 0; i0 < گنتی; i0 += 8) {
        __asm__ متلون(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [نتیجہ] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    واپس 0;
}
