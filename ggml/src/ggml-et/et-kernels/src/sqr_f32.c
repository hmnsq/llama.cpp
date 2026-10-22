//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
بنية ggml_et_sqr_params {
    بنية ggml_tensor src0;  // F32 input tensor
    بنية ggml_tensor dst;   // F32 output tensor
};

صحيح اندراج_نقطہ(بنية ggml_et_sqr_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    عائم * src0_انکڑے = (عائم *) src0->انکڑے;
    عائم * dst_انکڑے  = (عائم *) dst->انکڑے;

    إذا (!src0_انکڑے || !dst_انکڑے) {
        أرجع -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ثابت int64_t کل_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    ثابت int64_t cl_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_شروع      = thread_id * cl_فی_thread;
    int64_t       cl_سرا        = cl_شروع + cl_فی_thread;
    إذا (cl_سرا > کل_cachelines) {
        cl_سرا = کل_cachelines;
    }

    إذا (cl_شروع >= کل_cachelines) {
        أرجع 0;
    }

    ثابت int64_t elem_شروع = cl_شروع * elements_فی_cacheline;
    int64_t       elem_سرا   = cl_سرا * elements_فی_cacheline;
    إذا (elem_سرا > کل_elements) {
        elem_سرا = کل_elements;
    }

    ثابت عائم * src_ptr = src0_انکڑے + elem_شروع;
    عائم *       dst_ptr = dst_انکڑے + elem_شروع;
    ثابت int32_t گنتی   = (int32_t) (elem_سرا - elem_شروع);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    لكل (int32_t i0 = 0; i0 < گنتی; i0 += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [نتیجہ] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    أرجع 0;
}
