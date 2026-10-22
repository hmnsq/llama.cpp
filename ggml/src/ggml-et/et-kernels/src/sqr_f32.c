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

صحيح ثبت_ټکی(بنية ggml_et_sqr_params * params, فراغ * env) {
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
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ثابت int64_t ټول_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t ټول_cachelines       = (ټول_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    ثابت int64_t cl_فی_thread = (ټول_cachelines + عدد_تارونه - 1) / عدد_تارونه;
    ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
    int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
    إذا (cl_پای_ټکی > ټول_cachelines) {
        cl_پای_ټکی = ټول_cachelines;
    }

    إذا (cl_پیل >= ټول_cachelines) {
        أرجع 0;
    }

    ثابت int64_t elem_پیل = cl_پیل * elements_فی_cacheline;
    int64_t       elem_پای_ټکی   = cl_پای_ټکی * elements_فی_cacheline;
    إذا (elem_پای_ټکی > ټول_elements) {
        elem_پای_ټکی = ټول_elements;
    }

    ثابت عائم * src_ptr = src0_معلومات + elem_پیل;
    عائم *       dst_ptr = dst_معلومات + elem_پیل;
    ثابت int32_t شمېر   = (int32_t) (elem_پای_ټکی - elem_پیل);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    لكل (int32_t i0 = 0; i0 < شمېر; i0 += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [پایله] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    أرجع 0;
}
