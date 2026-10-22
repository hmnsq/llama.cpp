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

صحيح مدخل_نقطه(بنية ggml_et_sqr_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

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

    عائم * src0_داده = (عائم *) src0->داده;
    عائم * dst_داده  = (عائم *) dst->داده;

    إذا (!src0_داده || !dst_داده) {
        أرجع -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ثابت int64_t مجموع_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_در_هر_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t مجموع_cachelines       = (مجموع_elements + elements_در_هر_cacheline - 1) / elements_در_هر_cacheline;

    ثابت int64_t cl_در_هر_thread = (مجموع_cachelines + رقم_نخها - 1) / رقم_نخها;
    ثابت int64_t cl_شروع      = thread_id * cl_در_هر_thread;
    int64_t       cl_انتها        = cl_شروع + cl_در_هر_thread;
    إذا (cl_انتها > مجموع_cachelines) {
        cl_انتها = مجموع_cachelines;
    }

    إذا (cl_شروع >= مجموع_cachelines) {
        أرجع 0;
    }

    ثابت int64_t elem_شروع = cl_شروع * elements_در_هر_cacheline;
    int64_t       elem_انتها   = cl_انتها * elements_در_هر_cacheline;
    إذا (elem_انتها > مجموع_elements) {
        elem_انتها = مجموع_elements;
    }

    ثابت عائم * src_ptr = src0_داده + elem_شروع;
    عائم *       dst_ptr = dst_داده + elem_شروع;
    ثابت int32_t شمار   = (int32_t) (elem_انتها - elem_شروع);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    لكل (int32_t i0 = 0; i0 < شمار; i0 += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [نتیجه] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    أرجع 0;
}
