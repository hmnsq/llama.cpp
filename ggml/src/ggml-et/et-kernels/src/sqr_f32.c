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

صحيح داخلا_نقطو(بنية ggml_et_sqr_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

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
    ثابت int64_t ڪل_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_في_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t ڪل_cachelines       = (ڪل_elements + elements_في_cacheline - 1) / elements_في_cacheline;

    ثابت int64_t cl_في_thread = (ڪل_cachelines + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
    int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
    إذا (cl_پڇاڙي > ڪل_cachelines) {
        cl_پڇاڙي = ڪل_cachelines;
    }

    إذا (cl_شروعات >= ڪل_cachelines) {
        أرجع 0;
    }

    ثابت int64_t elem_شروعات = cl_شروعات * elements_في_cacheline;
    int64_t       elem_پڇاڙي   = cl_پڇاڙي * elements_في_cacheline;
    إذا (elem_پڇاڙي > ڪل_elements) {
        elem_پڇاڙي = ڪل_elements;
    }

    ثابت عائم * src_ptr = src0_معلومات + elem_شروعات;
    عائم *       dst_ptr = dst_معلومات + elem_شروعات;
    ثابت int32_t ڳڻپ   = (int32_t) (elem_پڇاڙي - elem_شروعات);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    لكل (int32_t i0 = 0; i0 < ڳڻپ; i0 += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [نتيجو] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    أرجع 0;
}
