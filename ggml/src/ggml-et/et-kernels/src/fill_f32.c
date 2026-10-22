//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_fill_params {
    بنية ggml_tensor dst;  // F32 output tensor (contiguous)
    عائم              c;    // Constant value to fill
};

صحيح داخلا_نقطو(بنية ggml_et_fill_params * params, فراغ * env) {
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
        أرجع -1;
    }

    بنية ggml_tensor * dst = &params->dst;

    إذا (dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * dst_معلومات = (عائم *) dst->معلومات;
    إذا (!dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ڪل_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    إذا (ڪل_elements == 0) {
        أرجع 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_في_cl  = 16;
    ثابت int64_t ڪل_cl      = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;
    ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
    int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
    إذا (cl_پڇاڙي > ڪل_cl) {
        cl_پڇاڙي = ڪل_cl;
    }
    إذا (cl_شروعات >= ڪل_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_شروعات * elems_في_cl;
    int64_t       ee = cl_پڇاڙي * elems_في_cl;
    إذا (ee > ڪل_elements) {
        ee = ڪل_elements;
    }

    // Broadcast constant to all SIMD lanes
    عائم c = params->c;
    __asm__ متطاير("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_پڇاڙي = es + ((ee - es) / 8) * 8;
    لكل (; i < vec_پڇاڙي; i += 8) {
        __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_معلومات[i])::"f10");
    }
    // Scalar tail
    لكل (; i < ee; i++) {
        dst_معلومات[i] = c;
    }

    أرجع 0;
}
