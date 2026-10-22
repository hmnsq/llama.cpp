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

صحيح ثبت_ټکی(بنية ggml_et_fill_params * params, فراغ * env) {
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

    ثابت int64_t ټول_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    إذا (ټول_elements == 0) {
        أرجع 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_فی_cl  = 16;
    ثابت int64_t ټول_cl      = (ټول_elements + elems_فی_cl - 1) / elems_فی_cl;
    ثابت int64_t cl_فی_thread = (ټول_cl + عدد_تارونه - 1) / عدد_تارونه;
    ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
    int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
    إذا (cl_پای_ټکی > ټول_cl) {
        cl_پای_ټکی = ټول_cl;
    }
    إذا (cl_پیل >= ټول_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_پیل * elems_فی_cl;
    int64_t       ee = cl_پای_ټکی * elems_فی_cl;
    إذا (ee > ټول_elements) {
        ee = ټول_elements;
    }

    // Broadcast constant to all SIMD lanes
    عائم c = params->c;
    __asm__ متطاير("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_پای_ټکی = es + ((ee - es) / 8) * 8;
    لكل (; i < vec_پای_ټکی; i += 8) {
        __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_معلومات[i])::"f10");
    }
    // Scalar tail
    لكل (; i < ee; i++) {
        dst_معلومات[i] = c;
    }

    أرجع 0;
}
