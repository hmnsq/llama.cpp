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

صحيح مدخل_نقطه(بنية ggml_et_fill_params * params, فراغ * env) {
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
        أرجع -1;
    }

    بنية ggml_tensor * dst = &params->dst;

    إذا (dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * dst_داده = (عائم *) dst->داده;
    إذا (!dst_داده) {
        أرجع -1;
    }

    ثابت int64_t مجموع_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    إذا (مجموع_elements == 0) {
        أرجع 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_در_هر_cl  = 16;
    ثابت int64_t مجموع_cl      = (مجموع_elements + elems_در_هر_cl - 1) / elems_در_هر_cl;
    ثابت int64_t cl_در_هر_thread = (مجموع_cl + رقم_نخها - 1) / رقم_نخها;
    ثابت int64_t cl_شروع      = thread_id * cl_در_هر_thread;
    int64_t       cl_انتها        = cl_شروع + cl_در_هر_thread;
    إذا (cl_انتها > مجموع_cl) {
        cl_انتها = مجموع_cl;
    }
    إذا (cl_شروع >= مجموع_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_شروع * elems_در_هر_cl;
    int64_t       ee = cl_انتها * elems_در_هر_cl;
    إذا (ee > مجموع_elements) {
        ee = مجموع_elements;
    }

    // Broadcast constant to all SIMD lanes
    عائم c = params->c;
    __asm__ متطاير("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_انتها = es + ((ee - es) / 8) * 8;
    لكل (; i < vec_انتها; i += 8) {
        __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_داده[i])::"f10");
    }
    // Scalar tail
    لكل (; i < ee; i++) {
        dst_داده[i] = c;
    }

    أرجع 0;
}
