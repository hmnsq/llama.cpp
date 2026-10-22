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

صحيح اندراج_نقطہ(بنية ggml_et_fill_params * params, فراغ * env) {
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
        أرجع -1;
    }

    بنية ggml_tensor * dst = &params->dst;

    إذا (dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * dst_انکڑے = (عائم *) dst->انکڑے;
    إذا (!dst_انکڑے) {
        أرجع -1;
    }

    ثابت int64_t کل_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    إذا (کل_elements == 0) {
        أرجع 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_فی_cl  = 16;
    ثابت int64_t کل_cl      = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;
    ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_شروع      = thread_id * cl_فی_thread;
    int64_t       cl_سرا        = cl_شروع + cl_فی_thread;
    إذا (cl_سرا > کل_cl) {
        cl_سرا = کل_cl;
    }
    إذا (cl_شروع >= کل_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_شروع * elems_فی_cl;
    int64_t       ee = cl_سرا * elems_فی_cl;
    إذا (ee > کل_elements) {
        ee = کل_elements;
    }

    // Broadcast constant to all SIMD lanes
    عائم c = params->c;
    __asm__ متطاير("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_سرا = es + ((ee - es) / 8) * 8;
    لكل (; i < vec_سرا; i += 8) {
        __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_انکڑے[i])::"f10");
    }
    // Scalar tail
    لكل (; i < ee; i++) {
        dst_انکڑے[i] = c;
    }

    أرجع 0;
}
