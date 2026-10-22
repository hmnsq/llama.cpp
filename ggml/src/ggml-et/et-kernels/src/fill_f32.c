//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_fill_params {
    ساخت ggml_tensor dst;  // F32 output tensor (contiguous)
    اعشاری              c;    // Constant value to fill
};

صحیح اندراج_نقطہ(ساخت ggml_et_fill_params * params, خالی * env) {
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

    ساخت ggml_tensor * dst = &params->dst;

    اگر (dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * dst_ڈیٹا = (اعشاری *) dst->ڈیٹا;
    اگر (!dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t کل_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    اگر (کل_elements == 0) {
        واپس 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_فی_cl  = 16;
    ثابت int64_t کل_cl      = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;
    ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
    int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
    اگر (cl_انتہا > کل_cl) {
        cl_انتہا = کل_cl;
    }
    اگر (cl_آغاز >= کل_cl) {
        واپس 0;
    }

    ثابت int64_t es = cl_آغاز * elems_فی_cl;
    int64_t       ee = cl_انتہا * elems_فی_cl;
    اگر (ee > کل_elements) {
        ee = کل_elements;
    }

    // Broadcast constant to all SIMD lanes
    اعشاری c = params->c;
    __asm__ متلون("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_انتہا = es + ((ee - es) / 8) * 8;
    برائے (; i < vec_انتہا; i += 8) {
        __asm__ متلون("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_ڈیٹا[i])::"f10");
    }
    // Scalar tail
    برائے (; i < ee; i++) {
        dst_ڈیٹا[i] = c;
    }

    واپس 0;
}
