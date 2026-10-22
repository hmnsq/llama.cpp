//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_fill_params {
    ساختار ggml_tensor dst;  // F32 output tensor (contiguous)
    اعشاری              c;    // Constant value to fill
};

صحیح entry_point(ساختار ggml_et_fill_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * dst = &params->dst;

    اگر (dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    اعشاری * dst_data = (اعشاری *) dst->data;
    اگر (!dst_data) {
        بازگشت -1;
    }

    ثابت int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    اگر (total_elements == 0) {
        بازگشت 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_per_cl  = 16;
    ثابت int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    ثابت int64_t cl_per_thread = (total_cl + رقم_threads - 1) / رقم_threads;
    ثابت int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    اگر (cl_end > total_cl) {
        cl_end = total_cl;
    }
    اگر (cl_start >= total_cl) {
        بازگشت 0;
    }

    ثابت int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    اگر (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    اعشاری c = params->c;
    __asm__ فرار("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_end = es + ((ee - es) / 8) * 8;
    برای (; i < vec_end; i += 8) {
        __asm__ فرار("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    برای (; i < ee; i++) {
        dst_data[i] = c;
    }

    بازگشت 0;
}
