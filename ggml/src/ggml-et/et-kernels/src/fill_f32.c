//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_fill_params {
    কাঠামো ggml_tensor dst;  // F32 output tensor (contiguous)
    দশমিক              c;    // Constant value to fill
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_fill_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * dst = &params->dst;

    যদি (dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * dst_তথ্য = (দশমিক *) dst->তথ্য;
    যদি (!dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t যোগ_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    যদি (যোগ_elements == 0) {
        ফেরত 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ধ্রুবক int64_t elems_per_cl  = 16;
    ধ্রুবক int64_t যোগ_cl      = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;
    ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    যদি (cl_end > যোগ_cl) {
        cl_end = যোগ_cl;
    }
    যদি (cl_start >= যোগ_cl) {
        ফেরত 0;
    }

    ধ্রুবক int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    যদি (ee > যোগ_elements) {
        ee = যোগ_elements;
    }

    // Broadcast constant to all SIMD lanes
    দশমিক c = params->c;
    __asm__ অস্থির("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ধ্রুবক int64_t vec_end = es + ((ee - es) / 8) * 8;
    জন্য (; i < vec_end; i += 8) {
        __asm__ অস্থির("fsw.ps f10, %[d]\n" : [d] "=m"(*(দশমিক (*)[8]) & dst_তথ্য[i])::"f10");
    }
    // Scalar tail
    জন্য (; i < ee; i++) {
        dst_তথ্য[i] = c;
    }

    ফেরত 0;
}
