//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

সংরচনা ggml_et_fill_params {
    সংরচনা ggml_tensor dst;  // F32 output tensor (contiguous)
    ভিন্ন              c;    // Constant value to fill
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_fill_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;
    }

    সংরচনা ggml_tensor * dst = &params->dst;

    যদি (dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ভিন্ন * dst_data = (ভিন্ন *) dst->data;
    যদি (!dst_data) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    যদি (total_elements == 0) {
        প্রত্যাগম 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ধ্রুব int64_t elems_per_cl  = 16;
    ধ্রুব int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    ধ্রুব int64_t cl_per_thread = (total_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুব int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    যদি (cl_end > total_cl) {
        cl_end = total_cl;
    }
    যদি (cl_start >= total_cl) {
        প্রত্যাগম 0;
    }

    ধ্রুব int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    যদি (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    ভিন্ন c = params->c;
    __asm__ চঞ্চল("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ধ্রুব int64_t vec_end = es + ((ee - es) / 8) * 8;
    প্রতি (; i < vec_end; i += 8) {
        __asm__ চঞ্চল("fsw.ps f10, %[d]\n" : [d] "=m"(*(ভিন্ন (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    প্রতি (; i < ee; i++) {
        dst_data[i] = c;
    }

    প্রত্যাগম 0;
}
