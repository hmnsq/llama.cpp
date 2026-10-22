//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

সংরচনা ggml_et_clamp_params {
    সংরচনা ggml_tensor src0;  // F32 input  (contiguous)
    সংরচনা ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    ভিন্ন              min_val;
    ভিন্ন              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
স্থির অন্তঃস্থ শূন্য clamp_block_f32(ভিন্ন * dst, ধ্রুব ভিন্ন * src, ভিন্ন min_val, ভিন্ন max_val, int32_t n) {
    int32_t       i       = 0;
    ধ্রুব int32_t vec_end = (n / 8) * 8;

    যদি (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ চঞ্চল("mova.x.m %0" : "=r"(temp_mask));
        __asm__ চঞ্চল("mov.m.x m0, x0, 0xFF");

        প্রতি (; i < vec_end; i += 8) {
            __asm__ চঞ্চল(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(ভিন্ন (*)[8]) & dst[i])
                : [s] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ চঞ্চল("mova.m.x %0" ::"r"(temp_mask));
    }

    প্রতি (; i < n; i++) {
        ভিন্ন v = src[i];
        যদি (v < min_val) {
            v = min_val;
        }
        যদি (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_clamp_params * params, শূন্য * env) {
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

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    যদি (total_elements <= 0) {
        প্রত্যাগম 0;
    }

    ধ্রুব ভিন্ন min_val = params->min_val;
    ধ্রুব ভিন্ন max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ধ্রুব int64_t elems_per_cl = 16;
    ধ্রুব int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ধ্রুব int64_t cl_per_thread = (total_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুব int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
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

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    প্রত্যাগম 0;
}
