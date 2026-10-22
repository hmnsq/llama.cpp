//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_clamp_params {
    কাঠামো ggml_tensor src0;  // F32 input  (contiguous)
    কাঠামো ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    দশমিক              min_val;
    দশমিক              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
স্থির অন্তঃরেখ শূন্য clamp_block_f32(দশমিক * dst, ধ্রুবক দশমিক * src, দশমিক min_val, দশমিক max_val, int32_t n) {
    int32_t       i       = 0;
    ধ্রুবক int32_t vec_end = (n / 8) * 8;

    যদি (vec_end > 0) {
        unsigned long অস্থায়ী_mask;
        __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));
        __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

        জন্য (; i < vec_end; i += 8) {
            __asm__ অস্থির(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(দশমিক (*)[8]) & dst[i])
                : [s] "m"(*(ধ্রুবক দশমিক (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));
    }

    জন্য (; i < n; i++) {
        দশমিক v = src[i];
        যদি (v < min_val) {
            v = min_val;
        }
        যদি (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_clamp_params * params, শূন্য * env) {
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

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t যোগ_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    যদি (যোগ_elements <= 0) {
        ফেরত 0;
    }

    ধ্রুবক দশমিক min_val = params->min_val;
    ধ্রুবক দশমিক max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ধ্রুবক int64_t elems_per_cl = 16;
    ধ্রুবক int64_t যোগ_cl     = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;

    ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
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

    clamp_block_f32(dst_তথ্য + es, src0_তথ্য + es, min_val, max_val, (int32_t) (ee - es));
    ফেরত 0;
}
