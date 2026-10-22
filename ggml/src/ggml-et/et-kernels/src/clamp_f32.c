//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_clamp_params {
    بنية ggml_tensor src0;  // F32 input  (contiguous)
    بنية ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    عائم              min_val;
    عائم              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ساكن مضمن فراغ clamp_block_f32(عائم * dst, ثابت عائم * src, عائم min_val, عائم max_val, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_end = (n / 8) * 8;

    ܐܢ (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ متطاير("mova.x.m %0" : "=r"(temp_mask));
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");

        لكل (; i < vec_end; i += 8) {
            __asm__ متطاير(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(عائم (*)[8]) & dst[i])
                : [s] "m"(*(ثابت عائم (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ متطاير("mova.m.x %0" ::"r"(temp_mask));
    }

    لكل (; i < n; i++) {
        عائم v = src[i];
        ܐܢ (v < min_val) {
            v = min_val;
        }
        ܐܢ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

صحيح entry_point(بنية ggml_et_clamp_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ܐܢ (!kernel_env) {
        ܦܢܐ -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    ܐܢ (thread_id < 0) {
        ܦܢܐ 0;
    }

    ܐܢ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ܦܢܐ -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    ܐܢ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ܦܢܐ -1;
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    ܐܢ (!src0_data || !dst_data) {
        ܦܢܐ -1;
    }

    ثابت int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ܐܢ (total_elements <= 0) {
        ܦܢܐ 0;
    }

    ثابت عائم min_val = params->min_val;
    ثابت عائم max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ثابت int64_t elems_per_cl = 16;
    ثابت int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ثابت int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    ثابت int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ܐܢ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ܐܢ (cl_start >= total_cl) {
        ܦܢܐ 0;
    }

    ثابت int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ܐܢ (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    ܦܢܐ 0;
}
