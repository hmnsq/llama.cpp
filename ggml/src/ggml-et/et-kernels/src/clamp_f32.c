//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_clamp_params {
    संरचना ggml_tensor src0;  // F32 input  (contiguous)
    संरचना ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    भिन्न              min_val;
    भिन्न              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
स्थिर अन्तःस्थ शून्य clamp_block_f32(भिन्न * dst, ध्रुव भिन्न * src, भिन्न min_val, भिन्न max_val, int32_t n) {
    int32_t       i       = 0;
    ध्रुव int32_t vec_end = (n / 8) * 8;

    सचे (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ चञ्चल("mova.x.m %0" : "=r"(temp_mask));
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

        अत्थाय (; i < vec_end; i += 8) {
            __asm__ चञ्चल(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
                : [s] "m"(*(ध्रुव भिन्न (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ चञ्चल("mova.m.x %0" ::"r"(temp_mask));
    }

    अत्थाय (; i < n; i++) {
        भिन्न v = src[i];
        सचे (v < min_val) {
            v = min_val;
        }
        सचे (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

पूर्णाङ्क entry_point(संरचना ggml_et_clamp_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    सचे (!kernel_env) {
        पटिनिवत्त -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = get_अङ्कः_threads(kernel_env->shire_mask);

    सचे (thread_id < 0) {
        पटिनिवत्त 0;
    }

    सचे (params == 0 || ((uint64_t) params & 0x7) != 0) {
        पटिनिवत्त -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    सचे (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        पटिनिवत्त -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    सचे (!src0_data || !dst_data) {
        पटिनिवत्त -1;
    }

    ध्रुव int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    सचे (total_elements <= 0) {
        पटिनिवत्त 0;
    }

    ध्रुव भिन्न min_val = params->min_val;
    ध्रुव भिन्न max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ध्रुव int64_t elems_per_cl = 16;
    ध्रुव int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ध्रुव int64_t cl_per_thread = (total_cl + अङ्कः_threads - 1) / अङ्कः_threads;
    ध्रुव int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    सचे (cl_end > total_cl) {
        cl_end = total_cl;
    }
    सचे (cl_start >= total_cl) {
        पटिनिवत्त 0;
    }

    ध्रुव int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    सचे (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    पटिनिवत्त 0;
}
