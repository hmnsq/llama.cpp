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
स्थिर अन्तःस्थ शून्य clamp_खण्डम्_f32(भिन्न * dst, ध्रुव भिन्न * src, भिन्न min_val, भिन्न max_val, int32_t n) {
    int32_t       i       = 0;
    ध्रुव int32_t vec_अन्तः = (n / 8) * 8;

    यदि (vec_अन्तः > 0) {
        अचिह्नित दीर्घ अस्थायी_आवरणम्;
        __asm__ चञ्चल("mova.x.m %0" : "=r"(अस्थायी_आवरणम्));
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

        प्रति (; i < vec_अन्तः; i += 8) {
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

        __asm__ चञ्चल("mova.m.x %0" ::"r"(अस्थायी_आवरणम्));
    }

    प्रति (; i < n; i++) {
        भिन्न v = src[i];
        यदि (v < min_val) {
            v = min_val;
        }
        यदि (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_clamp_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t योग_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    यदि (योग_elements <= 0) {
        प्रत्यागम 0;
    }

    ध्रुव भिन्न min_val = params->min_val;
    ध्रुव भिन्न max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ध्रुव int64_t elems_प्रत्येकम्_cl = 16;
    ध्रुव int64_t योग_cl     = (योग_elements + elems_प्रत्येकम्_cl - 1) / elems_प्रत्येकम्_cl;

    ध्रुव int64_t cl_प्रत्येकम्_thread = (योग_cl + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
    ध्रुव int64_t cl_start      = (int64_t) thread_id * cl_प्रत्येकम्_thread;
    int64_t       cl_अन्तः        = cl_start + cl_प्रत्येकम्_thread;
    यदि (cl_अन्तः > योग_cl) {
        cl_अन्तः = योग_cl;
    }
    यदि (cl_start >= योग_cl) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t es = cl_start * elems_प्रत्येकम्_cl;
    int64_t       ee = cl_अन्तः * elems_प्रत्येकम्_cl;
    यदि (ee > योग_elements) {
        ee = योग_elements;
    }

    clamp_खण्डम्_f32(dst_दत्त + es, src0_दत्त + es, min_val, max_val, (int32_t) (ee - es));
    प्रत्यागम 0;
}
