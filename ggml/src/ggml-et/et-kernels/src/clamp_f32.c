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
स्थिर अन्तःस्थ शून्य clamp_block_f32(भिन्न * dst, अचर भिन्न * src, भिन्न min_val, भिन्न max_val, int32_t n) {
    int32_t       i       = 0;
    अचर int32_t vec_end = (n / 8) * 8;

    यदि (vec_end > 0) {
        अचिन्हित दीर्घ temp_mask;
        __asm__ चञ्चल("mova.x.m %0" : "=r"(temp_mask));
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

        लागि (; i < vec_end; i += 8) {
            __asm__ चञ्चल(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
                : [s] "m"(*(अचर भिन्न (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ चञ्चल("mova.m.x %0" ::"r"(temp_mask));
    }

    लागि (; i < n; i++) {
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

पूर्णाङ्क entry_point(संरचना ggml_et_clamp_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }

    अचर int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    यदि (total_elements <= 0) {
        फर्काऊ 0;
    }

    अचर भिन्न min_val = params->min_val;
    अचर भिन्न max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    अचर int64_t elems_per_cl = 16;
    अचर int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    अचर int64_t cl_per_thread = (total_cl + अङ्क_threads - 1) / अङ्क_threads;
    अचर int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    यदि (cl_end > total_cl) {
        cl_end = total_cl;
    }
    यदि (cl_start >= total_cl) {
        फर्काऊ 0;
    }

    अचर int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    यदि (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    फर्काऊ 0;
}
