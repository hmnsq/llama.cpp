//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_clamp_params {
    काष्ठा ggml_tensor src0;  // F32 input  (contiguous)
    काष्ठा ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    भग्न              min_val;
    भग्न              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
अटल अंतरभूत व्योम clamp_block_f32(भग्न * dst, स्थिर भग्न * src, भग्न min_val, भग्न max_val, int32_t n) {
    int32_t       i       = 0;
    स्थिर int32_t vec_end = (n / 8) * 8;

    अगर (vec_end > 0) {
        अचिन्हित दीर्घ अस्थायी_mask;
        __asm__ अस्थिर("mova.x.m %0" : "=r"(अस्थायी_mask));
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

        क्रम (; i < vec_end; i += 8) {
            __asm__ अस्थिर(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(भग्न (*)[8]) & dst[i])
                : [s] "m"(*(स्थिर भग्न (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ अस्थिर("mova.m.x %0" ::"r"(अस्थायी_mask));
    }

    क्रम (; i < n; i++) {
        भग्न v = src[i];
        अगर (v < min_val) {
            v = min_val;
        }
        अगर (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

पूर्णांक entry_point(काष्ठा ggml_et_clamp_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t योग_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    अगर (योग_elements <= 0) {
        वापस 0;
    }

    स्थिर भग्न min_val = params->min_val;
    स्थिर भग्न max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    स्थिर int64_t elems_per_cl = 16;
    स्थिर int64_t योग_cl     = (योग_elements + elems_per_cl - 1) / elems_per_cl;

    स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
    स्थिर int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    अगर (cl_end > योग_cl) {
        cl_end = योग_cl;
    }
    अगर (cl_start >= योग_cl) {
        वापस 0;
    }

    स्थिर int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    अगर (ee > योग_elements) {
        ee = योग_elements;
    }

    clamp_block_f32(dst_तथ्य + es, src0_तथ्य + es, min_val, max_val, (int32_t) (ee - es));
    वापस 0;
}
