//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_clamp_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input  (contiguous)
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    ಭಿನ್ನ              min_val;
    ಭಿನ್ನ              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ clamp_block_f32(ಭಿನ್ನ * dst, ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ min_val, ಭಿನ್ನ max_val, int32_t n) {
    int32_t       i       = 0;
    ಧ್ರುವ int32_t vec_end = (n / 8) * 8;

    ಯದಿ (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(temp_mask));
        __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");

        ಪ್ರತಿ (; i < vec_end; i += 8) {
            __asm__ ಚಞ್ಚಲ(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst[i])
                : [s] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(temp_mask));
    }

    ಪ್ರತಿ (; i < n; i++) {
        ಭಿನ್ನ v = src[i];
        ಯದಿ (v < min_val) {
            v = min_val;
        }
        ಯದಿ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_clamp_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಯದಿ (total_elements <= 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ ಭಿನ್ನ min_val = params->min_val;
    ಧ್ರುವ ಭಿನ್ನ max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ಧ್ರುವ int64_t elems_per_cl = 16;
    ಧ್ರುವ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ಧ್ರುವ int64_t cl_per_thread = (total_cl + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
    ಧ್ರುವ int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ಯದಿ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ಯದಿ (cl_start >= total_cl) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ಯದಿ (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    ಪ್ರತ್ಯಾಗಮ 0;
}
