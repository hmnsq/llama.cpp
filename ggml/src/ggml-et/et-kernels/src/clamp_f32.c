//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_clamp_params {
    கட்டமைப்பு ggml_tensor src0;  // F32 input  (contiguous)
    கட்டமைப்பு ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    தசம              min_val;
    தசம              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
நிலையான உள்வரி வெறுமை clamp_block_f32(தசம * dst, மாறிலி தசம * src, தசம min_val, தசம max_val, int32_t n) {
    int32_t       i       = 0;
    மாறிலி int32_t vec_end = (n / 8) * 8;

    எனில் (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ நிலையற்ற("mova.x.m %0" : "=r"(temp_mask));
        __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");

        க்கு (; i < vec_end; i += 8) {
            __asm__ நிலையற்ற(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(தசம (*)[8]) & dst[i])
                : [s] "m"(*(மாறிலி தசம (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ நிலையற்ற("mova.m.x %0" ::"r"(temp_mask));
    }

    க்கு (; i < n; i++) {
        தசம v = src[i];
        எனில் (v < min_val) {
            v = min_val;
        }
        எனில் (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

முழு entry_point(கட்டமைப்பு ggml_et_clamp_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    எனில் (total_elements <= 0) {
        திரும்பு 0;
    }

    மாறிலி தசம min_val = params->min_val;
    மாறிலி தசம max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    மாறிலி int64_t elems_per_cl = 16;
    மாறிலி int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    மாறிலி int64_t cl_per_thread = (total_cl + இலக்கம்_threads - 1) / இலக்கம்_threads;
    மாறிலி int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    எனில் (cl_end > total_cl) {
        cl_end = total_cl;
    }
    எனில் (cl_start >= total_cl) {
        திரும்பு 0;
    }

    மாறிலி int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    எனில் (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    திரும்பு 0;
}
