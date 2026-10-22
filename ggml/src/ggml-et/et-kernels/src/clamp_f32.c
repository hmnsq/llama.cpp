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
    दशांश              min_val;
    दशांश              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
स्थिर ओळीत शून्य clamp_block_f32(दशांश * dst, स्थिरांक दशांश * src, दशांश min_val, दशांश max_val, int32_t n) {
    int32_t       i       = 0;
    स्थिरांक int32_t vec_end = (n / 8) * 8;

    जर (vec_end > 0) {
        अचिन्हित दीर्घ temp_mask;
        __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

        साठी (; i < vec_end; i += 8) {
            __asm__ अस्थिर(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(दशांश (*)[8]) & dst[i])
                : [s] "m"(*(स्थिरांक दशांश (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));
    }

    साठी (; i < n; i++) {
        दशांश v = src[i];
        जर (v < min_val) {
            v = min_val;
        }
        जर (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

पूर्णांक entry_point(संरचना ggml_et_clamp_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    दशांश * src0_data = (दशांश *) src0->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    जर (total_elements <= 0) {
        परत 0;
    }

    स्थिरांक दशांश min_val = params->min_val;
    स्थिरांक दशांश max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    स्थिरांक int64_t elems_per_cl = 16;
    स्थिरांक int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    स्थिरांक int64_t cl_per_thread = (total_cl + अंक_threads - 1) / अंक_threads;
    स्थिरांक int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    जर (cl_end > total_cl) {
        cl_end = total_cl;
    }
    जर (cl_start >= total_cl) {
        परत 0;
    }

    स्थिरांक int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    जर (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    परत 0;
}
