//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_clamp_params {
    ഘടന ggml_tensor src0;  // F32 input  (contiguous)
    ഘടന ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    ദശാംശം              min_val;
    ദശാംശം              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
സ്ഥിതം വരിയിൽ ശൂന്യം clamp_block_f32(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, ദശാംശം min_val, ദശാംശം max_val, int32_t n) {
    int32_t       i       = 0;
    സ്ഥിരം int32_t vec_end = (n / 8) * 8;

    എങ്കിൽ (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(temp_mask));
        __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");

        വേണ്ടി (; i < vec_end; i += 8) {
            __asm__ അസ്ഥിര(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(ദശാംശം (*)[8]) & dst[i])
                : [s] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(temp_mask));
    }

    വേണ്ടി (; i < n; i++) {
        ദശാംശം v = src[i];
        എങ്കിൽ (v < min_val) {
            v = min_val;
        }
        എങ്കിൽ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_clamp_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    എങ്കിൽ (total_elements <= 0) {
        തിരികെ 0;
    }

    സ്ഥിരം ദശാംശം min_val = params->min_val;
    സ്ഥിരം ദശാംശം max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    സ്ഥിരം int64_t elems_per_cl = 16;
    സ്ഥിരം int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    സ്ഥിരം int64_t cl_per_thread = (total_cl + അക്കം_threads - 1) / അക്കം_threads;
    സ്ഥിരം int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    എങ്കിൽ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    എങ്കിൽ (cl_start >= total_cl) {
        തിരികെ 0;
    }

    സ്ഥിരം int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    എങ്കിൽ (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    തിരികെ 0;
}
