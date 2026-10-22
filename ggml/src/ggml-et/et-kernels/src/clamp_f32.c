//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_clamp_params {
    સંરચના ggml_tensor src0;  // F32 input  (contiguous)
    સંરચના ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    દશાંશ              min_val;
    દશાંશ              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
સ્થિર અંતઃરેખ શૂન્ય clamp_block_f32(દશાંશ * dst, અચલ દશાંશ * src, દશાંશ min_val, દશાંશ max_val, int32_t n) {
    int32_t       i       = 0;
    અચલ int32_t vec_end = (n / 8) * 8;

    જો (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
        __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

        માટે (; i < vec_end; i += 8) {
            __asm__ અસ્થિર(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(દશાંશ (*)[8]) & dst[i])
                : [s] "m"(*(અચલ દશાંશ (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));
    }

    માટે (; i < n; i++) {
        દશાંશ v = src[i];
        જો (v < min_val) {
            v = min_val;
        }
        જો (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

પૂર્ણાંક entry_point(સંરચના ggml_et_clamp_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    જો (total_elements <= 0) {
        પાછા 0;
    }

    અચલ દશાંશ min_val = params->min_val;
    અચલ દશાંશ max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    અચલ int64_t elems_per_cl = 16;
    અચલ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
    અચલ int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    જો (cl_end > total_cl) {
        cl_end = total_cl;
    }
    જો (cl_start >= total_cl) {
        પાછા 0;
    }

    અચલ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    જો (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    પાછા 0;
}
