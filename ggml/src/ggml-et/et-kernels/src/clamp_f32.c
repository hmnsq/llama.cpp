//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_clamp_params {
    ସଂରଚନା ggml_tensor src0;  // F32 input  (contiguous)
    ସଂରଚନା ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    ଦଶମିକ              min_val;
    ଦଶମିକ              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ clamp_block_f32(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, ଦଶମିକ min_val, ଦଶମିକ max_val, int32_t n) {
    int32_t       i       = 0;
    ଅଚଳ int32_t vec_end = (n / 8) * 8;

    ଯଦି (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
        __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

        ପାଇଁ (; i < vec_end; i += 8) {
            __asm__ ଅସ୍ଥିର(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst[i])
                : [s] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));
    }

    ପାଇଁ (; i < n; i++) {
        ଦଶମିକ v = src[i];
        ଯଦି (v < min_val) {
            v = min_val;
        }
        ଯଦି (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_clamp_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଯଦି (total_elements <= 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ଦଶମିକ min_val = params->min_val;
    ଅଚଳ ଦଶମିକ max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ଅଚଳ int64_t elems_per_cl = 16;
    ଅଚଳ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ଅଚଳ int64_t cl_per_thread = (total_cl + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
    ଅଚଳ int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ଯଦି (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ଯଦି (cl_start >= total_cl) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ଯଦି (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    ଫେରନ୍ତୁ 0;
}
