//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_clamp_params {
    నిర్మాణం ggml_tensor src0;  // F32 input  (contiguous)
    నిర్మాణం ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    దశాంశం              min_val;
    దశాంశం              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
స్థిర అంతఃవరుస శూన్య clamp_ఖండం_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src, దశాంశం min_val, దశాంశం max_val, int32_t n) {
    int32_t       i       = 0;
    స్థిరరాశి int32_t vec_చివర = (n / 8) * 8;

    ఒకవేళ (vec_చివర > 0) {
        గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
        __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
        __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

        కొరకు (; i < vec_చివర; i += 8) {
            __asm__ అస్థిర(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(దశాంశం (*)[8]) & dst[i])
                : [s] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));
    }

    కొరకు (; i < n; i++) {
        దశాంశం v = src[i];
        ఒకవేళ (v < min_val) {
            v = min_val;
        }
        ఒకవేళ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_clamp_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t మొత్తం_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ఒకవేళ (మొత్తం_elements <= 0) {
        తిరిగి 0;
    }

    స్థిరరాశి దశాంశం min_val = params->min_val;
    స్థిరరాశి దశాంశం max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    స్థిరరాశి int64_t elems_చొప్పున_cl = 16;
    స్థిరరాశి int64_t మొత్తం_cl     = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

    స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cl + అంకె_దారాలు - 1) / అంకె_దారాలు;
    స్థిరరాశి int64_t cl_ప్రారంభం      = (int64_t) thread_id * cl_చొప్పున_thread;
    int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
    ఒకవేళ (cl_చివర > మొత్తం_cl) {
        cl_చివర = మొత్తం_cl;
    }
    ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cl) {
        తిరిగి 0;
    }

    స్థిరరాశి int64_t es = cl_ప్రారంభం * elems_చొప్పున_cl;
    int64_t       ee = cl_చివర * elems_చొప్పున_cl;
    ఒకవేళ (ee > మొత్తం_elements) {
        ee = మొత్తం_elements;
    }

    clamp_ఖండం_f32(dst_దత్తాంశం + es, src0_దత్తాంశం + es, min_val, max_val, (int32_t) (ee - es));
    తిరిగి 0;
}
