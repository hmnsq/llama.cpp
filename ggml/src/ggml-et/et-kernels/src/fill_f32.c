//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_fill_params {
    నిర్మాణం ggml_tensor dst;  // F32 output tensor (contiguous)
    దశాంశం              c;    // Constant value to fill
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_fill_params * params, శూన్య * env) {
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

    నిర్మాణం ggml_tensor * dst = &params->dst;

    ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    దశాంశం * dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;
    ఒకవేళ (!dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t మొత్తం_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ఒకవేళ (మొత్తం_elements == 0) {
        తిరిగి 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    స్థిరరాశి int64_t elems_చొప్పున_cl  = 16;
    స్థిరరాశి int64_t మొత్తం_cl      = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;
    స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cl + అంకె_దారాలు - 1) / అంకె_దారాలు;
    స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
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

    // Broadcast constant to all SIMD lanes
    దశాంశం c = params->c;
    __asm__ అస్థిర("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    స్థిరరాశి int64_t vec_చివర = es + ((ee - es) / 8) * 8;
    కొరకు (; i < vec_చివర; i += 8) {
        __asm__ అస్థిర("fsw.ps f10, %[d]\n" : [d] "=m"(*(దశాంశం (*)[8]) & dst_దత్తాంశం[i])::"f10");
    }
    // Scalar tail
    కొరకు (; i < ee; i++) {
        dst_దత్తాంశం[i] = c;
    }

    తిరిగి 0;
}
