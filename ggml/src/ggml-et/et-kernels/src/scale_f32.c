//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_scale_params {
    నిర్మాణం ggml_tensor src0;   // F32 input tensor
    నిర్మాణం ggml_tensor dst;    // F32 output tensor
    దశాంశం              scale;  // Scale factor
    దశాంశం              bias;   // Bias (additive offset)
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_scale_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

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


    దశాంశం scale = params->scale;
    దశాంశం bias  = params->bias;

    // Total elements across all dimensions
    స్థిరరాశి int64_t మొత్తం_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    స్థిరరాశి int64_t elements_చొప్పున_cacheline = 16;
    స్థిరరాశి int64_t మొత్తం_cachelines       = (మొత్తం_elements + elements_చొప్పున_cacheline - 1) / elements_చొప్పున_cacheline;

    int64_t cachelines_చొప్పున_thread = (మొత్తం_cachelines + అంకె_దారాలు - 1) / అంకె_దారాలు;
    int64_t ప్రారంభం_cacheline       = thread_id * cachelines_చొప్పున_thread;
    int64_t చివర_cacheline         = ప్రారంభం_cacheline + cachelines_చొప్పున_thread;

    ఒకవేళ (చివర_cacheline > మొత్తం_cachelines) {
        చివర_cacheline = మొత్తం_cachelines;
    }

    ఒకవేళ (ప్రారంభం_cacheline >= మొత్తం_cachelines) {
        తిరిగి 0;
    }

    int64_t ప్రారంభం_elem = ప్రారంభం_cacheline * elements_చొప్పున_cacheline;
    int64_t చివర_elem   = చివర_cacheline * elements_చొప్పున_cacheline;
    ఒకవేళ (చివర_elem > మొత్తం_elements) {
        చివర_elem = మొత్తం_elements;
    }

    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");
    __asm__ అస్థిర("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ అస్థిర("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    కొరకు (int64_t i = ప్రారంభం_elem; i < చివర_elem; i += 8) {
        __asm__ అస్థిర(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_దత్తాంశం[i])
            : [src] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src0_దత్తాంశం[i])
            : "f10", "f20", "f21");
    }
    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));

    తిరిగి 0;
}
