//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
నిర్మాణం ggml_et_sqr_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor
    నిర్మాణం ggml_tensor dst;   // F32 output tensor
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_sqr_params * params, శూన్య * env) {
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
        తిరిగి -1;  // Invalid pointer
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;  // Unsupported type combination
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    స్థిరరాశి int64_t మొత్తం_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    స్థిరరాశి int64_t elements_చొప్పున_cacheline = 16;  // 64 bytes / 4 bytes per float
    స్థిరరాశి int64_t మొత్తం_cachelines       = (మొత్తం_elements + elements_చొప్పున_cacheline - 1) / elements_చొప్పున_cacheline;

    స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cachelines + అంకె_దారాలు - 1) / అంకె_దారాలు;
    స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
    int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
    ఒకవేళ (cl_చివర > మొత్తం_cachelines) {
        cl_చివర = మొత్తం_cachelines;
    }

    ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cachelines) {
        తిరిగి 0;
    }

    స్థిరరాశి int64_t elem_ప్రారంభం = cl_ప్రారంభం * elements_చొప్పున_cacheline;
    int64_t       elem_చివర   = cl_చివర * elements_చొప్పున_cacheline;
    ఒకవేళ (elem_చివర > మొత్తం_elements) {
        elem_చివర = మొత్తం_elements;
    }

    స్థిరరాశి దశాంశం * src_ptr = src0_దత్తాంశం + elem_ప్రారంభం;
    దశాంశం *       dst_ptr = dst_దత్తాంశం + elem_ప్రారంభం;
    స్థిరరాశి int32_t లెక్క   = (int32_t) (elem_చివర - elem_ప్రారంభం);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    కొరకు (int32_t i0 = 0; i0 < లెక్క; i0 += 8) {
        __asm__ అస్థిర(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [ఫలితం] "=m"(*(దశాంశం (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    తిరిగి 0;
}
