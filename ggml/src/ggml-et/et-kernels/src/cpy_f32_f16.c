//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_cont_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor dst;
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_cont_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env || !params) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        తిరిగి -1;
    }

    స్థిరరాశి అక్షరం * src_దత్తాంశం = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం;
    uint16_t *   dst_దత్తాంశం = (uint16_t *) dst->దత్తాంశం;

    ఒకవేళ (!src_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి int64_t nb00 = src0->nb[0];
    స్థిరరాశి int64_t nb01 = src0->nb[1];
    స్థిరరాశి int64_t nb02 = src0->nb[2];
    స్థిరరాశి int64_t nb03 = src0->nb[3];

    స్థిరరాశి int64_t మొత్తం_elements = ne00 * ne01 * ne02 * ne03;

    ఒకవేళ (మొత్తం_elements == 0) {
        తిరిగి 0;
    }

    // Check if src is contiguous F32
    స్థిరరాశి bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    స్థిరరాశి int64_t elems_చొప్పున_cl = 32;
    స్థిరరాశి int64_t మొత్తం_cl     = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

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

    ఒకవేళ (src_contiguous) {
        // Fast path: src is contiguous F32
        స్థిరరాశి దశాంశం * src_f32 = (స్థిరరాశి దశాంశం *) src_దత్తాంశం;
        కొరకు (int64_t i = es; i < ee; ++i) {
            dst_దత్తాంశం[i] = fp32_గమ్యం_fp16(src_f32[i]);
        }
    } లేకపోతే {
        // General path: stride-aware read
        కొరకు (int64_t idx = es; idx < ee; ++idx) {
            స్థిరరాశి int64_t i00  = idx % ne00;
            స్థిరరాశి int64_t rem1 = idx / ne00;
            స్థిరరాశి int64_t i01  = rem1 % ne01;
            స్థిరరాశి int64_t rem2 = rem1 / ne01;
            స్థిరరాశి int64_t i02  = rem2 % ne02;
            స్థిరరాశి int64_t i03  = rem2 / ne02;

            స్థిరరాశి దశాంశం val = *(స్థిరరాశి దశాంశం *) (src_దత్తాంశం + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_దత్తాంశం[idx]   = fp32_గమ్యం_fp16(val);
        }
    }

    తిరిగి 0;
}
