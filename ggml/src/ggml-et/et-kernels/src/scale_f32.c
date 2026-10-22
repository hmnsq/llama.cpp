//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_scale_params {
    ساخت ggml_tensor src0;   // F32 input tensor
    ساخت ggml_tensor dst;    // F32 output tensor
    اعشاری              scale;  // Scale factor
    اعشاری              bias;   // Bias (additive offset)
};

صحیح اندراج_نقطہ(ساخت ggml_et_scale_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }


    اعشاری scale = params->scale;
    اعشاری bias  = params->bias;

    // Total elements across all dimensions
    ثابت int64_t کل_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ثابت int64_t elements_فی_cacheline = 16;
    ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    int64_t cachelines_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
    int64_t آغاز_cacheline       = thread_id * cachelines_فی_thread;
    int64_t انتہا_cacheline         = آغاز_cacheline + cachelines_فی_thread;

    اگر (انتہا_cacheline > کل_cachelines) {
        انتہا_cacheline = کل_cachelines;
    }

    اگر (آغاز_cacheline >= کل_cachelines) {
        واپس 0;
    }

    int64_t آغاز_elem = آغاز_cacheline * elements_فی_cacheline;
    int64_t انتہا_elem   = انتہا_cacheline * elements_فی_cacheline;
    اگر (انتہا_elem > کل_elements) {
        انتہا_elem = کل_elements;
    }

    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");
    __asm__ متلون("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ متلون("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    برائے (int64_t i = آغاز_elem; i < انتہا_elem; i += 8) {
        __asm__ متلون(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_ڈیٹا[i])
            : [src] "m"(*(ثابت اعشاری (*)[8]) & src0_ڈیٹا[i])
            : "f10", "f20", "f21");
    }
    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));

    واپس 0;
}
