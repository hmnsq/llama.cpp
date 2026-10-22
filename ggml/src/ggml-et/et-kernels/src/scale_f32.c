//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_scale_params {
    بنية ggml_tensor src0;   // F32 input tensor
    بنية ggml_tensor dst;    // F32 output tensor
    عائم              scale;  // Scale factor
    عائم              bias;   // Bias (additive offset)
};

صحيح ثبت_ټکی(بنية ggml_et_scale_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;
    }


    عائم scale = params->scale;
    عائم bias  = params->bias;

    // Total elements across all dimensions
    ثابت int64_t ټول_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ثابت int64_t elements_فی_cacheline = 16;
    ثابت int64_t ټول_cachelines       = (ټول_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    int64_t cachelines_فی_thread = (ټول_cachelines + عدد_تارونه - 1) / عدد_تارونه;
    int64_t پیل_cacheline       = thread_id * cachelines_فی_thread;
    int64_t پای_ټکی_cacheline         = پیل_cacheline + cachelines_فی_thread;

    إذا (پای_ټکی_cacheline > ټول_cachelines) {
        پای_ټکی_cacheline = ټول_cachelines;
    }

    إذا (پیل_cacheline >= ټول_cachelines) {
        أرجع 0;
    }

    int64_t پیل_elem = پیل_cacheline * elements_فی_cacheline;
    int64_t پای_ټکی_elem   = پای_ټکی_cacheline * elements_فی_cacheline;
    إذا (پای_ټکی_elem > ټول_elements) {
        پای_ټکی_elem = ټول_elements;
    }

    بینښه اوږد لنډمهاله_ماسک;
    __asm__ متطاير("mova.x.m %0" : "=r"(لنډمهاله_ماسک));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");
    __asm__ متطاير("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ متطاير("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    لكل (int64_t i = پیل_elem; i < پای_ټکی_elem; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_بهر] "=m"(*(عائم (*)[8]) & dst_معلومات[i])
            : [src] "m"(*(ثابت عائم (*)[8]) & src0_معلومات[i])
            : "f10", "f20", "f21");
    }
    __asm__ متطاير("mova.m.x %0" ::"r"(لنډمهاله_ماسک));

    أرجع 0;
}
