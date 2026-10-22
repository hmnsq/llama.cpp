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

صحيح مدخل_نقطه(بنية ggml_et_scale_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_داده = (عائم *) src0->داده;
    عائم * dst_داده  = (عائم *) dst->داده;

    إذا (!src0_داده || !dst_داده) {
        أرجع -1;
    }


    عائم scale = params->scale;
    عائم bias  = params->bias;

    // Total elements across all dimensions
    ثابت int64_t مجموع_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ثابت int64_t elements_در_هر_cacheline = 16;
    ثابت int64_t مجموع_cachelines       = (مجموع_elements + elements_در_هر_cacheline - 1) / elements_در_هر_cacheline;

    int64_t cachelines_در_هر_thread = (مجموع_cachelines + رقم_نخها - 1) / رقم_نخها;
    int64_t شروع_cacheline       = thread_id * cachelines_در_هر_thread;
    int64_t انتها_cacheline         = شروع_cacheline + cachelines_در_هر_thread;

    إذا (انتها_cacheline > مجموع_cachelines) {
        انتها_cacheline = مجموع_cachelines;
    }

    إذا (شروع_cacheline >= مجموع_cachelines) {
        أرجع 0;
    }

    int64_t شروع_elem = شروع_cacheline * elements_در_هر_cacheline;
    int64_t انتها_elem   = انتها_cacheline * elements_در_هر_cacheline;
    إذا (انتها_elem > مجموع_elements) {
        انتها_elem = مجموع_elements;
    }

    بیعلامت دراز موقت_ماسک;
    __asm__ متطاير("mova.x.m %0" : "=r"(موقت_ماسک));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");
    __asm__ متطاير("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ متطاير("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    لكل (int64_t i = شروع_elem; i < انتها_elem; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_بیرون] "=m"(*(عائم (*)[8]) & dst_داده[i])
            : [src] "m"(*(ثابت عائم (*)[8]) & src0_داده[i])
            : "f10", "f20", "f21");
    }
    __asm__ متطاير("mova.m.x %0" ::"r"(موقت_ماسک));

    أرجع 0;
}
