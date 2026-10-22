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

صحيح داخلا_نقطو(بنية ggml_et_scale_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

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
    ثابت int64_t ڪل_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ثابت int64_t elements_في_cacheline = 16;
    ثابت int64_t ڪل_cachelines       = (ڪل_elements + elements_في_cacheline - 1) / elements_في_cacheline;

    int64_t cachelines_في_thread = (ڪل_cachelines + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    int64_t شروعات_cacheline       = thread_id * cachelines_في_thread;
    int64_t پڇاڙي_cacheline         = شروعات_cacheline + cachelines_في_thread;

    إذا (پڇاڙي_cacheline > ڪل_cachelines) {
        پڇاڙي_cacheline = ڪل_cachelines;
    }

    إذا (شروعات_cacheline >= ڪل_cachelines) {
        أرجع 0;
    }

    int64_t شروعات_elem = شروعات_cacheline * elements_في_cacheline;
    int64_t پڇاڙي_elem   = پڇاڙي_cacheline * elements_في_cacheline;
    إذا (پڇاڙي_elem > ڪل_elements) {
        پڇاڙي_elem = ڪل_elements;
    }

    بنانشان ڊگهو عارضي_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضي_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");
    __asm__ متطاير("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ متطاير("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    لكل (int64_t i = شروعات_elem; i < پڇاڙي_elem; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_ٻاهريون] "=m"(*(عائم (*)[8]) & dst_معلومات[i])
            : [src] "m"(*(ثابت عائم (*)[8]) & src0_معلومات[i])
            : "f10", "f20", "f21");
    }
    __asm__ متطاير("mova.m.x %0" ::"r"(عارضي_نقاب));

    أرجع 0;
}
