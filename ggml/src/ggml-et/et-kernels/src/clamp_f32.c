//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_clamp_params {
    بنية ggml_tensor src0;  // F32 input  (contiguous)
    بنية ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    عائم              min_val;
    عائم              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ساكن مضمن فراغ clamp_بلاڪ_f32(عائم * dst, ثابت عائم * src, عائم min_val, عائم max_val, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_پڇاڙي = (n / 8) * 8;

    إذا (vec_پڇاڙي > 0) {
        بنانشان ڊگهو عارضي_نقاب;
        __asm__ متطاير("mova.x.m %0" : "=r"(عارضي_نقاب));
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");

        لكل (; i < vec_پڇاڙي; i += 8) {
            __asm__ متطاير(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(عائم (*)[8]) & dst[i])
                : [s] "m"(*(ثابت عائم (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ متطاير("mova.m.x %0" ::"r"(عارضي_نقاب));
    }

    لكل (; i < n; i++) {
        عائم v = src[i];
        إذا (v < min_val) {
            v = min_val;
        }
        إذا (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

صحيح داخلا_نقطو(بنية ggml_et_clamp_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

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

    ثابت int64_t ڪل_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    إذا (ڪل_elements <= 0) {
        أرجع 0;
    }

    ثابت عائم min_val = params->min_val;
    ثابت عائم max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ثابت int64_t elems_في_cl = 16;
    ثابت int64_t ڪل_cl     = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;

    ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    ثابت int64_t cl_شروعات      = (int64_t) thread_id * cl_في_thread;
    int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
    إذا (cl_پڇاڙي > ڪل_cl) {
        cl_پڇاڙي = ڪل_cl;
    }
    إذا (cl_شروعات >= ڪل_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_شروعات * elems_في_cl;
    int64_t       ee = cl_پڇاڙي * elems_في_cl;
    إذا (ee > ڪل_elements) {
        ee = ڪل_elements;
    }

    clamp_بلاڪ_f32(dst_معلومات + es, src0_معلومات + es, min_val, max_val, (int32_t) (ee - es));
    أرجع 0;
}
