//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_clamp_params {
    ساخت ggml_tensor src0;  // F32 input  (contiguous)
    ساخت ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    اعشاری              min_val;
    اعشاری              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ساکن اندرونی خالی clamp_بلاک_f32(اعشاری * dst, ثابت اعشاری * src, اعشاری min_val, اعشاری max_val, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_انتہا = (n / 8) * 8;

    اگر (vec_انتہا > 0) {
        غیرعلامتی طویل عارضی_نقاب;
        __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
        __asm__ متلون("mov.m.x m0, x0, 0xFF");

        برائے (; i < vec_انتہا; i += 8) {
            __asm__ متلون(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(اعشاری (*)[8]) & dst[i])
                : [s] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));
    }

    برائے (; i < n; i++) {
        اعشاری v = src[i];
        اگر (v < min_val) {
            v = min_val;
        }
        اگر (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_clamp_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

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

    ثابت int64_t کل_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    اگر (کل_elements <= 0) {
        واپس 0;
    }

    ثابت اعشاری min_val = params->min_val;
    ثابت اعشاری max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ثابت int64_t elems_فی_cl = 16;
    ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

    ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_آغاز      = (int64_t) thread_id * cl_فی_thread;
    int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
    اگر (cl_انتہا > کل_cl) {
        cl_انتہا = کل_cl;
    }
    اگر (cl_آغاز >= کل_cl) {
        واپس 0;
    }

    ثابت int64_t es = cl_آغاز * elems_فی_cl;
    int64_t       ee = cl_انتہا * elems_فی_cl;
    اگر (ee > کل_elements) {
        ee = کل_elements;
    }

    clamp_بلاک_f32(dst_ڈیٹا + es, src0_ڈیٹا + es, min_val, max_val, (int32_t) (ee - es));
    واپس 0;
}
