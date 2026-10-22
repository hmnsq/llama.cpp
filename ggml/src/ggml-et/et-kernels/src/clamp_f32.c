//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_clamp_params {
    ساختار ggml_tensor src0;  // F32 input  (contiguous)
    ساختار ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    اعشاری              min_val;
    اعشاری              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ایستا درونخطی تهی clamp_block_f32(اعشاری * dst, ثابت اعشاری * src, اعشاری min_val, اعشاری max_val, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_end = (n / 8) * 8;

    اگر (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ فرار("mova.x.m %0" : "=r"(temp_mask));
        __asm__ فرار("mov.m.x m0, x0, 0xFF");

        برای (; i < vec_end; i += 8) {
            __asm__ فرار(
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

        __asm__ فرار("mova.m.x %0" ::"r"(temp_mask));
    }

    برای (; i < n; i++) {
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

صحیح entry_point(ساختار ggml_et_clamp_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;
    }

    ثابت int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    اگر (total_elements <= 0) {
        بازگشت 0;
    }

    ثابت اعشاری min_val = params->min_val;
    ثابت اعشاری max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ثابت int64_t elems_per_cl = 16;
    ثابت int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ثابت int64_t cl_per_thread = (total_cl + رقم_threads - 1) / رقم_threads;
    ثابت int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    اگر (cl_end > total_cl) {
        cl_end = total_cl;
    }
    اگر (cl_start >= total_cl) {
        بازگشت 0;
    }

    ثابت int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    اگر (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    بازگشت 0;
}
