//******************************************************************************
// Bare Metal PAD F32 Kernel
// Zero-pads an F32 tensor along dimensions 1-3.
//
// Constraints:
//   - No dim0 padding (lp[0]==0, rp[0]==0)
//   - dst contiguous
//   - src nb[0] == 4 (dim0 contiguous for vectorized reads)
//   - Zero-pad only (no circular mode)
//
// Two paths:
//   Aligned (ne0 % 16 == 0): rows distributed across harts, vectorized.
//   Small   (16 % ne0 == 0): cache-line distributed, scalar per-element.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_pad_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ساكن مضمن فراغ vec_copy_f32(عائم * dst, ثابت عائم * src, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_سرا = (n / 8) * 8;
    لكل (; i < vec_سرا; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(عائم (*)[8]) & dst[i])
            : [s] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f10");
    }
    لكل (; i < n; i++) {
        dst[i] = src[i];
    }
}

صحيح اندراج_نقطہ(بنية ggml_et_pad_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

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

    ثابت عائم * src0_انکڑے = (ثابت عائم *) src0->انکڑے;
    عائم *       dst_انکڑے  = (عائم *) dst->انکڑے;

    إذا (!src0_انکڑے || !dst_انکڑے) {
        أرجع -1;
    }

    // Dst dimensions
    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ثابت int64_t nb1_src = src0->nb[1];
    ثابت int64_t nb2_src = src0->nb[2];
    ثابت int64_t nb3_src = src0->nb[3];

    // Padding values
    ثابت int32_t lp1 = params->lp[1];
    ثابت int32_t rp1 = params->rp[1];
    ثابت int32_t lp2 = params->lp[2];
    ثابت int32_t rp2 = params->rp[2];
    ثابت int32_t lp3 = params->lp[3];
    ثابت int32_t rp3 = params->rp[3];

    ثابت int64_t کل_صفاں     = ne1 * ne2 * ne3;
    ثابت int64_t کل_elements = ne0 * کل_صفاں;

    إذا (کل_elements == 0) {
        أرجع 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    عائم zero = 0.0f;
    __asm__ متطاير("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    إذا (ne0 % 16 == 0) {
        لكل (int64_t صف = thread_id; صف < کل_صفاں; صف += عدد_دھاگے) {
            ثابت int64_t i3 = صف / (ne1 * ne2);
            ثابت int64_t i2 = (صف / ne1) % ne2;
            ثابت int64_t i1 = صف % ne1;

            عائم * dst_صف = dst_انکڑے + صف * ne0;

            إذا (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_صف, src_صف, (int32_t) ne0);
            } وإلا {
                int64_t       i       = 0;
                ثابت int64_t vec_سرا = (ne0 / 8) * 8;
                لكل (; i < vec_سرا; i += 8) {
                    __asm__ متطاير("fsw.ps f12, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_صف[i])::"f12");
                }
            }
        }
        أرجع 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ثابت int64_t elems_فی_cl = 16;
    ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

    ثابت int64_t ne1_انکڑے_سرا = ne1 - rp1;
    ثابت int64_t ne2_انکڑے_سرا = ne2 - rp2;
    ثابت int64_t ne3_انکڑے_سرا = ne3 - rp3;

    لكل (int64_t cl = thread_id; cl < کل_cl; cl += عدد_دھاگے) {
        ثابت int64_t elem_شروع = cl * elems_فی_cl;
        int64_t       elem_سرا   = elem_شروع + elems_فی_cl;
        إذا (elem_سرا > کل_elements) {
            elem_سرا = کل_elements;
        }

        لكل (int64_t idx = elem_شروع; idx < elem_سرا; idx++) {
            ثابت int64_t i0   = idx % ne0;
            ثابت int64_t rem  = idx / ne0;
            ثابت int64_t i1   = rem % ne1;
            ثابت int64_t rem2 = rem / ne1;
            ثابت int64_t i2   = rem2 % ne2;
            ثابت int64_t i3   = rem2 / ne2;

            إذا (i1 >= lp1 && i1 < ne1_انکڑے_سرا && i2 >= lp2 && i2 < ne2_انکڑے_سرا && i3 >= lp3 && i3 < ne3_انکڑے_سرا) {
                ثابت عائم * sp = (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_انکڑے[idx]    = *sp;
            } وإلا {
                dst_انکڑے[idx] = 0.0f;
            }
        }
    }

    أرجع 0;
}
