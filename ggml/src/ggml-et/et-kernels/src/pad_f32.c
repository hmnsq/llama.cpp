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

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_pad_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ساکن اندرونی خالی vec_copy_f32(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_انتہا = (n / 8) * 8;
    برائے (; i < vec_انتہا; i += 8) {
        __asm__ متلون(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [s] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f10");
    }
    برائے (; i < n; i++) {
        dst[i] = src[i];
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_pad_params * params, خالی * env) {
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

    ثابت اعشاری * src0_ڈیٹا = (ثابت اعشاری *) src0->ڈیٹا;
    اعشاری *       dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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

    ثابت int64_t کل_صفیں     = ne1 * ne2 * ne3;
    ثابت int64_t کل_elements = ne0 * کل_صفیں;

    اگر (کل_elements == 0) {
        واپس 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    اعشاری zero = 0.0f;
    __asm__ متلون("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    اگر (ne0 % 16 == 0) {
        برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
            ثابت int64_t i3 = صف / (ne1 * ne2);
            ثابت int64_t i2 = (صف / ne1) % ne2;
            ثابت int64_t i1 = صف % ne1;

            اعشاری * dst_صف = dst_ڈیٹا + صف * ne0;

            اگر (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_صف, src_صف, (int32_t) ne0);
            } ورنہ {
                int64_t       i       = 0;
                ثابت int64_t vec_انتہا = (ne0 / 8) * 8;
                برائے (; i < vec_انتہا; i += 8) {
                    __asm__ متلون("fsw.ps f12, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_صف[i])::"f12");
                }
            }
        }
        واپس 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ثابت int64_t elems_فی_cl = 16;
    ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

    ثابت int64_t ne1_ڈیٹا_انتہا = ne1 - rp1;
    ثابت int64_t ne2_ڈیٹا_انتہا = ne2 - rp2;
    ثابت int64_t ne3_ڈیٹا_انتہا = ne3 - rp3;

    برائے (int64_t cl = thread_id; cl < کل_cl; cl += عدد_دھاگے) {
        ثابت int64_t elem_آغاز = cl * elems_فی_cl;
        int64_t       elem_انتہا   = elem_آغاز + elems_فی_cl;
        اگر (elem_انتہا > کل_elements) {
            elem_انتہا = کل_elements;
        }

        برائے (int64_t idx = elem_آغاز; idx < elem_انتہا; idx++) {
            ثابت int64_t i0   = idx % ne0;
            ثابت int64_t rem  = idx / ne0;
            ثابت int64_t i1   = rem % ne1;
            ثابت int64_t rem2 = rem / ne1;
            ثابت int64_t i2   = rem2 % ne2;
            ثابت int64_t i3   = rem2 / ne2;

            اگر (i1 >= lp1 && i1 < ne1_ڈیٹا_انتہا && i2 >= lp2 && i2 < ne2_ڈیٹا_انتہا && i3 >= lp3 && i3 < ne3_ڈیٹا_انتہا) {
                ثابت اعشاری * sp = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_ڈیٹا[idx]    = *sp;
            } ورنہ {
                dst_ڈیٹا[idx] = 0.0f;
            }
        }
    }

    واپس 0;
}
