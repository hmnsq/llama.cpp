//******************************************************************************
// Tri F32 Kernel
// Triangular masking: zero out elements outside the triangular region.
//
// tri_type (matches ggml_tri_type enum):
//   0 = UPPER_DIAG:  keep where i0 >= i1
//   1 = UPPER:       keep where i0 >  i1
//   2 = LOWER_DIAG:  keep where i0 <= i1
//   3 = LOWER:       keep where i0 <  i1
//
// Distribution: cache-line aligned chunks of the flat contiguous dst.
// Each element is individually classified as keep or zero based on its
// (i0, i1) coordinates. This avoids cache-line sharing between threads
// when ne0 is not a multiple of 16.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

#تعریف_کړه TRI_TYPE_پاسنی_DIAG 0
#تعریف_کړه TRI_TYPE_پاسنی      1
#تعریف_کړه TRI_TYPE_ښکتنی_DIAG 2
#تعریف_کړه TRI_TYPE_ښکتنی      3

بنية ggml_et_tri_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            tri_type;
};

ساكن مضمن صحيح keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    بدل (tri_type) {
        حالة TRI_TYPE_ښکتنی:
            أرجع i0 < i1;
        حالة TRI_TYPE_ښکتنی_DIAG:
            أرجع i0 <= i1;
        حالة TRI_TYPE_پاسنی:
            أرجع i0 > i1;
        حالة TRI_TYPE_پاسنی_DIAG:
            أرجع i0 >= i1;
        افتراضي:
            أرجع 0;
    }
}

صحيح ثبت_ټکی(بنية ggml_et_tri_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0     = &params->src0;
    بنية ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ثابت int64_t ټول_لیکې = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    إذا (ne0 % 16 == 0) {
        عائم zero = 0.0f;
        __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        لكل (int64_t لیکه = thread_id; لیکه < ټول_لیکې; لیکه += عدد_تارونه) {
            ثابت int64_t i1 = لیکه % ne1;
            ثابت int64_t i2 = (لیکه / ne1) % ne2;
            ثابت int64_t i3 = لیکه / (ne1 * ne2);

            ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);
            عائم *       dst_لیکه = (عائم *) ((حرف *) dst_معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_پیل, keep_پای_ټکی;
            بدل (tri_type) {
                حالة TRI_TYPE_ښکتنی:
                    keep_پیل = 0;
                    keep_پای_ټکی   = i1;
                    اكسر;
                حالة TRI_TYPE_ښکتنی_DIAG:
                    keep_پیل = 0;
                    keep_پای_ټکی   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_پاسنی:
                    keep_پیل = i1 + 1;
                    keep_پای_ټکی   = ne0;
                    اكسر;
                حالة TRI_TYPE_پاسنی_DIAG:
                    keep_پیل = i1;
                    keep_پای_ټکی   = ne0;
                    اكسر;
                افتراضي:
                    أرجع -1;
            }
            إذا (keep_پای_ټکی > ne0) {
                keep_پای_ټکی = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            لكل (; i0 + 8 <= keep_پیل; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_لیکه[i0])::"f10");
            }
            لكل (; i0 < keep_پیل; i0++) {
                dst_لیکه[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            لكل (; i0 + 8 <= keep_پای_ټکی; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(عائم (*)[8]) & dst_لیکه[i0])
                    : [s] "m"(*(ثابت عائم (*)[8]) & src_لیکه[i0])
                    : "f11");
            }
            لكل (; i0 < keep_پای_ټکی; i0++) {
                dst_لیکه[i0] = src_لیکه[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            لكل (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_لیکه[i0])::"f10");
            }
            لكل (; i0 < ne0; i0++) {
                dst_لیکه[i0] = 0.0f;
            }
        }
        أرجع 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ثابت int64_t ټول_elements = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elems_فی_cl   = 16;
        ثابت int64_t ټول_cl       = (ټول_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (ټول_cl + عدد_تارونه - 1) / عدد_تارونه;
        ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
        int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
        إذا (cl_پای_ټکی > ټول_cl) {
            cl_پای_ټکی = ټول_cl;
        }
        إذا (cl_پیل >= ټول_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_پیل * elems_فی_cl;
        int64_t       ee = cl_پای_ټکی * elems_فی_cl;
        إذا (ee > ټول_elements) {
            ee = ټول_elements;
        }

        int64_t لیکه_idx = es / ne0;
        int64_t ستون     = es % ne0;

        int64_t pos = es;
        طالما (pos < ee) {
            ثابت int64_t i1 = لیکه_idx % ne1;
            ثابت int64_t i2 = (لیکه_idx / ne1) % ne2;
            ثابت int64_t i3 = لیکه_idx / (ne1 * ne2);

            ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t لیکه_پاتې_شوی   = ne0 - ستون;
            int64_t chunk_پاتې_شوی = ee - pos;
            int64_t n               = لیکه_پاتې_شوی < chunk_پاتې_شوی ? لیکه_پاتې_شوی : chunk_پاتې_شوی;

            int64_t keep_پیل, keep_پای_ټکی;
            بدل (tri_type) {
                حالة TRI_TYPE_ښکتنی:
                    keep_پیل = 0;
                    keep_پای_ټکی   = i1;
                    اكسر;
                حالة TRI_TYPE_ښکتنی_DIAG:
                    keep_پیل = 0;
                    keep_پای_ټکی   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_پاسنی:
                    keep_پیل = i1 + 1;
                    keep_پای_ټکی   = ne0;
                    اكسر;
                حالة TRI_TYPE_پاسنی_DIAG:
                    keep_پیل = i1;
                    keep_پای_ټکی   = ne0;
                    اكسر;
                افتراضي:
                    أرجع -1;
            }
            إذا (keep_پای_ټکی > ne0) {
                keep_پای_ټکی = ne0;
            }

            int64_t پای_ټکی_ستون = ستون + n;
            لكل (int64_t i0 = ستون; i0 < پای_ټکی_ستون; i0++) {
                إذا (i0 >= keep_پیل && i0 < keep_پای_ټکی) {
                    dst_معلومات[pos + (i0 - ستون)] = src_لیکه[i0];
                } وإلا {
                    dst_معلومات[pos + (i0 - ستون)] = 0.0f;
                }
            }

            pos += n;
            ستون = 0;
            لیکه_idx++;
        }
    }

    أرجع 0;
}
