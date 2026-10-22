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

#تعریف_کن TRI_TYPE_فوقانی_DIAG 0
#تعریف_کن TRI_TYPE_فوقانی      1
#تعریف_کن TRI_TYPE_پایینی_DIAG 2
#تعریف_کن TRI_TYPE_پایینی      3

بنية ggml_et_tri_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            tri_type;
};

ساكن مضمن صحيح keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    بدل (tri_type) {
        حالة TRI_TYPE_پایینی:
            أرجع i0 < i1;
        حالة TRI_TYPE_پایینی_DIAG:
            أرجع i0 <= i1;
        حالة TRI_TYPE_فوقانی:
            أرجع i0 > i1;
        حالة TRI_TYPE_فوقانی_DIAG:
            أرجع i0 >= i1;
        افتراضي:
            أرجع 0;
    }
}

صحيح مدخل_نقطه(بنية ggml_et_tri_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

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

    عائم * src0_داده = (عائم *) src0->داده;
    عائم * dst_داده  = (عائم *) dst->داده;

    إذا (!src0_داده || !dst_داده) {
        أرجع -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ثابت int64_t مجموع_ردیفها = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    إذا (ne0 % 16 == 0) {
        عائم zero = 0.0f;
        __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        لكل (int64_t ردیف = thread_id; ردیف < مجموع_ردیفها; ردیف += رقم_نخها) {
            ثابت int64_t i1 = ردیف % ne1;
            ثابت int64_t i2 = (ردیف / ne1) % ne2;
            ثابت int64_t i3 = ردیف / (ne1 * ne2);

            ثابت عائم * src_ردیف = (ثابت عائم *) ((ثابت حرف *) src0_داده + i1 * nb01 + i2 * nb02 + i3 * nb03);
            عائم *       dst_ردیف = (عائم *) ((حرف *) dst_داده + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_شروع, keep_انتها;
            بدل (tri_type) {
                حالة TRI_TYPE_پایینی:
                    keep_شروع = 0;
                    keep_انتها   = i1;
                    اكسر;
                حالة TRI_TYPE_پایینی_DIAG:
                    keep_شروع = 0;
                    keep_انتها   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_فوقانی:
                    keep_شروع = i1 + 1;
                    keep_انتها   = ne0;
                    اكسر;
                حالة TRI_TYPE_فوقانی_DIAG:
                    keep_شروع = i1;
                    keep_انتها   = ne0;
                    اكسر;
                افتراضي:
                    أرجع -1;
            }
            إذا (keep_انتها > ne0) {
                keep_انتها = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            لكل (; i0 + 8 <= keep_شروع; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_ردیف[i0])::"f10");
            }
            لكل (; i0 < keep_شروع; i0++) {
                dst_ردیف[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            لكل (; i0 + 8 <= keep_انتها; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(عائم (*)[8]) & dst_ردیف[i0])
                    : [s] "m"(*(ثابت عائم (*)[8]) & src_ردیف[i0])
                    : "f11");
            }
            لكل (; i0 < keep_انتها; i0++) {
                dst_ردیف[i0] = src_ردیف[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            لكل (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_ردیف[i0])::"f10");
            }
            لكل (; i0 < ne0; i0++) {
                dst_ردیف[i0] = 0.0f;
            }
        }
        أرجع 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ثابت int64_t مجموع_elements = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elems_در_هر_cl   = 16;
        ثابت int64_t مجموع_cl       = (مجموع_elements + elems_در_هر_cl - 1) / elems_در_هر_cl;

        ثابت int64_t cl_در_هر_thread = (مجموع_cl + رقم_نخها - 1) / رقم_نخها;
        ثابت int64_t cl_شروع      = thread_id * cl_در_هر_thread;
        int64_t       cl_انتها        = cl_شروع + cl_در_هر_thread;
        إذا (cl_انتها > مجموع_cl) {
            cl_انتها = مجموع_cl;
        }
        إذا (cl_شروع >= مجموع_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_شروع * elems_در_هر_cl;
        int64_t       ee = cl_انتها * elems_در_هر_cl;
        إذا (ee > مجموع_elements) {
            ee = مجموع_elements;
        }

        int64_t ردیف_idx = es / ne0;
        int64_t ستون     = es % ne0;

        int64_t pos = es;
        طالما (pos < ee) {
            ثابت int64_t i1 = ردیف_idx % ne1;
            ثابت int64_t i2 = (ردیف_idx / ne1) % ne2;
            ثابت int64_t i3 = ردیف_idx / (ne1 * ne2);

            ثابت عائم * src_ردیف = (ثابت عائم *) ((ثابت حرف *) src0_داده + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t ردیف_مانده   = ne0 - ستون;
            int64_t chunk_مانده = ee - pos;
            int64_t n               = ردیف_مانده < chunk_مانده ? ردیف_مانده : chunk_مانده;

            int64_t keep_شروع, keep_انتها;
            بدل (tri_type) {
                حالة TRI_TYPE_پایینی:
                    keep_شروع = 0;
                    keep_انتها   = i1;
                    اكسر;
                حالة TRI_TYPE_پایینی_DIAG:
                    keep_شروع = 0;
                    keep_انتها   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_فوقانی:
                    keep_شروع = i1 + 1;
                    keep_انتها   = ne0;
                    اكسر;
                حالة TRI_TYPE_فوقانی_DIAG:
                    keep_شروع = i1;
                    keep_انتها   = ne0;
                    اكسر;
                افتراضي:
                    أرجع -1;
            }
            إذا (keep_انتها > ne0) {
                keep_انتها = ne0;
            }

            int64_t انتها_ستون = ستون + n;
            لكل (int64_t i0 = ستون; i0 < انتها_ستون; i0++) {
                إذا (i0 >= keep_شروع && i0 < keep_انتها) {
                    dst_داده[pos + (i0 - ستون)] = src_ردیف[i0];
                } وإلا {
                    dst_داده[pos + (i0 - ستون)] = 0.0f;
                }
            }

            pos += n;
            ستون = 0;
            ردیف_idx++;
        }
    }

    أرجع 0;
}
