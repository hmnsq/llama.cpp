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

#وصف_ڪريو TRI_TYPE_مٿانهون_DIAG 0
#وصف_ڪريو TRI_TYPE_مٿانهون      1
#وصف_ڪريو TRI_TYPE_هيٺاهون_DIAG 2
#وصف_ڪريو TRI_TYPE_هيٺاهون      3

بنية ggml_et_tri_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            tri_type;
};

ساكن مضمن صحيح keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    بدل (tri_type) {
        حالة TRI_TYPE_هيٺاهون:
            أرجع i0 < i1;
        حالة TRI_TYPE_هيٺاهون_DIAG:
            أرجع i0 <= i1;
        حالة TRI_TYPE_مٿانهون:
            أرجع i0 > i1;
        حالة TRI_TYPE_مٿانهون_DIAG:
            أرجع i0 >= i1;
        افتراضي:
            أرجع 0;
    }
}

صحيح داخلا_نقطو(بنية ggml_et_tri_params * params, فراغ * env) {
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

    ثابت int64_t ڪل_صفون = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    إذا (ne0 % 16 == 0) {
        عائم zero = 0.0f;
        __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        لكل (int64_t صف = thread_id; صف < ڪل_صفون; صف += انگ_ڌاڳا) {
            ثابت int64_t i1 = صف % ne1;
            ثابت int64_t i2 = (صف / ne1) % ne2;
            ثابت int64_t i3 = صف / (ne1 * ne2);

            ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);
            عائم *       dst_صف = (عائم *) ((حرف *) dst_معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_شروعات, keep_پڇاڙي;
            بدل (tri_type) {
                حالة TRI_TYPE_هيٺاهون:
                    keep_شروعات = 0;
                    keep_پڇاڙي   = i1;
                    اكسر;
                حالة TRI_TYPE_هيٺاهون_DIAG:
                    keep_شروعات = 0;
                    keep_پڇاڙي   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_مٿانهون:
                    keep_شروعات = i1 + 1;
                    keep_پڇاڙي   = ne0;
                    اكسر;
                حالة TRI_TYPE_مٿانهون_DIAG:
                    keep_شروعات = i1;
                    keep_پڇاڙي   = ne0;
                    اكسر;
                افتراضي:
                    أرجع -1;
            }
            إذا (keep_پڇاڙي > ne0) {
                keep_پڇاڙي = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            لكل (; i0 + 8 <= keep_شروعات; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_صف[i0])::"f10");
            }
            لكل (; i0 < keep_شروعات; i0++) {
                dst_صف[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            لكل (; i0 + 8 <= keep_پڇاڙي; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(عائم (*)[8]) & dst_صف[i0])
                    : [s] "m"(*(ثابت عائم (*)[8]) & src_صف[i0])
                    : "f11");
            }
            لكل (; i0 < keep_پڇاڙي; i0++) {
                dst_صف[i0] = src_صف[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            لكل (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_صف[i0])::"f10");
            }
            لكل (; i0 < ne0; i0++) {
                dst_صف[i0] = 0.0f;
            }
        }
        أرجع 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ثابت int64_t ڪل_elements = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elems_في_cl   = 16;
        ثابت int64_t ڪل_cl       = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;

        ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
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

        int64_t صف_idx = es / ne0;
        int64_t کانو     = es % ne0;

        int64_t pos = es;
        طالما (pos < ee) {
            ثابت int64_t i1 = صف_idx % ne1;
            ثابت int64_t i2 = (صف_idx / ne1) % ne2;
            ثابت int64_t i3 = صف_idx / (ne1 * ne2);

            ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t صف_بچيل   = ne0 - کانو;
            int64_t chunk_بچيل = ee - pos;
            int64_t n               = صف_بچيل < chunk_بچيل ? صف_بچيل : chunk_بچيل;

            int64_t keep_شروعات, keep_پڇاڙي;
            بدل (tri_type) {
                حالة TRI_TYPE_هيٺاهون:
                    keep_شروعات = 0;
                    keep_پڇاڙي   = i1;
                    اكسر;
                حالة TRI_TYPE_هيٺاهون_DIAG:
                    keep_شروعات = 0;
                    keep_پڇاڙي   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_مٿانهون:
                    keep_شروعات = i1 + 1;
                    keep_پڇاڙي   = ne0;
                    اكسر;
                حالة TRI_TYPE_مٿانهون_DIAG:
                    keep_شروعات = i1;
                    keep_پڇاڙي   = ne0;
                    اكسر;
                افتراضي:
                    أرجع -1;
            }
            إذا (keep_پڇاڙي > ne0) {
                keep_پڇاڙي = ne0;
            }

            int64_t پڇاڙي_کانو = کانو + n;
            لكل (int64_t i0 = کانو; i0 < پڇاڙي_کانو; i0++) {
                إذا (i0 >= keep_شروعات && i0 < keep_پڇاڙي) {
                    dst_معلومات[pos + (i0 - کانو)] = src_صف[i0];
                } وإلا {
                    dst_معلومات[pos + (i0 - کانو)] = 0.0f;
                }
            }

            pos += n;
            کانو = 0;
            صف_idx++;
        }
    }

    أرجع 0;
}
