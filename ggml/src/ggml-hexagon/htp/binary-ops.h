#جے_بنا_تعریف HTP_ثنائی_OPS_H
#تعریف_کرو HTP_ثنائی_OPS_H

#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "htp-ops.h"
#ضمن "htp-vtcm.h"

تعداد htp_ثنائی_kernel_type {
    HTP_ثنائی_KERNEL_SAME_SHAPE = 0,
    HTP_ثنائی_KERNEL_صف_BCAST,
    HTP_ثنائی_KERNEL_SCALAR,
    HTP_ثنائی_KERNEL_جوڑو_ID,
    HTP_ثنائی_KERNEL_COMPLEX,
    HTP_ثنائی_KERNEL_REPEAT,
    HTP_ثنائی_KERNEL_CHUNKED,
};

بنية htp_ثنائی_kernel_params {
    uint32_t kernel_type;
    uint32_t n_دھاگے;
    uint32_t صفاں_فی_بفر;

    uint32_t src0_صف_ناپ_aligned;
    uint32_t src1_صف_ناپ_aligned;
    uint32_t dst_صف_ناپ_aligned;

    uint32_t src1_ناپ;
    uint32_t src2_ناپ;
    uint32_t vtcm_ناپ;

    uint32_t chunk_ناپ;
    uint32_t chunk_بائٹاں;
    uint32_t is_scalar;

    uint8_t  ونڈو_at_ne01;
    uint8_t  ونڈو_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ثنائی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

بنية ggml_tensor;
ساكن مضمن bool htp_ثنائی_strides_محفوظ(
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    size_t elem_ناپ
) {
    إذا ((src0->ne[0] != 1 && src0->nb[0] != elem_ناپ) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_ناپ) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_ناپ)) {
        أرجع false;
    }
    إذا ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_ناپ) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_ناپ) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_ناپ)) {
        أرجع false;
    }
    أرجع true;
}
#وإلا
_Static_assert(حجم(بنية htp_ثنائی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#جے_انت

بنية htp_ثنائی_vtcm_layout {
    size_t کل_بائٹاں;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_بائٹاں_فی_thread;
    size_t src1_بائٹاں_فی_thread;
    size_t dst_بائٹاں_فی_thread;

    size_t src0_spad_half_ناپ;
    size_t src1_spad_half_ناپ;
    size_t dst_spad_half_ناپ;

    size_t src1_ناپ;
    size_t src2_ناپ;
    uint32_t صفاں_فی_بفر;
};

ساكن مضمن فراغ htp_ثنائی_vtcm_layout_build(
    بنية htp_ثنائی_vtcm_layout * L,
    ثابت بنية htp_ثنائی_kernel_params * kparams,
    size_t vtcm_ناپ
) {
    memset(L, 0, حجم(*L));

    ثابت uint32_t n_دھاگے = kparams->n_دھاگے;
    إذا (n_دھاگے == 0) {
        أرجع;
    }

    إذا (kparams->kernel_type == HTP_ثنائی_KERNEL_CHUNKED) {
        ثابت size_t chunk_بائٹاں = kparams->chunk_بائٹاں;
        إذا (chunk_بائٹاں == 0) {
            أرجع;
        }

        L->src0_بائٹاں_فی_thread = 2 * chunk_بائٹاں;
        L->src1_بائٹاں_فی_thread = kparams->is_scalar ? 0 : (2 * chunk_بائٹاں);
        L->dst_بائٹاں_فی_thread  = 2 * chunk_بائٹاں;

        L->src0_spad_half_ناپ = chunk_بائٹاں;
        L->src1_spad_half_ناپ = kparams->is_scalar ? 0 : chunk_بائٹاں;
        L->dst_spad_half_ناپ  = chunk_بائٹاں;

        L->صفاں_فی_بفر = 1;
        L->src1_ناپ = 0;

        ثابت size_t src0_کل = n_دھاگے * L->src0_بائٹاں_فی_thread;
        ثابت size_t src1_کل = kparams->is_scalar ? 128 : (n_دھاگے * L->src1_بائٹاں_فی_thread);
        ثابت size_t dst_کل  = n_دھاگے * L->dst_بائٹاں_فی_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_کل);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_کل);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_کل);

        إذا (off > vtcm_ناپ) {
            أرجع;
        }

        L->کل_بائٹاں = off;
        أرجع;
    }

    ثابت size_t spad_صف_کل = (kparams->kernel_type == HTP_ثنائی_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_صف_ناپ_aligned + kparams->src1_صف_ناپ_aligned + kparams->dst_صف_ناپ_aligned)
        : 2 * (kparams->src0_صف_ناپ_aligned + kparams->dst_صف_ناپ_aligned);

    ثابت size_t static_vtcm_ناپ = kparams->src1_ناپ + kparams->src2_ناپ;
    إذا (spad_صف_کل == 0 || vtcm_ناپ < static_vtcm_ناپ) {
        أرجع;
    }

    ثابت size_t صفاں_فی_بفر = (vtcm_ناپ - static_vtcm_ناپ) / (n_دھاگے * spad_صف_کل);
    إذا (صفاں_فی_بفر == 0) {
        أرجع;
    }

    L->صفاں_فی_بفر = (uint32_t) صفاں_فی_بفر;
    L->src1_ناپ = kparams->src1_ناپ;
    L->src2_ناپ = kparams->src2_ناپ;

    L->src0_بائٹاں_فی_thread = صفاں_فی_بفر * 2 * kparams->src0_صف_ناپ_aligned;
    L->dst_بائٹاں_فی_thread  = صفاں_فی_بفر * 2 * kparams->dst_صف_ناپ_aligned;
    L->src1_بائٹاں_فی_thread = (kparams->kernel_type == HTP_ثنائی_KERNEL_SAME_SHAPE)
        ? صفاں_فی_بفر * 2 * kparams->src1_صف_ناپ_aligned
        : 0;

    L->src0_spad_half_ناپ = L->src0_بائٹاں_فی_thread / 2;
    L->src1_spad_half_ناپ = L->src1_بائٹاں_فی_thread / 2;
    L->dst_spad_half_ناپ  = L->dst_بائٹاں_فی_thread / 2;

    ثابت size_t src0_کل = n_دھاگے * L->src0_بائٹاں_فی_thread;
    ثابت size_t src1_کل = (kparams->src1_ناپ > 0)
        ? kparams->src1_ناپ
        : n_دھاگے * L->src1_بائٹاں_فی_thread;
    ثابت size_t src2_کل = kparams->src2_ناپ;
    ثابت size_t dst_کل  = n_دھاگے * L->dst_بائٹاں_فی_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_کل);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_کل);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_کل);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_کل);

    L->کل_بائٹاں = off;
}

#جے_انت
