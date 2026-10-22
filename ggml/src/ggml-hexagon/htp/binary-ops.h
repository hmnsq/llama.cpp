#که_نه_تعریف_شوی HTP_دوګونی_OPS_H
#تعریف_کړه HTP_دوګونی_OPS_H

#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "htp-ops.h"
#ضمن "htp-vtcm.h"

تعداد htp_دوګونی_kernel_type {
    HTP_دوګونی_KERNEL_SAME_SHAPE = 0,
    HTP_دوګونی_KERNEL_لیکه_BCAST,
    HTP_دوګونی_KERNEL_SCALAR,
    HTP_دوګونی_KERNEL_جمع_کړه_ID,
    HTP_دوګونی_KERNEL_COMPLEX,
    HTP_دوګونی_KERNEL_REPEAT,
    HTP_دوګونی_KERNEL_CHUNKED,
};

بنية htp_دوګونی_kernel_params {
    uint32_t kernel_type;
    uint32_t n_تارونه;
    uint32_t لیکې_فی_بفر;

    uint32_t src0_لیکه_اندازه_aligned;
    uint32_t src1_لیکه_اندازه_aligned;
    uint32_t dst_لیکه_اندازه_aligned;

    uint32_t src1_اندازه;
    uint32_t src2_اندازه;
    uint32_t vtcm_اندازه;

    uint32_t chunk_اندازه;
    uint32_t chunk_بایټونه;
    uint32_t is_scalar;

    uint8_t  وېشل_at_ne01;
    uint8_t  وېشل_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_دوګونی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

بنية ggml_tensor;
ساكن مضمن bool htp_دوګونی_strides_خوندي(
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    size_t elem_اندازه
) {
    إذا ((src0->ne[0] != 1 && src0->nb[0] != elem_اندازه) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_اندازه) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_اندازه)) {
        أرجع false;
    }
    إذا ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_اندازه) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_اندازه) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_اندازه)) {
        أرجع false;
    }
    أرجع true;
}
#وإلا
_Static_assert(حجم(بنية htp_دوګونی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#د_که_پای

بنية htp_دوګونی_vtcm_layout {
    size_t ټول_بایټونه;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_بایټونه_فی_thread;
    size_t src1_بایټونه_فی_thread;
    size_t dst_بایټونه_فی_thread;

    size_t src0_spad_half_اندازه;
    size_t src1_spad_half_اندازه;
    size_t dst_spad_half_اندازه;

    size_t src1_اندازه;
    size_t src2_اندازه;
    uint32_t لیکې_فی_بفر;
};

ساكن مضمن فراغ htp_دوګونی_vtcm_layout_build(
    بنية htp_دوګونی_vtcm_layout * L,
    ثابت بنية htp_دوګونی_kernel_params * kparams,
    size_t vtcm_اندازه
) {
    memset(L, 0, حجم(*L));

    ثابت uint32_t n_تارونه = kparams->n_تارونه;
    إذا (n_تارونه == 0) {
        أرجع;
    }

    إذا (kparams->kernel_type == HTP_دوګونی_KERNEL_CHUNKED) {
        ثابت size_t chunk_بایټونه = kparams->chunk_بایټونه;
        إذا (chunk_بایټونه == 0) {
            أرجع;
        }

        L->src0_بایټونه_فی_thread = 2 * chunk_بایټونه;
        L->src1_بایټونه_فی_thread = kparams->is_scalar ? 0 : (2 * chunk_بایټونه);
        L->dst_بایټونه_فی_thread  = 2 * chunk_بایټونه;

        L->src0_spad_half_اندازه = chunk_بایټونه;
        L->src1_spad_half_اندازه = kparams->is_scalar ? 0 : chunk_بایټونه;
        L->dst_spad_half_اندازه  = chunk_بایټونه;

        L->لیکې_فی_بفر = 1;
        L->src1_اندازه = 0;

        ثابت size_t src0_ټول = n_تارونه * L->src0_بایټونه_فی_thread;
        ثابت size_t src1_ټول = kparams->is_scalar ? 128 : (n_تارونه * L->src1_بایټونه_فی_thread);
        ثابت size_t dst_ټول  = n_تارونه * L->dst_بایټونه_فی_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_ټول);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_ټول);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ټول);

        إذا (off > vtcm_اندازه) {
            أرجع;
        }

        L->ټول_بایټونه = off;
        أرجع;
    }

    ثابت size_t spad_لیکه_ټول = (kparams->kernel_type == HTP_دوګونی_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_لیکه_اندازه_aligned + kparams->src1_لیکه_اندازه_aligned + kparams->dst_لیکه_اندازه_aligned)
        : 2 * (kparams->src0_لیکه_اندازه_aligned + kparams->dst_لیکه_اندازه_aligned);

    ثابت size_t static_vtcm_اندازه = kparams->src1_اندازه + kparams->src2_اندازه;
    إذا (spad_لیکه_ټول == 0 || vtcm_اندازه < static_vtcm_اندازه) {
        أرجع;
    }

    ثابت size_t لیکې_فی_بفر = (vtcm_اندازه - static_vtcm_اندازه) / (n_تارونه * spad_لیکه_ټول);
    إذا (لیکې_فی_بفر == 0) {
        أرجع;
    }

    L->لیکې_فی_بفر = (uint32_t) لیکې_فی_بفر;
    L->src1_اندازه = kparams->src1_اندازه;
    L->src2_اندازه = kparams->src2_اندازه;

    L->src0_بایټونه_فی_thread = لیکې_فی_بفر * 2 * kparams->src0_لیکه_اندازه_aligned;
    L->dst_بایټونه_فی_thread  = لیکې_فی_بفر * 2 * kparams->dst_لیکه_اندازه_aligned;
    L->src1_بایټونه_فی_thread = (kparams->kernel_type == HTP_دوګونی_KERNEL_SAME_SHAPE)
        ? لیکې_فی_بفر * 2 * kparams->src1_لیکه_اندازه_aligned
        : 0;

    L->src0_spad_half_اندازه = L->src0_بایټونه_فی_thread / 2;
    L->src1_spad_half_اندازه = L->src1_بایټونه_فی_thread / 2;
    L->dst_spad_half_اندازه  = L->dst_بایټونه_فی_thread / 2;

    ثابت size_t src0_ټول = n_تارونه * L->src0_بایټونه_فی_thread;
    ثابت size_t src1_ټول = (kparams->src1_اندازه > 0)
        ? kparams->src1_اندازه
        : n_تارونه * L->src1_بایټونه_فی_thread;
    ثابت size_t src2_ټول = kparams->src2_اندازه;
    ثابت size_t dst_ټول  = n_تارونه * L->dst_بایټونه_فی_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_ټول);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_ټول);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_ټول);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ټول);

    L->ټول_بایټونه = off;
}

#د_که_پای
