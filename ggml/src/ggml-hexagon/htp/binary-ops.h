#اگر_تعریف_نشده HTP_دودویی_OPS_H
#تعریف_کن HTP_دودویی_OPS_H

#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "htp-ops.h"
#ضمن "htp-vtcm.h"

تعداد htp_دودویی_kernel_type {
    HTP_دودویی_KERNEL_SAME_SHAPE = 0,
    HTP_دودویی_KERNEL_ردیف_BCAST,
    HTP_دودویی_KERNEL_SCALAR,
    HTP_دودویی_KERNEL_جمع_کن_ID,
    HTP_دودویی_KERNEL_COMPLEX,
    HTP_دودویی_KERNEL_REPEAT,
    HTP_دودویی_KERNEL_CHUNKED,
};

بنية htp_دودویی_kernel_params {
    uint32_t kernel_type;
    uint32_t n_نخها;
    uint32_t ردیفها_در_هر_میانگیر;

    uint32_t src0_ردیف_اندازه_aligned;
    uint32_t src1_ردیف_اندازه_aligned;
    uint32_t dst_ردیف_اندازه_aligned;

    uint32_t src1_اندازه;
    uint32_t src2_اندازه;
    uint32_t vtcm_اندازه;

    uint32_t chunk_اندازه;
    uint32_t chunk_بایتها;
    uint32_t is_scalar;

    uint8_t  تقسیم_at_ne01;
    uint8_t  تقسیم_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_دودویی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

بنية ggml_tensor;
ساكن مضمن bool htp_دودویی_strides_امن(
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
_Static_assert(حجم(بنية htp_دودویی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#پایان_اگر

بنية htp_دودویی_vtcm_layout {
    size_t مجموع_بایتها;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_بایتها_در_هر_thread;
    size_t src1_بایتها_در_هر_thread;
    size_t dst_بایتها_در_هر_thread;

    size_t src0_spad_half_اندازه;
    size_t src1_spad_half_اندازه;
    size_t dst_spad_half_اندازه;

    size_t src1_اندازه;
    size_t src2_اندازه;
    uint32_t ردیفها_در_هر_میانگیر;
};

ساكن مضمن فراغ htp_دودویی_vtcm_layout_build(
    بنية htp_دودویی_vtcm_layout * L,
    ثابت بنية htp_دودویی_kernel_params * kparams,
    size_t vtcm_اندازه
) {
    memset(L, 0, حجم(*L));

    ثابت uint32_t n_نخها = kparams->n_نخها;
    إذا (n_نخها == 0) {
        أرجع;
    }

    إذا (kparams->kernel_type == HTP_دودویی_KERNEL_CHUNKED) {
        ثابت size_t chunk_بایتها = kparams->chunk_بایتها;
        إذا (chunk_بایتها == 0) {
            أرجع;
        }

        L->src0_بایتها_در_هر_thread = 2 * chunk_بایتها;
        L->src1_بایتها_در_هر_thread = kparams->is_scalar ? 0 : (2 * chunk_بایتها);
        L->dst_بایتها_در_هر_thread  = 2 * chunk_بایتها;

        L->src0_spad_half_اندازه = chunk_بایتها;
        L->src1_spad_half_اندازه = kparams->is_scalar ? 0 : chunk_بایتها;
        L->dst_spad_half_اندازه  = chunk_بایتها;

        L->ردیفها_در_هر_میانگیر = 1;
        L->src1_اندازه = 0;

        ثابت size_t src0_مجموع = n_نخها * L->src0_بایتها_در_هر_thread;
        ثابت size_t src1_مجموع = kparams->is_scalar ? 128 : (n_نخها * L->src1_بایتها_در_هر_thread);
        ثابت size_t dst_مجموع  = n_نخها * L->dst_بایتها_در_هر_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_مجموع);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_مجموع);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_مجموع);

        إذا (off > vtcm_اندازه) {
            أرجع;
        }

        L->مجموع_بایتها = off;
        أرجع;
    }

    ثابت size_t spad_ردیف_مجموع = (kparams->kernel_type == HTP_دودویی_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_ردیف_اندازه_aligned + kparams->src1_ردیف_اندازه_aligned + kparams->dst_ردیف_اندازه_aligned)
        : 2 * (kparams->src0_ردیف_اندازه_aligned + kparams->dst_ردیف_اندازه_aligned);

    ثابت size_t static_vtcm_اندازه = kparams->src1_اندازه + kparams->src2_اندازه;
    إذا (spad_ردیف_مجموع == 0 || vtcm_اندازه < static_vtcm_اندازه) {
        أرجع;
    }

    ثابت size_t ردیفها_در_هر_میانگیر = (vtcm_اندازه - static_vtcm_اندازه) / (n_نخها * spad_ردیف_مجموع);
    إذا (ردیفها_در_هر_میانگیر == 0) {
        أرجع;
    }

    L->ردیفها_در_هر_میانگیر = (uint32_t) ردیفها_در_هر_میانگیر;
    L->src1_اندازه = kparams->src1_اندازه;
    L->src2_اندازه = kparams->src2_اندازه;

    L->src0_بایتها_در_هر_thread = ردیفها_در_هر_میانگیر * 2 * kparams->src0_ردیف_اندازه_aligned;
    L->dst_بایتها_در_هر_thread  = ردیفها_در_هر_میانگیر * 2 * kparams->dst_ردیف_اندازه_aligned;
    L->src1_بایتها_در_هر_thread = (kparams->kernel_type == HTP_دودویی_KERNEL_SAME_SHAPE)
        ? ردیفها_در_هر_میانگیر * 2 * kparams->src1_ردیف_اندازه_aligned
        : 0;

    L->src0_spad_half_اندازه = L->src0_بایتها_در_هر_thread / 2;
    L->src1_spad_half_اندازه = L->src1_بایتها_در_هر_thread / 2;
    L->dst_spad_half_اندازه  = L->dst_بایتها_در_هر_thread / 2;

    ثابت size_t src0_مجموع = n_نخها * L->src0_بایتها_در_هر_thread;
    ثابت size_t src1_مجموع = (kparams->src1_اندازه > 0)
        ? kparams->src1_اندازه
        : n_نخها * L->src1_بایتها_در_هر_thread;
    ثابت size_t src2_مجموع = kparams->src2_اندازه;
    ثابت size_t dst_مجموع  = n_نخها * L->dst_بایتها_در_هر_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_مجموع);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_مجموع);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_مجموع);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_مجموع);

    L->مجموع_بایتها = off;
}

#پایان_اگر
