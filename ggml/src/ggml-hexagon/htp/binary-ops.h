#جيڪڏهن_وصف_نه_ٿيل HTP_ٻٽو_OPS_H
#وصف_ڪريو HTP_ٻٽو_OPS_H

#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "htp-ops.h"
#ضمن "htp-vtcm.h"

تعداد htp_ٻٽو_kernel_type {
    HTP_ٻٽو_KERNEL_SAME_SHAPE = 0,
    HTP_ٻٽو_KERNEL_صف_BCAST,
    HTP_ٻٽو_KERNEL_SCALAR,
    HTP_ٻٽو_KERNEL_جوڙيو_ID,
    HTP_ٻٽو_KERNEL_COMPLEX,
    HTP_ٻٽو_KERNEL_REPEAT,
    HTP_ٻٽو_KERNEL_CHUNKED,
};

بنية htp_ٻٽو_kernel_params {
    uint32_t kernel_type;
    uint32_t n_ڌاڳا;
    uint32_t صفون_في_بفر;

    uint32_t src0_صف_سائيز_aligned;
    uint32_t src1_صف_سائيز_aligned;
    uint32_t dst_صف_سائيز_aligned;

    uint32_t src1_سائيز;
    uint32_t src2_سائيز;
    uint32_t vtcm_سائيز;

    uint32_t chunk_سائيز;
    uint32_t chunk_بائيٽون;
    uint32_t is_scalar;

    uint8_t  ورهايو_at_ne01;
    uint8_t  ورهايو_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ٻٽو_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

بنية ggml_tensor;
ساكن مضمن bool htp_ٻٽو_strides_محفوظ(
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    size_t elem_سائيز
) {
    إذا ((src0->ne[0] != 1 && src0->nb[0] != elem_سائيز) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_سائيز) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_سائيز)) {
        أرجع false;
    }
    إذا ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_سائيز) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_سائيز) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_سائيز)) {
        أرجع false;
    }
    أرجع true;
}
#وإلا
_Static_assert(حجم(بنية htp_ٻٽو_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

بنية htp_ٻٽو_vtcm_layout {
    size_t ڪل_بائيٽون;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_بائيٽون_في_thread;
    size_t src1_بائيٽون_في_thread;
    size_t dst_بائيٽون_في_thread;

    size_t src0_spad_half_سائيز;
    size_t src1_spad_half_سائيز;
    size_t dst_spad_half_سائيز;

    size_t src1_سائيز;
    size_t src2_سائيز;
    uint32_t صفون_في_بفر;
};

ساكن مضمن فراغ htp_ٻٽو_vtcm_layout_build(
    بنية htp_ٻٽو_vtcm_layout * L,
    ثابت بنية htp_ٻٽو_kernel_params * kparams,
    size_t vtcm_سائيز
) {
    memset(L, 0, حجم(*L));

    ثابت uint32_t n_ڌاڳا = kparams->n_ڌاڳا;
    إذا (n_ڌاڳا == 0) {
        أرجع;
    }

    إذا (kparams->kernel_type == HTP_ٻٽو_KERNEL_CHUNKED) {
        ثابت size_t chunk_بائيٽون = kparams->chunk_بائيٽون;
        إذا (chunk_بائيٽون == 0) {
            أرجع;
        }

        L->src0_بائيٽون_في_thread = 2 * chunk_بائيٽون;
        L->src1_بائيٽون_في_thread = kparams->is_scalar ? 0 : (2 * chunk_بائيٽون);
        L->dst_بائيٽون_في_thread  = 2 * chunk_بائيٽون;

        L->src0_spad_half_سائيز = chunk_بائيٽون;
        L->src1_spad_half_سائيز = kparams->is_scalar ? 0 : chunk_بائيٽون;
        L->dst_spad_half_سائيز  = chunk_بائيٽون;

        L->صفون_في_بفر = 1;
        L->src1_سائيز = 0;

        ثابت size_t src0_ڪل = n_ڌاڳا * L->src0_بائيٽون_في_thread;
        ثابت size_t src1_ڪل = kparams->is_scalar ? 128 : (n_ڌاڳا * L->src1_بائيٽون_في_thread);
        ثابت size_t dst_ڪل  = n_ڌاڳا * L->dst_بائيٽون_في_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_ڪل);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_ڪل);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ڪل);

        إذا (off > vtcm_سائيز) {
            أرجع;
        }

        L->ڪل_بائيٽون = off;
        أرجع;
    }

    ثابت size_t spad_صف_ڪل = (kparams->kernel_type == HTP_ٻٽو_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_صف_سائيز_aligned + kparams->src1_صف_سائيز_aligned + kparams->dst_صف_سائيز_aligned)
        : 2 * (kparams->src0_صف_سائيز_aligned + kparams->dst_صف_سائيز_aligned);

    ثابت size_t static_vtcm_سائيز = kparams->src1_سائيز + kparams->src2_سائيز;
    إذا (spad_صف_ڪل == 0 || vtcm_سائيز < static_vtcm_سائيز) {
        أرجع;
    }

    ثابت size_t صفون_في_بفر = (vtcm_سائيز - static_vtcm_سائيز) / (n_ڌاڳا * spad_صف_ڪل);
    إذا (صفون_في_بفر == 0) {
        أرجع;
    }

    L->صفون_في_بفر = (uint32_t) صفون_في_بفر;
    L->src1_سائيز = kparams->src1_سائيز;
    L->src2_سائيز = kparams->src2_سائيز;

    L->src0_بائيٽون_في_thread = صفون_في_بفر * 2 * kparams->src0_صف_سائيز_aligned;
    L->dst_بائيٽون_في_thread  = صفون_في_بفر * 2 * kparams->dst_صف_سائيز_aligned;
    L->src1_بائيٽون_في_thread = (kparams->kernel_type == HTP_ٻٽو_KERNEL_SAME_SHAPE)
        ? صفون_في_بفر * 2 * kparams->src1_صف_سائيز_aligned
        : 0;

    L->src0_spad_half_سائيز = L->src0_بائيٽون_في_thread / 2;
    L->src1_spad_half_سائيز = L->src1_بائيٽون_في_thread / 2;
    L->dst_spad_half_سائيز  = L->dst_بائيٽون_في_thread / 2;

    ثابت size_t src0_ڪل = n_ڌاڳا * L->src0_بائيٽون_في_thread;
    ثابت size_t src1_ڪل = (kparams->src1_سائيز > 0)
        ? kparams->src1_سائيز
        : n_ڌاڳا * L->src1_بائيٽون_في_thread;
    ثابت size_t src2_ڪل = kparams->src2_سائيز;
    ثابت size_t dst_ڪل  = n_ڌاڳا * L->dst_بائيٽون_في_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_ڪل);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_ڪل);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_ڪل);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ڪل);

    L->ڪل_بائيٽون = off;
}

#جيڪڏهن_پورو
