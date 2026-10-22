#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_ਦੁਆਧਾਰੀ_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ਦੁਆਧਾਰੀ_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

ਗਿਣਤੀ htp_ਦੁਆਧਾਰੀ_kernel_type {
    HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE = 0,
    HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST,
    HTP_ਦੁਆਧਾਰੀ_KERNEL_SCALAR,
    HTP_ਦੁਆਧਾਰੀ_KERNEL_ਜੋੜੋ_ID,
    HTP_ਦੁਆਧਾਰੀ_KERNEL_COMPLEX,
    HTP_ਦੁਆਧਾਰੀ_KERNEL_REPEAT,
    HTP_ਦੁਆਧਾਰੀ_KERNEL_CHUNKED,
};

ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params {
    uint32_t kernel_type;
    uint32_t n_ਧਾਗੇ;
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ;

    uint32_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    uint32_t src1_ਮਾਪ;
    uint32_t src2_ਮਾਪ;
    uint32_t vtcm_ਮਾਪ;

    uint32_t chunk_ਮਾਪ;
    uint32_t chunk_ਬਾਈਟਾਂ;
    uint32_t is_scalar;

    uint8_t  ਵੰਡੋ_at_ne01;
    uint8_t  ਵੰਡੋ_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

ਬਣਤਰ ggml_tensor;
ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_ਦੁਆਧਾਰੀ_strides_ਸੁਰੱਖਿਅਤ(
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    size_t elem_ਮਾਪ
) {
    ਜੇ ((src0->ne[0] != 1 && src0->nb[0] != elem_ਮਾਪ) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_ਮਾਪ) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_ਮਾਪ)) {
        ਵਾਪਸ false;
    }
    ਜੇ ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_ਮਾਪ) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_ਮਾਪ) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_ਮਾਪ)) {
        ਵਾਪਸ false;
    }
    ਵਾਪਸ true;
}
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    size_t src0_spad_half_ਮਾਪ;
    size_t src1_spad_half_ਮਾਪ;
    size_t dst_spad_half_ਮਾਪ;

    size_t src1_ਮਾਪ;
    size_t src2_ਮਾਪ;
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_ਦੁਆਧਾਰੀ_vtcm_layout_build(
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * L,
    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params * kparams,
    size_t vtcm_ਮਾਪ
) {
    memset(L, 0, ਆਕਾਰ(*L));

    ਅਚਲ uint32_t n_ਧਾਗੇ = kparams->n_ਧਾਗੇ;
    ਜੇ (n_ਧਾਗੇ == 0) {
        ਵਾਪਸ;
    }

    ਜੇ (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_CHUNKED) {
        ਅਚਲ size_t chunk_ਬਾਈਟਾਂ = kparams->chunk_ਬਾਈਟਾਂ;
        ਜੇ (chunk_ਬਾਈਟਾਂ == 0) {
            ਵਾਪਸ;
        }

        L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * chunk_ਬਾਈਟਾਂ;
        L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = kparams->is_scalar ? 0 : (2 * chunk_ਬਾਈਟਾਂ);
        L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  = 2 * chunk_ਬਾਈਟਾਂ;

        L->src0_spad_half_ਮਾਪ = chunk_ਬਾਈਟਾਂ;
        L->src1_spad_half_ਮਾਪ = kparams->is_scalar ? 0 : chunk_ਬਾਈਟਾਂ;
        L->dst_spad_half_ਮਾਪ  = chunk_ਬਾਈਟਾਂ;

        L->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ = 1;
        L->src1_ਮਾਪ = 0;

        ਅਚਲ size_t src0_ਕੁੱਲ = n_ਧਾਗੇ * L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
        ਅਚਲ size_t src1_ਕੁੱਲ = kparams->is_scalar ? 128 : (n_ਧਾਗੇ * L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
        ਅਚਲ size_t dst_ਕੁੱਲ  = n_ਧਾਗੇ * L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_ਕੁੱਲ);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_ਕੁੱਲ);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ਕੁੱਲ);

        ਜੇ (off > vtcm_ਮਾਪ) {
            ਵਾਪਸ;
        }

        L->ਕੁੱਲ_ਬਾਈਟਾਂ = off;
        ਵਾਪਸ;
    }

    ਅਚਲ size_t spad_ਪੰਕਤੀ_ਕੁੱਲ = (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned + kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned + kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned)
        : 2 * (kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned + kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ size_t static_vtcm_ਮਾਪ = kparams->src1_ਮਾਪ + kparams->src2_ਮਾਪ;
    ਜੇ (spad_ਪੰਕਤੀ_ਕੁੱਲ == 0 || vtcm_ਮਾਪ < static_vtcm_ਮਾਪ) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ = (vtcm_ਮਾਪ - static_vtcm_ਮਾਪ) / (n_ਧਾਗੇ * spad_ਪੰਕਤੀ_ਕੁੱਲ);
    ਜੇ (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ == 0) {
        ਵਾਪਸ;
    }

    L->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ = (uint32_t) ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ;
    L->src1_ਮਾਪ = kparams->src1_ਮਾਪ;
    L->src2_ਮਾਪ = kparams->src2_ਮਾਪ;

    L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ * 2 * kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ * 2 * kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE)
        ? ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ * 2 * kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned
        : 0;

    L->src0_spad_half_ਮਾਪ = L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;
    L->src1_spad_half_ਮਾਪ = L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;
    L->dst_spad_half_ਮਾਪ  = L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;

    ਅਚਲ size_t src0_ਕੁੱਲ = n_ਧਾਗੇ * L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ size_t src1_ਕੁੱਲ = (kparams->src1_ਮਾਪ > 0)
        ? kparams->src1_ਮਾਪ
        : n_ਧਾਗੇ * L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ size_t src2_ਕੁੱਲ = kparams->src2_ਮਾਪ;
    ਅਚਲ size_t dst_ਕੁੱਲ  = n_ਧਾਗੇ * L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_ਕੁੱਲ);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_ਕੁੱਲ);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_ਕੁੱਲ);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ਕੁੱਲ);

    L->ਕੁੱਲ_ਬਾਈਟਾਂ = off;
}

#ਜੇ_ਅੰਤ
