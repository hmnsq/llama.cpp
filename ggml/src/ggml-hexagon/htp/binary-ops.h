#اگر_غیر_تعریف_شدہ HTP_ثنائی_OPS_H
#تعریف_کرو HTP_ثنائی_OPS_H

#شامل <stddef.h>
#شامل <stdint.h>
#شامل <string.h>

#شامل "hex-common.h"
#شامل "htp-ops.h"
#شامل "htp-vtcm.h"

شمار htp_ثنائی_kernel_type {
    HTP_ثنائی_KERNEL_SAME_SHAPE = 0,
    HTP_ثنائی_KERNEL_صف_BCAST,
    HTP_ثنائی_KERNEL_SCALAR,
    HTP_ثنائی_KERNEL_جوڑو_ID,
    HTP_ثنائی_KERNEL_COMPLEX,
    HTP_ثنائی_KERNEL_REPEAT,
    HTP_ثنائی_KERNEL_CHUNKED,
};

ساخت htp_ثنائی_kernel_params {
    uint32_t kernel_type;
    uint32_t n_دھاگے;
    uint32_t صفیں_فی_بفر;

    uint32_t src0_صف_سائز_aligned;
    uint32_t src1_صف_سائز_aligned;
    uint32_t dst_صف_سائز_aligned;

    uint32_t src1_سائز;
    uint32_t src2_سائز;
    uint32_t vtcm_سائز;

    uint32_t chunk_سائز;
    uint32_t chunk_بائٹیں;
    uint32_t is_scalar;

    uint8_t  بانٹو_at_ne01;
    uint8_t  بانٹو_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_ثنائی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

ساخت ggml_tensor;
ساکن اندرونی bool htp_ثنائی_strides_محفوظ(
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    size_t elem_سائز
) {
    اگر ((src0->ne[0] != 1 && src0->nb[0] != elem_سائز) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_سائز) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_سائز)) {
        واپس false;
    }
    اگر ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_سائز) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_سائز) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_سائز)) {
        واپس false;
    }
    واپس true;
}
#ورنہ
_Static_assert(حجم(ساخت htp_ثنائی_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساخت htp_ثنائی_vtcm_layout {
    size_t کل_بائٹیں;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_بائٹیں_فی_thread;
    size_t src1_بائٹیں_فی_thread;
    size_t dst_بائٹیں_فی_thread;

    size_t src0_spad_half_سائز;
    size_t src1_spad_half_سائز;
    size_t dst_spad_half_سائز;

    size_t src1_سائز;
    size_t src2_سائز;
    uint32_t صفیں_فی_بفر;
};

ساکن اندرونی خالی htp_ثنائی_vtcm_layout_build(
    ساخت htp_ثنائی_vtcm_layout * L,
    ثابت ساخت htp_ثنائی_kernel_params * kparams,
    size_t vtcm_سائز
) {
    memset(L, 0, حجم(*L));

    ثابت uint32_t n_دھاگے = kparams->n_دھاگے;
    اگر (n_دھاگے == 0) {
        واپس;
    }

    اگر (kparams->kernel_type == HTP_ثنائی_KERNEL_CHUNKED) {
        ثابت size_t chunk_بائٹیں = kparams->chunk_بائٹیں;
        اگر (chunk_بائٹیں == 0) {
            واپس;
        }

        L->src0_بائٹیں_فی_thread = 2 * chunk_بائٹیں;
        L->src1_بائٹیں_فی_thread = kparams->is_scalar ? 0 : (2 * chunk_بائٹیں);
        L->dst_بائٹیں_فی_thread  = 2 * chunk_بائٹیں;

        L->src0_spad_half_سائز = chunk_بائٹیں;
        L->src1_spad_half_سائز = kparams->is_scalar ? 0 : chunk_بائٹیں;
        L->dst_spad_half_سائز  = chunk_بائٹیں;

        L->صفیں_فی_بفر = 1;
        L->src1_سائز = 0;

        ثابت size_t src0_کل = n_دھاگے * L->src0_بائٹیں_فی_thread;
        ثابت size_t src1_کل = kparams->is_scalar ? 128 : (n_دھاگے * L->src1_بائٹیں_فی_thread);
        ثابت size_t dst_کل  = n_دھاگے * L->dst_بائٹیں_فی_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_کل);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_کل);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_کل);

        اگر (off > vtcm_سائز) {
            واپس;
        }

        L->کل_بائٹیں = off;
        واپس;
    }

    ثابت size_t spad_صف_کل = (kparams->kernel_type == HTP_ثنائی_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_صف_سائز_aligned + kparams->src1_صف_سائز_aligned + kparams->dst_صف_سائز_aligned)
        : 2 * (kparams->src0_صف_سائز_aligned + kparams->dst_صف_سائز_aligned);

    ثابت size_t static_vtcm_سائز = kparams->src1_سائز + kparams->src2_سائز;
    اگر (spad_صف_کل == 0 || vtcm_سائز < static_vtcm_سائز) {
        واپس;
    }

    ثابت size_t صفیں_فی_بفر = (vtcm_سائز - static_vtcm_سائز) / (n_دھاگے * spad_صف_کل);
    اگر (صفیں_فی_بفر == 0) {
        واپس;
    }

    L->صفیں_فی_بفر = (uint32_t) صفیں_فی_بفر;
    L->src1_سائز = kparams->src1_سائز;
    L->src2_سائز = kparams->src2_سائز;

    L->src0_بائٹیں_فی_thread = صفیں_فی_بفر * 2 * kparams->src0_صف_سائز_aligned;
    L->dst_بائٹیں_فی_thread  = صفیں_فی_بفر * 2 * kparams->dst_صف_سائز_aligned;
    L->src1_بائٹیں_فی_thread = (kparams->kernel_type == HTP_ثنائی_KERNEL_SAME_SHAPE)
        ? صفیں_فی_بفر * 2 * kparams->src1_صف_سائز_aligned
        : 0;

    L->src0_spad_half_سائز = L->src0_بائٹیں_فی_thread / 2;
    L->src1_spad_half_سائز = L->src1_بائٹیں_فی_thread / 2;
    L->dst_spad_half_سائز  = L->dst_بائٹیں_فی_thread / 2;

    ثابت size_t src0_کل = n_دھاگے * L->src0_بائٹیں_فی_thread;
    ثابت size_t src1_کل = (kparams->src1_سائز > 0)
        ? kparams->src1_سائز
        : n_دھاگے * L->src1_بائٹیں_فی_thread;
    ثابت size_t src2_کل = kparams->src2_سائز;
    ثابت size_t dst_کل  = n_دھاگے * L->dst_بائٹیں_فی_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_کل);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_کل);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_کل);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_کل);

    L->کل_بائٹیں = off;
}

#اگر_ختم
