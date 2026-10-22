#اگر_غیر_تعریف_شدہ HTP_GET_صفیں_OPS_H
#تعریف_کرو HTP_GET_صفیں_OPS_H

#شامل <stdbool.h>
#شامل <string.h>

#شامل "hex-fastdiv.h"
#شامل "htp-tensor.h"
#شامل "htp-vtcm.h"

شمار htp_get_صفیں_kernel_type {
    HTP_GET_صفیں_KERNEL_SAMETYPE = 0,
    HTP_GET_صفیں_KERNEL_TILED,
    HTP_GET_صفیں_KERNEL_FLAT,
};

ساخت htp_get_صفیں_kernel_params {
    int32_t  n_دھاگے;
    int32_t  kernel_type;
    int32_t  کل_tasks;
    int32_t  tasks_فی_thread;
    int32_t  vtcm_سائز;

    // Fastdiv helpers
    ساخت fastdiv_اقدار div_ne10;
    ساخت fastdiv_اقدار div_ne10_ne11;
    ساخت fastdiv_اقدار div_ne02;
    ساخت fastdiv_اقدار div_ne03;
};

ساخت htp_get_صفیں_vtcm_layout {
    size_t کل_بائٹیں;
    size_t off_src1;
    size_t src1_بائٹیں;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بائٹیں_فی_thread;
    size_t dst_بائٹیں_فی_thread;

    size_t src0_spad_half_سائز;
    size_t dst_spad_half_سائز;
};

ساکن اندرونی خالی htp_get_صفیں_vtcm_layout_build(
    ساخت htp_get_صفیں_vtcm_layout * L,
    صحیح kernel_type,
    صحیح type,
    uint32_t ne00,
    uint32_t کل_tasks,
    صحیح idx_type,
    uint32_t n_دھاگے) {

    memset(L, 0, حجم(*L));

    ثابت size_t elem_سائز = (idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_raw_بائٹیں = کل_tasks * elem_سائز;
    ثابت size_t src1_بائٹیں = ((src1_raw_بائٹیں + 127) & ~127) + 128;

    L->src1_بائٹیں = src1_بائٹیں;
    ثابت size_t src1_alloc_بائٹیں = (src1_بائٹیں + 255) & ~255;

    اگر (kernel_type != HTP_GET_صفیں_KERNEL_SAMETYPE) {
        size_t src0_صف_سائز = 0;
        اگر (kernel_type == HTP_GET_صفیں_KERNEL_TILED) {
            ثابت size_t tile_stride = htp_tensor_get_tile_padded_سائز(type);
            ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_صف_سائز = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } ورنہ {
            src0_صف_سائز = htp_tensor_get_صف_سائز(type, ne00);
        }

        L->src0_spad_half_سائز   = (src0_صف_سائز + 255) & ~255;
        L->dst_spad_half_سائز    = (ne00 * حجم(اعشاری) + 255) & ~255;
        L->src0_بائٹیں_فی_thread = 2 * L->src0_spad_half_سائز;
        L->dst_بائٹیں_فی_thread  = 2 * L->dst_spad_half_سائز;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_بائٹیں);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_بائٹیں_فی_thread * n_دھاگے);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_بائٹیں_فی_thread  * n_دھاگے);
    L->کل_بائٹیں = off;
}

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_get_صفیں_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_get_صفیں_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#اگر_ختم

#اگر_ختم // HTP_GET_ROWS_OPS_H
