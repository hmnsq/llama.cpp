#جے_بنا_تعریف HTP_GET_صفاں_OPS_H
#تعریف_کرو HTP_GET_صفاں_OPS_H

#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

تعداد htp_get_صفاں_kernel_type {
    HTP_GET_صفاں_KERNEL_SAMETYPE = 0,
    HTP_GET_صفاں_KERNEL_TILED,
    HTP_GET_صفاں_KERNEL_FLAT,
};

بنية htp_get_صفاں_kernel_params {
    int32_t  n_دھاگے;
    int32_t  kernel_type;
    int32_t  کل_tasks;
    int32_t  tasks_فی_thread;
    int32_t  vtcm_ناپ;

    // Fastdiv helpers
    بنية fastdiv_اقدار div_ne10;
    بنية fastdiv_اقدار div_ne10_ne11;
    بنية fastdiv_اقدار div_ne02;
    بنية fastdiv_اقدار div_ne03;
};

بنية htp_get_صفاں_vtcm_layout {
    size_t کل_بائٹاں;
    size_t off_src1;
    size_t src1_بائٹاں;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بائٹاں_فی_thread;
    size_t dst_بائٹاں_فی_thread;

    size_t src0_spad_half_ناپ;
    size_t dst_spad_half_ناپ;
};

ساكن مضمن فراغ htp_get_صفاں_vtcm_layout_build(
    بنية htp_get_صفاں_vtcm_layout * L,
    صحيح kernel_type,
    صحيح type,
    uint32_t ne00,
    uint32_t کل_tasks,
    صحيح idx_type,
    uint32_t n_دھاگے) {

    memset(L, 0, حجم(*L));

    ثابت size_t elem_ناپ = (idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_raw_بائٹاں = کل_tasks * elem_ناپ;
    ثابت size_t src1_بائٹاں = ((src1_raw_بائٹاں + 127) & ~127) + 128;

    L->src1_بائٹاں = src1_بائٹاں;
    ثابت size_t src1_alloc_بائٹاں = (src1_بائٹاں + 255) & ~255;

    إذا (kernel_type != HTP_GET_صفاں_KERNEL_SAMETYPE) {
        size_t src0_صف_ناپ = 0;
        إذا (kernel_type == HTP_GET_صفاں_KERNEL_TILED) {
            ثابت size_t tile_stride = htp_tensor_get_tile_padded_ناپ(type);
            ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_صف_ناپ = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } وإلا {
            src0_صف_ناپ = htp_tensor_get_صف_ناپ(type, ne00);
        }

        L->src0_spad_half_ناپ   = (src0_صف_ناپ + 255) & ~255;
        L->dst_spad_half_ناپ    = (ne00 * حجم(عائم) + 255) & ~255;
        L->src0_بائٹاں_فی_thread = 2 * L->src0_spad_half_ناپ;
        L->dst_بائٹاں_فی_thread  = 2 * L->dst_spad_half_ناپ;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_بائٹاں);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_بائٹاں_فی_thread * n_دھاگے);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_بائٹاں_فی_thread  * n_دھاگے);
    L->کل_بائٹاں = off;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_get_صفاں_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_get_صفاں_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#جے_انت

#جے_انت // HTP_GET_ROWS_OPS_H
