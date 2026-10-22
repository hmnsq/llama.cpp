#నిర్వచితం_కాకపోతే HTP_GET_అడ్డువరుసలు_OPS_H
#నిర్వచించు HTP_GET_అడ్డువరుసలు_OPS_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp-vtcm.h"

గణన htp_get_అడ్డువరుసలు_kernel_type {
    HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE = 0,
    HTP_GET_అడ్డువరుసలు_KERNEL_TILED,
    HTP_GET_అడ్డువరుసలు_KERNEL_FLAT,
};

నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params {
    int32_t  n_దారాలు;
    int32_t  kernel_type;
    int32_t  మొత్తం_tasks;
    int32_t  tasks_చొప్పున_thread;
    int32_t  vtcm_కొలత;

    // Fastdiv helpers
    నిర్మాణం fastdiv_విలువలు div_ne10;
    నిర్మాణం fastdiv_విలువలు div_ne10_ne11;
    నిర్మాణం fastdiv_విలువలు div_ne02;
    నిర్మాణం fastdiv_విలువలు div_ne03;
};

నిర్మాణం htp_get_అడ్డువరుసలు_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t off_src1;
    size_t src1_బైట్లు;
    size_t off_src0;
    size_t off_dst;

    size_t src0_బైట్లు_చొప్పున_thread;
    size_t dst_బైట్లు_చొప్పున_thread;

    size_t src0_spad_half_కొలత;
    size_t dst_spad_half_కొలత;
};

స్థిర అంతఃవరుస శూన్య htp_get_అడ్డువరుసలు_vtcm_layout_build(
    నిర్మాణం htp_get_అడ్డువరుసలు_vtcm_layout * L,
    పూర్ణాంకం kernel_type,
    పూర్ణాంకం type,
    uint32_t ne00,
    uint32_t మొత్తం_tasks,
    పూర్ణాంకం idx_type,
    uint32_t n_దారాలు) {

    memset(L, 0, పరిమాణం(*L));

    స్థిరరాశి size_t elem_కొలత = (idx_type == HTP_TYPE_I64) ? పరిమాణం(int64_t) : పరిమాణం(int32_t);
    స్థిరరాశి size_t src1_raw_బైట్లు = మొత్తం_tasks * elem_కొలత;
    స్థిరరాశి size_t src1_బైట్లు = ((src1_raw_బైట్లు + 127) & ~127) + 128;

    L->src1_బైట్లు = src1_బైట్లు;
    స్థిరరాశి size_t src1_alloc_బైట్లు = (src1_బైట్లు + 255) & ~255;

    ఒకవేళ (kernel_type != HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE) {
        size_t src0_అడ్డువరుస_కొలత = 0;
        ఒకవేళ (kernel_type == HTP_GET_అడ్డువరుసలు_KERNEL_TILED) {
            స్థిరరాశి size_t tile_stride = htp_tensor_get_tile_padded_కొలత(type);
            స్థిరరాశి uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_అడ్డువరుస_కొలత = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } లేకపోతే {
            src0_అడ్డువరుస_కొలత = htp_tensor_get_అడ్డువరుస_కొలత(type, ne00);
        }

        L->src0_spad_half_కొలత   = (src0_అడ్డువరుస_కొలత + 255) & ~255;
        L->dst_spad_half_కొలత    = (ne00 * పరిమాణం(దశాంశం) + 255) & ~255;
        L->src0_బైట్లు_చొప్పున_thread = 2 * L->src0_spad_half_కొలత;
        L->dst_బైట్లు_చొప్పున_thread  = 2 * L->dst_spad_half_కొలత;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_బైట్లు);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_బైట్లు_చొప్పున_thread * n_దారాలు);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_బైట్లు_చొప్పున_thread  * n_దారాలు);
    L->మొత్తం_బైట్లు = off;
}

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_GET_ROWS_OPS_H
