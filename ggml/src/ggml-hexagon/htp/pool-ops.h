#నిర్వచితం_కాకపోతే HTP_POOL_OPS_H
#నిర్వచించు HTP_POOL_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-common.h"

నిర్మాణం htp_pool_2d_kernel_params {
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t kernel_x;
    uint32_t kernel_y;
    uint32_t stride_x;
    uint32_t stride_y;
    int32_t  pad_x;
    int32_t  pad_y;
    uint32_t src_plane_బైట్లు;
    uint32_t dst_plane_బైట్లు;
    uint32_t src_plane_బైట్లు_aligned;
    uint32_t dst_plane_బైట్లు_aligned;
    uint32_t n_దారాలు;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_మార్గం;
    uint32_t narrow_మార్గం;
    uint32_t global_మార్గం;
    uint32_t ఖండం_మార్గం;
    uint32_t avg_divide_లెక్క;
    uint32_t ox_lo;
    uint32_t ox_hi;
    దశాంశం    inv_kernel_area;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#అయితే_ముగింపు

నిర్మాణం htp_pool_vtcm_layout {
    size_t   మొత్తం_బైట్లు;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_బైట్లు_చొప్పున_thread;
    size_t   dst_బైట్లు_చొప్పున_thread;
    size_t   src_spad_half_కొలత;
    size_t   dst_spad_half_కొలత;
};

స్థిర అంతఃవరుస bool htp_pool_solve_layout(
    నిర్మాణం htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_దారాలు,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    స్థిరరాశి size_t src_plane_బైట్లు   = (size_t) src_x * src_y * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_plane_బైట్లు   = (size_t) dst_x * dst_y * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t src_plane_aligned = hex_విడత_పైకి((uint32_t) src_plane_బైట్లు + 256, 128);
    స్థిరరాశి size_t dst_plane_aligned = hex_విడత_పైకి((uint32_t) dst_plane_బైట్లు, 128);

    స్థిరరాశి size_t spad_చొప్పున_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    ఒకవేళ (spad_చొప్పున_thread * n_దారాలు > vtcm_budget) {
        తిరిగి false;
    }

    layout->src_spad_half_కొలత   = src_plane_aligned;
    layout->dst_spad_half_కొలత   = dst_plane_aligned;
    layout->src_బైట్లు_చొప్పున_thread = 2 * src_plane_aligned;
    layout->dst_బైట్లు_చొప్పున_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_బైట్లు_చొప్పున_thread * n_దారాలు;
    layout->మొత్తం_బైట్లు          = layout->off_dst + layout->dst_బైట్లు_చొప్పున_thread * n_దారాలు;
    తిరిగి true;
}

#అయితే_ముగింపు // HTP_POOL_OPS_H
