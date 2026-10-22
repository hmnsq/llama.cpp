#నిర్వచితం_కాకపోతే HTP_CPY_OPS_H
#నిర్వచించు HTP_CPY_OPS_H

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "gather-rows.h"
#అన్తర్భావ <stdint.h>

గణన htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_అడ్డువరుసలు        = 6,
};

నిర్మాణం htp_copy_convert_params {
    uint32_t              src0_buf_కొలత;
    uint32_t              dst_buf_కొలత;
    uint32_t              spad0_కొలత_చొప్పున_thread;
    uint32_t              spad1_కొలత_చొప్పున_thread;
    uint32_t              src0_అడ్డువరుస_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_అడ్డువరుస_stride;
    uint32_t              blk_అడ్డువరుసలు;         // rows moved per DMA descriptor
    నిర్మాణం fastdiv_విలువలు div_ne01;
    నిర్మాణం fastdiv_విలువలు div_ne02_ne01;
};

నిర్మాణం htp_copy_reshape_params {
    నిర్మాణం fastdiv_విలువలు div_ne0;
    నిర్మాణం fastdiv_విలువలు div_ne1_ne0;
    నిర్మాణం fastdiv_విలువలు div_ne2_ne1_ne0;
    నిర్మాణం fastdiv_విలువలు div_ne00;
    నిర్మాణం fastdiv_విలువలు div_ne01_ne00;
    నిర్మాణం fastdiv_విలువలు div_ne02_ne01_ne00;
};

నిర్మాణం htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_కొలత;
    uint8_t  dst_type_కొలత;
    uint8_t  n_దారాలు;

    uint32_t మొత్తం_elems;
    uint32_t మొత్తం_అడ్డువరుసలు;
    uint32_t vtcm_కొలత;

    సంఘం {
        నిర్మాణం htp_copy_convert_params convert;
        నిర్మాణం htp_copy_reshape_params reshape;
        నిర్మాణం htp_gather_అడ్డువరుసలు_params  gather;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#నిర్వచించు HTP_COPY_VTCM_BLK_బైట్లు (32 * 1024)

నిర్మాణం htp_copy_convert_vtcm_layout {
    uint32_t src0_అడ్డువరుస_stride;
    uint32_t dst_అడ్డువరుస_stride;
    uint32_t blk_అడ్డువరుసలు;
    uint32_t src0_buf_కొలత;
    uint32_t dst_buf_కొలత;
    uint32_t spad0_కొలత_చొప్పున_thread;
    uint32_t spad1_కొలత_చొప్పున_thread;
    uint32_t మొత్తం_బైట్లు;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
స్థిర అంతఃవరుస శూన్య htp_copy_convert_vtcm_layout_build(
    నిర్మాణం htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_కొలత,
    uint32_t dst_type_కొలత,
    uint32_t n_దారాలు) {

    layout->src0_అడ్డువరుస_stride = hex_విడత_పైకి(ne00 * src_type_కొలత, 256);
    layout->dst_అడ్డువరుస_stride  = hex_విడత_పైకి(ne00 * dst_type_కొలత, 256);

    స్థిరరాశి uint32_t max_అడ్డువరుస = layout->src0_అడ్డువరుస_stride > layout->dst_అడ్డువరుస_stride ? layout->src0_అడ్డువరుస_stride : layout->dst_అడ్డువరుస_stride;
    uint32_t blk_అడ్డువరుసలు = HTP_COPY_VTCM_BLK_బైట్లు / max_అడ్డువరుస;
    ఒకవేళ (blk_అడ్డువరుసలు > ne01) {
        blk_అడ్డువరుసలు = ne01;
    }
    ఒకవేళ (blk_అడ్డువరుసలు == 0) {
        blk_అడ్డువరుసలు = 1;
    }
    layout->blk_అడ్డువరుసలు = blk_అడ్డువరుసలు;

    layout->src0_buf_కొలత = blk_అడ్డువరుసలు * layout->src0_అడ్డువరుస_stride;
    layout->dst_buf_కొలత  = blk_అడ్డువరుసలు * layout->dst_అడ్డువరుస_stride;
    layout->spad0_కొలత_చొప్పున_thread = 2 * layout->src0_buf_కొలత;
    layout->spad1_కొలత_చొప్పున_thread = 2 * layout->dst_buf_కొలత;
    layout->మొత్తం_బైట్లు = n_దారాలు * (layout->spad0_కొలత_చొప్పున_thread + layout->spad1_కొలత_చొప్పున_thread);
}

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_CPY_OPS_H
