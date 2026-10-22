#నిర్వచితం_కాకపోతే HTP_ద్విమాన_OPS_H
#నిర్వచించు HTP_ద్విమాన_OPS_H

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-vtcm.h"

గణన htp_ద్విమాన_kernel_type {
    HTP_ద్విమాన_KERNEL_SAME_SHAPE = 0,
    HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST,
    HTP_ద్విమాన_KERNEL_SCALAR,
    HTP_ద్విమాన_KERNEL_కూడు_ID,
    HTP_ద్విమాన_KERNEL_COMPLEX,
    HTP_ద్విమాన_KERNEL_REPEAT,
    HTP_ద్విమాన_KERNEL_CHUNKED,
};

నిర్మాణం htp_ద్విమాన_kernel_params {
    uint32_t kernel_type;
    uint32_t n_దారాలు;
    uint32_t అడ్డువరుసలు_చొప్పున_బఫరు;

    uint32_t src0_అడ్డువరుస_కొలత_aligned;
    uint32_t src1_అడ్డువరుస_కొలత_aligned;
    uint32_t dst_అడ్డువరుస_కొలత_aligned;

    uint32_t src1_కొలత;
    uint32_t src2_కొలత;
    uint32_t vtcm_కొలత;

    uint32_t chunk_కొలత;
    uint32_t chunk_బైట్లు;
    uint32_t is_scalar;

    uint8_t  విభజించు_at_ne01;
    uint8_t  విభజించు_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_ద్విమాన_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

నిర్మాణం ggml_tensor;
స్థిర అంతఃవరుస bool htp_ద్విమాన_strides_సురక్షిత(
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    size_t elem_కొలత
) {
    ఒకవేళ ((src0->ne[0] != 1 && src0->nb[0] != elem_కొలత) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_కొలత) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_కొలత)) {
        తిరిగి false;
    }
    ఒకవేళ ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_కొలత) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_కొలత) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_కొలత)) {
        తిరిగి false;
    }
    తిరిగి true;
}
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_ద్విమాన_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

నిర్మాణం htp_ద్విమాన_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_బైట్లు_చొప్పున_thread;
    size_t src1_బైట్లు_చొప్పున_thread;
    size_t dst_బైట్లు_చొప్పున_thread;

    size_t src0_spad_half_కొలత;
    size_t src1_spad_half_కొలత;
    size_t dst_spad_half_కొలత;

    size_t src1_కొలత;
    size_t src2_కొలత;
    uint32_t అడ్డువరుసలు_చొప్పున_బఫరు;
};

స్థిర అంతఃవరుస శూన్య htp_ద్విమాన_vtcm_layout_build(
    నిర్మాణం htp_ద్విమాన_vtcm_layout * L,
    స్థిరరాశి నిర్మాణం htp_ద్విమాన_kernel_params * kparams,
    size_t vtcm_కొలత
) {
    memset(L, 0, పరిమాణం(*L));

    స్థిరరాశి uint32_t n_దారాలు = kparams->n_దారాలు;
    ఒకవేళ (n_దారాలు == 0) {
        తిరిగి;
    }

    ఒకవేళ (kparams->kernel_type == HTP_ద్విమాన_KERNEL_CHUNKED) {
        స్థిరరాశి size_t chunk_బైట్లు = kparams->chunk_బైట్లు;
        ఒకవేళ (chunk_బైట్లు == 0) {
            తిరిగి;
        }

        L->src0_బైట్లు_చొప్పున_thread = 2 * chunk_బైట్లు;
        L->src1_బైట్లు_చొప్పున_thread = kparams->is_scalar ? 0 : (2 * chunk_బైట్లు);
        L->dst_బైట్లు_చొప్పున_thread  = 2 * chunk_బైట్లు;

        L->src0_spad_half_కొలత = chunk_బైట్లు;
        L->src1_spad_half_కొలత = kparams->is_scalar ? 0 : chunk_బైట్లు;
        L->dst_spad_half_కొలత  = chunk_బైట్లు;

        L->అడ్డువరుసలు_చొప్పున_బఫరు = 1;
        L->src1_కొలత = 0;

        స్థిరరాశి size_t src0_మొత్తం = n_దారాలు * L->src0_బైట్లు_చొప్పున_thread;
        స్థిరరాశి size_t src1_మొత్తం = kparams->is_scalar ? 128 : (n_దారాలు * L->src1_బైట్లు_చొప్పున_thread);
        స్థిరరాశి size_t dst_మొత్తం  = n_దారాలు * L->dst_బైట్లు_చొప్పున_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_మొత్తం);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_మొత్తం);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_మొత్తం);

        ఒకవేళ (off > vtcm_కొలత) {
            తిరిగి;
        }

        L->మొత్తం_బైట్లు = off;
        తిరిగి;
    }

    స్థిరరాశి size_t spad_అడ్డువరుస_మొత్తం = (kparams->kernel_type == HTP_ద్విమాన_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_అడ్డువరుస_కొలత_aligned + kparams->src1_అడ్డువరుస_కొలత_aligned + kparams->dst_అడ్డువరుస_కొలత_aligned)
        : 2 * (kparams->src0_అడ్డువరుస_కొలత_aligned + kparams->dst_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి size_t static_vtcm_కొలత = kparams->src1_కొలత + kparams->src2_కొలత;
    ఒకవేళ (spad_అడ్డువరుస_మొత్తం == 0 || vtcm_కొలత < static_vtcm_కొలత) {
        తిరిగి;
    }

    స్థిరరాశి size_t అడ్డువరుసలు_చొప్పున_బఫరు = (vtcm_కొలత - static_vtcm_కొలత) / (n_దారాలు * spad_అడ్డువరుస_మొత్తం);
    ఒకవేళ (అడ్డువరుసలు_చొప్పున_బఫరు == 0) {
        తిరిగి;
    }

    L->అడ్డువరుసలు_చొప్పున_బఫరు = (uint32_t) అడ్డువరుసలు_చొప్పున_బఫరు;
    L->src1_కొలత = kparams->src1_కొలత;
    L->src2_కొలత = kparams->src2_కొలత;

    L->src0_బైట్లు_చొప్పున_thread = అడ్డువరుసలు_చొప్పున_బఫరు * 2 * kparams->src0_అడ్డువరుస_కొలత_aligned;
    L->dst_బైట్లు_చొప్పున_thread  = అడ్డువరుసలు_చొప్పున_బఫరు * 2 * kparams->dst_అడ్డువరుస_కొలత_aligned;
    L->src1_బైట్లు_చొప్పున_thread = (kparams->kernel_type == HTP_ద్విమాన_KERNEL_SAME_SHAPE)
        ? అడ్డువరుసలు_చొప్పున_బఫరు * 2 * kparams->src1_అడ్డువరుస_కొలత_aligned
        : 0;

    L->src0_spad_half_కొలత = L->src0_బైట్లు_చొప్పున_thread / 2;
    L->src1_spad_half_కొలత = L->src1_బైట్లు_చొప్పున_thread / 2;
    L->dst_spad_half_కొలత  = L->dst_బైట్లు_చొప్పున_thread / 2;

    స్థిరరాశి size_t src0_మొత్తం = n_దారాలు * L->src0_బైట్లు_చొప్పున_thread;
    స్థిరరాశి size_t src1_మొత్తం = (kparams->src1_కొలత > 0)
        ? kparams->src1_కొలత
        : n_దారాలు * L->src1_బైట్లు_చొప్పున_thread;
    స్థిరరాశి size_t src2_మొత్తం = kparams->src2_కొలత;
    స్థిరరాశి size_t dst_మొత్తం  = n_దారాలు * L->dst_బైట్లు_చొప్పున_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_మొత్తం);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_మొత్తం);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_మొత్తం);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_మొత్తం);

    L->మొత్తం_బైట్లు = off;
}

#అయితే_ముగింపు
