#నిర్వచితం_కాకపోతే HTP_UNARY_OPS_H
#నిర్వచించు HTP_UNARY_OPS_H

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-ops.h"

// Op-specific struct for precomputed unary params
నిర్మాణం htp_unary_kernel_params {
    uint32_t  n_దారాలు;
    uint32_t  నిలువు_tile;
    uint32_t  vtcm_అడ్డువరుస_చొప్పున_thread;
    uint32_t  ఖండం;
    uint32_t  ప్రసారం_బరువు;

    uint32_t  vtcm_src0_కొలత_చొప్పున_thread;
    uint32_t  vtcm_src1_కొలత_చొప్పున_thread;
    uint32_t  vtcm_dst_కొలత_చొప్పున_thread;

    uint32_t  vtcm_src0_కొలత;
    uint32_t  vtcm_src1_కొలత;
    uint32_t  vtcm_dst_కొలత;

    uint32_t  src0_అడ్డువరుస_కొలత_aligned;
    uint32_t  src1_అడ్డువరుస_కొలత_aligned;
    uint32_t  dst_అడ్డువరుస_కొలత_aligned;

    uint32_t  vtcm_కొలత;

    // Fastdiv helpers
    నిర్మాణం fastdiv_విలువలు div_ne01;
    నిర్మాణం fastdiv_విలువలు div_ne02;
    నిర్మాణం fastdiv_విలువలు div_ne012;
    నిర్మాణం fastdiv_విలువలు div_tpr;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

స్థిర అంతఃవరుస bool htp_op_is_unary(uint32_t opcode) {
    ఎంపిక (opcode) {
        స్థితి HTP_OP_CLAMP:
        స్థితి HTP_OP_LEAKY_RELU:
        స్థితి HTP_OP_NORM:
        స్థితి HTP_OP_RMS_NORM:
        స్థితి HTP_OP_RMS_NORM_MUL:
        స్థితి HTP_OP_SCALE:
        స్థితి HTP_OP_SQR:
        స్థితి HTP_OP_SQRT:
        స్థితి HTP_OP_UNARY_NEG:
        స్థితి HTP_OP_UNARY_EXP:
        స్థితి HTP_OP_UNARY_SIGMOID:
        స్థితి HTP_OP_UNARY_SILU:
        స్థితి HTP_OP_UNARY_GELU:
        స్థితి HTP_OP_UNARY_GELU_శీఘ్ర:
        స్థితి HTP_OP_UNARY_GELU_ERF:
        స్థితి HTP_OP_UNARY_SOFTPLUS:
        స్థితి HTP_OP_UNARY_TANH:
        స్థితి HTP_OP_UNARY_ABS:
        స్థితి HTP_OP_UNARY_LOG:
        స్థితి HTP_OP_UNARY_RELU:
        స్థితి HTP_OP_UNARY_అడుగు:
        స్థితి HTP_OP_L2_NORM:
        స్థితి HTP_OP_TRI:
        స్థితి HTP_OP_HADAMARD:
            తిరిగి true;
        అప్రమేయ:
            తిరిగి false;
    }
}

నిర్మాణం htp_unary_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_బైట్లు;
    size_t src1_బైట్లు;
    size_t dst_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య htp_unary_vtcm_layout_build(
    నిర్మాణం htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool ప్రసారం_బరువు,
    uint32_t n_దారాలు,
    size_t vtcm_కొలత,
    size_t elem_కొలత,
    uint32_t * బహిర్గతం_నిలువు_tile,
    uint32_t * బహిర్గతం_vtcm_అడ్డువరుస_చొప్పున_thread
) {
    స్థిరరాశి size_t dst_elem_కొలత      = (op == HTP_OP_HADAMARD) ? పరిమాణం(దశాంశం) : elem_కొలత;
    స్థిరరాశి size_t src0_దత్తాంశం_అడ్డువరుస_కొలత = ne00 * elem_కొలత;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత  = ne10 * dst_elem_కొలత;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src0_దత్తాంశం_అడ్డువరుస_కొలత, 128);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = hex_విడత_పైకి(dst_దత్తాంశం_అడ్డువరుస_కొలత,  128);

    size_t src1_అడ్డువరుస_కొలత_aligned = 0;
    ఒకవేళ (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        స్థిరరాశి size_t src1_దత్తాంశం_అడ్డువరుస_కొలత = ne11 * పరిమాణం(దశాంశం);
        src1_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src1_దత్తాంశం_అడ్డువరుస_కొలత, 128);
    }

    size_t vtcm_కొలత_చొప్పున_అడ్డువరుస = 0;
    size_t vtcm_అడ్డువరుస_చొప్పున_thread = 0;

    ఒకవేళ (op == HTP_OP_RMS_NORM_MUL) {
        ఒకవేళ (ప్రసారం_బరువు) {
            size_t అందుబాటు_vtcm = vtcm_కొలత;
            size_t src1_vtcm_మొత్తం = n_దారాలు * src1_అడ్డువరుస_కొలత_aligned;
            ఒకవేళ (అందుబాటు_vtcm > src1_vtcm_మొత్తం) {
                అందుబాటు_vtcm -= src1_vtcm_మొత్తం;
            } లేకపోతే {
                అందుబాటు_vtcm = 0;
            }
            vtcm_కొలత_చొప్పున_అడ్డువరుస = 2 * (src0_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned);
            vtcm_అడ్డువరుస_చొప్పున_thread = అందుబాటు_vtcm / (n_దారాలు * vtcm_కొలత_చొప్పున_అడ్డువరుస);
        } లేకపోతే {
            vtcm_కొలత_చొప్పున_అడ్డువరుస = 2 * (src0_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned + src1_అడ్డువరుస_కొలత_aligned);
            vtcm_అడ్డువరుస_చొప్పున_thread = vtcm_కొలత / (n_దారాలు * vtcm_కొలత_చొప్పున_అడ్డువరుస);
        }
    } లేకపోతే {
        vtcm_కొలత_చొప్పున_అడ్డువరుస   = 2 * (src0_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned);
        vtcm_అడ్డువరుస_చొప్పున_thread = vtcm_కొలత / (n_దారాలు * vtcm_కొలత_చొప్పున_అడ్డువరుస);
    }

    స్థిరరాశి bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    స్థిరరాశి bool is_f16 = (elem_కొలత == పరిమాణం(_Float16));
    uint32_t నిలువు_tile = 0;

    ఒకవేళ (vtcm_అడ్డువరుస_చొప్పున_thread == 0 && !is_reduction && !is_f16) {
        స్థిరరాశి size_t చొప్పున_thread_budget = vtcm_కొలత / n_దారాలు;
        స్థిరరాశి size_t నిలువు_tile_బైట్లు = hex_align_కిందికి(చొప్పున_thread_budget / 4, 128);
        నిలువు_tile = (uint32_t) (నిలువు_tile_బైట్లు / elem_కొలత);

        L->src0_బైట్లు = నిలువు_tile_బైట్లు * 2;
        L->dst_బైట్లు  = నిలువు_tile_బైట్లు * 2;
        L->src1_బైట్లు = 0;
    } లేకపోతే {
        L->src0_బైట్లు = src0_అడ్డువరుస_కొలత_aligned * vtcm_అడ్డువరుస_చొప్పున_thread * 2;
        L->dst_బైట్లు  = dst_అడ్డువరుస_కొలత_aligned * vtcm_అడ్డువరుస_చొప్పున_thread * 2;
        ఒకవేళ (op == HTP_OP_RMS_NORM_MUL) {
            ఒకవేళ (ప్రసారం_బరువు) {
                L->src1_బైట్లు = src1_అడ్డువరుస_కొలత_aligned;
            } లేకపోతే {
                L->src1_బైట్లు = src1_అడ్డువరుస_కొలత_aligned * vtcm_అడ్డువరుస_చొప్పున_thread * 2;
            }
        } లేకపోతే {
            L->src1_బైట్లు = 0;
        }
    }

    L->off_src0 = 0;
    ఒకవేళ (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_బైట్లు * n_దారాలు;
        L->off_dst  = L->off_src1 + L->src1_బైట్లు * n_దారాలు;
    } లేకపోతే {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_బైట్లు * n_దారాలు;
    }

    L->మొత్తం_బైట్లు = L->off_dst + L->dst_బైట్లు * n_దారాలు;

    *బహిర్గతం_నిలువు_tile = నిలువు_tile;
    *బహిర్గతం_vtcm_అడ్డువరుస_చొప్పున_thread = vtcm_అడ్డువరుస_చొప్పున_thread;
}

#అయితే_ముగింపు /* HTP_UNARY_OPS_H */
