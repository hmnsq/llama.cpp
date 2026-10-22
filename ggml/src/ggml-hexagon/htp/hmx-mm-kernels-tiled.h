#అన్తర్భావ "hmx-utils.h"
#అన్తర్భావ "hmx-queue.h"

// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
స్థిర స్థిరరాశి __fp16 mxfp4_గమ్యం_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

స్థిర స్థిరరాశి __fp16 iq4_nl_గమ్యం_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

రకనిర్వచనం నిర్మాణం tiled_dequantize_state tiled_dequantize_state_t;
రకనిర్వచనం శూన్య (*tiled_dequant_task_fn_t)(స్థిరరాశి tiled_dequantize_state_t * state, uint32_t ప్రారంభం_tile, uint32_t చివర_tile);

నిర్మాణం tiled_dequantize_state {
    నిర్మాణం htp_context      * ctx;
    నిర్మాణం htp_thread_trace * traces;
    __fp16                  * dst;
    స్థిరరాశి uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    నిర్మాణం fastdiv_విలువలు     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_బరువు_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_చొప్పున_task;
    uint32_t                  tile_కొలత;
    uint32_t                  aligned_tile_కొలత;
    uint32_t                  n_tasks;
    uint32_t                  n_నిలువుల_సంఖ్య;
    uint32_t                  k_ఖండం;
    size_t                    అడ్డువరుస_stride;
    uint32_t                  బరువు_type;
};

స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q1_0(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ v_neg_one = Q6_Vb_vsplat_R(-1);
    స్థిరరాశి HVX_సదిశ v_pos_one = Q6_Vb_vsplat_R(1);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        స్థిరరాశి HVX_సదిశ v_sc = hvx_vmem(tile_src + 512);
        స్థిరరాశి HVX_సదిశ v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        కొరకు (uint32_t q = 0; q < 4; q++) {
            స్థిరరాశి HVX_సదిశ v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #సూచన unroll
            కొరకు (పూర్ణాంకం g = 0; g < 8; g++) {
                స్థిరరాశి HVX_సదిశ     v_q  = unpack_q1_0_సమూహం(v_plane, g, v_neg_one, v_pos_one);
                స్థిరరాశి HVX_సదిశ_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                స్థిరరాశి HVX_సదిశ_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q4_0(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    స్థిరరాశి HVX_సదిశ i8 = Q6_Vb_vsplat_R(8);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v_sc = hvx_vmem(tile_src + 512);
        HVX_సదిశ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_సదిశ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_సదిశ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_సదిశ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_సదిశ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_సదిశ v_lo0 = Q6_V_vand_VV(vq0, ముసుగు_h4);
        HVX_సదిశ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_సదిశ v_lo1 = Q6_V_vand_VV(vq1, ముసుగు_h4);
        HVX_సదిశ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_సదిశ v_lo2 = Q6_V_vand_VV(vq2, ముసుగు_h4);
        HVX_సదిశ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_సదిశ v_lo3 = Q6_V_vand_VV(vq3, ముసుగు_h4);
        HVX_సదిశ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Offsetting (-8)
        v_lo0 = Q6_Vb_vsub_VbVb(v_lo0, i8);
        v_hi0 = Q6_Vb_vsub_VbVb(v_hi0, i8);
        v_lo1 = Q6_Vb_vsub_VbVb(v_lo1, i8);
        v_hi1 = Q6_Vb_vsub_VbVb(v_hi1, i8);
        v_lo2 = Q6_Vb_vsub_VbVb(v_lo2, i8);
        v_hi2 = Q6_Vb_vsub_VbVb(v_hi2, i8);
        v_lo3 = Q6_Vb_vsub_VbVb(v_lo3, i8);
        v_hi3 = Q6_Vb_vsub_VbVb(v_hi3, i8);

        // Shuffling
        HVX_సదిశ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_సదిశ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_సదిశ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_సదిశ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_సదిశ_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_సదిశ_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_సదిశ_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_సదిశ_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_సదిశ_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_సదిశ_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_సదిశ_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_సదిశ_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_సదిశ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_సదిశ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_సదిశ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_సదిశ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_సదిశ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_సదిశ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_సదిశ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_సదిశ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_సదిశ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_సదిశ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_సదిశ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_సదిశ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_సదిశ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_సదిశ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_సదిశ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_సదిశ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q4_1(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ vscale_స్థానభ్రంశం = hvx_vmem(tile_src + 512);
        HVX_సదిశ_Pair dm_deal = Q6_W_vdeal_VVR(vscale_స్థానభ్రంశం, vscale_స్థానభ్రంశం, -2);
        HVX_సదిశ vd = Q6_V_lo_W(dm_deal);
        HVX_సదిశ vm = Q6_V_hi_W(dm_deal);

        HVX_సదిశ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_సదిశ v_స్థానభ్రంశం_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_సదిశ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_సదిశ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_సదిశ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_సదిశ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_సదిశ v_lo0 = Q6_V_vand_VV(vq0, ముసుగు_h4);
        HVX_సదిశ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_సదిశ v_lo1 = Q6_V_vand_VV(vq1, ముసుగు_h4);
        HVX_సదిశ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_సదిశ v_lo2 = Q6_V_vand_VV(vq2, ముసుగు_h4);
        HVX_సదిశ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_సదిశ v_lo3 = Q6_V_vand_VV(vq3, ముసుగు_h4);
        HVX_సదిశ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_సదిశ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_సదిశ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_సదిశ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_సదిశ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_సదిశ_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_సదిశ_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_సదిశ_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_సదిశ_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_సదిశ_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_సదిశ_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_సదిశ_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_సదిశ_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_సదిశ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        HVX_సదిశ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        HVX_సదిశ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        HVX_సదిశ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        // Parallel Stores
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_iq4_nl(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    స్థిరరాశి HVX_సదిశ vlut_cvt = hvx_vmem(iq4_nl_గమ్యం_fp16_lut);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v_sc = hvx_vmem(tile_src + 512);
        HVX_సదిశ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_సదిశ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_సదిశ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_సదిశ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_సదిశ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_సదిశ v_lo0 = Q6_V_vand_VV(vq0, ముసుగు_h4);
        HVX_సదిశ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_సదిశ v_lo1 = Q6_V_vand_VV(vq1, ముసుగు_h4);
        HVX_సదిశ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_సదిశ v_lo2 = Q6_V_vand_VV(vq2, ముసుగు_h4);
        HVX_సదిశ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_సదిశ v_lo3 = Q6_V_vand_VV(vq3, ముసుగు_h4);
        HVX_సదిశ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_సదిశ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_సదిశ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_సదిశ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_సదిశ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_సదిశ v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_సదిశ v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_సదిశ v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_సదిశ v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_సదిశ v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_సదిశ v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_సదిశ v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_సదిశ v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_సదిశ_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_సదిశ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_సదిశ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_సదిశ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_సదిశ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_సదిశ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_సదిశ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_సదిశ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_సదిశ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_సదిశ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_సదిశ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_సదిశ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_సదిశ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_సదిశ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_సదిశ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_సదిశ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_సదిశ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_mxfp4(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    స్థిరరాశి HVX_సదిశ vlut_cvt = hvx_vmem(mxfp4_గమ్యం_fp16_lut);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v = hvx_vmem(tile_src + 512);
        HVX_సదిశ vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_సదిశ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_సదిశ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_సదిశ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_సదిశ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_సదిశ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_సదిశ v_lo0 = Q6_V_vand_VV(vq0, ముసుగు_h4);
        HVX_సదిశ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_సదిశ v_lo1 = Q6_V_vand_VV(vq1, ముసుగు_h4);
        HVX_సదిశ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_సదిశ v_lo2 = Q6_V_vand_VV(vq2, ముసుగు_h4);
        HVX_సదిశ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_సదిశ v_lo3 = Q6_V_vand_VV(vq3, ముసుగు_h4);
        HVX_సదిశ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_సదిశ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_సదిశ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_సదిశ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_సదిశ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_సదిశ v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_సదిశ v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_సదిశ v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_సదిశ v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_సదిశ v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_సదిశ v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_సదిశ v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_సదిశ v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_సదిశ_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_సదిశ_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_సదిశ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_సదిశ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_సదిశ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_సదిశ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_సదిశ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_సదిశ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_సదిశ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_సదిశ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_సదిశ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_సదిశ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_సదిశ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_సదిశ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_సదిశ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_సదిశ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_సదిశ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_సదిశ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q8_0(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v_sc = hvx_vmem(tile_src + 1024);
        HVX_సదిశ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_సదిశ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_సదిశ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_సదిశ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_సదిశ vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_సదిశ_Pair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_సదిశ_Pair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_సదిశ_Pair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_సదిశ_Pair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_సదిశ vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_సదిశ vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_సదిశ vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_సదిశ vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_సదిశ_Pair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_సదిశ_Pair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_సదిశ_Pair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_సదిశ_Pair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_సదిశ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_సదిశ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_సదిశ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_సదిశ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_సదిశ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_సదిశ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_సదిశ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_సదిశ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

        // Store groups 0-3
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  4 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp3_1;

        // Convert and scale multiply for groups 4-7
        HVX_సదిశ v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_సదిశ v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_సదిశ v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_సదిశ v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_సదిశ v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_సదిశ v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_సదిశ v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_సదిశ v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

        // Store groups 4-7
        hvx_vmem(dst_ptr +  8 * 64) = v_grp4_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp4_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp5_0;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp5_1;
        hvx_vmem(dst_ptr + 12 * 64) = v_grp6_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp6_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp7_0;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp7_1;
    }
}

స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q5_k(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile_src = state->src + t * state->aligned_tile_కొలత;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ vscale_స్థానభ్రంశం = hvx_vmem(tile_src + 512);
        HVX_సదిశ_Pair dm_deal = Q6_W_vdeal_VVR(vscale_స్థానభ్రంశం, vscale_స్థానభ్రంశం, -2);
        HVX_సదిశ vd = Q6_V_lo_W(dm_deal);
        HVX_సదిశ vm = Q6_V_hi_W(dm_deal);

        HVX_సదిశ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_సదిశ v_స్థానభ్రంశం_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_సదిశ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_సదిశ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_సదిశ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_సదిశ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_సదిశ v_lo0 = Q6_V_vand_VV(vq0, ముసుగు_h4);
        HVX_సదిశ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_సదిశ v_lo1 = Q6_V_vand_VV(vq1, ముసుగు_h4);
        HVX_సదిశ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_సదిశ v_lo2 = Q6_V_vand_VV(vq2, ముసుగు_h4);
        HVX_సదిశ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_సదిశ v_lo3 = Q6_V_vand_VV(vq3, ముసుగు_h4);
        HVX_సదిశ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Q5_K: OR in the 5th bit from the plane
        HVX_సదిశ v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

        // Shuffling
        HVX_సదిశ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_సదిశ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_సదిశ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_సదిశ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_సదిశ_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_సదిశ_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_సదిశ_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_సదిశ_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_సదిశ_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_సదిశ_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_సదిశ_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_సదిశ_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_సదిశ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        HVX_సదిశ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        HVX_సదిశ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        HVX_సదిశ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));
        HVX_సదిశ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_స్థానభ్రంశం_duplicated));

        // Parallel Stores
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

// Q6_K stores 6-bit weights and one fp16 scale per 16 k, see HTP_TENSOR_TILE_SIZE_Q6_K.
// A k-group holds 4 k per row, the HMX tile holds 2, so each group is dealt into two tiles.
స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q6_k(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_0f = Q6_Vb_vsplat_R(0x0F);
    స్థిరరాశి HVX_సదిశ ముసుగు_03 = Q6_Vb_vsplat_R(0x03);
    స్థిరరాశి HVX_సదిశ i32     = Q6_Vb_vsplat_R(32);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి HVX_సదిశ * vptr = (స్థిరరాశి HVX_సదిశ *) (state->src + t * state->aligned_tile_కొలత);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v_sc      = vptr[6];
        HVX_సదిశ v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_సదిశ v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_సదిశ v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #సూచన unroll
        కొరకు (పూర్ణాంకం g = 0; g < 8; g += 4) {
            స్థిరరాశి HVX_సదిశ v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_సదిశ v_q0 = unpack_q6_k_సమూహం(vptr, g + 0, ముసుగు_0f, ముసుగు_03, i32);
            HVX_సదిశ v_q1 = unpack_q6_k_సమూహం(vptr, g + 1, ముసుగు_0f, ముసుగు_03, i32);
            HVX_సదిశ v_q2 = unpack_q6_k_సమూహం(vptr, g + 2, ముసుగు_0f, ముసుగు_03, i32);
            HVX_సదిశ v_q3 = unpack_q6_k_సమూహం(vptr, g + 3, ముసుగు_0f, ముసుగు_03, i32);

            HVX_సదిశ_Pair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_సదిశ_Pair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_సదిశ_Pair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_సదిశ_Pair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_సదిశ_Pair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_సదిశ_Pair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_సదిశ_Pair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_సదిశ_Pair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_సదిశ v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_సదిశ v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_సదిశ v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_సదిశ v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_సదిశ v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_సదిశ v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_సదిశ v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_సదిశ v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = v_out00;
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = v_out01;
            hvx_vmem(dst_ptr + (2 * g + 2) * 64) = v_out10;
            hvx_vmem(dst_ptr + (2 * g + 3) * 64) = v_out11;
            hvx_vmem(dst_ptr + (2 * g + 4) * 64) = v_out20;
            hvx_vmem(dst_ptr + (2 * g + 5) * 64) = v_out21;
            hvx_vmem(dst_ptr + (2 * g + 6) * 64) = v_out30;
            hvx_vmem(dst_ptr + (2 * g + 7) * 64) = v_out31;
        }
    }
}

// Q3_K stores 3-bit weights and one fp16 scale per 16 k, see HTP_TENSOR_TILE_SIZE_Q3_K.
స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q3_k(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_03 = Q6_Vb_vsplat_R(0x03);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి HVX_సదిశ * vptr = (స్థిరరాశి HVX_సదిశ *) (state->src + t * state->aligned_tile_కొలత);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v_sc      = vptr[3];
        HVX_సదిశ v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_సదిశ v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_సదిశ v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #సూచన unroll
        కొరకు (పూర్ణాంకం g = 0; g < 8; g++) {
            స్థిరరాశి HVX_సదిశ v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_సదిశ     v_q   = unpack_q3_k_సమూహం(vptr, g, ముసుగు_03);
            HVX_సదిశ_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_సదిశ_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q2_k(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_03 = Q6_Vb_vsplat_R(0x03);

    HVX_సదిశ v_d = Q6_V_vzero();
    HVX_సదిశ v_dm = Q6_V_vzero();

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి uint8_t * tile = state->src + t * state->aligned_tile_కొలత;
        స్థిరరాశి HVX_సదిశ * vptr = (స్థిరరాశి HVX_సదిశ *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        ఒకవేళ (t == ప్రారంభం_tile || (t & 7) == 0) {
            q2_k_ఎక్కించు_dmin(tile - (t & 7) * state->aligned_tile_కొలత, &v_d, &v_dm);
        }

        HVX_సదిశ v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_సదిశ v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_సదిశ v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_సదిశ v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_సదిశ v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_సదిశ v_స్థానభ్రంశం_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_సదిశ v_స్థానభ్రంశం_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #సూచన unroll
        కొరకు (పూర్ణాంకం g = 0; g < 8; g++) {
            స్థిరరాశి HVX_సదిశ v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            స్థిరరాశి HVX_సదిశ v_స్థానభ్రంశం = (g < 4) ? v_స్థానభ్రంశం_k0 : v_స్థానభ్రంశం_k16;

            HVX_సదిశ     v_q   = unpack_q3_k_low2(vptr, g, ముసుగు_03);
            HVX_సదిశ_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_సదిశ_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_స్థానభ్రంశం));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_స్థానభ్రంశం));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
స్థిర శూన్య dequantize_tiled_బరువు_గమ్యం_fp16_task_q2_0(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి HVX_సదిశ ముసుగు_03 = Q6_Vb_vsplat_R(0x03);
    స్థిరరాశి HVX_సదిశ i1      = Q6_Vb_vsplat_R(1);

    కొరకు (uint32_t t = ప్రారంభం_tile; t < చివర_tile; t++) {
        స్థిరరాశి HVX_సదిశ * vptr = (స్థిరరాశి HVX_సదిశ *) (state->src + t * state->aligned_tile_కొలత);
        __fp16 * dst_పునాది = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_సదిశ v_sc    = vptr[4];
        HVX_సదిశ v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        కొరకు (పూర్ణాంకం half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_పునాది + half * HTP_MM_HMX_TILE_N_ELMS;

            #సూచన unroll
            కొరకు (పూర్ణాంకం g = 0; g < 8; g++) {
                HVX_సదిశ     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, ముసుగు_03), i1);
                HVX_సదిశ_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_సదిశ_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

స్థిర __attribute__((noinline))
శూన్య convert_f16_బరువు_గమ్యం_fp16_tiles_task(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి uint32_t n_k_tiles = state->n_k_tiles;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు n_k_tiles_div = state->n_k_tiles_div;

    స్థిరరాశి HVX_సదిశ v_scat_పునాది  = hvx_vmem(hmx_transpose_scatter_offsets);
    స్థిరరాశి HVX_సదిశ v_scat_అడుగు  = Q6_V_vsplat_R(4);
    స్థిరరాశి HVX_సదిశ_Pred q_mask64 = Q6_Q_vsetq_R(64);

    గుర్తులేని ct = fastdiv((గుర్తులేని)ప్రారంభం_tile, &n_k_tiles_div);
    గుర్తులేని kt = fastmodulo((గుర్తులేని)ప్రారంభం_tile, n_k_tiles, &n_k_tiles_div);

    కొరకు (గుర్తులేని t = ప్రారంభం_tile; t < (గుర్తులేని)చివర_tile; ) {
        ఒకవేళ (kt >= (గుర్తులేని)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_పునాది = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * పరిమాణం(__fp16);

            HVX_సదిశ v_off = v_scat_పునాది;
            కొరకు (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_అడ్డువరుసలు; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య + r;
                uint32_t row1 = row0 + 1;

                స్థిరరాశి uint8_t *r0 = state->src + row0 * state->అడ్డువరుస_stride;
                స్థిరరాశి uint8_t *r1 = state->src + row1 * state->అడ్డువరుస_stride;

                HVX_సదిశ v0 = (row0 < state->n_నిలువుల_సంఖ్య) ? hvx_vmemu((స్థిరరాశి __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_సదిశ v1 = (row1 < state->n_నిలువుల_సంఖ్య) ? hvx_vmemu((స్థిరరాశి __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_పునాది, HTP_MM_HMX_TILE_కొలత - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_అడుగు);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_పునాది, HTP_MM_HMX_TILE_కొలత - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_అడుగు);
            }
        }
        ++t; ++kt;
    }
}

స్థిర __attribute__((noinline))
శూన్య quantize_f32_బరువు_గమ్యం_fp16_tiles_task(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి uint32_t n_k_tiles = state->n_k_tiles;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు n_k_tiles_div = state->n_k_tiles_div;

    స్థిరరాశి HVX_సదిశ v_scat_పునాది  = hvx_vmem(hmx_transpose_scatter_offsets);
    స్థిరరాశి HVX_సదిశ v_scat_అడుగు  = Q6_V_vsplat_R(4);
    స్థిరరాశి HVX_సదిశ_Pred q_mask64 = Q6_Q_vsetq_R(64);

    గుర్తులేని ct = fastdiv((గుర్తులేని)ప్రారంభం_tile, &n_k_tiles_div);
    గుర్తులేని kt = fastmodulo((గుర్తులేని)ప్రారంభం_tile, n_k_tiles, &n_k_tiles_div);

    కొరకు (గుర్తులేని t = ప్రారంభం_tile; t < (గుర్తులేని)చివర_tile; ) {
        ఒకవేళ (kt >= (గుర్తులేని)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_పునాది = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * పరిమాణం(దశాంశం);

            HVX_సదిశ v_off = v_scat_పునాది;
            కొరకు (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_అడ్డువరుసలు; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య + r;
                uint32_t row1 = row0 + 1;

                స్థిరరాశి uint8_t *r0 = state->src + row0 * state->అడ్డువరుస_stride;
                స్థిరరాశి uint8_t *r1 = state->src + row1 * state->అడ్డువరుస_stride;

                HVX_సదిశ v0_f32 = (row0 < state->n_నిలువుల_సంఖ్య) ? hvx_vmem((స్థిరరాశి దశాంశం *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_సదిశ v1_f32 = (row1 < state->n_నిలువుల_సంఖ్య) ? hvx_vmem((స్థిరరాశి దశాంశం *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_పునాది, HTP_MM_HMX_TILE_కొలత - 1, v_off, v_బహిర్గతం);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_అడుగు);

                HVX_సదిశ v_బహిర్గతం_hi = Q6_V_vror_VR(v_బహిర్గతం, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_పునాది, HTP_MM_HMX_TILE_కొలత - 1, v_off, v_బహిర్గతం_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_అడుగు);
            }
        }
        ++t; ++kt;
    }
}

స్థిర __attribute__((noinline))
శూన్య convert_bf16_బరువు_గమ్యం_fp16_tiles_task(
        స్థిరరాశి tiled_dequantize_state_t *state,
        uint32_t ప్రారంభం_tile, uint32_t చివర_tile) {

    స్థిరరాశి uint32_t n_k_tiles = state->n_k_tiles;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు n_k_tiles_div = state->n_k_tiles_div;

    స్థిరరాశి HVX_సదిశ v_scat_పునాది  = hvx_vmem(hmx_transpose_scatter_offsets);
    స్థిరరాశి HVX_సదిశ v_scat_అడుగు  = Q6_V_vsplat_R(4);
    స్థిరరాశి HVX_సదిశ_Pred q_mask64 = Q6_Q_vsetq_R(64);

    గుర్తులేని ct = fastdiv((గుర్తులేని)ప్రారంభం_tile, &n_k_tiles_div);
    గుర్తులేని kt = fastmodulo((గుర్తులేని)ప్రారంభం_tile, n_k_tiles, &n_k_tiles_div);

    కొరకు (గుర్తులేని t = ప్రారంభం_tile; t < (గుర్తులేని)చివర_tile; ) {
        ఒకవేళ (kt >= (గుర్తులేని)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_పునాది = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * పరిమాణం(uint16_t);

            HVX_సదిశ v_off = v_scat_పునాది;
            కొరకు (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_అడ్డువరుసలు; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య + r;
                uint32_t row1 = row0 + 1;

                స్థిరరాశి uint8_t *r0 = state->src + row0 * state->అడ్డువరుస_stride;
                స్థిరరాశి uint8_t *r1 = state->src + row1 * state->అడ్డువరుస_stride;

                HVX_సదిశ v0_raw = (row0 < state->n_నిలువుల_సంఖ్య) ? hvx_vmemu((స్థిరరాశి శూన్య *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_సదిశ v1_raw = (row1 < state->n_నిలువుల_సంఖ్య) ? hvx_vmemu((స్థిరరాశి శూన్య *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_సదిశ v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_సదిశ v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_పునాది, HTP_MM_HMX_TILE_కొలత - 1, v_off, v_బహిర్గతం);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_అడుగు);

                HVX_సదిశ v_బహిర్గతం_hi = Q6_V_vror_VR(v_బహిర్గతం, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_పునాది, HTP_MM_HMX_TILE_కొలత - 1, v_off, v_బహిర్గతం_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_అడుగు);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

స్థిర శూన్య core_dot_chunk_fp16_short(__fp16 *పరిమిత output, స్థిరరాశి __fp16 *పరిమిత activation,
                                స్థిరరాశి __fp16 *పరిమిత బరువు, స్థిరరాశి __fp16 *పరిమిత scales,
                                uint32_t n_అడ్డువరుస_tiles, uint32_t n_నిలువు_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(n_నిలువు_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)scales));

    స్థిరరాశి size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    స్థిరరాశి uint32_t range = 2048u * n_dot_tiles - 1;

    కొరకు (uint32_t r = 0; r < n_అడ్డువరుస_tiles; ++r) {
        స్థిరరాశి __fp16 *అడ్డువరుస_పునాది = activation + r * dot_stride;
        స్థిరరాశి __fp16 *నిలువు_పునాది = బరువు;
        __fp16 *బహిర్గతం_tile = output + r * n_నిలువు_tiles * HTP_MM_HMX_TILE_N_ELMS;

        కొరకు (size_t c = 0; c < n_నిలువు_tiles; ++c) {
            asm అస్థిర(HMX_CLRACC_F16());
            asm అస్థిర(HMX_ఎక్కించు_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(అడ్డువరుస_పునాది), "r"(నిలువు_పునాది));
            asm అస్థిర(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(బహిర్గతం_tile), "r"(0) : "memory");
            నిలువు_పునాది += dot_stride;
            బహిర్గతం_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

స్థిర శూన్య core_dot_chunk_fp16(__fp16 *పరిమిత output, స్థిరరాశి __fp16 *పరిమిత activation,
                          స్థిరరాశి __fp16 *పరిమిత బరువు, స్థిరరాశి __fp16 *పరిమిత scales,
                          uint32_t n_అడ్డువరుస_tiles, uint32_t n_నిలువు_tiles, uint32_t n_dot_tiles) {
    ఒకవేళ (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, బరువు, scales, n_అడ్డువరుస_tiles, n_నిలువు_tiles, n_dot_tiles);
        తిరిగి;
    }
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(n_నిలువు_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)scales));

    స్థిరరాశి size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    కొరకు (uint32_t r = 0; r < n_అడ్డువరుస_tiles; ++r) {
        స్థిరరాశి __fp16 *అడ్డువరుస_పునాది = activation + r * dot_stride;
        స్థిరరాశి __fp16 *నిలువు_పునాది = బరువు;
        __fp16 *బహిర్గతం_tile = output + r * n_నిలువు_tiles * HTP_MM_HMX_TILE_N_ELMS;

        కొరకు (size_t c = 0; c < n_నిలువు_tiles; ++c) {
            స్థిరరాశి __fp16 *అడ్డువరుస_tiles = అడ్డువరుస_పునాది;
            స్థిరరాశి __fp16 *నిలువు_tiles = నిలువు_పునాది;

            asm అస్థిర(HMX_CLRACC_F16());

            స్థిరరాశి uint32_t n_loops = n_dot_tiles / 32;
            స్థిరరాశి uint32_t rem = n_dot_tiles % 32;

            కొరకు (uint32_t l = 0; l < n_loops; ++l) {
                asm అస్థిర(HMX_ఎక్కించు_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(అడ్డువరుస_tiles), "r"(నిలువు_tiles));
                అడ్డువరుస_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                నిలువు_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            ఒకవేళ (rem > 0) {
                స్థిరరాశి uint32_t range = 2048u * rem - 1;
                asm అస్థిర(HMX_ఎక్కించు_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(అడ్డువరుస_tiles), "r"(నిలువు_tiles));
            }

            asm అస్థిర(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(బహిర్గతం_tile), "r"(0) : "memory");

            నిలువు_పునాది += dot_stride;
            బహిర్గతం_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

స్థిర శూన్య core_mma_chunk_fp16_short(__fp16 *పరిమిత c, స్థిరరాశి __fp16 *పరిమిత a, స్థిరరాశి __fp16 *పరిమిత b,
                                స్థిరరాశి __fp16 *పరిమిత నిలువు_scales, స్థిరరాశి __fp16 *పరిమిత eye_tile,
                                uint32_t n_అడ్డువరుస_tiles, uint32_t n_నిలువు_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(n_నిలువు_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)నిలువు_scales));

    స్థిరరాశి size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    స్థిరరాశి uint32_t range = 2048u * n_dot_tiles - 1;

    కొరకు (size_t i = 0; i < n_అడ్డువరుస_tiles; ++i) {
        స్థిరరాశి __fp16 *అడ్డువరుస_పునాది = a + i * dot_tile_stride;
        __fp16 *res_పునాది = c + i * n_నిలువు_tiles * HTP_MM_HMX_TILE_N_ELMS;
        స్థిరరాశి __fp16 *నిలువు_పునాది = b;
        __fp16 *accum_tile = res_పునాది;

        కొరకు (size_t j = 0; j < n_నిలువు_tiles; ++j) {
            asm అస్థిర(HMX_CLRACC_F16());

            ఒకవేళ (!zero_init) {
                asm అస్థిర(HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm అస్థిర(HMX_ఎక్కించు_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(అడ్డువరుస_పునాది), "r"(నిలువు_పునాది));

            asm అస్థిర(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            నిలువు_పునాది   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

స్థిర శూన్య core_mma_chunk_fp16(__fp16 *పరిమిత c, స్థిరరాశి __fp16 *పరిమిత a, స్థిరరాశి __fp16 *పరిమిత b,
                          స్థిరరాశి __fp16 *పరిమిత నిలువు_scales, స్థిరరాశి __fp16 *పరిమిత eye_tile,
                          uint32_t n_అడ్డువరుస_tiles, uint32_t n_నిలువు_tiles, uint32_t n_dot_tiles, bool zero_init) {
    ఒకవేళ (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, నిలువు_scales, eye_tile, n_అడ్డువరుస_tiles, n_నిలువు_tiles, n_dot_tiles, zero_init);
        తిరిగి;
    }
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(n_నిలువు_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)నిలువు_scales));

    స్థిరరాశి size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    కొరకు (size_t i = 0; i < n_అడ్డువరుస_tiles; ++i) {
        స్థిరరాశి __fp16 *అడ్డువరుస_పునాది = a + i * dot_tile_stride;
        __fp16 *res_పునాది = c + i * n_నిలువు_tiles * HTP_MM_HMX_TILE_N_ELMS;
        స్థిరరాశి __fp16 *నిలువు_పునాది = b;
        __fp16 *accum_tile = res_పునాది;

        కొరకు (size_t j = 0; j < n_నిలువు_tiles; ++j) {
            స్థిరరాశి __fp16 *నిలువు_tiles = నిలువు_పునాది;
            స్థిరరాశి __fp16 *అడ్డువరుస_tiles = అడ్డువరుస_పునాది;

            asm అస్థిర(HMX_CLRACC_F16());

            ఒకవేళ (!zero_init) {
                asm అస్థిర(HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            స్థిరరాశి uint32_t n_loops = n_dot_tiles / 32;
            స్థిరరాశి uint32_t rem = n_dot_tiles % 32;

            కొరకు (uint32_t l = 0; l < n_loops; ++l) {
                asm అస్థిర(HMX_ఎక్కించు_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(అడ్డువరుస_tiles), "r"(నిలువు_tiles));
                అడ్డువరుస_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                నిలువు_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            ఒకవేళ (rem > 0) {
                స్థిరరాశి uint32_t range = 2048u * rem - 1;
                asm అస్థిర(HMX_ఎక్కించు_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(అడ్డువరుస_tiles), "r"(నిలువు_tiles));
            }

            asm అస్థిర(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            నిలువు_పునాది += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32p

స్థిర శూన్య transfer_output_chunk_fp16_గమ్యం_fp32_నిలువు_chunk(
    దశాంశం *పరిమిత dst,
    స్థిరరాశి దశాంశం *పరిమిత src2,
    స్థిరరాశి __fp16 *పరిమిత vtcm_src,
    uint32_t ప్రారంభం_అడ్డువరుస,
    uint32_t n_అడ్డువరుసలు,
    uint32_t c_len,
    uint32_t మొత్తం_n_నిలువుల_సంఖ్య,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_నిలువుల_సంఖ్య
) {
    స్థిరరాశి size_t tile_అడ్డువరుస_stride = hmx_ceil_div(మొత్తం_n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) * HTP_MM_HMX_TILE_N_ELMS;

    స్థిరరాశి HVX_సదిశ one = hvx_vec_splat_f16(1.0);

    స్థిరరాశి size_t హద్దు_c         = hex_smin(c_len, dst_నిలువుల_సంఖ్య);
    స్థిరరాశి size_t హద్దు_c_aligned = (హద్దు_c & ~31);

    కొరకు (size_t r = 0; r < n_అడ్డువరుసలు; r += 2) {
        స్థిరరాశి size_t r_idx0 = ప్రారంభం_అడ్డువరుస + r + 0;
        స్థిరరాశి size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;
        స్థిరరాశి size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_అడ్డువరుసలు) / 2;  // index of the row pair within the tile
        స్థిరరాశి __fp16 *అడ్డువరుస_పునాది = vtcm_src + r0 * tile_అడ్డువరుస_stride;
        దశాంశం *output_అడ్డువరుస_పునాది = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        స్థిరరాశి దశాంశం *src2_అడ్డువరుస_పునాది = src2 ? (src2 + r * src2_stride) : ఖాళీ;

        #సూచన unroll(4)
        కొరకు (size_t c = 0; c < హద్దు_c_aligned; c += HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) {
            స్థిరరాశి size_t c0    = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            స్థిరరాశి __fp16 *tile = అడ్డువరుస_పునాది + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_సదిశ v = ((స్థిరరాశి HVX_సదిశ *) tile)[r1];
            HVX_సదిశ_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_సదిశ v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ఒకవేళ (src2_అడ్డువరుస_పునాది) {
                HVX_సదిశ v_src2_0 = hvx_vmemu(src2_అడ్డువరుస_పునాది + c + 0);
                v_out0 = hvx_vec_కూడు_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_అడ్డువరుస_పునాది + c + 0) = v_out0;

            ఒకవేళ (r + 1 < n_అడ్డువరుసలు) {
                HVX_సదిశ v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                ఒకవేళ (src2_అడ్డువరుస_పునాది) {
                    HVX_సదిశ v_src2_1 = hvx_vmemu(src2_అడ్డువరుస_పునాది + c + src2_stride);
                    v_out1 = hvx_vec_కూడు_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_అడ్డువరుస_పునాది + c + dst_stride) = v_out1;
            }
        }

        ఒకవేళ (హద్దు_c_aligned < హద్దు_c) {
            size_t c = హద్దు_c_aligned;
            size_t valid_c = హద్దు_c - c;
            స్థిరరాశి size_t c0 = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            స్థిరరాశి __fp16 *tile = అడ్డువరుస_పునాది + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_సదిశ v = ((స్థిరరాశి HVX_సదిశ *) tile)[r1];
            HVX_సదిశ_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_సదిశ v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ఒకవేళ (src2_అడ్డువరుస_పునాది) {
                HVX_సదిశ v_src2_0 = hvx_vmemu(src2_అడ్డువరుస_పునాది + c + 0);
                v_out0 = hvx_vec_కూడు_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_అడ్డువరుస_పునాది + c, valid_c * పరిమాణం(దశాంశం), v_out0);

            ఒకవేళ (r + 1 < n_అడ్డువరుసలు) {
                HVX_సదిశ v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                ఒకవేళ (src2_అడ్డువరుస_పునాది) {
                    HVX_సదిశ v_src2_1 = hvx_vmemu(src2_అడ్డువరుస_పునాది + c + src2_stride);
                    v_out1 = hvx_vec_కూడు_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_అడ్డువరుస_పునాది + c + dst_stride, valid_c * పరిమాణం(దశాంశం), v_out1);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య transfer_output_chunk_fp16_గమ్యం_fp32(
    దశాంశం *పరిమిత dst,
    స్థిరరాశి దశాంశం *పరిమిత src2,
    స్థిరరాశి __fp16 *పరిమిత vtcm_src,
    uint32_t ప్రారంభం_అడ్డువరుస,
    uint32_t n_అడ్డువరుసలు,
    uint32_t n_నిలువుల_సంఖ్య,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_నిలువుల_సంఖ్య
) {
    transfer_output_chunk_fp16_గమ్యం_fp32_నిలువు_chunk(
        dst, src2, vtcm_src, ప్రారంభం_అడ్డువరుస, n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య, n_నిలువుల_సంఖ్య, dst_stride, src2_stride, dst_నిలువుల_సంఖ్య
    );
}

రకనిర్వచనం నిర్మాణం {
    స్థిరరాశి __fp16  *vtcm_src;
    దశాంశం         *dst;
    స్థిరరాశి దశాంశం   *src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_చొప్పున_task;
    uint32_t       n_నిలువుల_సంఖ్య;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_నిలువుల_సంఖ్య;    // Actual output columns
    నిర్మాణం htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

స్థిర శూన్య transfer_activation_అడ్డువరుస_pair_fp32_గమ్యం_fp16(
        __fp16 *పరిమిత vtcm_dst,
        స్థిరరాశి దశాంశం *పరిమిత row0,
        స్థిరరాశి దశాంశం *పరిమిత row1,
        uint32_t r,
        uint32_t k_ఖండం,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

    uint32_t c = 0;
    కొరకు (; c + 32 <= k_valid; c += 32) {
        HVX_సదిశ v0 = Q6_V_vzero();
        HVX_సదిశ v1 = Q6_V_vzero();
        ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0 + c);
        ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1 + c);

        HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;  // tile column index
        uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

        HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_బహిర్గతం;
    }
    ఒకవేళ (c < k_ఖండం) {
        HVX_సదిశ v0 = Q6_V_vzero();
        HVX_సదిశ v1 = Q6_V_vzero();
        ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0 + c);
        ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem > 0 ? rem * పరిమాణం(దశాంశం) : 0);
        v0 = Q6_V_vmux_QVV(ముసుగు, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ముసుగు, v1, Q6_V_vzero());

        HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;  // tile column index
        uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

        HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_బహిర్గతం;
    }
}

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
స్థిర శూన్య transfer_activation_అడ్డువరుస_pair_f16_గమ్యం_f16(__fp16 * పరిమిత vtcm_dst,
                                                    స్థిరరాశి __fp16 * పరిమిత row0,
                                                    స్థిరరాశి __fp16 * పరిమిత row1,
                                                    uint32_t r,
                                                    uint32_t k_ఖండం,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

    స్థిరరాశి uint32_t n_tile_నిలువుల_సంఖ్య = k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
    __fp16 * పరిమిత tile_అడ్డువరుస = vtcm_dst + (size_t) r0 * n_tile_నిలువుల_సంఖ్య * HTP_MM_HMX_TILE_N_ELMS;

    స్థిరరాశి HVX_UVector * pv0 = row0_valid ? (స్థిరరాశి HVX_UVector *) row0 : ఖాళీ;
    స్థిరరాశి HVX_UVector * pv1 = row1_valid ? (స్థిరరాశి HVX_UVector *) row1 : ఖాళీ;

    uint32_t c = 0;
    కొరకు (; c + 64 <= k_valid; c += 64) {
        HVX_సదిశ     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_సదిశ     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;

        HVX_సదిశ * tile0 = (HVX_సదిశ *) (tile_అడ్డువరుస + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_సదిశ * tile1 = (HVX_సదిశ *) (tile_అడ్డువరుస + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    కొరకు (; c < k_ఖండం; c += 64) {
        HVX_సదిశ v0 = Q6_V_vzero();
        HVX_సదిశ v1 = Q6_V_vzero();

        ఒకవేళ (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem * పరిమాణం(__fp16));
            ఒకవేళ (pv0) {
                v0 = Q6_V_vmux_QVV(ముసుగు, pv0[c / 64], Q6_V_vzero());
            }
            ఒకవేళ (pv1) {
                v1 = Q6_V_vmux_QVV(ముసుగు, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;

        HVX_సదిశ * tile0 = (HVX_సదిశ *) (tile_అడ్డువరుస + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        ఒకవేళ (c0 + 1 < n_tile_నిలువుల_సంఖ్య) {
            HVX_సదిశ * tile1 = (HVX_సదిశ *) (tile_అడ్డువరుస + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

స్థిర శూన్య transfer_activation_అడ్డువరుస_pair_fp32_గమ్యం_fp16_నిలువు_chunk(
        __fp16 *పరిమిత vtcm_dst,
        స్థిరరాశి దశాంశం *పరిమిత row0, // offset by c_first
        స్థిరరాశి దశాంశం *పరిమిత row1, // offset by c_first
        uint32_t r,
        uint32_t k_ఖండం,
        uint32_t c_మొదటి,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

    uint32_t c = 0;
    కొరకు (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_సదిశ v0 = Q6_V_vzero();
        HVX_సదిశ v1 = Q6_V_vzero();
        ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0 + c);
        ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1 + c);

        HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

        uint32_t c0       = (c_మొదటి + c) / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;  // tile column index
        uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

        HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_బహిర్గతం;
    }
    ఒకవేళ (c < c_len) {
        HVX_సదిశ v0 = Q6_V_vzero();
        HVX_సదిశ v1 = Q6_V_vzero();
        ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0 + c);
        ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem > 0 ? rem * పరిమాణం(దశాంశం) : 0);
        v0 = Q6_V_vmux_QVV(ముసుగు, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ముసుగు, v1, Q6_V_vzero());

        HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

        uint32_t c0       = (c_మొదటి + c) / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;  // tile column index
        uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

        HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_బహిర్గతం;
    }
}

స్థిర శూన్య transfer_activation_chunk_fp32_గమ్యం_fp16_gathered(
            __fp16 *పరిమిత vtcm_dst,
            స్థిరరాశి దశాంశం *పరిమిత src,
            uint32_t ప్రారంభం_అడ్డువరుస,
            uint32_t vtcm_ప్రారంభం_అడ్డువరుస,
            uint32_t n_అడ్డువరుసలు,
            uint32_t k_ఖండం,
            స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping *మాతృక_అడ్డువరుసలు,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            స్థిరరాశి నిర్మాణం fastdiv_విలువలు * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_padded = hex_align_పైకి(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_tiled  = (n_అడ్డువరుసలు / HTP_MM_HMX_TILE_N_అడ్డువరుసలు) * HTP_MM_HMX_TILE_N_అడ్డువరుసలు;

    uint32_t r = 0;

    #సూచన unroll(2)
    కొరకు (r = 0; r < n_అడ్డువరుసలు_tiled; r += 2) {
        uint32_t r_idx0 = ప్రారంభం_అడ్డువరుస + r + 0;
        uint32_t r_idx1 = ప్రారంభం_అడ్డువరుస + r + 1;
        uint32_t lr = vtcm_ప్రారంభం_అడ్డువరుస + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

        నిర్మాణం mmid_అడ్డువరుస_mapping mapping0 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + r_idx0];
        నిర్మాణం mmid_అడ్డువరుస_mapping mapping1 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        స్థిరరాశి దశాంశం *row0_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        స్థిరరాశి దశాంశం *row1_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        కొరకు (; c + 32 <= k_valid; c += 32) {
            HVX_సదిశ v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            HVX_సదిశ v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);
            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
        ఒకవేళ (c < k_ఖండం) {
            HVX_సదిశ v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            HVX_సదిశ v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem > 0 ? rem * పరిమాణం(దశాంశం) : 0);
            v0 = Q6_V_vmux_QVV(ముసుగు, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ముసుగు, v1, Q6_V_vzero());

            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
    }

    కొరకు (; r < n_అడ్డువరుసలు_padded; r += 2) {
        uint32_t lr = vtcm_ప్రారంభం_అడ్డువరుస + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

        స్థిరరాశి bool row0_valid = (ప్రారంభం_అడ్డువరుస + r + 0) < cne1;
        స్థిరరాశి bool row1_valid = (ప్రారంభం_అడ్డువరుస + r + 1) < cne1;

        స్థిరరాశి దశాంశం *row0_ptr = ఖాళీ;
        స్థిరరాశి దశాంశం *row1_ptr = ఖాళీ;

        ఒకవేళ (row0_valid) {
            నిర్మాణం mmid_అడ్డువరుస_mapping mapping0 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + (ప్రారంభం_అడ్డువరుస + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        ఒకవేళ (row1_valid) {
            నిర్మాణం mmid_అడ్డువరుస_mapping mapping1 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + (ప్రారంభం_అడ్డువరుస + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        కొరకు (; c + 32 <= k_valid; c += 32) {
            HVX_సదిశ v0 = Q6_V_vzero();
            HVX_సదిశ v1 = Q6_V_vzero();
            ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);

            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
        ఒకవేళ (c < k_ఖండం) {
            HVX_సదిశ v0 = Q6_V_vzero();
            HVX_సదిశ v1 = Q6_V_vzero();
            ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem > 0 ? rem * పరిమాణం(దశాంశం) : 0);
            v0 = Q6_V_vmux_QVV(ముసుగు, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ముసుగు, v1, Q6_V_vzero());

            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
    }
}

స్థిర శూన్య transfer_activation_chunk_fp32_గమ్యం_fp16_gathered_flat(
            __fp16 *పరిమిత vtcm_dst,
            స్థిరరాశి దశాంశం *పరిమిత src,
            uint32_t ప్రారంభం_అడ్డువరుస,
            uint32_t vtcm_ప్రారంభం_అడ్డువరుస,
            uint32_t n_అడ్డువరుసలు,
            uint32_t k_ఖండం,
            స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping *మాతృక_అడ్డువరుసలు,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_padded = hex_align_పైకి(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_tiled  = (n_అడ్డువరుసలు / HTP_MM_HMX_TILE_N_అడ్డువరుసలు) * HTP_MM_HMX_TILE_N_అడ్డువరుసలు;

    uint32_t r = 0;

    #సూచన unroll(2)
    కొరకు (r = 0; r < n_అడ్డువరుసలు_tiled; r += 2) {
        uint32_t r_idx0 = ప్రారంభం_అడ్డువరుస + r + 0;
        uint32_t r_idx1 = ప్రారంభం_అడ్డువరుస + r + 1;
        uint32_t lr = vtcm_ప్రారంభం_అడ్డువరుస + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

        నిర్మాణం mmid_అడ్డువరుస_mapping mapping0 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + r_idx0];
        నిర్మాణం mmid_అడ్డువరుస_mapping mapping1 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + r_idx1];

        స్థిరరాశి దశాంశం *row0_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + mapping0.i2 * nb12);
        స్థిరరాశి దశాంశం *row1_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        కొరకు (; c + 32 <= k_valid; c += 32) {
            HVX_సదిశ v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            HVX_సదిశ v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);
            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
        ఒకవేళ (c < k_ఖండం) {
            HVX_సదిశ v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            HVX_సదిశ v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem > 0 ? rem * పరిమాణం(దశాంశం) : 0);
            v0 = Q6_V_vmux_QVV(ముసుగు, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ముసుగు, v1, Q6_V_vzero());

            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
    }

    కొరకు (; r < n_అడ్డువరుసలు_padded; r += 2) {
        uint32_t lr = vtcm_ప్రారంభం_అడ్డువరుస + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_అడ్డువరుసలు;  // intra-tile row idx

        స్థిరరాశి bool row0_valid = (ప్రారంభం_అడ్డువరుస + r + 0) < cne1;
        స్థిరరాశి bool row1_valid = (ప్రారంభం_అడ్డువరుస + r + 1) < cne1;

        స్థిరరాశి దశాంశం *row0_ptr = ఖాళీ;
        స్థిరరాశి దశాంశం *row1_ptr = ఖాళీ;

        ఒకవేళ (row0_valid) {
            నిర్మాణం mmid_అడ్డువరుస_mapping mapping0 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + (ప్రారంభం_అడ్డువరుస + r + 0)];
            row0_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + mapping0.i2 * nb12);
        }
        ఒకవేళ (row1_valid) {
            నిర్మాణం mmid_అడ్డువరుస_mapping mapping1 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + (ప్రారంభం_అడ్డువరుస + r + 1)];
            row1_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        కొరకు (; c + 32 <= k_valid; c += 32) {
            HVX_సదిశ v0 = Q6_V_vzero();
            HVX_సదిశ v1 = Q6_V_vzero();
            ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);

            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
        ఒకవేళ (c < k_ఖండం) {
            HVX_సదిశ v0 = Q6_V_vzero();
            HVX_సదిశ v1 = Q6_V_vzero();
            ఒకవేళ (row0_valid) v0 = *(స్థిరరాశి HVX_సదిశ *)(row0_ptr + c);
            ఒకవేళ (row1_valid) v1 = *(స్థిరరాశి HVX_సదిశ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(rem > 0 ? rem * పరిమాణం(దశాంశం) : 0);
            v0 = Q6_V_vmux_QVV(ముసుగు, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ముసుగు, v1, Q6_V_vzero());

            HVX_సదిశ v_బహిర్గతం = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            uint32_t tile_idx = r0 * (k_ఖండం / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) + c0;

            HVX_సదిశ *tile = (HVX_సదిశ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_బహిర్గతం;
        }
    }
}

స్థిర శూన్య transfer_output_chunk_fp16_గమ్యం_fp32_scattered(
            దశాంశం *పరిమిత dst,
            స్థిరరాశి __fp16 *పరిమిత vtcm_src,
            uint32_t ప్రారంభం_అడ్డువరుస,
            uint32_t vtcm_ప్రారంభం_అడ్డువరుస,
            uint32_t n_అడ్డువరుసలు,
            uint32_t n_నిలువుల_సంఖ్య,
            స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping *మాతృక_అడ్డువరుసలు,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    ప్రతిజ్ఞా(n_నిలువుల_సంఖ్య % HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య == 0);
    స్థిరరాశి size_t tile_అడ్డువరుస_stride = (n_నిలువుల_సంఖ్య / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) * HTP_MM_HMX_TILE_N_ELMS;

    స్థిరరాశి HVX_సదిశ one = hvx_vec_splat_f16(1.0);

    కొరకు (size_t r = 0; r < n_అడ్డువరుసలు; r += 2) {
        uint32_t r_idx0 = ప్రారంభం_అడ్డువరుస + r + 0;
        uint32_t r_idx1 = ప్రారంభం_అడ్డువరుస + r + 1;
        uint32_t     lr = vtcm_ప్రారంభం_అడ్డువరుస + r;                 // vtcm-local row
        స్థిరరాశి size_t r0 = (lr / HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
        స్థిరరాశి size_t r1 = (lr % HTP_MM_HMX_TILE_N_అడ్డువరుసలు) / 2;  // index of the row pair within the tile
        స్థిరరాశి __fp16 *అడ్డువరుస_పునాది = vtcm_src + r0 * tile_అడ్డువరుస_stride;

        ఒకవేళ (r_idx0 >= cne1) ఆపు;

        నిర్మాణం mmid_అడ్డువరుస_mapping mapping0 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + r_idx0];
        దశాంశం *output_row0 = (దశాంశం *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        దశాంశం *output_row1 = ఖాళీ;
        ఒకవేళ (r_idx1 < cne1) {
            నిర్మాణం mmid_అడ్డువరుస_mapping mapping1 = మాతృక_అడ్డువరుసలు[cur_a * mapping_stride + r_idx1];
            output_row1 = (దశాంశం *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #సూచన unroll(4)
        కొరకు (size_t c = 0; c < (size_t)n_నిలువుల_సంఖ్య; c += HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) {
            స్థిరరాశి size_t c0 = c / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
            స్థిరరాశి __fp16 *tile = అడ్డువరుస_పునాది + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_సదిశ v = ((స్థిరరాశి HVX_సదిశ *) tile)[r1];
            HVX_సదిశ_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ఒకవేళ (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
