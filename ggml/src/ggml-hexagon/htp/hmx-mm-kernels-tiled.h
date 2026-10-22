#अन्तर्भाव "hmx-utils.h"
#अन्तर्भाव "hmx-queue.h"

// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
स्थिर ध्रुव __fp16 mxfp4_गन्तव्यम्_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

स्थिर ध्रुव __fp16 iq4_nl_गन्तव्यम्_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

प्रकारनिर्देश संरचना tiled_dequantize_state tiled_dequantize_state_t;
प्रकारनिर्देश शून्य (*tiled_dequant_task_fn_t)(ध्रुव tiled_dequantize_state_t * state, uint32_t start_tile, uint32_t अन्तः_tile);

संरचना tiled_dequantize_state {
    संरचना htp_context      * ctx;
    संरचना htp_thread_trace * traces;
    __fp16                  * dst;
    ध्रुव uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    संरचना fastdiv_मानानि     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_weight_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_प्रत्येकम्_task;
    uint32_t                  tile_आकार;
    uint32_t                  aligned_tile_आकार;
    uint32_t                  n_tasks;
    uint32_t                  n_स्तम्भसङ्ख्या;
    uint32_t                  k_खण्डम्;
    size_t                    row_stride;
    uint32_t                  weight_type;
};

स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q1_0(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् v_neg_one = Q6_Vb_vsplat_R(-1);
    ध्रुव HVX_सदिशम् v_pos_one = Q6_Vb_vsplat_R(1);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        ध्रुव HVX_सदिशम् v_sc = hvx_vmem(tile_src + 512);
        ध्रुव HVX_सदिशम् v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        प्रति (uint32_t q = 0; q < 4; q++) {
            ध्रुव HVX_सदिशम् v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #निर्देशः unroll
            प्रति (पूर्णाङ्क g = 0; g < 8; g++) {
                ध्रुव HVX_सदिशम्     v_q  = unpack_q1_0_गणः(v_plane, g, v_neg_one, v_pos_one);
                ध्रुव HVX_सदिशम्_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                ध्रुव HVX_सदिशम्_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q4_0(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    ध्रुव HVX_सदिशम् i8 = Q6_Vb_vsplat_R(8);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v_sc = hvx_vmem(tile_src + 512);
        HVX_सदिशम् v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_सदिशम् vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_सदिशम् vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_सदिशम् vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_सदिशम् vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_सदिशम् v_lo0 = Q6_V_vand_VV(vq0, आवरणम्_h4);
        HVX_सदिशम् v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_सदिशम् v_lo1 = Q6_V_vand_VV(vq1, आवरणम्_h4);
        HVX_सदिशम् v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_सदिशम् v_lo2 = Q6_V_vand_VV(vq2, आवरणम्_h4);
        HVX_सदिशम् v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_सदिशम् v_lo3 = Q6_V_vand_VV(vq3, आवरणम्_h4);
        HVX_सदिशम् v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

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
        HVX_सदिशम्_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_सदिशम्_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_सदिशम्_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_सदिशम्_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_सदिशम्_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_सदिशम्_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_सदिशम्_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_सदिशम्_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_सदिशम्_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_सदिशम्_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_सदिशम्_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_सदिशम्_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_सदिशम् v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_सदिशम् v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_सदिशम् v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_सदिशम् v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_सदिशम् v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_सदिशम् v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_सदिशम् v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_सदिशम् v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_सदिशम् v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_सदिशम् v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_सदिशम् v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_सदिशम् v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_सदिशम् v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_सदिशम् v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_सदिशम् v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_सदिशम् v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

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

स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q4_1(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् vscale_विस्थापनम् = hvx_vmem(tile_src + 512);
        HVX_सदिशम्_Pair dm_deal = Q6_W_vdeal_VVR(vscale_विस्थापनम्, vscale_विस्थापनम्, -2);
        HVX_सदिशम् vd = Q6_V_lo_W(dm_deal);
        HVX_सदिशम् vm = Q6_V_hi_W(dm_deal);

        HVX_सदिशम् v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_सदिशम् v_विस्थापनम्_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_सदिशम् vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_सदिशम् vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_सदिशम् vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_सदिशम् vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_सदिशम् v_lo0 = Q6_V_vand_VV(vq0, आवरणम्_h4);
        HVX_सदिशम् v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_सदिशम् v_lo1 = Q6_V_vand_VV(vq1, आवरणम्_h4);
        HVX_सदिशम् v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_सदिशम् v_lo2 = Q6_V_vand_VV(vq2, आवरणम्_h4);
        HVX_सदिशम् v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_सदिशम् v_lo3 = Q6_V_vand_VV(vq3, आवरणम्_h4);
        HVX_सदिशम् v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_सदिशम्_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_सदिशम्_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_सदिशम्_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_सदिशम्_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_सदिशम्_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_सदिशम्_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_सदिशम्_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_सदिशम्_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_सदिशम्_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_सदिशम्_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_सदिशम्_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_सदिशम्_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_सदिशम् v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_विस्थापनम्_duplicated));

        HVX_सदिशम् v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_विस्थापनम्_duplicated));

        HVX_सदिशम् v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_विस्थापनम्_duplicated));

        HVX_सदिशम् v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_विस्थापनम्_duplicated));

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

स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_iq4_nl(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    ध्रुव HVX_सदिशम् vlut_cvt = hvx_vmem(iq4_nl_गन्तव्यम्_fp16_lut);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v_sc = hvx_vmem(tile_src + 512);
        HVX_सदिशम् v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_सदिशम् vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_सदिशम् vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_सदिशम् vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_सदिशम् vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_सदिशम् v_lo0 = Q6_V_vand_VV(vq0, आवरणम्_h4);
        HVX_सदिशम् v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_सदिशम् v_lo1 = Q6_V_vand_VV(vq1, आवरणम्_h4);
        HVX_सदिशम् v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_सदिशम् v_lo2 = Q6_V_vand_VV(vq2, आवरणम्_h4);
        HVX_सदिशम् v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_सदिशम् v_lo3 = Q6_V_vand_VV(vq3, आवरणम्_h4);
        HVX_सदिशम् v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_सदिशम्_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_सदिशम्_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_सदिशम्_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_सदिशम्_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_सदिशम् v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_सदिशम् v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_सदिशम् v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_सदिशम् v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_सदिशम् v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_सदिशम् v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_सदिशम् v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_सदिशम् v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_सदिशम्_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_सदिशम् v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_सदिशम् v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_सदिशम् v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_सदिशम् v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_सदिशम् v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_सदिशम् v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_सदिशम् v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_सदिशम् v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_सदिशम् v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_सदिशम् v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_सदिशम् v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_सदिशम् v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_सदिशम् v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_सदिशम् v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_सदिशम् v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_सदिशम् v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_mxfp4(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    ध्रुव HVX_सदिशम् vlut_cvt = hvx_vmem(mxfp4_गन्तव्यम्_fp16_lut);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v = hvx_vmem(tile_src + 512);
        HVX_सदिशम् vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_सदिशम् v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_सदिशम् vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_सदिशम् vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_सदिशम् vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_सदिशम् vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_सदिशम् v_lo0 = Q6_V_vand_VV(vq0, आवरणम्_h4);
        HVX_सदिशम् v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_सदिशम् v_lo1 = Q6_V_vand_VV(vq1, आवरणम्_h4);
        HVX_सदिशम् v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_सदिशम् v_lo2 = Q6_V_vand_VV(vq2, आवरणम्_h4);
        HVX_सदिशम् v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_सदिशम् v_lo3 = Q6_V_vand_VV(vq3, आवरणम्_h4);
        HVX_सदिशम् v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_सदिशम्_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_सदिशम्_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_सदिशम्_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_सदिशम्_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_सदिशम् v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_सदिशम् v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_सदिशम् v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_सदिशम् v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_सदिशम् v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_सदिशम् v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_सदिशम् v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_सदिशम् v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_सदिशम्_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_सदिशम्_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_सदिशम् v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_सदिशम् v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_सदिशम् v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_सदिशम् v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_सदिशम् v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_सदिशम् v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_सदिशम् v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_सदिशम् v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_सदिशम् v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_सदिशम् v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_सदिशम् v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_सदिशम् v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_सदिशम् v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_सदिशम् v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_सदिशम् v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_सदिशम् v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q8_0(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v_sc = hvx_vmem(tile_src + 1024);
        HVX_सदिशम् v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_सदिशम् vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_सदिशम् vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_सदिशम् vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_सदिशम् vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_सदिशम्_Pair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_सदिशम्_Pair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_सदिशम्_Pair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_सदिशम्_Pair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_सदिशम् vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_सदिशम् vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_सदिशम् vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_सदिशम् vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_सदिशम्_Pair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_सदिशम्_Pair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_सदिशम्_Pair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_सदिशम्_Pair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_सदिशम् v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_सदिशम् v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_सदिशम् v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_सदिशम् v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_सदिशम् v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_सदिशम् v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_सदिशम् v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_सदिशम् v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

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
        HVX_सदिशम् v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_सदिशम् v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_सदिशम् v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_सदिशम् v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_सदिशम् v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_सदिशम् v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_सदिशम् v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_सदिशम् v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

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

स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q5_k(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile_src = state->src + t * state->aligned_tile_आकार;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् vscale_विस्थापनम् = hvx_vmem(tile_src + 512);
        HVX_सदिशम्_Pair dm_deal = Q6_W_vdeal_VVR(vscale_विस्थापनम्, vscale_विस्थापनम्, -2);
        HVX_सदिशम् vd = Q6_V_lo_W(dm_deal);
        HVX_सदिशम् vm = Q6_V_hi_W(dm_deal);

        HVX_सदिशम् v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_सदिशम् v_विस्थापनम्_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_सदिशम् vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_सदिशम् vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_सदिशम् vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_सदिशम् vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_सदिशम् v_lo0 = Q6_V_vand_VV(vq0, आवरणम्_h4);
        HVX_सदिशम् v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_सदिशम् v_lo1 = Q6_V_vand_VV(vq1, आवरणम्_h4);
        HVX_सदिशम् v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_सदिशम् v_lo2 = Q6_V_vand_VV(vq2, आवरणम्_h4);
        HVX_सदिशम् v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_सदिशम् v_lo3 = Q6_V_vand_VV(vq3, आवरणम्_h4);
        HVX_सदिशम् v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Q5_K: OR in the 5th bit from the plane
        HVX_सदिशम् v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

        // Shuffling
        HVX_सदिशम्_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_सदिशम्_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_सदिशम्_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_सदिशम्_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_सदिशम्_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_सदिशम्_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_सदिशम्_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_सदिशम्_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_सदिशम्_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_सदिशम्_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_सदिशम्_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_सदिशम्_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_सदिशम् v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_विस्थापनम्_duplicated));

        HVX_सदिशम् v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_विस्थापनम्_duplicated));

        HVX_सदिशम् v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_विस्थापनम्_duplicated));

        HVX_सदिशम् v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_विस्थापनम्_duplicated));
        HVX_सदिशम् v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_विस्थापनम्_duplicated));

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
स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q6_k(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_0f = Q6_Vb_vsplat_R(0x0F);
    ध्रुव HVX_सदिशम् आवरणम्_03 = Q6_Vb_vsplat_R(0x03);
    ध्रुव HVX_सदिशम् i32     = Q6_Vb_vsplat_R(32);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव HVX_सदिशम् * vptr = (ध्रुव HVX_सदिशम् *) (state->src + t * state->aligned_tile_आकार);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v_sc      = vptr[6];
        HVX_सदिशम् v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_सदिशम् v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_सदिशम् v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #निर्देशः unroll
        प्रति (पूर्णाङ्क g = 0; g < 8; g += 4) {
            ध्रुव HVX_सदिशम् v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_सदिशम् v_q0 = unpack_q6_k_गणः(vptr, g + 0, आवरणम्_0f, आवरणम्_03, i32);
            HVX_सदिशम् v_q1 = unpack_q6_k_गणः(vptr, g + 1, आवरणम्_0f, आवरणम्_03, i32);
            HVX_सदिशम् v_q2 = unpack_q6_k_गणः(vptr, g + 2, आवरणम्_0f, आवरणम्_03, i32);
            HVX_सदिशम् v_q3 = unpack_q6_k_गणः(vptr, g + 3, आवरणम्_0f, आवरणम्_03, i32);

            HVX_सदिशम्_Pair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_सदिशम्_Pair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_सदिशम्_Pair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_सदिशम्_Pair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_सदिशम्_Pair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_सदिशम्_Pair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_सदिशम्_Pair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_सदिशम्_Pair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_सदिशम् v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_सदिशम् v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_सदिशम् v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_सदिशम् v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_सदिशम् v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_सदिशम् v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_सदिशम् v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_सदिशम् v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

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
स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q3_k(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_03 = Q6_Vb_vsplat_R(0x03);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव HVX_सदिशम् * vptr = (ध्रुव HVX_सदिशम् *) (state->src + t * state->aligned_tile_आकार);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v_sc      = vptr[3];
        HVX_सदिशम् v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_सदिशम् v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_सदिशम् v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #निर्देशः unroll
        प्रति (पूर्णाङ्क g = 0; g < 8; g++) {
            ध्रुव HVX_सदिशम् v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_सदिशम्     v_q   = unpack_q3_k_गणः(vptr, g, आवरणम्_03);
            HVX_सदिशम्_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_सदिशम्_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q2_k(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_03 = Q6_Vb_vsplat_R(0x03);

    HVX_सदिशम् v_d = Q6_V_vzero();
    HVX_सदिशम् v_dm = Q6_V_vzero();

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव uint8_t * tile = state->src + t * state->aligned_tile_आकार;
        ध्रुव HVX_सदिशम् * vptr = (ध्रुव HVX_सदिशम् *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        यदि (t == start_tile || (t & 7) == 0) {
            q2_k_आरोपय_dmin(tile - (t & 7) * state->aligned_tile_आकार, &v_d, &v_dm);
        }

        HVX_सदिशम् v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_सदिशम् v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_सदिशम् v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_सदिशम् v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_सदिशम् v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_सदिशम् v_विस्थापनम्_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_सदिशम् v_विस्थापनम्_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #निर्देशः unroll
        प्रति (पूर्णाङ्क g = 0; g < 8; g++) {
            ध्रुव HVX_सदिशम् v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            ध्रुव HVX_सदिशम् v_विस्थापनम् = (g < 4) ? v_विस्थापनम्_k0 : v_विस्थापनम्_k16;

            HVX_सदिशम्     v_q   = unpack_q3_k_low2(vptr, g, आवरणम्_03);
            HVX_सदिशम्_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_सदिशम्_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_विस्थापनम्));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_विस्थापनम्));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
स्थिर शून्य dequantize_tiled_weight_गन्तव्यम्_fp16_task_q2_0(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव HVX_सदिशम् आवरणम्_03 = Q6_Vb_vsplat_R(0x03);
    ध्रुव HVX_सदिशम् i1      = Q6_Vb_vsplat_R(1);

    प्रति (uint32_t t = start_tile; t < अन्तः_tile; t++) {
        ध्रुव HVX_सदिशम् * vptr = (ध्रुव HVX_सदिशम् *) (state->src + t * state->aligned_tile_आकार);
        __fp16 * dst_आधारः = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_सदिशम् v_sc    = vptr[4];
        HVX_सदिशम् v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        प्रति (पूर्णाङ्क half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_आधारः + half * HTP_MM_HMX_TILE_N_ELMS;

            #निर्देशः unroll
            प्रति (पूर्णाङ्क g = 0; g < 8; g++) {
                HVX_सदिशम्     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, आवरणम्_03), i1);
                HVX_सदिशम्_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_सदिशम्_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

स्थिर __attribute__((noinline))
शून्य convert_f16_weight_गन्तव्यम्_fp16_tiles_task(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव uint32_t n_k_tiles = state->n_k_tiles;
    ध्रुव संरचना fastdiv_मानानि n_k_tiles_div = state->n_k_tiles_div;

    ध्रुव HVX_सदिशम् v_scat_आधारः  = hvx_vmem(hmx_transpose_scatter_offsets);
    ध्रुव HVX_सदिशम् v_scat_चरणः  = Q6_V_vsplat_R(4);
    ध्रुव HVX_सदिशम्_Pred q_mask64 = Q6_Q_vsetq_R(64);

    अचिह्नित ct = fastdiv((अचिह्नित)start_tile, &n_k_tiles_div);
    अचिह्नित kt = fastmodulo((अचिह्नित)start_tile, n_k_tiles, &n_k_tiles_div);

    प्रति (अचिह्नित t = start_tile; t < (अचिह्नित)अन्तः_tile; ) {
        यदि (kt >= (अचिह्नित)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_आधारः = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * परिमाण(__fp16);

            HVX_सदिशम् v_off = v_scat_आधारः;
            प्रति (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_पङ्क्तिकाः; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या + r;
                uint32_t row1 = row0 + 1;

                ध्रुव uint8_t *r0 = state->src + row0 * state->row_stride;
                ध्रुव uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_सदिशम् v0 = (row0 < state->n_स्तम्भसङ्ख्या) ? hvx_vmemu((ध्रुव __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_सदिशम् v1 = (row1 < state->n_स्तम्भसङ्ख्या) ? hvx_vmemu((ध्रुव __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_आधारः, HTP_MM_HMX_TILE_आकार - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_चरणः);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_आधारः, HTP_MM_HMX_TILE_आकार - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_चरणः);
            }
        }
        ++t; ++kt;
    }
}

स्थिर __attribute__((noinline))
शून्य quantize_f32_weight_गन्तव्यम्_fp16_tiles_task(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव uint32_t n_k_tiles = state->n_k_tiles;
    ध्रुव संरचना fastdiv_मानानि n_k_tiles_div = state->n_k_tiles_div;

    ध्रुव HVX_सदिशम् v_scat_आधारः  = hvx_vmem(hmx_transpose_scatter_offsets);
    ध्रुव HVX_सदिशम् v_scat_चरणः  = Q6_V_vsplat_R(4);
    ध्रुव HVX_सदिशम्_Pred q_mask64 = Q6_Q_vsetq_R(64);

    अचिह्नित ct = fastdiv((अचिह्नित)start_tile, &n_k_tiles_div);
    अचिह्नित kt = fastmodulo((अचिह्नित)start_tile, n_k_tiles, &n_k_tiles_div);

    प्रति (अचिह्नित t = start_tile; t < (अचिह्नित)अन्तः_tile; ) {
        यदि (kt >= (अचिह्नित)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_आधारः = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * परिमाण(भिन्न);

            HVX_सदिशम् v_off = v_scat_आधारः;
            प्रति (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_पङ्क्तिकाः; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या + r;
                uint32_t row1 = row0 + 1;

                ध्रुव uint8_t *r0 = state->src + row0 * state->row_stride;
                ध्रुव uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_सदिशम् v0_f32 = (row0 < state->n_स्तम्भसङ्ख्या) ? hvx_vmem((ध्रुव भिन्न *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_सदिशम् v1_f32 = (row1 < state->n_स्तम्भसङ्ख्या) ? hvx_vmem((ध्रुव भिन्न *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_आधारः, HTP_MM_HMX_TILE_आकार - 1, v_off, v_बहिर्गमनम्);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_चरणः);

                HVX_सदिशम् v_बहिर्गमनम्_hi = Q6_V_vror_VR(v_बहिर्गमनम्, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_आधारः, HTP_MM_HMX_TILE_आकार - 1, v_off, v_बहिर्गमनम्_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_चरणः);
            }
        }
        ++t; ++kt;
    }
}

स्थिर __attribute__((noinline))
शून्य convert_bf16_weight_गन्तव्यम्_fp16_tiles_task(
        ध्रुव tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t अन्तः_tile) {

    ध्रुव uint32_t n_k_tiles = state->n_k_tiles;
    ध्रुव संरचना fastdiv_मानानि n_k_tiles_div = state->n_k_tiles_div;

    ध्रुव HVX_सदिशम् v_scat_आधारः  = hvx_vmem(hmx_transpose_scatter_offsets);
    ध्रुव HVX_सदिशम् v_scat_चरणः  = Q6_V_vsplat_R(4);
    ध्रुव HVX_सदिशम्_Pred q_mask64 = Q6_Q_vsetq_R(64);

    अचिह्नित ct = fastdiv((अचिह्नित)start_tile, &n_k_tiles_div);
    अचिह्नित kt = fastmodulo((अचिह्नित)start_tile, n_k_tiles, &n_k_tiles_div);

    प्रति (अचिह्नित t = start_tile; t < (अचिह्नित)अन्तः_tile; ) {
        यदि (kt >= (अचिह्नित)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_आधारः = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * परिमाण(uint16_t);

            HVX_सदिशम् v_off = v_scat_आधारः;
            प्रति (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_पङ्क्तिकाः; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या + r;
                uint32_t row1 = row0 + 1;

                ध्रुव uint8_t *r0 = state->src + row0 * state->row_stride;
                ध्रुव uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_सदिशम् v0_raw = (row0 < state->n_स्तम्भसङ्ख्या) ? hvx_vmemu((ध्रुव शून्य *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_सदिशम् v1_raw = (row1 < state->n_स्तम्भसङ्ख्या) ? hvx_vmemu((ध्रुव शून्य *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_सदिशम् v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_सदिशम् v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_आधारः, HTP_MM_HMX_TILE_आकार - 1, v_off, v_बहिर्गमनम्);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_चरणः);

                HVX_सदिशम् v_बहिर्गमनम्_hi = Q6_V_vror_VR(v_बहिर्गमनम्, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_आधारः, HTP_MM_HMX_TILE_आकार - 1, v_off, v_बहिर्गमनम्_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_चरणः);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

स्थिर शून्य core_dot_chunk_fp16_short(__fp16 *सीमित output, ध्रुव __fp16 *सीमित activation,
                                ध्रुव __fp16 *सीमित weight, ध्रुव __fp16 *सीमित scales,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)scales));

    ध्रुव size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ध्रुव uint32_t range = 2048u * n_dot_tiles - 1;

    प्रति (uint32_t r = 0; r < n_row_tiles; ++r) {
        ध्रुव __fp16 *row_आधारः = activation + r * dot_stride;
        ध्रुव __fp16 *col_आधारः = weight;
        __fp16 *बहिर्गमनम्_tile = output + r * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;

        प्रति (size_t c = 0; c < n_col_tiles; ++c) {
            asm चञ्चल(HMX_CLRACC_F16());
            asm चञ्चल(HMX_आरोपय_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_आधारः), "r"(col_आधारः));
            asm चञ्चल(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(बहिर्गमनम्_tile), "r"(0) : "memory");
            col_आधारः += dot_stride;
            बहिर्गमनम्_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

स्थिर शून्य core_dot_chunk_fp16(__fp16 *सीमित output, ध्रुव __fp16 *सीमित activation,
                          ध्रुव __fp16 *सीमित weight, ध्रुव __fp16 *सीमित scales,
                          uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    यदि (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, weight, scales, n_row_tiles, n_col_tiles, n_dot_tiles);
        प्रत्यागम;
    }
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)scales));

    ध्रुव size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    प्रति (uint32_t r = 0; r < n_row_tiles; ++r) {
        ध्रुव __fp16 *row_आधारः = activation + r * dot_stride;
        ध्रुव __fp16 *col_आधारः = weight;
        __fp16 *बहिर्गमनम्_tile = output + r * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;

        प्रति (size_t c = 0; c < n_col_tiles; ++c) {
            ध्रुव __fp16 *row_tiles = row_आधारः;
            ध्रुव __fp16 *col_tiles = col_आधारः;

            asm चञ्चल(HMX_CLRACC_F16());

            ध्रुव uint32_t n_loops = n_dot_tiles / 32;
            ध्रुव uint32_t rem = n_dot_tiles % 32;

            प्रति (uint32_t l = 0; l < n_loops; ++l) {
                asm चञ्चल(HMX_आरोपय_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            यदि (rem > 0) {
                ध्रुव uint32_t range = 2048u * rem - 1;
                asm चञ्चल(HMX_आरोपय_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
            }

            asm चञ्चल(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(बहिर्गमनम्_tile), "r"(0) : "memory");

            col_आधारः += dot_stride;
            बहिर्गमनम्_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

स्थिर शून्य core_mma_chunk_fp16_short(__fp16 *सीमित c, ध्रुव __fp16 *सीमित a, ध्रुव __fp16 *सीमित b,
                                ध्रुव __fp16 *सीमित col_scales, ध्रुव __fp16 *सीमित eye_tile,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)col_scales));

    ध्रुव size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ध्रुव uint32_t range = 2048u * n_dot_tiles - 1;

    प्रति (size_t i = 0; i < n_row_tiles; ++i) {
        ध्रुव __fp16 *row_आधारः = a + i * dot_tile_stride;
        __fp16 *res_आधारः = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ध्रुव __fp16 *col_आधारः = b;
        __fp16 *accum_tile = res_आधारः;

        प्रति (size_t j = 0; j < n_col_tiles; ++j) {
            asm चञ्चल(HMX_CLRACC_F16());

            यदि (!zero_init) {
                asm चञ्चल(HMX_आरोपय_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm चञ्चल(HMX_आरोपय_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_आधारः), "r"(col_आधारः));

            asm चञ्चल(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_आधारः   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

स्थिर शून्य core_mma_chunk_fp16(__fp16 *सीमित c, ध्रुव __fp16 *सीमित a, ध्रुव __fp16 *सीमित b,
                          ध्रुव __fp16 *सीमित col_scales, ध्रुव __fp16 *सीमित eye_tile,
                          uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    यदि (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, col_scales, eye_tile, n_row_tiles, n_col_tiles, n_dot_tiles, zero_init);
        प्रत्यागम;
    }
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)col_scales));

    ध्रुव size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    प्रति (size_t i = 0; i < n_row_tiles; ++i) {
        ध्रुव __fp16 *row_आधारः = a + i * dot_tile_stride;
        __fp16 *res_आधारः = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ध्रुव __fp16 *col_आधारः = b;
        __fp16 *accum_tile = res_आधारः;

        प्रति (size_t j = 0; j < n_col_tiles; ++j) {
            ध्रुव __fp16 *col_tiles = col_आधारः;
            ध्रुव __fp16 *row_tiles = row_आधारः;

            asm चञ्चल(HMX_CLRACC_F16());

            यदि (!zero_init) {
                asm चञ्चल(HMX_आरोपय_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            ध्रुव uint32_t n_loops = n_dot_tiles / 32;
            ध्रुव uint32_t rem = n_dot_tiles % 32;

            प्रति (uint32_t l = 0; l < n_loops; ++l) {
                asm चञ्चल(HMX_आरोपय_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            यदि (rem > 0) {
                ध्रुव uint32_t range = 2048u * rem - 1;
                asm चञ्चल(HMX_आरोपय_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
            }

            asm चञ्चल(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_आधारः += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32p

स्थिर शून्य transfer_output_chunk_fp16_गन्तव्यम्_fp32_col_chunk(
    भिन्न *सीमित dst,
    ध्रुव भिन्न *सीमित src2,
    ध्रुव __fp16 *सीमित vtcm_src,
    uint32_t start_row,
    uint32_t n_पङ्क्तिकाः,
    uint32_t c_len,
    uint32_t योग_n_स्तम्भसङ्ख्या,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_स्तम्भसङ्ख्या
) {
    ध्रुव size_t tile_row_stride = hmx_ceil_div(योग_n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) * HTP_MM_HMX_TILE_N_ELMS;

    ध्रुव HVX_सदिशम् one = hvx_vec_splat_f16(1.0);

    ध्रुव size_t सीमा_c         = hex_smin(c_len, dst_स्तम्भसङ्ख्या);
    ध्रुव size_t सीमा_c_aligned = (सीमा_c & ~31);

    प्रति (size_t r = 0; r < n_पङ्क्तिकाः; r += 2) {
        ध्रुव size_t r_idx0 = start_row + r + 0;
        ध्रुव size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;
        ध्रुव size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_पङ्क्तिकाः) / 2;  // index of the row pair within the tile
        ध्रुव __fp16 *row_आधारः = vtcm_src + r0 * tile_row_stride;
        भिन्न *output_row_आधारः = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        ध्रुव भिन्न *src2_row_आधारः = src2 ? (src2 + r * src2_stride) : रिक्त;

        #निर्देशः unroll(4)
        प्रति (size_t c = 0; c < सीमा_c_aligned; c += HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) {
            ध्रुव size_t c0    = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            ध्रुव __fp16 *tile = row_आधारः + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_सदिशम् v = ((ध्रुव HVX_सदिशम् *) tile)[r1];
            HVX_सदिशम्_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_सदिशम् v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            यदि (src2_row_आधारः) {
                HVX_सदिशम् v_src2_0 = hvx_vmemu(src2_row_आधारः + c + 0);
                v_out0 = hvx_vec_सङ्कलय_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_row_आधारः + c + 0) = v_out0;

            यदि (r + 1 < n_पङ्क्तिकाः) {
                HVX_सदिशम् v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                यदि (src2_row_आधारः) {
                    HVX_सदिशम् v_src2_1 = hvx_vmemu(src2_row_आधारः + c + src2_stride);
                    v_out1 = hvx_vec_सङ्कलय_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_row_आधारः + c + dst_stride) = v_out1;
            }
        }

        यदि (सीमा_c_aligned < सीमा_c) {
            size_t c = सीमा_c_aligned;
            size_t valid_c = सीमा_c - c;
            ध्रुव size_t c0 = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            ध्रुव __fp16 *tile = row_आधारः + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_सदिशम् v = ((ध्रुव HVX_सदिशम् *) tile)[r1];
            HVX_सदिशम्_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_सदिशम् v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            यदि (src2_row_आधारः) {
                HVX_सदिशम् v_src2_0 = hvx_vmemu(src2_row_आधारः + c + 0);
                v_out0 = hvx_vec_सङ्कलय_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_row_आधारः + c, valid_c * परिमाण(भिन्न), v_out0);

            यदि (r + 1 < n_पङ्क्तिकाः) {
                HVX_सदिशम् v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                यदि (src2_row_आधारः) {
                    HVX_सदिशम् v_src2_1 = hvx_vmemu(src2_row_आधारः + c + src2_stride);
                    v_out1 = hvx_vec_सङ्कलय_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_row_आधारः + c + dst_stride, valid_c * परिमाण(भिन्न), v_out1);
            }
        }
    }
}

स्थिर अन्तःस्थ शून्य transfer_output_chunk_fp16_गन्तव्यम्_fp32(
    भिन्न *सीमित dst,
    ध्रुव भिन्न *सीमित src2,
    ध्रुव __fp16 *सीमित vtcm_src,
    uint32_t start_row,
    uint32_t n_पङ्क्तिकाः,
    uint32_t n_स्तम्भसङ्ख्या,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_स्तम्भसङ्ख्या
) {
    transfer_output_chunk_fp16_गन्तव्यम्_fp32_col_chunk(
        dst, src2, vtcm_src, start_row, n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या, n_स्तम्भसङ्ख्या, dst_stride, src2_stride, dst_स्तम्भसङ्ख्या
    );
}

प्रकारनिर्देश संरचना {
    ध्रुव __fp16  *vtcm_src;
    भिन्न         *dst;
    ध्रुव भिन्न   *src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_प्रत्येकम्_task;
    uint32_t       n_स्तम्भसङ्ख्या;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_स्तम्भसङ्ख्या;    // Actual output columns
    संरचना htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

स्थिर शून्य transfer_activation_row_pair_fp32_गन्तव्यम्_fp16(
        __fp16 *सीमित vtcm_dst,
        ध्रुव भिन्न *सीमित row0,
        ध्रुव भिन्न *सीमित row1,
        uint32_t r,
        uint32_t k_खण्डम्,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

    uint32_t c = 0;
    प्रति (; c + 32 <= k_valid; c += 32) {
        HVX_सदिशम् v0 = Q6_V_vzero();
        HVX_सदिशम् v1 = Q6_V_vzero();
        यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0 + c);
        यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1 + c);

        HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;  // tile column index
        uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

        HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_बहिर्गमनम्;
    }
    यदि (c < k_खण्डम्) {
        HVX_सदिशम् v0 = Q6_V_vzero();
        HVX_सदिशम् v1 = Q6_V_vzero();
        यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0 + c);
        यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem > 0 ? rem * परिमाण(भिन्न) : 0);
        v0 = Q6_V_vmux_QVV(आवरणम्, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(आवरणम्, v1, Q6_V_vzero());

        HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;  // tile column index
        uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

        HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_बहिर्गमनम्;
    }
}

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
स्थिर शून्य transfer_activation_row_pair_f16_गन्तव्यम्_f16(__fp16 * सीमित vtcm_dst,
                                                    ध्रुव __fp16 * सीमित row0,
                                                    ध्रुव __fp16 * सीमित row1,
                                                    uint32_t r,
                                                    uint32_t k_खण्डम्,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

    ध्रुव uint32_t n_tile_स्तम्भसङ्ख्या = k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
    __fp16 * सीमित tile_row = vtcm_dst + (size_t) r0 * n_tile_स्तम्भसङ्ख्या * HTP_MM_HMX_TILE_N_ELMS;

    ध्रुव HVX_UVector * pv0 = row0_valid ? (ध्रुव HVX_UVector *) row0 : रिक्त;
    ध्रुव HVX_UVector * pv1 = row1_valid ? (ध्रुव HVX_UVector *) row1 : रिक्त;

    uint32_t c = 0;
    प्रति (; c + 64 <= k_valid; c += 64) {
        HVX_सदिशम्     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_सदिशम्     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;

        HVX_सदिशम् * tile0 = (HVX_सदिशम् *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_सदिशम् * tile1 = (HVX_सदिशम् *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    प्रति (; c < k_खण्डम्; c += 64) {
        HVX_सदिशम् v0 = Q6_V_vzero();
        HVX_सदिशम् v1 = Q6_V_vzero();

        यदि (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem * परिमाण(__fp16));
            यदि (pv0) {
                v0 = Q6_V_vmux_QVV(आवरणम्, pv0[c / 64], Q6_V_vzero());
            }
            यदि (pv1) {
                v1 = Q6_V_vmux_QVV(आवरणम्, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;

        HVX_सदिशम् * tile0 = (HVX_सदिशम् *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        यदि (c0 + 1 < n_tile_स्तम्भसङ्ख्या) {
            HVX_सदिशम् * tile1 = (HVX_सदिशम् *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

स्थिर शून्य transfer_activation_row_pair_fp32_गन्तव्यम्_fp16_col_chunk(
        __fp16 *सीमित vtcm_dst,
        ध्रुव भिन्न *सीमित row0, // offset by c_first
        ध्रुव भिन्न *सीमित row1, // offset by c_first
        uint32_t r,
        uint32_t k_खण्डम्,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

    uint32_t c = 0;
    प्रति (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_सदिशम् v0 = Q6_V_vzero();
        HVX_सदिशम् v1 = Q6_V_vzero();
        यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0 + c);
        यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1 + c);

        HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

        uint32_t c0       = (c_first + c) / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;  // tile column index
        uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

        HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_बहिर्गमनम्;
    }
    यदि (c < c_len) {
        HVX_सदिशम् v0 = Q6_V_vzero();
        HVX_सदिशम् v1 = Q6_V_vzero();
        यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0 + c);
        यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem > 0 ? rem * परिमाण(भिन्न) : 0);
        v0 = Q6_V_vmux_QVV(आवरणम्, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(आवरणम्, v1, Q6_V_vzero());

        HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

        uint32_t c0       = (c_first + c) / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;  // tile column index
        uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

        HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_बहिर्गमनम्;
    }
}

स्थिर शून्य transfer_activation_chunk_fp32_गन्तव्यम्_fp16_gathered(
            __fp16 *सीमित vtcm_dst,
            ध्रुव भिन्न *सीमित src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_पङ्क्तिकाः,
            uint32_t k_खण्डम्,
            ध्रुव संरचना mmid_row_mapping *matrix_पङ्क्तिकाः,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            ध्रुव संरचना fastdiv_मानानि * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ध्रुव uint32_t n_पङ्क्तिकाः_padded = hex_align_ऊर्ध्वम्(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
    ध्रुव uint32_t n_पङ्क्तिकाः_tiled  = (n_पङ्क्तिकाः / HTP_MM_HMX_TILE_N_पङ्क्तिकाः) * HTP_MM_HMX_TILE_N_पङ्क्तिकाः;

    uint32_t r = 0;

    #निर्देशः unroll(2)
    प्रति (r = 0; r < n_पङ्क्तिकाः_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

        संरचना mmid_row_mapping mapping0 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + r_idx0];
        संरचना mmid_row_mapping mapping1 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        ध्रुव भिन्न *row0_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        ध्रुव भिन्न *row1_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        प्रति (; c + 32 <= k_valid; c += 32) {
            HVX_सदिशम् v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            HVX_सदिशम् v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);
            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
        यदि (c < k_खण्डम्) {
            HVX_सदिशम् v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            HVX_सदिशम् v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem > 0 ? rem * परिमाण(भिन्न) : 0);
            v0 = Q6_V_vmux_QVV(आवरणम्, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(आवरणम्, v1, Q6_V_vzero());

            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
    }

    प्रति (; r < n_पङ्क्तिकाः_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

        ध्रुव bool row0_valid = (start_row + r + 0) < cne1;
        ध्रुव bool row1_valid = (start_row + r + 1) < cne1;

        ध्रुव भिन्न *row0_ptr = रिक्त;
        ध्रुव भिन्न *row1_ptr = रिक्त;

        यदि (row0_valid) {
            संरचना mmid_row_mapping mapping0 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + (start_row + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        यदि (row1_valid) {
            संरचना mmid_row_mapping mapping1 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + (start_row + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        प्रति (; c + 32 <= k_valid; c += 32) {
            HVX_सदिशम् v0 = Q6_V_vzero();
            HVX_सदिशम् v1 = Q6_V_vzero();
            यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);

            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
        यदि (c < k_खण्डम्) {
            HVX_सदिशम् v0 = Q6_V_vzero();
            HVX_सदिशम् v1 = Q6_V_vzero();
            यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem > 0 ? rem * परिमाण(भिन्न) : 0);
            v0 = Q6_V_vmux_QVV(आवरणम्, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(आवरणम्, v1, Q6_V_vzero());

            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
    }
}

स्थिर शून्य transfer_activation_chunk_fp32_गन्तव्यम्_fp16_gathered_flat(
            __fp16 *सीमित vtcm_dst,
            ध्रुव भिन्न *सीमित src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_पङ्क्तिकाः,
            uint32_t k_खण्डम्,
            ध्रुव संरचना mmid_row_mapping *matrix_पङ्क्तिकाः,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ध्रुव uint32_t n_पङ्क्तिकाः_padded = hex_align_ऊर्ध्वम्(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
    ध्रुव uint32_t n_पङ्क्तिकाः_tiled  = (n_पङ्क्तिकाः / HTP_MM_HMX_TILE_N_पङ्क्तिकाः) * HTP_MM_HMX_TILE_N_पङ्क्तिकाः;

    uint32_t r = 0;

    #निर्देशः unroll(2)
    प्रति (r = 0; r < n_पङ्क्तिकाः_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

        संरचना mmid_row_mapping mapping0 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + r_idx0];
        संरचना mmid_row_mapping mapping1 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + r_idx1];

        ध्रुव भिन्न *row0_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + mapping0.i2 * nb12);
        ध्रुव भिन्न *row1_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        प्रति (; c + 32 <= k_valid; c += 32) {
            HVX_सदिशम् v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            HVX_सदिशम् v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);
            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
        यदि (c < k_खण्डम्) {
            HVX_सदिशम् v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            HVX_सदिशम् v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem > 0 ? rem * परिमाण(भिन्न) : 0);
            v0 = Q6_V_vmux_QVV(आवरणम्, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(आवरणम्, v1, Q6_V_vzero());

            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
    }

    प्रति (; r < n_पङ्क्तिकाः_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_पङ्क्तिकाः;  // intra-tile row idx

        ध्रुव bool row0_valid = (start_row + r + 0) < cne1;
        ध्रुव bool row1_valid = (start_row + r + 1) < cne1;

        ध्रुव भिन्न *row0_ptr = रिक्त;
        ध्रुव भिन्न *row1_ptr = रिक्त;

        यदि (row0_valid) {
            संरचना mmid_row_mapping mapping0 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + (start_row + r + 0)];
            row0_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + mapping0.i2 * nb12);
        }
        यदि (row1_valid) {
            संरचना mmid_row_mapping mapping1 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + (start_row + r + 1)];
            row1_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        प्रति (; c + 32 <= k_valid; c += 32) {
            HVX_सदिशम् v0 = Q6_V_vzero();
            HVX_सदिशम् v1 = Q6_V_vzero();
            यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);

            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
        यदि (c < k_खण्डम्) {
            HVX_सदिशम् v0 = Q6_V_vzero();
            HVX_सदिशम् v1 = Q6_V_vzero();
            यदि (row0_valid) v0 = *(ध्रुव HVX_सदिशम् *)(row0_ptr + c);
            यदि (row1_valid) v1 = *(ध्रुव HVX_सदिशम् *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(rem > 0 ? rem * परिमाण(भिन्न) : 0);
            v0 = Q6_V_vmux_QVV(आवरणम्, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(आवरणम्, v1, Q6_V_vzero());

            HVX_सदिशम् v_बहिर्गमनम् = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            uint32_t tile_idx = r0 * (k_खण्डम् / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) + c0;

            HVX_सदिशम् *tile = (HVX_सदिशम् *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_बहिर्गमनम्;
        }
    }
}

स्थिर शून्य transfer_output_chunk_fp16_गन्तव्यम्_fp32_scattered(
            भिन्न *सीमित dst,
            ध्रुव __fp16 *सीमित vtcm_src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_पङ्क्तिकाः,
            uint32_t n_स्तम्भसङ्ख्या,
            ध्रुव संरचना mmid_row_mapping *matrix_पङ्क्तिकाः,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    प्रतिज्ञा(n_स्तम्भसङ्ख्या % HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या == 0);
    ध्रुव size_t tile_row_stride = (n_स्तम्भसङ्ख्या / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) * HTP_MM_HMX_TILE_N_ELMS;

    ध्रुव HVX_सदिशम् one = hvx_vec_splat_f16(1.0);

    प्रति (size_t r = 0; r < n_पङ्क्तिकाः; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t     lr = vtcm_start_row + r;                 // vtcm-local row
        ध्रुव size_t r0 = (lr / HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
        ध्रुव size_t r1 = (lr % HTP_MM_HMX_TILE_N_पङ्क्तिकाः) / 2;  // index of the row pair within the tile
        ध्रुव __fp16 *row_आधारः = vtcm_src + r0 * tile_row_stride;

        यदि (r_idx0 >= cne1) भङ्ग;

        संरचना mmid_row_mapping mapping0 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + r_idx0];
        भिन्न *output_row0 = (भिन्न *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        भिन्न *output_row1 = रिक्त;
        यदि (r_idx1 < cne1) {
            संरचना mmid_row_mapping mapping1 = matrix_पङ्क्तिकाः[cur_a * mapping_stride + r_idx1];
            output_row1 = (भिन्न *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #निर्देशः unroll(4)
        प्रति (size_t c = 0; c < (size_t)n_स्तम्भसङ्ख्या; c += HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) {
            ध्रुव size_t c0 = c / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
            ध्रुव __fp16 *tile = row_आधारः + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_सदिशम् v = ((ध्रुव HVX_सदिशम् *) tile)[r1];
            HVX_सदिशम्_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            यदि (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
