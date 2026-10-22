#ಅನ್ತರ್ಭಾವ "hmx-utils.h"
#ಅನ್ತರ್ಭಾವ "hmx-queue.h"

// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ __fp16 mxfp4_ಗಮ್ಯ_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ __fp16 iq4_nl_ಗಮ್ಯ_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ tiled_dequantize_state tiled_dequantize_state_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*tiled_dequant_task_fn_t)(ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t * state, uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile);

ರಚನೆ tiled_dequantize_state {
    ರಚನೆ htp_context      * ctx;
    ರಚನೆ htp_thread_trace * traces;
    __fp16                  * dst;
    ಸ್ಥಿರಾಂಕ uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_ತೂಕ_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_ಪ್ರತಿ_task;
    uint32_t                  tile_ಗಾತ್ರ;
    uint32_t                  aligned_tile_ಗಾತ್ರ;
    uint32_t                  n_tasks;
    uint32_t                  n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    uint32_t                  k_ಖಂಡ;
    size_t                    ಅಡ್ಡಸಾಲು_stride;
    uint32_t                  ತೂಕ_type;
};

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q1_0(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg_one = Q6_Vb_vsplat_R(-1);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_pos_one = Q6_Vb_vsplat_R(1);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_sc = hvx_vmem(tile_src + 512);
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        ಗಾಗಿ (uint32_t q = 0; q < 4; q++) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 8; g++) {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     v_q  = unpack_q1_0_ಗುಂಪು(v_plane, g, v_neg_one, v_pos_one);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q4_0(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_h4 = Q6_Vb_vsplat_R(0x0F);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ i8 = Q6_Vb_vsplat_R(8);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v_sc = hvx_vmem(tile_src + 512);
        HVX_ಸದಿಶ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_ಸದಿಶ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ಸದಿಶ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ಸದಿಶ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ಸದಿಶ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ಸದಿಶ v_lo0 = Q6_V_vand_VV(vq0, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ಸದಿಶ v_lo1 = Q6_V_vand_VV(vq1, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ಸದಿಶ v_lo2 = Q6_V_vand_VV(vq2, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ಸದಿಶ v_lo3 = Q6_V_vand_VV(vq3, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

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
        HVX_ಸದಿಶ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ಸದಿಶ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ಸದಿಶ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ಸದಿಶ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ಸದಿಶ_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ಸದಿಶ_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ಸದಿಶ_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ಸದಿಶ_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ಸದಿಶ_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ಸದಿಶ_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ಸದಿಶ_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ಸದಿಶ_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_ಸದಿಶ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

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

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q4_1(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_h4 = Q6_Vb_vsplat_R(0x0F);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ vscale_ಸ್ಥಾನಾಂತರ = hvx_vmem(tile_src + 512);
        HVX_ಸದಿಶ_Pair dm_deal = Q6_W_vdeal_VVR(vscale_ಸ್ಥಾನಾಂತರ, vscale_ಸ್ಥಾನಾಂತರ, -2);
        HVX_ಸದಿಶ vd = Q6_V_lo_W(dm_deal);
        HVX_ಸದಿಶ vm = Q6_V_hi_W(dm_deal);

        HVX_ಸದಿಶ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_ಸದಿಶ v_ಸ್ಥಾನಾಂತರ_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_ಸದಿಶ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ಸದಿಶ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ಸದಿಶ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ಸದಿಶ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ಸದಿಶ v_lo0 = Q6_V_vand_VV(vq0, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ಸದಿಶ v_lo1 = Q6_V_vand_VV(vq1, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ಸದಿಶ v_lo2 = Q6_V_vand_VV(vq2, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ಸದಿಶ v_lo3 = Q6_V_vand_VV(vq3, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ಸದಿಶ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ಸದಿಶ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ಸದಿಶ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ಸದಿಶ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ಸದಿಶ_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ಸದಿಶ_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ಸದಿಶ_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ಸದಿಶ_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ಸದಿಶ_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ಸದಿಶ_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ಸದಿಶ_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ಸದಿಶ_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_ಸದಿಶ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

        HVX_ಸದಿಶ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

        HVX_ಸದಿಶ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

        HVX_ಸದಿಶ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

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

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_iq4_nl(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_h4 = Q6_Vb_vsplat_R(0x0F);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vlut_cvt = hvx_vmem(iq4_nl_ಗಮ್ಯ_fp16_lut);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v_sc = hvx_vmem(tile_src + 512);
        HVX_ಸದಿಶ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_ಸದಿಶ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ಸದಿಶ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ಸದಿಶ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ಸದಿಶ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ಸದಿಶ v_lo0 = Q6_V_vand_VV(vq0, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ಸದಿಶ v_lo1 = Q6_V_vand_VV(vq1, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ಸದಿಶ v_lo2 = Q6_V_vand_VV(vq2, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ಸದಿಶ v_lo3 = Q6_V_vand_VV(vq3, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ಸದಿಶ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ಸದಿಶ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ಸದಿಶ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ಸದಿಶ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_ಸದಿಶ v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ಸದಿಶ v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ಸದಿಶ v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ಸದಿಶ v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ಸದಿಶ v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ಸದಿಶ v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ಸದಿಶ v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ಸದಿಶ v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_ಸದಿಶ_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_ಸದಿಶ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_mxfp4(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_h4 = Q6_Vb_vsplat_R(0x0F);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vlut_cvt = hvx_vmem(mxfp4_ಗಮ್ಯ_fp16_lut);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v = hvx_vmem(tile_src + 512);
        HVX_ಸದಿಶ vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_ಸದಿಶ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_ಸದಿಶ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ಸದಿಶ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ಸದಿಶ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ಸದಿಶ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ಸದಿಶ v_lo0 = Q6_V_vand_VV(vq0, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ಸದಿಶ v_lo1 = Q6_V_vand_VV(vq1, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ಸದಿಶ v_lo2 = Q6_V_vand_VV(vq2, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ಸದಿಶ v_lo3 = Q6_V_vand_VV(vq3, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ಸದಿಶ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ಸದಿಶ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ಸದಿಶ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ಸದಿಶ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_ಸದಿಶ v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ಸದಿಶ v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ಸದಿಶ v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ಸದಿಶ v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ಸದಿಶ v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ಸದಿಶ v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ಸದಿಶ v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ಸದಿಶ v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_ಸದಿಶ_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_ಸದಿಶ_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_ಸದಿಶ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_ಸದಿಶ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q8_0(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v_sc = hvx_vmem(tile_src + 1024);
        HVX_ಸದಿಶ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_ಸದಿಶ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ಸದಿಶ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ಸದಿಶ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ಸದಿಶ vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_ಸದಿಶ_Pair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_ಸದಿಶ_Pair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_ಸದಿಶ_Pair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_ಸದಿಶ_Pair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_ಸದಿಶ vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_ಸದಿಶ vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_ಸದಿಶ vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_ಸದಿಶ vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_ಸದಿಶ_Pair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_ಸದಿಶ_Pair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_ಸದಿಶ_Pair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_ಸದಿಶ_Pair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_ಸದಿಶ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

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
        HVX_ಸದಿಶ v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_ಸದಿಶ v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

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

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q5_k(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_h4 = Q6_Vb_vsplat_R(0x0F);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile_src = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ vscale_ಸ್ಥಾನಾಂತರ = hvx_vmem(tile_src + 512);
        HVX_ಸದಿಶ_Pair dm_deal = Q6_W_vdeal_VVR(vscale_ಸ್ಥಾನಾಂತರ, vscale_ಸ್ಥಾನಾಂತರ, -2);
        HVX_ಸದಿಶ vd = Q6_V_lo_W(dm_deal);
        HVX_ಸದಿಶ vm = Q6_V_hi_W(dm_deal);

        HVX_ಸದಿಶ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_ಸದಿಶ v_ಸ್ಥಾನಾಂತರ_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_ಸದಿಶ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ಸದಿಶ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ಸದಿಶ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ಸದಿಶ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ಸದಿಶ v_lo0 = Q6_V_vand_VV(vq0, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ಸದಿಶ v_lo1 = Q6_V_vand_VV(vq1, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ಸದಿಶ v_lo2 = Q6_V_vand_VV(vq2, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ಸದಿಶ v_lo3 = Q6_V_vand_VV(vq3, ಮುಖವಾಡ_h4);
        HVX_ಸದಿಶ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Q5_K: OR in the 5th bit from the plane
        HVX_ಸದಿಶ v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

        // Shuffling
        HVX_ಸದಿಶ_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ಸದಿಶ_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ಸದಿಶ_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ಸದಿಶ_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ಸದಿಶ_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ಸದಿಶ_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ಸದಿಶ_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ಸದಿಶ_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ಸದಿಶ_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ಸದಿಶ_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ಸದಿಶ_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ಸದಿಶ_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_ಸದಿಶ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

        HVX_ಸದಿಶ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

        HVX_ಸದಿಶ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

        HVX_ಸದಿಶ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));
        HVX_ಸದಿಶ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_ಸ್ಥಾನಾಂತರ_duplicated));

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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q6_k(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_0f = Q6_Vb_vsplat_R(0x0F);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_03 = Q6_Vb_vsplat_R(0x03);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ i32     = Q6_Vb_vsplat_R(32);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vptr = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (state->src + t * state->aligned_tile_ಗಾತ್ರ);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v_sc      = vptr[6];
        HVX_ಸದಿಶ v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_ಸದಿಶ v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ಸದಿಶ v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 8; g += 4) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_ಸದಿಶ v_q0 = unpack_q6_k_ಗುಂಪು(vptr, g + 0, ಮುಖವಾಡ_0f, ಮುಖವಾಡ_03, i32);
            HVX_ಸದಿಶ v_q1 = unpack_q6_k_ಗುಂಪು(vptr, g + 1, ಮುಖವಾಡ_0f, ಮುಖವಾಡ_03, i32);
            HVX_ಸದಿಶ v_q2 = unpack_q6_k_ಗುಂಪು(vptr, g + 2, ಮುಖವಾಡ_0f, ಮುಖವಾಡ_03, i32);
            HVX_ಸದಿಶ v_q3 = unpack_q6_k_ಗುಂಪು(vptr, g + 3, ಮುಖವಾಡ_0f, ಮುಖವಾಡ_03, i32);

            HVX_ಸದಿಶ_Pair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_ಸದಿಶ_Pair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_ಸದಿಶ_Pair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_ಸದಿಶ_Pair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_ಸದಿಶ_Pair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_ಸದಿಶ_Pair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_ಸದಿಶ_Pair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_ಸದಿಶ_Pair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_ಸದಿಶ v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_ಸದಿಶ v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_ಸದಿಶ v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_ಸದಿಶ v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_ಸದಿಶ v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_ಸದಿಶ v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_ಸದಿಶ v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_ಸದಿಶ v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q3_k(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_03 = Q6_Vb_vsplat_R(0x03);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vptr = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (state->src + t * state->aligned_tile_ಗಾತ್ರ);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v_sc      = vptr[3];
        HVX_ಸದಿಶ v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_ಸದಿಶ v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ಸದಿಶ v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 8; g++) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_ಸದಿಶ     v_q   = unpack_q3_k_ಗುಂಪು(vptr, g, ಮುಖವಾಡ_03);
            HVX_ಸದಿಶ_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_ಸದಿಶ_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q2_k(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_03 = Q6_Vb_vsplat_R(0x03);

    HVX_ಸದಿಶ v_d = Q6_V_vzero();
    HVX_ಸದಿಶ v_dm = Q6_V_vzero();

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ uint8_t * tile = state->src + t * state->aligned_tile_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vptr = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        ಆದರೆ (t == ಆರಂಭ_tile || (t & 7) == 0) {
            q2_k_ತುಂಬು_dmin(tile - (t & 7) * state->aligned_tile_ಗಾತ್ರ, &v_d, &v_dm);
        }

        HVX_ಸದಿಶ v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_ಸದಿಶ v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_ಸದಿಶ v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_ಸದಿಶ v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ಸದಿಶ v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_ಸದಿಶ v_ಸ್ಥಾನಾಂತರ_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_ಸದಿಶ v_ಸ್ಥಾನಾಂತರ_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 8; g++) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_ಸ್ಥಾನಾಂತರ = (g < 4) ? v_ಸ್ಥಾನಾಂತರ_k0 : v_ಸ್ಥಾನಾಂತರ_k16;

            HVX_ಸದಿಶ     v_q   = unpack_q3_k_low2(vptr, g, ಮುಖವಾಡ_03);
            HVX_ಸದಿಶ_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_ಸದಿಶ_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_ಸ್ಥಾನಾಂತರ));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_ಸ್ಥಾನಾಂತರ));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q2_0(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_03 = Q6_Vb_vsplat_R(0x03);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ i1      = Q6_Vb_vsplat_R(1);

    ಗಾಗಿ (uint32_t t = ಆರಂಭ_tile; t < ಕೊನೆ_tile; t++) {
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vptr = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (state->src + t * state->aligned_tile_ಗಾತ್ರ);
        __fp16 * dst_ತಳ = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ಸದಿಶ v_sc    = vptr[4];
        HVX_ಸದಿಶ v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_ತಳ + half * HTP_MM_HMX_TILE_N_ELMS;

            #ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 8; g++) {
                HVX_ಸದಿಶ     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, ಮುಖವಾಡ_03), i1);
                HVX_ಸದಿಶ_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_ಸದಿಶ_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

ಸ್ಥಿರ __attribute__((noinline))
ಶೂನ್ಯ convert_f16_ತೂಕ_ಗಮ್ಯ_fp16_tiles_task(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles = state->n_k_tiles;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div = state->n_k_tiles_div;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scat_ತಳ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scat_ಹೆಜ್ಜೆ  = Q6_V_vsplat_R(4);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_mask64 = Q6_Q_vsetq_R(64);

    ಚಿಹ್ನೆರಹಿತ ct = fastdiv((ಚಿಹ್ನೆರಹಿತ)ಆರಂಭ_tile, &n_k_tiles_div);
    ಚಿಹ್ನೆರಹಿತ kt = fastmodulo((ಚಿಹ್ನೆರಹಿತ)ಆರಂಭ_tile, n_k_tiles, &n_k_tiles_div);

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ t = ಆರಂಭ_tile; t < (ಚಿಹ್ನೆರಹಿತ)ಕೊನೆ_tile; ) {
        ಆದರೆ (kt >= (ಚಿಹ್ನೆರಹಿತ)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_ತಳ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * ಗಾತ್ರ(__fp16);

            HVX_ಸದಿಶ v_off = v_scat_ತಳ;
            ಗಾಗಿ (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ + r;
                uint32_t row1 = row0 + 1;

                ಸ್ಥಿರಾಂಕ uint8_t *r0 = state->src + row0 * state->ಅಡ್ಡಸಾಲು_stride;
                ಸ್ಥಿರಾಂಕ uint8_t *r1 = state->src + row1 * state->ಅಡ್ಡಸಾಲು_stride;

                HVX_ಸದಿಶ v0 = (row0 < state->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? hvx_vmemu((ಸ್ಥಿರಾಂಕ __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ಸದಿಶ v1 = (row1 < state->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? hvx_vmemu((ಸ್ಥಿರಾಂಕ __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ತಳ, HTP_MM_HMX_TILE_ಗಾತ್ರ - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ಹೆಜ್ಜೆ);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ತಳ, HTP_MM_HMX_TILE_ಗಾತ್ರ - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ಹೆಜ್ಜೆ);
            }
        }
        ++t; ++kt;
    }
}

ಸ್ಥಿರ __attribute__((noinline))
ಶೂನ್ಯ quantize_f32_ತೂಕ_ಗಮ್ಯ_fp16_tiles_task(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles = state->n_k_tiles;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div = state->n_k_tiles_div;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scat_ತಳ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scat_ಹೆಜ್ಜೆ  = Q6_V_vsplat_R(4);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_mask64 = Q6_Q_vsetq_R(64);

    ಚಿಹ್ನೆರಹಿತ ct = fastdiv((ಚಿಹ್ನೆರಹಿತ)ಆರಂಭ_tile, &n_k_tiles_div);
    ಚಿಹ್ನೆರಹಿತ kt = fastmodulo((ಚಿಹ್ನೆರಹಿತ)ಆರಂಭ_tile, n_k_tiles, &n_k_tiles_div);

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ t = ಆರಂಭ_tile; t < (ಚಿಹ್ನೆರಹಿತ)ಕೊನೆ_tile; ) {
        ಆದರೆ (kt >= (ಚಿಹ್ನೆರಹಿತ)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_ತಳ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * ಗಾತ್ರ(ದಶಮಾಂಶ);

            HVX_ಸದಿಶ v_off = v_scat_ತಳ;
            ಗಾಗಿ (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ + r;
                uint32_t row1 = row0 + 1;

                ಸ್ಥಿರಾಂಕ uint8_t *r0 = state->src + row0 * state->ಅಡ್ಡಸಾಲು_stride;
                ಸ್ಥಿರಾಂಕ uint8_t *r1 = state->src + row1 * state->ಅಡ್ಡಸಾಲು_stride;

                HVX_ಸದಿಶ v0_f32 = (row0 < state->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? hvx_vmem((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ಸದಿಶ v1_f32 = (row1 < state->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? hvx_vmem((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ತಳ, HTP_MM_HMX_TILE_ಗಾತ್ರ - 1, v_off, v_ಹೊರಹೋಗು);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ಹೆಜ್ಜೆ);

                HVX_ಸದಿಶ v_ಹೊರಹೋಗು_hi = Q6_V_vror_VR(v_ಹೊರಹೋಗು, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ತಳ, HTP_MM_HMX_TILE_ಗಾತ್ರ - 1, v_off, v_ಹೊರಹೋಗು_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ಹೆಜ್ಜೆ);
            }
        }
        ++t; ++kt;
    }
}

ಸ್ಥಿರ __attribute__((noinline))
ಶೂನ್ಯ convert_bf16_ತೂಕ_ಗಮ್ಯ_fp16_tiles_task(
        ಸ್ಥಿರಾಂಕ tiled_dequantize_state_t *state,
        uint32_t ಆರಂಭ_tile, uint32_t ಕೊನೆ_tile) {

    ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles = state->n_k_tiles;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div = state->n_k_tiles_div;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scat_ತಳ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scat_ಹೆಜ್ಜೆ  = Q6_V_vsplat_R(4);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_mask64 = Q6_Q_vsetq_R(64);

    ಚಿಹ್ನೆರಹಿತ ct = fastdiv((ಚಿಹ್ನೆರಹಿತ)ಆರಂಭ_tile, &n_k_tiles_div);
    ಚಿಹ್ನೆರಹಿತ kt = fastmodulo((ಚಿಹ್ನೆರಹಿತ)ಆರಂಭ_tile, n_k_tiles, &n_k_tiles_div);

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ t = ಆರಂಭ_tile; t < (ಚಿಹ್ನೆರಹಿತ)ಕೊನೆ_tile; ) {
        ಆದರೆ (kt >= (ಚಿಹ್ನೆರಹಿತ)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_ತಳ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * ಗಾತ್ರ(uint16_t);

            HVX_ಸದಿಶ v_off = v_scat_ತಳ;
            ಗಾಗಿ (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ + r;
                uint32_t row1 = row0 + 1;

                ಸ್ಥಿರಾಂಕ uint8_t *r0 = state->src + row0 * state->ಅಡ್ಡಸಾಲು_stride;
                ಸ್ಥಿರಾಂಕ uint8_t *r1 = state->src + row1 * state->ಅಡ್ಡಸಾಲು_stride;

                HVX_ಸದಿಶ v0_raw = (row0 < state->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? hvx_vmemu((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ಸದಿಶ v1_raw = (row1 < state->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? hvx_vmemu((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_ಸದಿಶ v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_ಸದಿಶ v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ತಳ, HTP_MM_HMX_TILE_ಗಾತ್ರ - 1, v_off, v_ಹೊರಹೋಗು);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ಹೆಜ್ಜೆ);

                HVX_ಸದಿಶ v_ಹೊರಹೋಗು_hi = Q6_V_vror_VR(v_ಹೊರಹೋಗು, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ತಳ, HTP_MM_HMX_TILE_ಗಾತ್ರ - 1, v_off, v_ಹೊರಹೋಗು_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ಹೆಜ್ಜೆ);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

ಸ್ಥಿರ ಶೂನ್ಯ core_dot_chunk_fp16_short(__fp16 *ಸೀಮಿತ output, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ activation,
                                ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ ತೂಕ, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ scales,
                                uint32_t n_ಅಡ್ಡಸಾಲು_tiles, uint32_t n_ಲಂಬಸಾಲು_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(n_ಲಂಬಸಾಲು_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)scales));

    ಸ್ಥಿರಾಂಕ size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ಸ್ಥಿರಾಂಕ uint32_t range = 2048u * n_dot_tiles - 1;

    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲು_tiles; ++r) {
        ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_ತಳ = activation + r * dot_stride;
        ಸ್ಥಿರಾಂಕ __fp16 *ಲಂಬಸಾಲು_ತಳ = ತೂಕ;
        __fp16 *ಹೊರಹೋಗು_tile = output + r * n_ಲಂಬಸಾಲು_tiles * HTP_MM_HMX_TILE_N_ELMS;

        ಗಾಗಿ (size_t c = 0; c < n_ಲಂಬಸಾಲು_tiles; ++c) {
            asm ಅಸ್ಥಿರ(HMX_CLRACC_F16());
            asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ಅಡ್ಡಸಾಲು_ತಳ), "r"(ಲಂಬಸಾಲು_ತಳ));
            asm ಅಸ್ಥಿರ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(ಹೊರಹೋಗು_tile), "r"(0) : "memory");
            ಲಂಬಸಾಲು_ತಳ += dot_stride;
            ಹೊರಹೋಗು_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ core_dot_chunk_fp16(__fp16 *ಸೀಮಿತ output, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ activation,
                          ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ ತೂಕ, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ scales,
                          uint32_t n_ಅಡ್ಡಸಾಲು_tiles, uint32_t n_ಲಂಬಸಾಲು_tiles, uint32_t n_dot_tiles) {
    ಆದರೆ (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, ತೂಕ, scales, n_ಅಡ್ಡಸಾಲು_tiles, n_ಲಂಬಸಾಲು_tiles, n_dot_tiles);
        ಹಿಂತಿರುಗಿ;
    }
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(n_ಲಂಬಸಾಲು_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)scales));

    ಸ್ಥಿರಾಂಕ size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲು_tiles; ++r) {
        ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_ತಳ = activation + r * dot_stride;
        ಸ್ಥಿರಾಂಕ __fp16 *ಲಂಬಸಾಲು_ತಳ = ತೂಕ;
        __fp16 *ಹೊರಹೋಗು_tile = output + r * n_ಲಂಬಸಾಲು_tiles * HTP_MM_HMX_TILE_N_ELMS;

        ಗಾಗಿ (size_t c = 0; c < n_ಲಂಬಸಾಲು_tiles; ++c) {
            ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_tiles = ಅಡ್ಡಸಾಲು_ತಳ;
            ಸ್ಥಿರಾಂಕ __fp16 *ಲಂಬಸಾಲು_tiles = ಲಂಬಸಾಲು_ತಳ;

            asm ಅಸ್ಥಿರ(HMX_CLRACC_F16());

            ಸ್ಥಿರಾಂಕ uint32_t n_loops = n_dot_tiles / 32;
            ಸ್ಥಿರಾಂಕ uint32_t rem = n_dot_tiles % 32;

            ಗಾಗಿ (uint32_t l = 0; l < n_loops; ++l) {
                asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(ಅಡ್ಡಸಾಲು_tiles), "r"(ಲಂಬಸಾಲು_tiles));
                ಅಡ್ಡಸಾಲು_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                ಲಂಬಸಾಲು_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            ಆದರೆ (rem > 0) {
                ಸ್ಥಿರಾಂಕ uint32_t range = 2048u * rem - 1;
                asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ಅಡ್ಡಸಾಲು_tiles), "r"(ಲಂಬಸಾಲು_tiles));
            }

            asm ಅಸ್ಥಿರ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(ಹೊರಹೋಗು_tile), "r"(0) : "memory");

            ಲಂಬಸಾಲು_ತಳ += dot_stride;
            ಹೊರಹೋಗು_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ core_mma_chunk_fp16_short(__fp16 *ಸೀಮಿತ c, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ a, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ b,
                                ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ ಲಂಬಸಾಲು_scales, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ eye_tile,
                                uint32_t n_ಅಡ್ಡಸಾಲು_tiles, uint32_t n_ಲಂಬಸಾಲು_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(n_ಲಂಬಸಾಲು_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಲಂಬಸಾಲು_scales));

    ಸ್ಥಿರಾಂಕ size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ಸ್ಥಿರಾಂಕ uint32_t range = 2048u * n_dot_tiles - 1;

    ಗಾಗಿ (size_t i = 0; i < n_ಅಡ್ಡಸಾಲು_tiles; ++i) {
        ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_ತಳ = a + i * dot_tile_stride;
        __fp16 *res_ತಳ = c + i * n_ಲಂಬಸಾಲು_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ __fp16 *ಲಂಬಸಾಲು_ತಳ = b;
        __fp16 *accum_tile = res_ತಳ;

        ಗಾಗಿ (size_t j = 0; j < n_ಲಂಬಸಾಲು_tiles; ++j) {
            asm ಅಸ್ಥಿರ(HMX_CLRACC_F16());

            ಆದರೆ (!zero_init) {
                asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ಅಡ್ಡಸಾಲು_ತಳ), "r"(ಲಂಬಸಾಲು_ತಳ));

            asm ಅಸ್ಥಿರ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            ಲಂಬಸಾಲು_ತಳ   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ core_mma_chunk_fp16(__fp16 *ಸೀಮಿತ c, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ a, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ b,
                          ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ ಲಂಬಸಾಲು_scales, ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ eye_tile,
                          uint32_t n_ಅಡ್ಡಸಾಲು_tiles, uint32_t n_ಲಂಬಸಾಲು_tiles, uint32_t n_dot_tiles, bool zero_init) {
    ಆದರೆ (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, ಲಂಬಸಾಲು_scales, eye_tile, n_ಅಡ್ಡಸಾಲು_tiles, n_ಲಂಬಸಾಲು_tiles, n_dot_tiles, zero_init);
        ಹಿಂತಿರುಗಿ;
    }
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(n_ಲಂಬಸಾಲು_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಲಂಬಸಾಲು_scales));

    ಸ್ಥಿರಾಂಕ size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    ಗಾಗಿ (size_t i = 0; i < n_ಅಡ್ಡಸಾಲು_tiles; ++i) {
        ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_ತಳ = a + i * dot_tile_stride;
        __fp16 *res_ತಳ = c + i * n_ಲಂಬಸಾಲು_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ __fp16 *ಲಂಬಸಾಲು_ತಳ = b;
        __fp16 *accum_tile = res_ತಳ;

        ಗಾಗಿ (size_t j = 0; j < n_ಲಂಬಸಾಲು_tiles; ++j) {
            ಸ್ಥಿರಾಂಕ __fp16 *ಲಂಬಸಾಲು_tiles = ಲಂಬಸಾಲು_ತಳ;
            ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_tiles = ಅಡ್ಡಸಾಲು_ತಳ;

            asm ಅಸ್ಥಿರ(HMX_CLRACC_F16());

            ಆದರೆ (!zero_init) {
                asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            ಸ್ಥಿರಾಂಕ uint32_t n_loops = n_dot_tiles / 32;
            ಸ್ಥಿರಾಂಕ uint32_t rem = n_dot_tiles % 32;

            ಗಾಗಿ (uint32_t l = 0; l < n_loops; ++l) {
                asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(ಅಡ್ಡಸಾಲು_tiles), "r"(ಲಂಬಸಾಲು_tiles));
                ಅಡ್ಡಸಾಲು_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                ಲಂಬಸಾಲು_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            ಆದರೆ (rem > 0) {
                ಸ್ಥಿರಾಂಕ uint32_t range = 2048u * rem - 1;
                asm ಅಸ್ಥಿರ(HMX_ತುಂಬು_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ಅಡ್ಡಸಾಲು_tiles), "r"(ಲಂಬಸಾಲು_tiles));
            }

            asm ಅಸ್ಥಿರ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            ಲಂಬಸಾಲು_ತಳ += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32p

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_fp16_ಗಮ್ಯ_fp32_ಲಂಬಸಾಲು_chunk(
    ದಶಮಾಂಶ *ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ src2,
    ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ vtcm_src,
    uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t c_len,
    uint32_t ಒಟ್ಟು_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ
) {
    ಸ್ಥಿರಾಂಕ size_t tile_ಅಡ್ಡಸಾಲು_stride = hmx_ceil_div(ಒಟ್ಟು_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) * HTP_MM_HMX_TILE_N_ELMS;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);

    ಸ್ಥಿರಾಂಕ size_t ಎಲ್ಲೆ_c         = hex_smin(c_len, dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
    ಸ್ಥಿರಾಂಕ size_t ಎಲ್ಲೆ_c_aligned = (ಎಲ್ಲೆ_c & ~31);

    ಗಾಗಿ (size_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r += 2) {
        ಸ್ಥಿರಾಂಕ size_t r_idx0 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0;
        ಸ್ಥಿರಾಂಕ size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿರಾಂಕ size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು) / 2;  // index of the row pair within the tile
        ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_ತಳ = vtcm_src + r0 * tile_ಅಡ್ಡಸಾಲು_stride;
        ದಶಮಾಂಶ *output_ಅಡ್ಡಸಾಲು_ತಳ = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src2_ಅಡ್ಡಸಾಲು_ತಳ = src2 ? (src2 + r * src2_stride) : ಖಾಲಿ;

        #ಸೂಚನೆ unroll(4)
        ಗಾಗಿ (size_t c = 0; c < ಎಲ್ಲೆ_c_aligned; c += HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
            ಸ್ಥಿರಾಂಕ size_t c0    = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ __fp16 *tile = ಅಡ್ಡಸಾಲು_ತಳ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ಸದಿಶ v = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) tile)[r1];
            HVX_ಸದಿಶ_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_ಸದಿಶ v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ಆದರೆ (src2_ಅಡ್ಡಸಾಲು_ತಳ) {
                HVX_ಸದಿಶ v_src2_0 = hvx_vmemu(src2_ಅಡ್ಡಸಾಲು_ತಳ + c + 0);
                v_out0 = hvx_vec_ಕೂಡಿಸು_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_ಅಡ್ಡಸಾಲು_ತಳ + c + 0) = v_out0;

            ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು) {
                HVX_ಸದಿಶ v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                ಆದರೆ (src2_ಅಡ್ಡಸಾಲು_ತಳ) {
                    HVX_ಸದಿಶ v_src2_1 = hvx_vmemu(src2_ಅಡ್ಡಸಾಲು_ತಳ + c + src2_stride);
                    v_out1 = hvx_vec_ಕೂಡಿಸು_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_ಅಡ್ಡಸಾಲು_ತಳ + c + dst_stride) = v_out1;
            }
        }

        ಆದರೆ (ಎಲ್ಲೆ_c_aligned < ಎಲ್ಲೆ_c) {
            size_t c = ಎಲ್ಲೆ_c_aligned;
            size_t valid_c = ಎಲ್ಲೆ_c - c;
            ಸ್ಥಿರಾಂಕ size_t c0 = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ __fp16 *tile = ಅಡ್ಡಸಾಲು_ತಳ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ಸದಿಶ v = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) tile)[r1];
            HVX_ಸದಿಶ_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_ಸದಿಶ v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ಆದರೆ (src2_ಅಡ್ಡಸಾಲು_ತಳ) {
                HVX_ಸದಿಶ v_src2_0 = hvx_vmemu(src2_ಅಡ್ಡಸಾಲು_ತಳ + c + 0);
                v_out0 = hvx_vec_ಕೂಡಿಸು_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_ಅಡ್ಡಸಾಲು_ತಳ + c, valid_c * ಗಾತ್ರ(ದಶಮಾಂಶ), v_out0);

            ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು) {
                HVX_ಸದಿಶ v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                ಆದರೆ (src2_ಅಡ್ಡಸಾಲು_ತಳ) {
                    HVX_ಸದಿಶ v_src2_1 = hvx_vmemu(src2_ಅಡ್ಡಸಾಲು_ತಳ + c + src2_stride);
                    v_out1 = hvx_vec_ಕೂಡಿಸು_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_ಅಡ್ಡಸಾಲು_ತಳ + c + dst_stride, valid_c * ಗಾತ್ರ(ದಶಮಾಂಶ), v_out1);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ transfer_output_chunk_fp16_ಗಮ್ಯ_fp32(
    ದಶಮಾಂಶ *ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ src2,
    ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ vtcm_src,
    uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ
) {
    transfer_output_chunk_fp16_ಗಮ್ಯ_fp32_ಲಂಬಸಾಲು_chunk(
        dst, src2, vtcm_src, ಆರಂಭ_ಅಡ್ಡಸಾಲು, n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, dst_stride, src2_stride, dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ
    );
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಸ್ಥಿರಾಂಕ __fp16  *vtcm_src;
    ದಶಮಾಂಶ         *dst;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   *src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_ಪ್ರತಿ_task;
    uint32_t       n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;    // Actual output columns
    ರಚನೆ htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_ಅಡ್ಡಸಾಲು_pair_fp32_ಗಮ್ಯ_fp16(
        __fp16 *ಸೀಮಿತ vtcm_dst,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ row0,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ row1,
        uint32_t r,
        uint32_t k_ಖಂಡ,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

    uint32_t c = 0;
    ಗಾಗಿ (; c + 32 <= k_valid; c += 32) {
        HVX_ಸದಿಶ v0 = Q6_V_vzero();
        HVX_ಸದಿಶ v1 = Q6_V_vzero();
        ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0 + c);
        ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1 + c);

        HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;  // tile column index
        uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

        HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ಹೊರಹೋಗು;
    }
    ಆದರೆ (c < k_ಖಂಡ) {
        HVX_ಸದಿಶ v0 = Q6_V_vzero();
        HVX_ಸದಿಶ v1 = Q6_V_vzero();
        ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0 + c);
        ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem > 0 ? rem * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0);
        v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v1, Q6_V_vzero());

        HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;  // tile column index
        uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

        HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ಹೊರಹೋಗು;
    }
}

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_ಅಡ್ಡಸಾಲು_pair_f16_ಗಮ್ಯ_f16(__fp16 * ಸೀಮಿತ vtcm_dst,
                                                    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ row0,
                                                    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ row1,
                                                    uint32_t r,
                                                    uint32_t k_ಖಂಡ,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

    ಸ್ಥಿರಾಂಕ uint32_t n_tile_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    __fp16 * ಸೀಮಿತ tile_ಅಡ್ಡಸಾಲು = vtcm_dst + (size_t) r0 * n_tile_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * HTP_MM_HMX_TILE_N_ELMS;

    ಸ್ಥಿರಾಂಕ HVX_UVector * pv0 = row0_valid ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) row0 : ಖಾಲಿ;
    ಸ್ಥಿರಾಂಕ HVX_UVector * pv1 = row1_valid ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) row1 : ಖಾಲಿ;

    uint32_t c = 0;
    ಗಾಗಿ (; c + 64 <= k_valid; c += 64) {
        HVX_ಸದಿಶ     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_ಸದಿಶ     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

        HVX_ಸದಿಶ * tile0 = (HVX_ಸದಿಶ *) (tile_ಅಡ್ಡಸಾಲು + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_ಸದಿಶ * tile1 = (HVX_ಸದಿಶ *) (tile_ಅಡ್ಡಸಾಲು + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    ಗಾಗಿ (; c < k_ಖಂಡ; c += 64) {
        HVX_ಸದಿಶ v0 = Q6_V_vzero();
        HVX_ಸದಿಶ v1 = Q6_V_vzero();

        ಆದರೆ (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem * ಗಾತ್ರ(__fp16));
            ಆದರೆ (pv0) {
                v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, pv0[c / 64], Q6_V_vzero());
            }
            ಆದರೆ (pv1) {
                v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

        HVX_ಸದಿಶ * tile0 = (HVX_ಸದಿಶ *) (tile_ಅಡ್ಡಸಾಲು + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        ಆದರೆ (c0 + 1 < n_tile_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
            HVX_ಸದಿಶ * tile1 = (HVX_ಸದಿಶ *) (tile_ಅಡ್ಡಸಾಲು + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_ಅಡ್ಡಸಾಲು_pair_fp32_ಗಮ್ಯ_fp16_ಲಂಬಸಾಲು_chunk(
        __fp16 *ಸೀಮಿತ vtcm_dst,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ row0, // offset by c_first
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ row1, // offset by c_first
        uint32_t r,
        uint32_t k_ಖಂಡ,
        uint32_t c_ಮೊದಲ,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

    uint32_t c = 0;
    ಗಾಗಿ (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_ಸದಿಶ v0 = Q6_V_vzero();
        HVX_ಸದಿಶ v1 = Q6_V_vzero();
        ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0 + c);
        ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1 + c);

        HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

        uint32_t c0       = (c_ಮೊದಲ + c) / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;  // tile column index
        uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

        HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ಹೊರಹೋಗು;
    }
    ಆದರೆ (c < c_len) {
        HVX_ಸದಿಶ v0 = Q6_V_vzero();
        HVX_ಸದಿಶ v1 = Q6_V_vzero();
        ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0 + c);
        ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem > 0 ? rem * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0);
        v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v1, Q6_V_vzero());

        HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

        uint32_t c0       = (c_ಮೊದಲ + c) / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;  // tile column index
        uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

        HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ಹೊರಹೋಗು;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_fp32_ಗಮ್ಯ_fp16_gathered(
            __fp16 *ಸೀಮಿತ vtcm_dst,
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ src,
            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            uint32_t vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
            uint32_t k_ಖಂಡ,
            ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_padded = hex_align_ಮೇಲೆ(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_tiled  = (n_ಅಡ್ಡಸಾಲುಗಳು / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು) * HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;

    uint32_t r = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು_tiled; r += 2) {
        uint32_t r_idx0 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0;
        uint32_t r_idx1 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1;
        uint32_t lr = vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

        ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping0 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + r_idx0];
        ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row0_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row1_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        ಗಾಗಿ (; c + 32 <= k_valid; c += 32) {
            HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);
            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
        ಆದರೆ (c < k_ಖಂಡ) {
            HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem > 0 ? rem * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0);
            v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v1, Q6_V_vzero());

            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
    }

    ಗಾಗಿ (; r < n_ಅಡ್ಡಸಾಲುಗಳು_padded; r += 2) {
        uint32_t lr = vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

        ಸ್ಥಿರಾಂಕ bool row0_valid = (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0) < cne1;
        ಸ್ಥಿರಾಂಕ bool row1_valid = (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1) < cne1;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row0_ptr = ಖಾಲಿ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row1_ptr = ಖಾಲಿ;

        ಆದರೆ (row0_valid) {
            ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping0 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        ಆದರೆ (row1_valid) {
            ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        ಗಾಗಿ (; c + 32 <= k_valid; c += 32) {
            HVX_ಸದಿಶ v0 = Q6_V_vzero();
            HVX_ಸದಿಶ v1 = Q6_V_vzero();
            ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);

            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
        ಆದರೆ (c < k_ಖಂಡ) {
            HVX_ಸದಿಶ v0 = Q6_V_vzero();
            HVX_ಸದಿಶ v1 = Q6_V_vzero();
            ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem > 0 ? rem * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0);
            v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v1, Q6_V_vzero());

            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_fp32_ಗಮ್ಯ_fp16_gathered_flat(
            __fp16 *ಸೀಮಿತ vtcm_dst,
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *ಸೀಮಿತ src,
            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            uint32_t vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
            uint32_t k_ಖಂಡ,
            ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_padded = hex_align_ಮೇಲೆ(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_tiled  = (n_ಅಡ್ಡಸಾಲುಗಳು / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು) * HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;

    uint32_t r = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು_tiled; r += 2) {
        uint32_t r_idx0 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0;
        uint32_t r_idx1 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1;
        uint32_t lr = vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

        ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping0 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + r_idx0];
        ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + r_idx1];

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row0_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + mapping0.i2 * nb12);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row1_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        ಗಾಗಿ (; c + 32 <= k_valid; c += 32) {
            HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);
            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
        ಆದರೆ (c < k_ಖಂಡ) {
            HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem > 0 ? rem * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0);
            v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v1, Q6_V_vzero());

            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
    }

    ಗಾಗಿ (; r < n_ಅಡ್ಡಸಾಲುಗಳು_padded; r += 2) {
        uint32_t lr = vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // intra-tile row idx

        ಸ್ಥಿರಾಂಕ bool row0_valid = (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0) < cne1;
        ಸ್ಥಿರಾಂಕ bool row1_valid = (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1) < cne1;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row0_ptr = ಖಾಲಿ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *row1_ptr = ಖಾಲಿ;

        ಆದರೆ (row0_valid) {
            ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping0 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0)];
            row0_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + mapping0.i2 * nb12);
        }
        ಆದರೆ (row1_valid) {
            ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + (ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1)];
            row1_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        ಗಾಗಿ (; c + 32 <= k_valid; c += 32) {
            HVX_ಸದಿಶ v0 = Q6_V_vzero();
            HVX_ಸದಿಶ v1 = Q6_V_vzero();
            ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);

            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
        ಆದರೆ (c < k_ಖಂಡ) {
            HVX_ಸದಿಶ v0 = Q6_V_vzero();
            HVX_ಸದಿಶ v1 = Q6_V_vzero();
            ಆದರೆ (row0_valid) v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row0_ptr + c);
            ಆದರೆ (row1_valid) v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem > 0 ? rem * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0);
            v0 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ಮುಖವಾಡ, v1, Q6_V_vzero());

            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            uint32_t tile_idx = r0 * (k_ಖಂಡ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) + c0;

            HVX_ಸದಿಶ *tile = (HVX_ಸದಿಶ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ಹೊರಹೋಗು;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_fp16_ಗಮ್ಯ_fp32_scattered(
            ದಶಮಾಂಶ *ಸೀಮಿತ dst,
            ಸ್ಥಿರಾಂಕ __fp16 *ಸೀಮಿತ vtcm_src,
            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            uint32_t vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
            uint32_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
            ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    ಪ್ರತಿಜ್ಞಾ(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ % HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == 0);
    ಸ್ಥಿರಾಂಕ size_t tile_ಅಡ್ಡಸಾಲು_stride = (n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) * HTP_MM_HMX_TILE_N_ELMS;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);

    ಗಾಗಿ (size_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r += 2) {
        uint32_t r_idx0 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 0;
        uint32_t r_idx1 = ಆರಂಭ_ಅಡ್ಡಸಾಲು + r + 1;
        uint32_t     lr = vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು + r;                 // vtcm-local row
        ಸ್ಥಿರಾಂಕ size_t r0 = (lr / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
        ಸ್ಥಿರಾಂಕ size_t r1 = (lr % HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು) / 2;  // index of the row pair within the tile
        ಸ್ಥಿರಾಂಕ __fp16 *ಅಡ್ಡಸಾಲು_ತಳ = vtcm_src + r0 * tile_ಅಡ್ಡಸಾಲು_stride;

        ಆದರೆ (r_idx0 >= cne1) ನಿಲ್ಲಿಸು;

        ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping0 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + r_idx0];
        ದಶಮಾಂಶ *output_row0 = (ದಶಮಾಂಶ *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        ದಶಮಾಂಶ *output_row1 = ಖಾಲಿ;
        ಆದರೆ (r_idx1 < cne1) {
            ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping mapping1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a * mapping_stride + r_idx1];
            output_row1 = (ದಶಮಾಂಶ *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #ಸೂಚನೆ unroll(4)
        ಗಾಗಿ (size_t c = 0; c < (size_t)n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ; c += HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
            ಸ್ಥಿರಾಂಕ size_t c0 = c / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ __fp16 *tile = ಅಡ್ಡಸಾಲು_ತಳ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ಸದಿಶ v = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) tile)[r1];
            HVX_ಸದಿಶ_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ಆದರೆ (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
