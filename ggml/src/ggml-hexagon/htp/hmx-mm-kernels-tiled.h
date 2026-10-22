#شامل "hmx-utils.h"
#شامل "hmx-queue.h"

// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
ایستا ثابت __fp16 mxfp4_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

ایستا ثابت __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

تعریفنوع ساختار tiled_dequantize_state tiled_dequantize_state_t;
تعریفنوع تهی (*tiled_dequant_task_fn_t)(ثابت tiled_dequantize_state_t * state, uint32_t start_tile, uint32_t end_tile);

ساختار tiled_dequantize_state {
    ساختار htp_context      * ctx;
    ساختار htp_thread_trace * traces;
    __fp16                  * dst;
    ثابت uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    ساختار fastdiv_مقادیر     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_weight_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_per_task;
    uint32_t                  tile_size;
    uint32_t                  aligned_tile_size;
    uint32_t                  n_tasks;
    uint32_t                  n_cols;
    uint32_t                  k_block;
    size_t                    row_stride;
    uint32_t                  weight_type;
};

ایستا تهی dequantize_tiled_weight_to_fp16_task_q1_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector v_neg_one = Q6_Vb_vsplat_R(-1);
    ثابت HVX_Vector v_pos_one = Q6_Vb_vsplat_R(1);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        ثابت HVX_Vector v_sc = hvx_vmem(tile_src + 512);
        ثابت HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        برای (uint32_t q = 0; q < 4; q++) {
            ثابت HVX_Vector v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #pragma unroll
            برای (صحیح g = 0; g < 8; g++) {
                ثابت HVX_Vector     v_q  = unpack_q1_0_group(v_plane, g, v_neg_one, v_pos_one);
                ثابت HVX_VectorPair vp16 = Q6_Wh_vunpack_Vb(v_q);
                ثابت HVX_VectorPair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
ایستا تهی dequantize_tiled_weight_to_fp16_task_q4_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 512);
        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

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
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_VectorPair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_VectorPair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_VectorPair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_VectorPair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_VectorPair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_VectorPair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_VectorPair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_VectorPair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

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

ایستا تهی dequantize_tiled_weight_to_fp16_task_q4_1(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector vscale_offset = hvx_vmem(tile_src + 512);
        HVX_VectorPair dm_deal = Q6_W_vdeal_VVR(vscale_offset, vscale_offset, -2);
        HVX_Vector vd = Q6_V_lo_W(dm_deal);
        HVX_Vector vm = Q6_V_hi_W(dm_deal);

        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_Vector v_offset_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_VectorPair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_VectorPair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_VectorPair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_VectorPair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_VectorPair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_VectorPair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_VectorPair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_VectorPair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));

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

ایستا تهی dequantize_tiled_weight_to_fp16_task_iq4_nl(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_Vector vlut_cvt = hvx_vmem(iq4_nl_to_fp16_lut);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 512);
        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_Vector v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_Vector v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_Vector v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_Vector v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_Vector v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_Vector v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_Vector v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_Vector v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_VectorPair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_VectorPair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_VectorPair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_VectorPair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_VectorPair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_VectorPair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_VectorPair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_VectorPair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ایستا تهی dequantize_tiled_weight_to_fp16_task_mxfp4(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_Vector vlut_cvt = hvx_vmem(mxfp4_to_fp16_lut);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v = hvx_vmem(tile_src + 512);
        HVX_Vector vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_Vector v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_Vector v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_Vector v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_Vector v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_Vector v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_Vector v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_Vector v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_Vector v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_VectorPair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_VectorPair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_VectorPair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_VectorPair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_VectorPair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_VectorPair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_VectorPair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_VectorPair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ایستا تهی dequantize_tiled_weight_to_fp16_task_q8_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 1024);
        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_VectorPair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_VectorPair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_VectorPair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_VectorPair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_Vector vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_Vector vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_Vector vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_Vector vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_VectorPair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_VectorPair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_VectorPair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_VectorPair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

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
        HVX_Vector v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_Vector v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_Vector v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_Vector v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_Vector v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_Vector v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_Vector v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_Vector v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

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

ایستا تهی dequantize_tiled_weight_to_fp16_task_q5_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector vscale_offset = hvx_vmem(tile_src + 512);
        HVX_VectorPair dm_deal = Q6_W_vdeal_VVR(vscale_offset, vscale_offset, -2);
        HVX_Vector vd = Q6_V_lo_W(dm_deal);
        HVX_Vector vm = Q6_V_hi_W(dm_deal);

        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_Vector v_offset_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Q5_K: OR in the 5th bit from the plane
        HVX_Vector v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_VectorPair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_VectorPair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_VectorPair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_VectorPair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_VectorPair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_VectorPair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_VectorPair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_VectorPair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));

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
ایستا تهی dequantize_tiled_weight_to_fp16_task_q6_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_Vector i32     = Q6_Vb_vsplat_R(32);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت HVX_Vector * vptr = (ثابت HVX_Vector *) (state->src + t * state->aligned_tile_size);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc      = vptr[6];
        HVX_Vector v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #pragma unroll
        برای (صحیح g = 0; g < 8; g += 4) {
            ثابت HVX_Vector v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_Vector v_q0 = unpack_q6_k_group(vptr, g + 0, mask_0f, mask_03, i32);
            HVX_Vector v_q1 = unpack_q6_k_group(vptr, g + 1, mask_0f, mask_03, i32);
            HVX_Vector v_q2 = unpack_q6_k_group(vptr, g + 2, mask_0f, mask_03, i32);
            HVX_Vector v_q3 = unpack_q6_k_group(vptr, g + 3, mask_0f, mask_03, i32);

            HVX_VectorPair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_VectorPair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_VectorPair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_VectorPair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_VectorPair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_VectorPair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_VectorPair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_VectorPair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_Vector v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_Vector v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_Vector v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_Vector v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_Vector v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_Vector v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_Vector v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_Vector v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

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
ایستا تهی dequantize_tiled_weight_to_fp16_task_q3_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت HVX_Vector * vptr = (ثابت HVX_Vector *) (state->src + t * state->aligned_tile_size);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc      = vptr[3];
        HVX_Vector v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #pragma unroll
        برای (صحیح g = 0; g < 8; g++) {
            ثابت HVX_Vector v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_Vector     v_q   = unpack_q3_k_group(vptr, g, mask_03);
            HVX_VectorPair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_VectorPair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
ایستا تهی dequantize_tiled_weight_to_fp16_task_q2_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    HVX_Vector v_d = Q6_V_vzero();
    HVX_Vector v_dm = Q6_V_vzero();

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت uint8_t * tile = state->src + t * state->aligned_tile_size;
        ثابت HVX_Vector * vptr = (ثابت HVX_Vector *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        اگر (t == start_tile || (t & 7) == 0) {
            q2_k_load_dmin(tile - (t & 7) * state->aligned_tile_size, &v_d, &v_dm);
        }

        HVX_Vector v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_Vector v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_Vector v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_Vector v_offset_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_Vector v_offset_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #pragma unroll
        برای (صحیح g = 0; g < 8; g++) {
            ثابت HVX_Vector v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            ثابت HVX_Vector v_offset = (g < 4) ? v_offset_k0 : v_offset_k16;

            HVX_Vector     v_q   = unpack_q3_k_low2(vptr, g, mask_03);
            HVX_VectorPair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_VectorPair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_offset));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_offset));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
ایستا تهی dequantize_tiled_weight_to_fp16_task_q2_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_Vector i1      = Q6_Vb_vsplat_R(1);

    برای (uint32_t t = start_tile; t < end_tile; t++) {
        ثابت HVX_Vector * vptr = (ثابت HVX_Vector *) (state->src + t * state->aligned_tile_size);
        __fp16 * dst_base = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc    = vptr[4];
        HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        برای (صحیح half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_base + half * HTP_MM_HMX_TILE_N_ELMS;

            #pragma unroll
            برای (صحیح g = 0; g < 8; g++) {
                HVX_Vector     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, mask_03), i1);
                HVX_VectorPair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_VectorPair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

ایستا __attribute__((noinline))
تهی convert_f16_weight_to_fp16_tiles_task(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت uint32_t n_k_tiles = state->n_k_tiles;
    ثابت ساختار fastdiv_مقادیر n_k_tiles_div = state->n_k_tiles_div;

    ثابت HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    ثابت HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    برای (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        اگر (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * اندازه(__fp16);

            HVX_Vector v_off = v_scat_base;
            برای (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_COLS + r;
                uint32_t row1 = row0 + 1;

                ثابت uint8_t *r0 = state->src + row0 * state->row_stride;
                ثابت uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0 = (row0 < state->n_cols) ? hvx_vmemu((ثابت __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_Vector v1 = (row1 < state->n_cols) ? hvx_vmemu((ثابت __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
        ++t; ++kt;
    }
}

ایستا __attribute__((noinline))
تهی quantize_f32_weight_to_fp16_tiles_task(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت uint32_t n_k_tiles = state->n_k_tiles;
    ثابت ساختار fastdiv_مقادیر n_k_tiles_div = state->n_k_tiles_div;

    ثابت HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    ثابت HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    برای (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        اگر (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * اندازه(اعشاری);

            HVX_Vector v_off = v_scat_base;
            برای (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_COLS + r;
                uint32_t row1 = row0 + 1;

                ثابت uint8_t *r0 = state->src + row0 * state->row_stride;
                ثابت uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0_f32 = (row0 < state->n_cols) ? hvx_vmem((ثابت اعشاری *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_Vector v1_f32 = (row1 < state->n_cols) ? hvx_vmem((ثابت اعشاری *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

                HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
        ++t; ++kt;
    }
}

ایستا __attribute__((noinline))
تهی convert_bf16_weight_to_fp16_tiles_task(
        ثابت tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    ثابت uint32_t n_k_tiles = state->n_k_tiles;
    ثابت ساختار fastdiv_مقادیر n_k_tiles_div = state->n_k_tiles_div;

    ثابت HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    ثابت HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    برای (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        اگر (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * اندازه(uint16_t);

            HVX_Vector v_off = v_scat_base;
            برای (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_COLS + r;
                uint32_t row1 = row0 + 1;

                ثابت uint8_t *r0 = state->src + row0 * state->row_stride;
                ثابت uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0_raw = (row0 < state->n_cols) ? hvx_vmemu((ثابت تهی *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_Vector v1_raw = (row1 < state->n_cols) ? hvx_vmemu((ثابت تهی *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_Vector v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_Vector v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

                HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

ایستا تهی core_dot_chunk_fp16_short(__fp16 *restrict output, ثابت __fp16 *restrict activation,
                                ثابت __fp16 *restrict weight, ثابت __fp16 *restrict scales,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm فرار(HMX_SET_BIAS("%0") :: "r"((unsigned صحیح)scales));

    ثابت size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ثابت uint32_t range = 2048u * n_dot_tiles - 1;

    برای (uint32_t r = 0; r < n_row_tiles; ++r) {
        ثابت __fp16 *row_base = activation + r * dot_stride;
        ثابت __fp16 *col_base = weight;
        __fp16 *out_tile = output + r * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;

        برای (size_t c = 0; c < n_col_tiles; ++c) {
            asm فرار(HMX_CLRACC_F16());
            asm فرار(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_base), "r"(col_base));
            asm فرار(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");
            col_base += dot_stride;
            out_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ایستا تهی core_dot_chunk_fp16(__fp16 *restrict output, ثابت __fp16 *restrict activation,
                          ثابت __fp16 *restrict weight, ثابت __fp16 *restrict scales,
                          uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    اگر (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, weight, scales, n_row_tiles, n_col_tiles, n_dot_tiles);
        بازگشت;
    }
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm فرار(HMX_SET_BIAS("%0") :: "r"((unsigned صحیح)scales));

    ثابت size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    برای (uint32_t r = 0; r < n_row_tiles; ++r) {
        ثابت __fp16 *row_base = activation + r * dot_stride;
        ثابت __fp16 *col_base = weight;
        __fp16 *out_tile = output + r * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;

        برای (size_t c = 0; c < n_col_tiles; ++c) {
            ثابت __fp16 *row_tiles = row_base;
            ثابت __fp16 *col_tiles = col_base;

            asm فرار(HMX_CLRACC_F16());

            ثابت uint32_t n_loops = n_dot_tiles / 32;
            ثابت uint32_t rem = n_dot_tiles % 32;

            برای (uint32_t l = 0; l < n_loops; ++l) {
                asm فرار(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            اگر (rem > 0) {
                ثابت uint32_t range = 2048u * rem - 1;
                asm فرار(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
            }

            asm فرار(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");

            col_base += dot_stride;
            out_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ایستا تهی core_mma_chunk_fp16_short(__fp16 *restrict c, ثابت __fp16 *restrict a, ثابت __fp16 *restrict b,
                                ثابت __fp16 *restrict col_scales, ثابت __fp16 *restrict eye_tile,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm فرار(HMX_SET_BIAS("%0") :: "r"((unsigned صحیح)col_scales));

    ثابت size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ثابت uint32_t range = 2048u * n_dot_tiles - 1;

    برای (size_t i = 0; i < n_row_tiles; ++i) {
        ثابت __fp16 *row_base = a + i * dot_tile_stride;
        __fp16 *res_base = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ثابت __fp16 *col_base = b;
        __fp16 *accum_tile = res_base;

        برای (size_t j = 0; j < n_col_tiles; ++j) {
            asm فرار(HMX_CLRACC_F16());

            اگر (!zero_init) {
                asm فرار(HMX_LOAD_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm فرار(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_base), "r"(col_base));

            asm فرار(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_base   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ایستا تهی core_mma_chunk_fp16(__fp16 *restrict c, ثابت __fp16 *restrict a, ثابت __fp16 *restrict b,
                          ثابت __fp16 *restrict col_scales, ثابت __fp16 *restrict eye_tile,
                          uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    اگر (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, col_scales, eye_tile, n_row_tiles, n_col_tiles, n_dot_tiles, zero_init);
        بازگشت;
    }
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm فرار(HMX_SET_BIAS("%0") :: "r"((unsigned صحیح)col_scales));

    ثابت size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    برای (size_t i = 0; i < n_row_tiles; ++i) {
        ثابت __fp16 *row_base = a + i * dot_tile_stride;
        __fp16 *res_base = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ثابت __fp16 *col_base = b;
        __fp16 *accum_tile = res_base;

        برای (size_t j = 0; j < n_col_tiles; ++j) {
            ثابت __fp16 *col_tiles = col_base;
            ثابت __fp16 *row_tiles = row_base;

            asm فرار(HMX_CLRACC_F16());

            اگر (!zero_init) {
                asm فرار(HMX_LOAD_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            ثابت uint32_t n_loops = n_dot_tiles / 32;
            ثابت uint32_t rem = n_dot_tiles % 32;

            برای (uint32_t l = 0; l < n_loops; ++l) {
                asm فرار(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            اگر (rem > 0) {
                ثابت uint32_t range = 2048u * rem - 1;
                asm فرار(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
            }

            asm فرار(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_base += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32p

ایستا تهی transfer_output_chunk_fp16_to_fp32_col_chunk(
    اعشاری *restrict dst,
    ثابت اعشاری *restrict src2,
    ثابت __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t c_len,
    uint32_t total_n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols
) {
    ثابت size_t tile_row_stride = hmx_ceil_div(total_n_cols, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;

    ثابت HVX_Vector one = hvx_vec_splat_f16(1.0);

    ثابت size_t limit_c         = hex_smin(c_len, dst_cols);
    ثابت size_t limit_c_aligned = (limit_c & ~31);

    برای (size_t r = 0; r < n_rows; r += 2) {
        ثابت size_t r_idx0 = start_row + r + 0;
        ثابت size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;
        ثابت size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        ثابت __fp16 *row_base = vtcm_src + r0 * tile_row_stride;
        اعشاری *output_row_base = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        ثابت اعشاری *src2_row_base = src2 ? (src2 + r * src2_stride) : پوچ;

        #pragma unroll(4)
        برای (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {
            ثابت size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;
            ثابت __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((ثابت HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            اگر (src2_row_base) {
                HVX_Vector v_src2_0 = hvx_vmemu(src2_row_base + c + 0);
                v_out0 = hvx_vec_add_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_row_base + c + 0) = v_out0;

            اگر (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                اگر (src2_row_base) {
                    HVX_Vector v_src2_1 = hvx_vmemu(src2_row_base + c + src2_stride);
                    v_out1 = hvx_vec_add_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_row_base + c + dst_stride) = v_out1;
            }
        }

        اگر (limit_c_aligned < limit_c) {
            size_t c = limit_c_aligned;
            size_t valid_c = limit_c - c;
            ثابت size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            ثابت __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((ثابت HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            اگر (src2_row_base) {
                HVX_Vector v_src2_0 = hvx_vmemu(src2_row_base + c + 0);
                v_out0 = hvx_vec_add_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_row_base + c, valid_c * اندازه(اعشاری), v_out0);

            اگر (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                اگر (src2_row_base) {
                    HVX_Vector v_src2_1 = hvx_vmemu(src2_row_base + c + src2_stride);
                    v_out1 = hvx_vec_add_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_row_base + c + dst_stride, valid_c * اندازه(اعشاری), v_out1);
            }
        }
    }
}

ایستا درونخطی تهی transfer_output_chunk_fp16_to_fp32(
    اعشاری *restrict dst,
    ثابت اعشاری *restrict src2,
    ثابت __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols
) {
    transfer_output_chunk_fp16_to_fp32_col_chunk(
        dst, src2, vtcm_src, start_row, n_rows, n_cols, n_cols, dst_stride, src2_stride, dst_cols
    );
}

تعریفنوع ساختار {
    ثابت __fp16  *vtcm_src;
    اعشاری         *dst;
    ثابت اعشاری   *src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_per_task;
    uint32_t       n_cols;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_cols;    // Actual output columns
    ساختار htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

ایستا تهی transfer_activation_row_pair_fp32_to_fp16(
        __fp16 *restrict vtcm_dst,
        ثابت اعشاری *restrict row0,
        ثابت اعشاری *restrict row1,
        uint32_t r,
        uint32_t k_block,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    uint32_t c = 0;
    برای (; c + 32 <= k_valid; c += 32) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0 + c);
        اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1 + c);

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
    اگر (c < k_block) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0 + c);
        اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * اندازه(اعشاری) : 0);
        v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
}

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
ایستا تهی transfer_activation_row_pair_f16_to_f16(__fp16 * restrict vtcm_dst,
                                                    ثابت __fp16 * restrict row0,
                                                    ثابت __fp16 * restrict row1,
                                                    uint32_t r,
                                                    uint32_t k_block,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    ثابت uint32_t n_tile_cols = k_block / HTP_MM_HMX_TILE_N_COLS;
    __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

    ثابت HVX_UVector * pv0 = row0_valid ? (ثابت HVX_UVector *) row0 : پوچ;
    ثابت HVX_UVector * pv1 = row1_valid ? (ثابت HVX_UVector *) row1 : پوچ;

    uint32_t c = 0;
    برای (; c + 64 <= k_valid; c += 64) {
        HVX_Vector     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_Vector     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;

        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    برای (; c < k_block; c += 64) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();

        اگر (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * اندازه(__fp16));
            اگر (pv0) {
                v0 = Q6_V_vmux_QVV(mask, pv0[c / 64], Q6_V_vzero());
            }
            اگر (pv1) {
                v1 = Q6_V_vmux_QVV(mask, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;

        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        اگر (c0 + 1 < n_tile_cols) {
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

ایستا تهی transfer_activation_row_pair_fp32_to_fp16_col_chunk(
        __fp16 *restrict vtcm_dst,
        ثابت اعشاری *restrict row0, // offset by c_first
        ثابت اعشاری *restrict row1, // offset by c_first
        uint32_t r,
        uint32_t k_block,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    uint32_t c = 0;
    برای (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0 + c);
        اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1 + c);

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = (c_first + c) / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
    اگر (c < c_len) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0 + c);
        اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * اندازه(اعشاری) : 0);
        v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = (c_first + c) / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
}

ایستا تهی transfer_activation_chunk_fp32_to_fp16_gathered(
            __fp16 *restrict vtcm_dst,
            ثابت اعشاری *restrict src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_rows,
            uint32_t k_block,
            ثابت ساختار mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            ثابت ساختار fastdiv_مقادیر * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ثابت uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    ثابت uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    برای (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        ساختار mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        ساختار mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        ثابت اعشاری *row0_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        ثابت اعشاری *row1_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        برای (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(ثابت HVX_Vector *)(row1_ptr + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        اگر (c < k_block) {
            HVX_Vector v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(ثابت HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * اندازه(اعشاری) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    برای (; r < n_rows_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        ثابت bool row0_valid = (start_row + r + 0) < cne1;
        ثابت bool row1_valid = (start_row + r + 1) < cne1;

        ثابت اعشاری *row0_ptr = پوچ;
        ثابت اعشاری *row1_ptr = پوچ;

        اگر (row0_valid) {
            ساختار mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        اگر (row1_valid) {
            ساختار mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        برای (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1_ptr + c);

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        اگر (c < k_block) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * اندازه(اعشاری) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

ایستا تهی transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            __fp16 *restrict vtcm_dst,
            ثابت اعشاری *restrict src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_rows,
            uint32_t k_block,
            ثابت ساختار mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ثابت uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    ثابت uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    برای (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        ساختار mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        ساختار mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        ثابت اعشاری *row0_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + mapping0.i2 * nb12);
        ثابت اعشاری *row1_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        برای (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(ثابت HVX_Vector *)(row1_ptr + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        اگر (c < k_block) {
            HVX_Vector v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(ثابت HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * اندازه(اعشاری) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    برای (; r < n_rows_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        ثابت bool row0_valid = (start_row + r + 0) < cne1;
        ثابت bool row1_valid = (start_row + r + 1) < cne1;

        ثابت اعشاری *row0_ptr = پوچ;
        ثابت اعشاری *row1_ptr = پوچ;

        اگر (row0_valid) {
            ساختار mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            row0_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + mapping0.i2 * nb12);
        }
        اگر (row1_valid) {
            ساختار mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            row1_ptr = (ثابت اعشاری *) ((ثابت uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        برای (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1_ptr + c);

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        اگر (c < k_block) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            اگر (row0_valid) v0 = *(ثابت HVX_Vector *)(row0_ptr + c);
            اگر (row1_valid) v1 = *(ثابت HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * اندازه(اعشاری) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

ایستا تهی transfer_output_chunk_fp16_to_fp32_scattered(
            اعشاری *restrict dst,
            ثابت __fp16 *restrict vtcm_src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_rows,
            uint32_t n_cols,
            ثابت ساختار mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    أكد(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);
    ثابت size_t tile_row_stride = (n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;

    ثابت HVX_Vector one = hvx_vec_splat_f16(1.0);

    برای (size_t r = 0; r < n_rows; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t     lr = vtcm_start_row + r;                 // vtcm-local row
        ثابت size_t r0 = (lr / HTP_MM_HMX_TILE_N_ROWS);
        ثابت size_t r1 = (lr % HTP_MM_HMX_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        ثابت __fp16 *row_base = vtcm_src + r0 * tile_row_stride;

        اگر (r_idx0 >= cne1) بشکن;

        ساختار mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        اعشاری *output_row0 = (اعشاری *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        اعشاری *output_row1 = پوچ;
        اگر (r_idx1 < cne1) {
            ساختار mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];
            output_row1 = (اعشاری *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #pragma unroll(4)
        برای (size_t c = 0; c < (size_t)n_cols; c += HTP_MM_HMX_TILE_N_COLS) {
            ثابت size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            ثابت __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((ثابت HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            اگر (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
