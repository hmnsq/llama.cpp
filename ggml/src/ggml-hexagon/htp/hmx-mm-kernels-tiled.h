#ضمن "hmx-utils.h"
#ضمن "hmx-queue.h"

// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
ساكن ثابت __fp16 mxfp4_منزل_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

ساكن ثابت __fp16 iq4_nl_منزل_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

تعريفنوع بنية tiled_dequantize_state tiled_dequantize_state_t;
تعريفنوع فراغ (*tiled_dequant_task_fn_t)(ثابت tiled_dequantize_state_t * state, uint32_t پیل_tile, uint32_t پای_ټکی_tile);

بنية tiled_dequantize_state {
    بنية htp_context      * ctx;
    بنية htp_thread_trace * traces;
    __fp16                  * dst;
    ثابت uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    بنية fastdiv_ارزښتونه     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_وزن_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_فی_task;
    uint32_t                  tile_اندازه;
    uint32_t                  aligned_tile_اندازه;
    uint32_t                  n_tasks;
    uint32_t                  n_ستون_شمېر;
    uint32_t                  k_بلاک;
    size_t                    لیکه_stride;
    uint32_t                  وزن_type;
};

ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q1_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور v_neg_one = Q6_Vb_vsplat_R(-1);
    ثابت HVX_ویکټور v_pos_one = Q6_Vb_vsplat_R(1);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        ثابت HVX_ویکټور v_sc = hvx_vmem(tile_src + 512);
        ثابت HVX_ویکټور v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        لكل (uint32_t q = 0; q < 4; q++) {
            ثابت HVX_ویکټور v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #لارښوونه unroll
            لكل (صحيح g = 0; g < 8; g++) {
                ثابت HVX_ویکټور     v_q  = unpack_q1_0_ډله(v_plane, g, v_neg_one, v_pos_one);
                ثابت HVX_ویکټور_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                ثابت HVX_ویکټور_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q4_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_h4 = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_ویکټور i8 = Q6_Vb_vsplat_R(8);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v_sc = hvx_vmem(tile_src + 512);
        HVX_ویکټور v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_ویکټور vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ویکټور vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ویکټور vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ویکټور vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ویکټور v_lo0 = Q6_V_vand_VV(vq0, ماسک_h4);
        HVX_ویکټور v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ویکټور v_lo1 = Q6_V_vand_VV(vq1, ماسک_h4);
        HVX_ویکټور v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ویکټور v_lo2 = Q6_V_vand_VV(vq2, ماسک_h4);
        HVX_ویکټور v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ویکټور v_lo3 = Q6_V_vand_VV(vq3, ماسک_h4);
        HVX_ویکټور v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

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
        HVX_ویکټور_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ویکټور_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ویکټور_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ویکټور_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ویکټور_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ویکټور_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ویکټور_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ویکټور_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ویکټور_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ویکټور_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ویکټور_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ویکټور_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_ویکټور v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_ویکټور v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_ویکټور v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_ویکټور v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_ویکټور v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_ویکټور v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_ویکټور v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_ویکټور v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_ویکټور v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_ویکټور v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_ویکټور v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_ویکټور v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_ویکټور v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_ویکټور v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_ویکټور v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_ویکټور v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

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

ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q4_1(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_h4 = Q6_Vb_vsplat_R(0x0F);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور vscale_دننه_واټن = hvx_vmem(tile_src + 512);
        HVX_ویکټور_Pair dm_deal = Q6_W_vdeal_VVR(vscale_دننه_واټن, vscale_دننه_واټن, -2);
        HVX_ویکټور vd = Q6_V_lo_W(dm_deal);
        HVX_ویکټور vm = Q6_V_hi_W(dm_deal);

        HVX_ویکټور v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_ویکټور v_دننه_واټن_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_ویکټور vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ویکټور vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ویکټور vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ویکټور vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ویکټور v_lo0 = Q6_V_vand_VV(vq0, ماسک_h4);
        HVX_ویکټور v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ویکټور v_lo1 = Q6_V_vand_VV(vq1, ماسک_h4);
        HVX_ویکټور v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ویکټور v_lo2 = Q6_V_vand_VV(vq2, ماسک_h4);
        HVX_ویکټور v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ویکټور v_lo3 = Q6_V_vand_VV(vq3, ماسک_h4);
        HVX_ویکټور v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ویکټور_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ویکټور_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ویکټور_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ویکټور_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ویکټور_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ویکټور_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ویکټور_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ویکټور_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ویکټور_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ویکټور_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ویکټور_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ویکټور_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_ویکټور v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_دننه_واټن_duplicated));

        HVX_ویکټور v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_دننه_واټن_duplicated));

        HVX_ویکټور v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_دننه_واټن_duplicated));

        HVX_ویکټور v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_دننه_واټن_duplicated));

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

ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_iq4_nl(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_h4 = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_ویکټور vlut_cvt = hvx_vmem(iq4_nl_منزل_fp16_lut);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v_sc = hvx_vmem(tile_src + 512);
        HVX_ویکټور v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_ویکټور vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ویکټور vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ویکټور vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ویکټور vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ویکټور v_lo0 = Q6_V_vand_VV(vq0, ماسک_h4);
        HVX_ویکټور v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ویکټور v_lo1 = Q6_V_vand_VV(vq1, ماسک_h4);
        HVX_ویکټور v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ویکټور v_lo2 = Q6_V_vand_VV(vq2, ماسک_h4);
        HVX_ویکټور v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ویکټور v_lo3 = Q6_V_vand_VV(vq3, ماسک_h4);
        HVX_ویکټور v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ویکټور_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ویکټور_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ویکټور_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ویکټور_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_ویکټور v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ویکټور v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ویکټور v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ویکټور v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ویکټور v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ویکټور v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ویکټور v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ویکټور v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_ویکټور_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_ویکټور v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_ویکټور v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_ویکټور v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_ویکټور v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_ویکټور v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_ویکټور v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_ویکټور v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_ویکټور v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_ویکټور v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_ویکټور v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_ویکټور v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_ویکټور v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_ویکټور v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_ویکټور v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_ویکټور v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_ویکټور v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_mxfp4(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_h4 = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_ویکټور vlut_cvt = hvx_vmem(mxfp4_منزل_fp16_lut);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v = hvx_vmem(tile_src + 512);
        HVX_ویکټور vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_ویکټور v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_ویکټور vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ویکټور vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ویکټور vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ویکټور vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ویکټور v_lo0 = Q6_V_vand_VV(vq0, ماسک_h4);
        HVX_ویکټور v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ویکټور v_lo1 = Q6_V_vand_VV(vq1, ماسک_h4);
        HVX_ویکټور v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ویکټور v_lo2 = Q6_V_vand_VV(vq2, ماسک_h4);
        HVX_ویکټور v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ویکټور v_lo3 = Q6_V_vand_VV(vq3, ماسک_h4);
        HVX_ویکټور v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ویکټور_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ویکټور_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ویکټور_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ویکټور_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_ویکټور v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ویکټور v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ویکټور v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ویکټور v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ویکټور v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ویکټور v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ویکټور v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ویکټور v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_ویکټور_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_ویکټور_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_ویکټور v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_ویکټور v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_ویکټور v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_ویکټور v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_ویکټور v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_ویکټور v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_ویکټور v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_ویکټور v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_ویکټور v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_ویکټور v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_ویکټور v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_ویکټور v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_ویکټور v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_ویکټور v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_ویکټور v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_ویکټور v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q8_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v_sc = hvx_vmem(tile_src + 1024);
        HVX_ویکټور v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_ویکټور vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ویکټور vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ویکټور vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ویکټور vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_ویکټور_Pair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_ویکټور_Pair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_ویکټور_Pair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_ویکټور_Pair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_ویکټور vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_ویکټور vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_ویکټور vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_ویکټور vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_ویکټور_Pair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_ویکټور_Pair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_ویکټور_Pair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_ویکټور_Pair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_ویکټور v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_ویکټور v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_ویکټور v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_ویکټور v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_ویکټور v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_ویکټور v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_ویکټور v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_ویکټور v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

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
        HVX_ویکټور v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_ویکټور v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_ویکټور v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_ویکټور v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_ویکټور v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_ویکټور v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_ویکټور v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_ویکټور v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

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

ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q5_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_h4 = Q6_Vb_vsplat_R(0x0F);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile_src = state->src + t * state->aligned_tile_اندازه;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور vscale_دننه_واټن = hvx_vmem(tile_src + 512);
        HVX_ویکټور_Pair dm_deal = Q6_W_vdeal_VVR(vscale_دننه_واټن, vscale_دننه_واټن, -2);
        HVX_ویکټور vd = Q6_V_lo_W(dm_deal);
        HVX_ویکټور vm = Q6_V_hi_W(dm_deal);

        HVX_ویکټور v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_ویکټور v_دننه_واټن_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_ویکټور vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ویکټور vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ویکټور vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ویکټور vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ویکټور v_lo0 = Q6_V_vand_VV(vq0, ماسک_h4);
        HVX_ویکټور v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ویکټور v_lo1 = Q6_V_vand_VV(vq1, ماسک_h4);
        HVX_ویکټور v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ویکټور v_lo2 = Q6_V_vand_VV(vq2, ماسک_h4);
        HVX_ویکټور v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ویکټور v_lo3 = Q6_V_vand_VV(vq3, ماسک_h4);
        HVX_ویکټور v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Q5_K: OR in the 5th bit from the plane
        HVX_ویکټور v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

        // Shuffling
        HVX_ویکټور_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ویکټور_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ویکټور_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ویکټور_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ویکټور_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ویکټور_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ویکټور_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ویکټور_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ویکټور_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ویکټور_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ویکټور_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ویکټور_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_ویکټور v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_دننه_واټن_duplicated));

        HVX_ویکټور v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_دننه_واټن_duplicated));

        HVX_ویکټور v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_دننه_واټن_duplicated));

        HVX_ویکټور v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_دننه_واټن_duplicated));
        HVX_ویکټور v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_دننه_واټن_duplicated));

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
ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q6_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_ویکټور ماسک_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_ویکټور i32     = Q6_Vb_vsplat_R(32);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت HVX_ویکټور * vptr = (ثابت HVX_ویکټور *) (state->src + t * state->aligned_tile_اندازه);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v_sc      = vptr[6];
        HVX_ویکټور v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_ویکټور v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ویکټور v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #لارښوونه unroll
        لكل (صحيح g = 0; g < 8; g += 4) {
            ثابت HVX_ویکټور v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_ویکټور v_q0 = unpack_q6_k_ډله(vptr, g + 0, ماسک_0f, ماسک_03, i32);
            HVX_ویکټور v_q1 = unpack_q6_k_ډله(vptr, g + 1, ماسک_0f, ماسک_03, i32);
            HVX_ویکټور v_q2 = unpack_q6_k_ډله(vptr, g + 2, ماسک_0f, ماسک_03, i32);
            HVX_ویکټور v_q3 = unpack_q6_k_ډله(vptr, g + 3, ماسک_0f, ماسک_03, i32);

            HVX_ویکټور_Pair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_ویکټور_Pair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_ویکټور_Pair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_ویکټور_Pair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_ویکټور_Pair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_ویکټور_Pair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_ویکټور_Pair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_ویکټور_Pair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_ویکټور v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_ویکټور v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_ویکټور v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_ویکټور v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_ویکټور v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_ویکټور v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_ویکټور v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_ویکټور v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

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
ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q3_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_03 = Q6_Vb_vsplat_R(0x03);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت HVX_ویکټور * vptr = (ثابت HVX_ویکټور *) (state->src + t * state->aligned_tile_اندازه);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v_sc      = vptr[3];
        HVX_ویکټور v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_ویکټور v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ویکټور v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #لارښوونه unroll
        لكل (صحيح g = 0; g < 8; g++) {
            ثابت HVX_ویکټور v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_ویکټور     v_q   = unpack_q3_k_ډله(vptr, g, ماسک_03);
            HVX_ویکټور_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_ویکټور_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q2_k(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_03 = Q6_Vb_vsplat_R(0x03);

    HVX_ویکټور v_d = Q6_V_vzero();
    HVX_ویکټور v_dm = Q6_V_vzero();

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت uint8_t * tile = state->src + t * state->aligned_tile_اندازه;
        ثابت HVX_ویکټور * vptr = (ثابت HVX_ویکټور *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        إذا (t == پیل_tile || (t & 7) == 0) {
            q2_k_بارول_dmin(tile - (t & 7) * state->aligned_tile_اندازه, &v_d, &v_dm);
        }

        HVX_ویکټور v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_ویکټور v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_ویکټور v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_ویکټور v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ویکټور v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_ویکټور v_دننه_واټن_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_ویکټور v_دننه_واټن_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #لارښوونه unroll
        لكل (صحيح g = 0; g < 8; g++) {
            ثابت HVX_ویکټور v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            ثابت HVX_ویکټور v_دننه_واټن = (g < 4) ? v_دننه_واټن_k0 : v_دننه_واټن_k16;

            HVX_ویکټور     v_q   = unpack_q3_k_low2(vptr, g, ماسک_03);
            HVX_ویکټور_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_ویکټور_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_دننه_واټن));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_دننه_واټن));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
ساكن فراغ dequantize_tiled_وزن_منزل_fp16_task_q2_0(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت HVX_ویکټور ماسک_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_ویکټور i1      = Q6_Vb_vsplat_R(1);

    لكل (uint32_t t = پیل_tile; t < پای_ټکی_tile; t++) {
        ثابت HVX_ویکټور * vptr = (ثابت HVX_ویکټور *) (state->src + t * state->aligned_tile_اندازه);
        __fp16 * dst_بنسټ = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ویکټور v_sc    = vptr[4];
        HVX_ویکټور v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        لكل (صحيح half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_بنسټ + half * HTP_MM_HMX_TILE_N_ELMS;

            #لارښوونه unroll
            لكل (صحيح g = 0; g < 8; g++) {
                HVX_ویکټور     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, ماسک_03), i1);
                HVX_ویکټور_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_ویکټور_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

ساكن __attribute__((noinline))
فراغ convert_f16_وزن_منزل_fp16_tiles_task(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت uint32_t n_k_tiles = state->n_k_tiles;
    ثابت بنية fastdiv_ارزښتونه n_k_tiles_div = state->n_k_tiles_div;

    ثابت HVX_ویکټور v_scat_بنسټ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_ویکټور v_scat_ګام  = Q6_V_vsplat_R(4);
    ثابت HVX_ویکټور_Pred q_mask64 = Q6_Q_vsetq_R(64);

    بینښه ct = fastdiv((بینښه)پیل_tile, &n_k_tiles_div);
    بینښه kt = fastmodulo((بینښه)پیل_tile, n_k_tiles, &n_k_tiles_div);

    لكل (بینښه t = پیل_tile; t < (بینښه)پای_ټکی_tile; ) {
        إذا (kt >= (بینښه)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_بنسټ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * حجم(__fp16);

            HVX_ویکټور v_off = v_scat_بنسټ;
            لكل (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_لیکې; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ستون_شمېر + r;
                uint32_t row1 = row0 + 1;

                ثابت uint8_t *r0 = state->src + row0 * state->لیکه_stride;
                ثابت uint8_t *r1 = state->src + row1 * state->لیکه_stride;

                HVX_ویکټور v0 = (row0 < state->n_ستون_شمېر) ? hvx_vmemu((ثابت __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ویکټور v1 = (row1 < state->n_ستون_شمېر) ? hvx_vmemu((ثابت __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_بنسټ, HTP_MM_HMX_TILE_اندازه - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ګام);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_بنسټ, HTP_MM_HMX_TILE_اندازه - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ګام);
            }
        }
        ++t; ++kt;
    }
}

ساكن __attribute__((noinline))
فراغ quantize_f32_وزن_منزل_fp16_tiles_task(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت uint32_t n_k_tiles = state->n_k_tiles;
    ثابت بنية fastdiv_ارزښتونه n_k_tiles_div = state->n_k_tiles_div;

    ثابت HVX_ویکټور v_scat_بنسټ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_ویکټور v_scat_ګام  = Q6_V_vsplat_R(4);
    ثابت HVX_ویکټور_Pred q_mask64 = Q6_Q_vsetq_R(64);

    بینښه ct = fastdiv((بینښه)پیل_tile, &n_k_tiles_div);
    بینښه kt = fastmodulo((بینښه)پیل_tile, n_k_tiles, &n_k_tiles_div);

    لكل (بینښه t = پیل_tile; t < (بینښه)پای_ټکی_tile; ) {
        إذا (kt >= (بینښه)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_بنسټ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * حجم(عائم);

            HVX_ویکټور v_off = v_scat_بنسټ;
            لكل (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_لیکې; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ستون_شمېر + r;
                uint32_t row1 = row0 + 1;

                ثابت uint8_t *r0 = state->src + row0 * state->لیکه_stride;
                ثابت uint8_t *r1 = state->src + row1 * state->لیکه_stride;

                HVX_ویکټور v0_f32 = (row0 < state->n_ستون_شمېر) ? hvx_vmem((ثابت عائم *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ویکټور v1_f32 = (row1 < state->n_ستون_شمېر) ? hvx_vmem((ثابت عائم *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_بنسټ, HTP_MM_HMX_TILE_اندازه - 1, v_off, v_بهر);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ګام);

                HVX_ویکټور v_بهر_hi = Q6_V_vror_VR(v_بهر, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_بنسټ, HTP_MM_HMX_TILE_اندازه - 1, v_off, v_بهر_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ګام);
            }
        }
        ++t; ++kt;
    }
}

ساكن __attribute__((noinline))
فراغ convert_bf16_وزن_منزل_fp16_tiles_task(
        ثابت tiled_dequantize_state_t *state,
        uint32_t پیل_tile, uint32_t پای_ټکی_tile) {

    ثابت uint32_t n_k_tiles = state->n_k_tiles;
    ثابت بنية fastdiv_ارزښتونه n_k_tiles_div = state->n_k_tiles_div;

    ثابت HVX_ویکټور v_scat_بنسټ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_ویکټور v_scat_ګام  = Q6_V_vsplat_R(4);
    ثابت HVX_ویکټور_Pred q_mask64 = Q6_Q_vsetq_R(64);

    بینښه ct = fastdiv((بینښه)پیل_tile, &n_k_tiles_div);
    بینښه kt = fastmodulo((بینښه)پیل_tile, n_k_tiles, &n_k_tiles_div);

    لكل (بینښه t = پیل_tile; t < (بینښه)پای_ټکی_tile; ) {
        إذا (kt >= (بینښه)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_بنسټ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * حجم(uint16_t);

            HVX_ویکټور v_off = v_scat_بنسټ;
            لكل (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_لیکې; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ستون_شمېر + r;
                uint32_t row1 = row0 + 1;

                ثابت uint8_t *r0 = state->src + row0 * state->لیکه_stride;
                ثابت uint8_t *r1 = state->src + row1 * state->لیکه_stride;

                HVX_ویکټور v0_raw = (row0 < state->n_ستون_شمېر) ? hvx_vmemu((ثابت فراغ *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ویکټور v1_raw = (row1 < state->n_ستون_شمېر) ? hvx_vmemu((ثابت فراغ *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_ویکټور v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_ویکټور v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_بنسټ, HTP_MM_HMX_TILE_اندازه - 1, v_off, v_بهر);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ګام);

                HVX_ویکټور v_بهر_hi = Q6_V_vror_VR(v_بهر, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_بنسټ, HTP_MM_HMX_TILE_اندازه - 1, v_off, v_بهر_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ګام);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

ساكن فراغ core_dot_chunk_fp16_short(__fp16 *محدود output, ثابت __fp16 *محدود activation,
                                ثابت __fp16 *محدود وزن, ثابت __fp16 *محدود scales,
                                uint32_t n_لیکه_tiles, uint32_t n_ستون_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_لیکه_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm متطاير(HMX_سېټ_BIAS("%0") :: "r"((بینښه صحيح)scales));

    ثابت size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ثابت uint32_t range = 2048u * n_dot_tiles - 1;

    لكل (uint32_t r = 0; r < n_لیکه_tiles; ++r) {
        ثابت __fp16 *لیکه_بنسټ = activation + r * dot_stride;
        ثابت __fp16 *ستون_بنسټ = وزن;
        __fp16 *بهر_tile = output + r * n_ستون_tiles * HTP_MM_HMX_TILE_N_ELMS;

        لكل (size_t c = 0; c < n_ستون_tiles; ++c) {
            asm متطاير(HMX_CLRACC_F16());
            asm متطاير(HMX_بارول_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(لیکه_بنسټ), "r"(ستون_بنسټ));
            asm متطاير(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(بهر_tile), "r"(0) : "memory");
            ستون_بنسټ += dot_stride;
            بهر_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ساكن فراغ core_dot_chunk_fp16(__fp16 *محدود output, ثابت __fp16 *محدود activation,
                          ثابت __fp16 *محدود وزن, ثابت __fp16 *محدود scales,
                          uint32_t n_لیکه_tiles, uint32_t n_ستون_tiles, uint32_t n_dot_tiles) {
    إذا (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, وزن, scales, n_لیکه_tiles, n_ستون_tiles, n_dot_tiles);
        أرجع;
    }
    __builtin_assume(n_لیکه_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm متطاير(HMX_سېټ_BIAS("%0") :: "r"((بینښه صحيح)scales));

    ثابت size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    لكل (uint32_t r = 0; r < n_لیکه_tiles; ++r) {
        ثابت __fp16 *لیکه_بنسټ = activation + r * dot_stride;
        ثابت __fp16 *ستون_بنسټ = وزن;
        __fp16 *بهر_tile = output + r * n_ستون_tiles * HTP_MM_HMX_TILE_N_ELMS;

        لكل (size_t c = 0; c < n_ستون_tiles; ++c) {
            ثابت __fp16 *لیکه_tiles = لیکه_بنسټ;
            ثابت __fp16 *ستون_tiles = ستون_بنسټ;

            asm متطاير(HMX_CLRACC_F16());

            ثابت uint32_t n_loops = n_dot_tiles / 32;
            ثابت uint32_t rem = n_dot_tiles % 32;

            لكل (uint32_t l = 0; l < n_loops; ++l) {
                asm متطاير(HMX_بارول_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(لیکه_tiles), "r"(ستون_tiles));
                لیکه_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                ستون_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            إذا (rem > 0) {
                ثابت uint32_t range = 2048u * rem - 1;
                asm متطاير(HMX_بارول_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(لیکه_tiles), "r"(ستون_tiles));
            }

            asm متطاير(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(بهر_tile), "r"(0) : "memory");

            ستون_بنسټ += dot_stride;
            بهر_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ساكن فراغ core_mma_chunk_fp16_short(__fp16 *محدود c, ثابت __fp16 *محدود a, ثابت __fp16 *محدود b,
                                ثابت __fp16 *محدود ستون_scales, ثابت __fp16 *محدود eye_tile,
                                uint32_t n_لیکه_tiles, uint32_t n_ستون_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_لیکه_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm متطاير(HMX_سېټ_BIAS("%0") :: "r"((بینښه صحيح)ستون_scales));

    ثابت size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ثابت uint32_t range = 2048u * n_dot_tiles - 1;

    لكل (size_t i = 0; i < n_لیکه_tiles; ++i) {
        ثابت __fp16 *لیکه_بنسټ = a + i * dot_tile_stride;
        __fp16 *res_بنسټ = c + i * n_ستون_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ثابت __fp16 *ستون_بنسټ = b;
        __fp16 *accum_tile = res_بنسټ;

        لكل (size_t j = 0; j < n_ستون_tiles; ++j) {
            asm متطاير(HMX_CLRACC_F16());

            إذا (!zero_init) {
                asm متطاير(HMX_بارول_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm متطاير(HMX_بارول_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(لیکه_بنسټ), "r"(ستون_بنسټ));

            asm متطاير(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            ستون_بنسټ   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ساكن فراغ core_mma_chunk_fp16(__fp16 *محدود c, ثابت __fp16 *محدود a, ثابت __fp16 *محدود b,
                          ثابت __fp16 *محدود ستون_scales, ثابت __fp16 *محدود eye_tile,
                          uint32_t n_لیکه_tiles, uint32_t n_ستون_tiles, uint32_t n_dot_tiles, bool zero_init) {
    إذا (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, ستون_scales, eye_tile, n_لیکه_tiles, n_ستون_tiles, n_dot_tiles, zero_init);
        أرجع;
    }
    __builtin_assume(n_لیکه_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm متطاير(HMX_سېټ_BIAS("%0") :: "r"((بینښه صحيح)ستون_scales));

    ثابت size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    لكل (size_t i = 0; i < n_لیکه_tiles; ++i) {
        ثابت __fp16 *لیکه_بنسټ = a + i * dot_tile_stride;
        __fp16 *res_بنسټ = c + i * n_ستون_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ثابت __fp16 *ستون_بنسټ = b;
        __fp16 *accum_tile = res_بنسټ;

        لكل (size_t j = 0; j < n_ستون_tiles; ++j) {
            ثابت __fp16 *ستون_tiles = ستون_بنسټ;
            ثابت __fp16 *لیکه_tiles = لیکه_بنسټ;

            asm متطاير(HMX_CLRACC_F16());

            إذا (!zero_init) {
                asm متطاير(HMX_بارول_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            ثابت uint32_t n_loops = n_dot_tiles / 32;
            ثابت uint32_t rem = n_dot_tiles % 32;

            لكل (uint32_t l = 0; l < n_loops; ++l) {
                asm متطاير(HMX_بارول_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(لیکه_tiles), "r"(ستون_tiles));
                لیکه_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                ستون_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            إذا (rem > 0) {
                ثابت uint32_t range = 2048u * rem - 1;
                asm متطاير(HMX_بارول_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(لیکه_tiles), "r"(ستون_tiles));
            }

            asm متطاير(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            ستون_بنسټ += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32p

ساكن فراغ transfer_output_chunk_fp16_منزل_fp32_ستون_chunk(
    عائم *محدود dst,
    ثابت عائم *محدود src2,
    ثابت __fp16 *محدود vtcm_src,
    uint32_t پیل_لیکه,
    uint32_t n_لیکې,
    uint32_t c_len,
    uint32_t ټول_n_ستون_شمېر,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_ستون_شمېر
) {
    ثابت size_t tile_لیکه_stride = hmx_ceil_div(ټول_n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر) * HTP_MM_HMX_TILE_N_ELMS;

    ثابت HVX_ویکټور one = hvx_vec_splat_f16(1.0);

    ثابت size_t حد_c         = hex_smin(c_len, dst_ستون_شمېر);
    ثابت size_t حد_c_aligned = (حد_c & ~31);

    لكل (size_t r = 0; r < n_لیکې; r += 2) {
        ثابت size_t r_idx0 = پیل_لیکه + r + 0;
        ثابت size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_لیکې;
        ثابت size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_لیکې) / 2;  // index of the row pair within the tile
        ثابت __fp16 *لیکه_بنسټ = vtcm_src + r0 * tile_لیکه_stride;
        عائم *output_لیکه_بنسټ = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        ثابت عائم *src2_لیکه_بنسټ = src2 ? (src2 + r * src2_stride) : عدم;

        #لارښوونه unroll(4)
        لكل (size_t c = 0; c < حد_c_aligned; c += HTP_MM_HMX_TILE_N_ستون_شمېر) {
            ثابت size_t c0    = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            ثابت __fp16 *tile = لیکه_بنسټ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ویکټور v = ((ثابت HVX_ویکټور *) tile)[r1];
            HVX_ویکټور_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_ویکټور v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            إذا (src2_لیکه_بنسټ) {
                HVX_ویکټور v_src2_0 = hvx_vmemu(src2_لیکه_بنسټ + c + 0);
                v_out0 = hvx_vec_جمع_کړه_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_لیکه_بنسټ + c + 0) = v_out0;

            إذا (r + 1 < n_لیکې) {
                HVX_ویکټور v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                إذا (src2_لیکه_بنسټ) {
                    HVX_ویکټور v_src2_1 = hvx_vmemu(src2_لیکه_بنسټ + c + src2_stride);
                    v_out1 = hvx_vec_جمع_کړه_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_لیکه_بنسټ + c + dst_stride) = v_out1;
            }
        }

        إذا (حد_c_aligned < حد_c) {
            size_t c = حد_c_aligned;
            size_t valid_c = حد_c - c;
            ثابت size_t c0 = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            ثابت __fp16 *tile = لیکه_بنسټ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ویکټور v = ((ثابت HVX_ویکټور *) tile)[r1];
            HVX_ویکټور_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_ویکټور v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            إذا (src2_لیکه_بنسټ) {
                HVX_ویکټور v_src2_0 = hvx_vmemu(src2_لیکه_بنسټ + c + 0);
                v_out0 = hvx_vec_جمع_کړه_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_لیکه_بنسټ + c, valid_c * حجم(عائم), v_out0);

            إذا (r + 1 < n_لیکې) {
                HVX_ویکټور v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                إذا (src2_لیکه_بنسټ) {
                    HVX_ویکټور v_src2_1 = hvx_vmemu(src2_لیکه_بنسټ + c + src2_stride);
                    v_out1 = hvx_vec_جمع_کړه_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_لیکه_بنسټ + c + dst_stride, valid_c * حجم(عائم), v_out1);
            }
        }
    }
}

ساكن مضمن فراغ transfer_output_chunk_fp16_منزل_fp32(
    عائم *محدود dst,
    ثابت عائم *محدود src2,
    ثابت __fp16 *محدود vtcm_src,
    uint32_t پیل_لیکه,
    uint32_t n_لیکې,
    uint32_t n_ستون_شمېر,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_ستون_شمېر
) {
    transfer_output_chunk_fp16_منزل_fp32_ستون_chunk(
        dst, src2, vtcm_src, پیل_لیکه, n_لیکې, n_ستون_شمېر, n_ستون_شمېر, dst_stride, src2_stride, dst_ستون_شمېر
    );
}

تعريفنوع بنية {
    ثابت __fp16  *vtcm_src;
    عائم         *dst;
    ثابت عائم   *src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_فی_task;
    uint32_t       n_ستون_شمېر;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_ستون_شمېر;    // Actual output columns
    بنية htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

ساكن فراغ transfer_activation_لیکه_pair_fp32_منزل_fp16(
        __fp16 *محدود vtcm_dst,
        ثابت عائم *محدود row0,
        ثابت عائم *محدود row1,
        uint32_t r,
        uint32_t k_بلاک,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

    uint32_t c = 0;
    لكل (; c + 32 <= k_valid; c += 32) {
        HVX_ویکټور v0 = Q6_V_vzero();
        HVX_ویکټور v1 = Q6_V_vzero();
        إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0 + c);
        إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1 + c);

        HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;  // tile column index
        uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

        HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_بهر;
    }
    إذا (c < k_بلاک) {
        HVX_ویکټور v0 = Q6_V_vzero();
        HVX_ویکټور v1 = Q6_V_vzero();
        إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0 + c);
        إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem > 0 ? rem * حجم(عائم) : 0);
        v0 = Q6_V_vmux_QVV(ماسک, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ماسک, v1, Q6_V_vzero());

        HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;  // tile column index
        uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

        HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_بهر;
    }
}

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
ساكن فراغ transfer_activation_لیکه_pair_f16_منزل_f16(__fp16 * محدود vtcm_dst,
                                                    ثابت __fp16 * محدود row0,
                                                    ثابت __fp16 * محدود row1,
                                                    uint32_t r,
                                                    uint32_t k_بلاک,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

    ثابت uint32_t n_tile_ستون_شمېر = k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر;
    __fp16 * محدود tile_لیکه = vtcm_dst + (size_t) r0 * n_tile_ستون_شمېر * HTP_MM_HMX_TILE_N_ELMS;

    ثابت HVX_UVector * pv0 = row0_valid ? (ثابت HVX_UVector *) row0 : عدم;
    ثابت HVX_UVector * pv1 = row1_valid ? (ثابت HVX_UVector *) row1 : عدم;

    uint32_t c = 0;
    لكل (; c + 64 <= k_valid; c += 64) {
        HVX_ویکټور     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_ویکټور     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_ستون_شمېر;

        HVX_ویکټور * tile0 = (HVX_ویکټور *) (tile_لیکه + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_ویکټور * tile1 = (HVX_ویکټور *) (tile_لیکه + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    لكل (; c < k_بلاک; c += 64) {
        HVX_ویکټور v0 = Q6_V_vzero();
        HVX_ویکټور v1 = Q6_V_vzero();

        إذا (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem * حجم(__fp16));
            إذا (pv0) {
                v0 = Q6_V_vmux_QVV(ماسک, pv0[c / 64], Q6_V_vzero());
            }
            إذا (pv1) {
                v1 = Q6_V_vmux_QVV(ماسک, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_ستون_شمېر;

        HVX_ویکټور * tile0 = (HVX_ویکټور *) (tile_لیکه + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        إذا (c0 + 1 < n_tile_ستون_شمېر) {
            HVX_ویکټور * tile1 = (HVX_ویکټور *) (tile_لیکه + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

ساكن فراغ transfer_activation_لیکه_pair_fp32_منزل_fp16_ستون_chunk(
        __fp16 *محدود vtcm_dst,
        ثابت عائم *محدود row0, // offset by c_first
        ثابت عائم *محدود row1, // offset by c_first
        uint32_t r,
        uint32_t k_بلاک,
        uint32_t c_لومړی,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

    uint32_t c = 0;
    لكل (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_ویکټور v0 = Q6_V_vzero();
        HVX_ویکټور v1 = Q6_V_vzero();
        إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0 + c);
        إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1 + c);

        HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

        uint32_t c0       = (c_لومړی + c) / HTP_MM_HMX_TILE_N_ستون_شمېر;  // tile column index
        uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

        HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_بهر;
    }
    إذا (c < c_len) {
        HVX_ویکټور v0 = Q6_V_vzero();
        HVX_ویکټور v1 = Q6_V_vzero();
        إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0 + c);
        إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem > 0 ? rem * حجم(عائم) : 0);
        v0 = Q6_V_vmux_QVV(ماسک, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ماسک, v1, Q6_V_vzero());

        HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

        uint32_t c0       = (c_لومړی + c) / HTP_MM_HMX_TILE_N_ستون_شمېر;  // tile column index
        uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

        HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_بهر;
    }
}

ساكن فراغ transfer_activation_chunk_fp32_منزل_fp16_gathered(
            __fp16 *محدود vtcm_dst,
            ثابت عائم *محدود src,
            uint32_t پیل_لیکه,
            uint32_t vtcm_پیل_لیکه,
            uint32_t n_لیکې,
            uint32_t k_بلاک,
            ثابت بنية mmid_لیکه_mapping *ماتریکس_لیکې,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            ثابت بنية fastdiv_ارزښتونه * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ثابت uint32_t n_لیکې_padded = hex_align_پورته(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);
    ثابت uint32_t n_لیکې_tiled  = (n_لیکې / HTP_MM_HMX_TILE_N_لیکې) * HTP_MM_HMX_TILE_N_لیکې;

    uint32_t r = 0;

    #لارښوونه unroll(2)
    لكل (r = 0; r < n_لیکې_tiled; r += 2) {
        uint32_t r_idx0 = پیل_لیکه + r + 0;
        uint32_t r_idx1 = پیل_لیکه + r + 1;
        uint32_t lr = vtcm_پیل_لیکه + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

        بنية mmid_لیکه_mapping mapping0 = ماتریکس_لیکې[cur_a * mapping_stride + r_idx0];
        بنية mmid_لیکه_mapping mapping1 = ماتریکس_لیکې[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        ثابت عائم *row0_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        ثابت عائم *row1_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        لكل (; c + 32 <= k_valid; c += 32) {
            HVX_ویکټور v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            HVX_ویکټور v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);
            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
        إذا (c < k_بلاک) {
            HVX_ویکټور v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            HVX_ویکټور v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem > 0 ? rem * حجم(عائم) : 0);
            v0 = Q6_V_vmux_QVV(ماسک, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ماسک, v1, Q6_V_vzero());

            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
    }

    لكل (; r < n_لیکې_padded; r += 2) {
        uint32_t lr = vtcm_پیل_لیکه + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

        ثابت bool row0_valid = (پیل_لیکه + r + 0) < cne1;
        ثابت bool row1_valid = (پیل_لیکه + r + 1) < cne1;

        ثابت عائم *row0_ptr = عدم;
        ثابت عائم *row1_ptr = عدم;

        إذا (row0_valid) {
            بنية mmid_لیکه_mapping mapping0 = ماتریکس_لیکې[cur_a * mapping_stride + (پیل_لیکه + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        إذا (row1_valid) {
            بنية mmid_لیکه_mapping mapping1 = ماتریکس_لیکې[cur_a * mapping_stride + (پیل_لیکه + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        لكل (; c + 32 <= k_valid; c += 32) {
            HVX_ویکټور v0 = Q6_V_vzero();
            HVX_ویکټور v1 = Q6_V_vzero();
            إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);

            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
        إذا (c < k_بلاک) {
            HVX_ویکټور v0 = Q6_V_vzero();
            HVX_ویکټور v1 = Q6_V_vzero();
            إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem > 0 ? rem * حجم(عائم) : 0);
            v0 = Q6_V_vmux_QVV(ماسک, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ماسک, v1, Q6_V_vzero());

            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
    }
}

ساكن فراغ transfer_activation_chunk_fp32_منزل_fp16_gathered_flat(
            __fp16 *محدود vtcm_dst,
            ثابت عائم *محدود src,
            uint32_t پیل_لیکه,
            uint32_t vtcm_پیل_لیکه,
            uint32_t n_لیکې,
            uint32_t k_بلاک,
            ثابت بنية mmid_لیکه_mapping *ماتریکس_لیکې,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ثابت uint32_t n_لیکې_padded = hex_align_پورته(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);
    ثابت uint32_t n_لیکې_tiled  = (n_لیکې / HTP_MM_HMX_TILE_N_لیکې) * HTP_MM_HMX_TILE_N_لیکې;

    uint32_t r = 0;

    #لارښوونه unroll(2)
    لكل (r = 0; r < n_لیکې_tiled; r += 2) {
        uint32_t r_idx0 = پیل_لیکه + r + 0;
        uint32_t r_idx1 = پیل_لیکه + r + 1;
        uint32_t lr = vtcm_پیل_لیکه + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

        بنية mmid_لیکه_mapping mapping0 = ماتریکس_لیکې[cur_a * mapping_stride + r_idx0];
        بنية mmid_لیکه_mapping mapping1 = ماتریکس_لیکې[cur_a * mapping_stride + r_idx1];

        ثابت عائم *row0_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + mapping0.i2 * nb12);
        ثابت عائم *row1_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        لكل (; c + 32 <= k_valid; c += 32) {
            HVX_ویکټور v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            HVX_ویکټور v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);
            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
        إذا (c < k_بلاک) {
            HVX_ویکټور v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            HVX_ویکټور v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem > 0 ? rem * حجم(عائم) : 0);
            v0 = Q6_V_vmux_QVV(ماسک, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ماسک, v1, Q6_V_vzero());

            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
    }

    لكل (; r < n_لیکې_padded; r += 2) {
        uint32_t lr = vtcm_پیل_لیکه + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_لیکې;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_لیکې;  // intra-tile row idx

        ثابت bool row0_valid = (پیل_لیکه + r + 0) < cne1;
        ثابت bool row1_valid = (پیل_لیکه + r + 1) < cne1;

        ثابت عائم *row0_ptr = عدم;
        ثابت عائم *row1_ptr = عدم;

        إذا (row0_valid) {
            بنية mmid_لیکه_mapping mapping0 = ماتریکس_لیکې[cur_a * mapping_stride + (پیل_لیکه + r + 0)];
            row0_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + mapping0.i2 * nb12);
        }
        إذا (row1_valid) {
            بنية mmid_لیکه_mapping mapping1 = ماتریکس_لیکې[cur_a * mapping_stride + (پیل_لیکه + r + 1)];
            row1_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        لكل (; c + 32 <= k_valid; c += 32) {
            HVX_ویکټور v0 = Q6_V_vzero();
            HVX_ویکټور v1 = Q6_V_vzero();
            إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);

            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
        إذا (c < k_بلاک) {
            HVX_ویکټور v0 = Q6_V_vzero();
            HVX_ویکټور v1 = Q6_V_vzero();
            إذا (row0_valid) v0 = *(ثابت HVX_ویکټور *)(row0_ptr + c);
            إذا (row1_valid) v1 = *(ثابت HVX_ویکټور *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(rem > 0 ? rem * حجم(عائم) : 0);
            v0 = Q6_V_vmux_QVV(ماسک, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ماسک, v1, Q6_V_vzero());

            HVX_ویکټور v_بهر = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            uint32_t tile_idx = r0 * (k_بلاک / HTP_MM_HMX_TILE_N_ستون_شمېر) + c0;

            HVX_ویکټور *tile = (HVX_ویکټور *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_بهر;
        }
    }
}

ساكن فراغ transfer_output_chunk_fp16_منزل_fp32_scattered(
            عائم *محدود dst,
            ثابت __fp16 *محدود vtcm_src,
            uint32_t پیل_لیکه,
            uint32_t vtcm_پیل_لیکه,
            uint32_t n_لیکې,
            uint32_t n_ستون_شمېر,
            ثابت بنية mmid_لیکه_mapping *ماتریکس_لیکې,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    أكد(n_ستون_شمېر % HTP_MM_HMX_TILE_N_ستون_شمېر == 0);
    ثابت size_t tile_لیکه_stride = (n_ستون_شمېر / HTP_MM_HMX_TILE_N_ستون_شمېر) * HTP_MM_HMX_TILE_N_ELMS;

    ثابت HVX_ویکټور one = hvx_vec_splat_f16(1.0);

    لكل (size_t r = 0; r < n_لیکې; r += 2) {
        uint32_t r_idx0 = پیل_لیکه + r + 0;
        uint32_t r_idx1 = پیل_لیکه + r + 1;
        uint32_t     lr = vtcm_پیل_لیکه + r;                 // vtcm-local row
        ثابت size_t r0 = (lr / HTP_MM_HMX_TILE_N_لیکې);
        ثابت size_t r1 = (lr % HTP_MM_HMX_TILE_N_لیکې) / 2;  // index of the row pair within the tile
        ثابت __fp16 *لیکه_بنسټ = vtcm_src + r0 * tile_لیکه_stride;

        إذا (r_idx0 >= cne1) اكسر;

        بنية mmid_لیکه_mapping mapping0 = ماتریکس_لیکې[cur_a * mapping_stride + r_idx0];
        عائم *output_row0 = (عائم *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        عائم *output_row1 = عدم;
        إذا (r_idx1 < cne1) {
            بنية mmid_لیکه_mapping mapping1 = ماتریکس_لیکې[cur_a * mapping_stride + r_idx1];
            output_row1 = (عائم *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #لارښوونه unroll(4)
        لكل (size_t c = 0; c < (size_t)n_ستون_شمېر; c += HTP_MM_HMX_TILE_N_ستون_شمېر) {
            ثابت size_t c0 = c / HTP_MM_HMX_TILE_N_ستون_شمېر;
            ثابت __fp16 *tile = لیکه_بنسټ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ویکټور v = ((ثابت HVX_ویکټور *) tile)[r1];
            HVX_ویکټور_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            إذا (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
