#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-queue.h"

// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
ਸਥਿਰ ਅਚਲ __fp16 mxfp4_ਮੰਜ਼ਿਲ_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

ਸਥਿਰ ਅਚਲ __fp16 iq4_nl_ਮੰਜ਼ਿਲ_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ tiled_dequantize_state tiled_dequantize_state_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*tiled_dequant_task_fn_t)(ਅਚਲ tiled_dequantize_state_t * state, uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile);

ਬਣਤਰ tiled_dequantize_state {
    ਬਣਤਰ htp_context      * ctx;
    ਬਣਤਰ htp_thread_trace * traces;
    __fp16                  * dst;
    ਅਚਲ uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_ਭਾਰ_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_ਪ੍ਰਤੀ_task;
    uint32_t                  tile_ਮਾਪ;
    uint32_t                  aligned_tile_ਮਾਪ;
    uint32_t                  n_tasks;
    uint32_t                  n_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t                  k_ਖੰਡ;
    size_t                    ਪੰਕਤੀ_stride;
    uint32_t                  ਭਾਰ_type;
};

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q1_0(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ v_neg_one = Q6_Vb_vsplat_R(-1);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_pos_one = Q6_Vb_vsplat_R(1);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        ਅਚਲ HVX_ਸਦਿਸ਼ v_sc = hvx_vmem(tile_src + 512);
        ਅਚਲ HVX_ਸਦਿਸ਼ v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        ਲਈ (uint32_t q = 0; q < 4; q++) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ g = 0; g < 8; g++) {
                ਅਚਲ HVX_ਸਦਿਸ਼     v_q  = unpack_q1_0_ਸਮੂਹ(v_plane, g, v_neg_one, v_pos_one);
                ਅਚਲ HVX_ਸਦਿਸ਼_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                ਅਚਲ HVX_ਸਦਿਸ਼_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q4_0(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_h4 = Q6_Vb_vsplat_R(0x0F);
    ਅਚਲ HVX_ਸਦਿਸ਼ i8 = Q6_Vb_vsplat_R(8);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v_sc = hvx_vmem(tile_src + 512);
        HVX_ਸਦਿਸ਼ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_ਸਦਿਸ਼ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ਸਦਿਸ਼ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ਸਦਿਸ਼ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ਸਦਿਸ਼ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ਸਦਿਸ਼ v_lo0 = Q6_V_vand_VV(vq0, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ਸਦਿਸ਼ v_lo1 = Q6_V_vand_VV(vq1, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ਸਦਿਸ਼ v_lo2 = Q6_V_vand_VV(vq2, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ਸਦਿਸ਼ v_lo3 = Q6_V_vand_VV(vq3, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

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
        HVX_ਸਦਿਸ਼_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_ਸਦਿਸ਼ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

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

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q4_1(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_h4 = Q6_Vb_vsplat_R(0x0F);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ vscale_ਵਿੱਥ = hvx_vmem(tile_src + 512);
        HVX_ਸਦਿਸ਼_Pair dm_deal = Q6_W_vdeal_VVR(vscale_ਵਿੱਥ, vscale_ਵਿੱਥ, -2);
        HVX_ਸਦਿਸ਼ vd = Q6_V_lo_W(dm_deal);
        HVX_ਸਦਿਸ਼ vm = Q6_V_hi_W(dm_deal);

        HVX_ਸਦਿਸ਼ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_ਸਦਿਸ਼ v_ਵਿੱਥ_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_ਸਦਿਸ਼ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ਸਦਿਸ਼ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ਸਦਿਸ਼ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ਸਦਿਸ਼ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ਸਦਿਸ਼ v_lo0 = Q6_V_vand_VV(vq0, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ਸਦਿਸ਼ v_lo1 = Q6_V_vand_VV(vq1, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ਸਦਿਸ਼ v_lo2 = Q6_V_vand_VV(vq2, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ਸਦਿਸ਼ v_lo3 = Q6_V_vand_VV(vq3, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ਸਦਿਸ਼_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_ਸਦਿਸ਼ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

        HVX_ਸਦਿਸ਼ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

        HVX_ਸਦਿਸ਼ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

        HVX_ਸਦਿਸ਼ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

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

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_iq4_nl(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_h4 = Q6_Vb_vsplat_R(0x0F);
    ਅਚਲ HVX_ਸਦਿਸ਼ vlut_cvt = hvx_vmem(iq4_nl_ਮੰਜ਼ਿਲ_fp16_lut);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v_sc = hvx_vmem(tile_src + 512);
        HVX_ਸਦਿਸ਼ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_ਸਦਿਸ਼ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ਸਦਿਸ਼ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ਸਦਿਸ਼ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ਸਦਿਸ਼ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ਸਦਿਸ਼ v_lo0 = Q6_V_vand_VV(vq0, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ਸਦਿਸ਼ v_lo1 = Q6_V_vand_VV(vq1, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ਸਦਿਸ਼ v_lo2 = Q6_V_vand_VV(vq2, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ਸਦਿਸ਼ v_lo3 = Q6_V_vand_VV(vq3, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ਸਦਿਸ਼_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_ਸਦਿਸ਼ v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ਸਦਿਸ਼ v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ਸਦਿਸ਼ v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ਸਦਿਸ਼ v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ਸਦਿਸ਼ v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ਸਦਿਸ਼ v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ਸਦਿਸ਼ v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ਸਦਿਸ਼ v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_ਸਦਿਸ਼_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_ਸਦਿਸ਼ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_mxfp4(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_h4 = Q6_Vb_vsplat_R(0x0F);
    ਅਚਲ HVX_ਸਦਿਸ਼ vlut_cvt = hvx_vmem(mxfp4_ਮੰਜ਼ਿਲ_fp16_lut);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v = hvx_vmem(tile_src + 512);
        HVX_ਸਦਿਸ਼ vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_ਸਦਿਸ਼ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_ਸਦਿਸ਼ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ਸਦਿਸ਼ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ਸਦਿਸ਼ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ਸਦਿਸ਼ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ਸਦਿਸ਼ v_lo0 = Q6_V_vand_VV(vq0, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ਸਦਿਸ਼ v_lo1 = Q6_V_vand_VV(vq1, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ਸਦਿਸ਼ v_lo2 = Q6_V_vand_VV(vq2, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ਸਦਿਸ਼ v_lo3 = Q6_V_vand_VV(vq3, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_ਸਦਿਸ਼_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_ਸਦਿਸ਼ v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ਸਦਿਸ਼ v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ਸਦਿਸ਼ v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ਸਦਿਸ਼ v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ਸਦਿਸ਼ v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ਸਦਿਸ਼ v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ਸਦਿਸ਼ v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ਸਦਿਸ਼ v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_ਸਦਿਸ਼_Pair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_ਸਦਿਸ਼_Pair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_ਸਦਿਸ਼ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_ਸਦਿਸ਼ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

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

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q8_0(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v_sc = hvx_vmem(tile_src + 1024);
        HVX_ਸਦਿਸ਼ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_ਸਦਿਸ਼ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ਸਦਿਸ਼ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ਸਦਿਸ਼ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ਸਦਿਸ਼ vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_ਸਦਿਸ਼_Pair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_ਸਦਿਸ਼_Pair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_ਸਦਿਸ਼_Pair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_ਸਦਿਸ਼_Pair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_ਸਦਿਸ਼ vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_ਸਦਿਸ਼ vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_ਸਦਿਸ਼ vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_ਸਦਿਸ਼ vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_ਸਦਿਸ਼_Pair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_ਸਦਿਸ਼_Pair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_ਸਦਿਸ਼_Pair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_ਸਦਿਸ਼_Pair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_ਸਦਿਸ਼ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

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
        HVX_ਸਦਿਸ਼ v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_ਸਦਿਸ਼ v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

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

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q5_k(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_h4 = Q6_Vb_vsplat_R(0x0F);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile_src = state->src + t * state->aligned_tile_ਮਾਪ;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ vscale_ਵਿੱਥ = hvx_vmem(tile_src + 512);
        HVX_ਸਦਿਸ਼_Pair dm_deal = Q6_W_vdeal_VVR(vscale_ਵਿੱਥ, vscale_ਵਿੱਥ, -2);
        HVX_ਸਦਿਸ਼ vd = Q6_V_lo_W(dm_deal);
        HVX_ਸਦਿਸ਼ vm = Q6_V_hi_W(dm_deal);

        HVX_ਸਦਿਸ਼ v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_ਸਦਿਸ਼ v_ਵਿੱਥ_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_ਸਦਿਸ਼ vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_ਸਦਿਸ਼ vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_ਸਦਿਸ਼ vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_ਸਦਿਸ਼ vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_ਸਦਿਸ਼ v_lo0 = Q6_V_vand_VV(vq0, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_ਸਦਿਸ਼ v_lo1 = Q6_V_vand_VV(vq1, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_ਸਦਿਸ਼ v_lo2 = Q6_V_vand_VV(vq2, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_ਸਦਿਸ਼ v_lo3 = Q6_V_vand_VV(vq3, ਨਕਾਬ_h4);
        HVX_ਸਦਿਸ਼ v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Q5_K: OR in the 5th bit from the plane
        HVX_ਸਦਿਸ਼ v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

        // Shuffling
        HVX_ਸਦਿਸ਼_Pair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_ਸਦਿਸ਼_Pair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_ਸਦਿਸ਼_Pair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_ਸਦਿਸ਼_Pair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_ਸਦਿਸ਼ v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

        HVX_ਸਦਿਸ਼ v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

        HVX_ਸਦਿਸ਼ v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

        HVX_ਸਦਿਸ਼ v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));
        HVX_ਸਦਿਸ਼ v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_ਵਿੱਥ_duplicated));

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
ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q6_k(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_0f = Q6_Vb_vsplat_R(0x0F);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);
    ਅਚਲ HVX_ਸਦਿਸ਼ i32     = Q6_Vb_vsplat_R(32);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ HVX_ਸਦਿਸ਼ * vptr = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (state->src + t * state->aligned_tile_ਮਾਪ);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v_sc      = vptr[6];
        HVX_ਸਦਿਸ਼ v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_ਸਦਿਸ਼ v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ਸਦਿਸ਼ v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ g = 0; g < 8; g += 4) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_ਸਦਿਸ਼ v_q0 = unpack_q6_k_ਸਮੂਹ(vptr, g + 0, ਨਕਾਬ_0f, ਨਕਾਬ_03, i32);
            HVX_ਸਦਿਸ਼ v_q1 = unpack_q6_k_ਸਮੂਹ(vptr, g + 1, ਨਕਾਬ_0f, ਨਕਾਬ_03, i32);
            HVX_ਸਦਿਸ਼ v_q2 = unpack_q6_k_ਸਮੂਹ(vptr, g + 2, ਨਕਾਬ_0f, ਨਕਾਬ_03, i32);
            HVX_ਸਦਿਸ਼ v_q3 = unpack_q6_k_ਸਮੂਹ(vptr, g + 3, ਨਕਾਬ_0f, ਨਕਾਬ_03, i32);

            HVX_ਸਦਿਸ਼_Pair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_ਸਦਿਸ਼_Pair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_ਸਦਿਸ਼_Pair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_ਸਦਿਸ਼_Pair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_ਸਦਿਸ਼_Pair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_ਸਦਿਸ਼_Pair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_ਸਦਿਸ਼_Pair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_ਸਦਿਸ਼_Pair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_ਸਦਿਸ਼ v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_ਸਦਿਸ਼ v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_ਸਦਿਸ਼ v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_ਸਦਿਸ਼ v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_ਸਦਿਸ਼ v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_ਸਦਿਸ਼ v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_ਸਦਿਸ਼ v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_ਸਦਿਸ਼ v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

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
ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q3_k(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ HVX_ਸਦਿਸ਼ * vptr = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (state->src + t * state->aligned_tile_ਮਾਪ);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v_sc      = vptr[3];
        HVX_ਸਦਿਸ਼ v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_ਸਦਿਸ਼ v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ਸਦਿਸ਼ v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ g = 0; g < 8; g++) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_ਸਦਿਸ਼     v_q   = unpack_q3_k_ਸਮੂਹ(vptr, g, ਨਕਾਬ_03);
            HVX_ਸਦਿਸ਼_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_ਸਦਿਸ਼_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q2_k(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);

    HVX_ਸਦਿਸ਼ v_d = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ v_dm = Q6_V_vzero();

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ uint8_t * tile = state->src + t * state->aligned_tile_ਮਾਪ;
        ਅਚਲ HVX_ਸਦਿਸ਼ * vptr = (ਅਚਲ HVX_ਸਦਿਸ਼ *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        ਜੇ (t == ਸ਼ੁਰੂ_tile || (t & 7) == 0) {
            q2_k_ਲੱਦੋ_dmin(tile - (t & 7) * state->aligned_tile_ਮਾਪ, &v_d, &v_dm);
        }

        HVX_ਸਦਿਸ਼ v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_ਸਦਿਸ਼ v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_ਸਦਿਸ਼ v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_ਸਦਿਸ਼ v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_ਸਦਿਸ਼ v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_ਸਦਿਸ਼ v_ਵਿੱਥ_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_ਸਦਿਸ਼ v_ਵਿੱਥ_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ g = 0; g < 8; g++) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            ਅਚਲ HVX_ਸਦਿਸ਼ v_ਵਿੱਥ = (g < 4) ? v_ਵਿੱਥ_k0 : v_ਵਿੱਥ_k16;

            HVX_ਸਦਿਸ਼     v_q   = unpack_q3_k_low2(vptr, g, ਨਕਾਬ_03);
            HVX_ਸਦਿਸ਼_Pair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_ਸਦਿਸ਼_Pair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_ਵਿੱਥ));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_ਵਿੱਥ));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q2_0(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);
    ਅਚਲ HVX_ਸਦਿਸ਼ i1      = Q6_Vb_vsplat_R(1);

    ਲਈ (uint32_t t = ਸ਼ੁਰੂ_tile; t < ਸਿਰਾ_tile; t++) {
        ਅਚਲ HVX_ਸਦਿਸ਼ * vptr = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (state->src + t * state->aligned_tile_ਮਾਪ);
        __fp16 * dst_ਅਧਾਰ = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_ਸਦਿਸ਼ v_sc    = vptr[4];
        HVX_ਸਦਿਸ਼ v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        ਲਈ (ਪੂਰਨ half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_ਅਧਾਰ + half * HTP_MM_HMX_TILE_N_ELMS;

            #ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ g = 0; g < 8; g++) {
                HVX_ਸਦਿਸ਼     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, ਨਕਾਬ_03), i1);
                HVX_ਸਦਿਸ਼_Pair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_ਸਦਿਸ਼_Pair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

ਸਥਿਰ __attribute__((noinline))
ਖਾਲੀ convert_f16_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_tiles_task(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ uint32_t n_k_tiles = state->n_k_tiles;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div = state->n_k_tiles_div;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_scat_ਅਧਾਰ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_scat_ਕਦਮ  = Q6_V_vsplat_R(4);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_mask64 = Q6_Q_vsetq_R(64);

    ਬਿਨਾਨਿਸ਼ਾਨੀ ct = fastdiv((ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸ਼ੁਰੂ_tile, &n_k_tiles_div);
    ਬਿਨਾਨਿਸ਼ਾਨੀ kt = fastmodulo((ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸ਼ੁਰੂ_tile, n_k_tiles, &n_k_tiles_div);

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ t = ਸ਼ੁਰੂ_tile; t < (ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸਿਰਾ_tile; ) {
        ਜੇ (kt >= (ਬਿਨਾਨਿਸ਼ਾਨੀ)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_ਅਧਾਰ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * ਆਕਾਰ(__fp16);

            HVX_ਸਦਿਸ਼ v_off = v_scat_ਅਧਾਰ;
            ਲਈ (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ + r;
                uint32_t row1 = row0 + 1;

                ਅਚਲ uint8_t *r0 = state->src + row0 * state->ਪੰਕਤੀ_stride;
                ਅਚਲ uint8_t *r1 = state->src + row1 * state->ਪੰਕਤੀ_stride;

                HVX_ਸਦਿਸ਼ v0 = (row0 < state->n_ਖਾਨੇ_ਗਿਣਤੀ) ? hvx_vmemu((ਅਚਲ __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ਸਦਿਸ਼ v1 = (row1 < state->n_ਖਾਨੇ_ਗਿਣਤੀ) ? hvx_vmemu((ਅਚਲ __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ਅਧਾਰ, HTP_MM_HMX_TILE_ਮਾਪ - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ਕਦਮ);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ਅਧਾਰ, HTP_MM_HMX_TILE_ਮਾਪ - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ਕਦਮ);
            }
        }
        ++t; ++kt;
    }
}

ਸਥਿਰ __attribute__((noinline))
ਖਾਲੀ quantize_f32_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_tiles_task(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ uint32_t n_k_tiles = state->n_k_tiles;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div = state->n_k_tiles_div;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_scat_ਅਧਾਰ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_scat_ਕਦਮ  = Q6_V_vsplat_R(4);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_mask64 = Q6_Q_vsetq_R(64);

    ਬਿਨਾਨਿਸ਼ਾਨੀ ct = fastdiv((ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸ਼ੁਰੂ_tile, &n_k_tiles_div);
    ਬਿਨਾਨਿਸ਼ਾਨੀ kt = fastmodulo((ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸ਼ੁਰੂ_tile, n_k_tiles, &n_k_tiles_div);

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ t = ਸ਼ੁਰੂ_tile; t < (ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸਿਰਾ_tile; ) {
        ਜੇ (kt >= (ਬਿਨਾਨਿਸ਼ਾਨੀ)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_ਅਧਾਰ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

            HVX_ਸਦਿਸ਼ v_off = v_scat_ਅਧਾਰ;
            ਲਈ (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ + r;
                uint32_t row1 = row0 + 1;

                ਅਚਲ uint8_t *r0 = state->src + row0 * state->ਪੰਕਤੀ_stride;
                ਅਚਲ uint8_t *r1 = state->src + row1 * state->ਪੰਕਤੀ_stride;

                HVX_ਸਦਿਸ਼ v0_f32 = (row0 < state->n_ਖਾਨੇ_ਗਿਣਤੀ) ? hvx_vmem((ਅਚਲ ਦਸ਼ਮਲਵ *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ਸਦਿਸ਼ v1_f32 = (row1 < state->n_ਖਾਨੇ_ਗਿਣਤੀ) ? hvx_vmem((ਅਚਲ ਦਸ਼ਮਲਵ *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ਅਧਾਰ, HTP_MM_HMX_TILE_ਮਾਪ - 1, v_off, v_ਬਾਹਰਲਾ);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ਕਦਮ);

                HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ_hi = Q6_V_vror_VR(v_ਬਾਹਰਲਾ, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ਅਧਾਰ, HTP_MM_HMX_TILE_ਮਾਪ - 1, v_off, v_ਬਾਹਰਲਾ_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ਕਦਮ);
            }
        }
        ++t; ++kt;
    }
}

ਸਥਿਰ __attribute__((noinline))
ਖਾਲੀ convert_bf16_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_tiles_task(
        ਅਚਲ tiled_dequantize_state_t *state,
        uint32_t ਸ਼ੁਰੂ_tile, uint32_t ਸਿਰਾ_tile) {

    ਅਚਲ uint32_t n_k_tiles = state->n_k_tiles;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div = state->n_k_tiles_div;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_scat_ਅਧਾਰ  = hvx_vmem(hmx_transpose_scatter_offsets);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_scat_ਕਦਮ  = Q6_V_vsplat_R(4);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_mask64 = Q6_Q_vsetq_R(64);

    ਬਿਨਾਨਿਸ਼ਾਨੀ ct = fastdiv((ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸ਼ੁਰੂ_tile, &n_k_tiles_div);
    ਬਿਨਾਨਿਸ਼ਾਨੀ kt = fastmodulo((ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸ਼ੁਰੂ_tile, n_k_tiles, &n_k_tiles_div);

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ t = ਸ਼ੁਰੂ_tile; t < (ਬਿਨਾਨਿਸ਼ਾਨੀ)ਸਿਰਾ_tile; ) {
        ਜੇ (kt >= (ਬਿਨਾਨਿਸ਼ਾਨੀ)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_ਅਧਾਰ = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * ਆਕਾਰ(uint16_t);

            HVX_ਸਦਿਸ਼ v_off = v_scat_ਅਧਾਰ;
            ਲਈ (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ + r;
                uint32_t row1 = row0 + 1;

                ਅਚਲ uint8_t *r0 = state->src + row0 * state->ਪੰਕਤੀ_stride;
                ਅਚਲ uint8_t *r1 = state->src + row1 * state->ਪੰਕਤੀ_stride;

                HVX_ਸਦਿਸ਼ v0_raw = (row0 < state->n_ਖਾਨੇ_ਗਿਣਤੀ) ? hvx_vmemu((ਅਚਲ ਖਾਲੀ *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_ਸਦਿਸ਼ v1_raw = (row1 < state->n_ਖਾਨੇ_ਗਿਣਤੀ) ? hvx_vmemu((ਅਚਲ ਖਾਲੀ *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_ਸਦਿਸ਼ v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_ਸਦਿਸ਼ v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ਅਧਾਰ, HTP_MM_HMX_TILE_ਮਾਪ - 1, v_off, v_ਬਾਹਰਲਾ);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ਕਦਮ);

                HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ_hi = Q6_V_vror_VR(v_ਬਾਹਰਲਾ, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_ਅਧਾਰ, HTP_MM_HMX_TILE_ਮਾਪ - 1, v_off, v_ਬਾਹਰਲਾ_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_ਕਦਮ);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

ਸਥਿਰ ਖਾਲੀ core_dot_chunk_fp16_short(__fp16 *ਸੀਮਿਤ output, ਅਚਲ __fp16 *ਸੀਮਿਤ activation,
                                ਅਚਲ __fp16 *ਸੀਮਿਤ ਭਾਰ, ਅਚਲ __fp16 *ਸੀਮਿਤ scales,
                                uint32_t n_ਪੰਕਤੀ_tiles, uint32_t n_ਖਾਨਾ_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(n_ਖਾਨਾ_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)scales));

    ਅਚਲ size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ਅਚਲ uint32_t range = 2048u * n_dot_tiles - 1;

    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀ_tiles; ++r) {
        ਅਚਲ __fp16 *ਪੰਕਤੀ_ਅਧਾਰ = activation + r * dot_stride;
        ਅਚਲ __fp16 *ਖਾਨਾ_ਅਧਾਰ = ਭਾਰ;
        __fp16 *ਬਾਹਰਲਾ_tile = output + r * n_ਖਾਨਾ_tiles * HTP_MM_HMX_TILE_N_ELMS;

        ਲਈ (size_t c = 0; c < n_ਖਾਨਾ_tiles; ++c) {
            asm ਅਸਥਿਰ(HMX_CLRACC_F16());
            asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ਪੰਕਤੀ_ਅਧਾਰ), "r"(ਖਾਨਾ_ਅਧਾਰ));
            asm ਅਸਥਿਰ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(ਬਾਹਰਲਾ_tile), "r"(0) : "memory");
            ਖਾਨਾ_ਅਧਾਰ += dot_stride;
            ਬਾਹਰਲਾ_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ core_dot_chunk_fp16(__fp16 *ਸੀਮਿਤ output, ਅਚਲ __fp16 *ਸੀਮਿਤ activation,
                          ਅਚਲ __fp16 *ਸੀਮਿਤ ਭਾਰ, ਅਚਲ __fp16 *ਸੀਮਿਤ scales,
                          uint32_t n_ਪੰਕਤੀ_tiles, uint32_t n_ਖਾਨਾ_tiles, uint32_t n_dot_tiles) {
    ਜੇ (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, ਭਾਰ, scales, n_ਪੰਕਤੀ_tiles, n_ਖਾਨਾ_tiles, n_dot_tiles);
        ਵਾਪਸ;
    }
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(n_ਖਾਨਾ_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)scales));

    ਅਚਲ size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀ_tiles; ++r) {
        ਅਚਲ __fp16 *ਪੰਕਤੀ_ਅਧਾਰ = activation + r * dot_stride;
        ਅਚਲ __fp16 *ਖਾਨਾ_ਅਧਾਰ = ਭਾਰ;
        __fp16 *ਬਾਹਰਲਾ_tile = output + r * n_ਖਾਨਾ_tiles * HTP_MM_HMX_TILE_N_ELMS;

        ਲਈ (size_t c = 0; c < n_ਖਾਨਾ_tiles; ++c) {
            ਅਚਲ __fp16 *ਪੰਕਤੀ_tiles = ਪੰਕਤੀ_ਅਧਾਰ;
            ਅਚਲ __fp16 *ਖਾਨਾ_tiles = ਖਾਨਾ_ਅਧਾਰ;

            asm ਅਸਥਿਰ(HMX_CLRACC_F16());

            ਅਚਲ uint32_t n_loops = n_dot_tiles / 32;
            ਅਚਲ uint32_t rem = n_dot_tiles % 32;

            ਲਈ (uint32_t l = 0; l < n_loops; ++l) {
                asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(ਪੰਕਤੀ_tiles), "r"(ਖਾਨਾ_tiles));
                ਪੰਕਤੀ_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                ਖਾਨਾ_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            ਜੇ (rem > 0) {
                ਅਚਲ uint32_t range = 2048u * rem - 1;
                asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ਪੰਕਤੀ_tiles), "r"(ਖਾਨਾ_tiles));
            }

            asm ਅਸਥਿਰ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(ਬਾਹਰਲਾ_tile), "r"(0) : "memory");

            ਖਾਨਾ_ਅਧਾਰ += dot_stride;
            ਬਾਹਰਲਾ_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ core_mma_chunk_fp16_short(__fp16 *ਸੀਮਿਤ c, ਅਚਲ __fp16 *ਸੀਮਿਤ a, ਅਚਲ __fp16 *ਸੀਮਿਤ b,
                                ਅਚਲ __fp16 *ਸੀਮਿਤ ਖਾਨਾ_scales, ਅਚਲ __fp16 *ਸੀਮਿਤ eye_tile,
                                uint32_t n_ਪੰਕਤੀ_tiles, uint32_t n_ਖਾਨਾ_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(n_ਖਾਨਾ_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਖਾਨਾ_scales));

    ਅਚਲ size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    ਅਚਲ uint32_t range = 2048u * n_dot_tiles - 1;

    ਲਈ (size_t i = 0; i < n_ਪੰਕਤੀ_tiles; ++i) {
        ਅਚਲ __fp16 *ਪੰਕਤੀ_ਅਧਾਰ = a + i * dot_tile_stride;
        __fp16 *res_ਅਧਾਰ = c + i * n_ਖਾਨਾ_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ਅਚਲ __fp16 *ਖਾਨਾ_ਅਧਾਰ = b;
        __fp16 *accum_tile = res_ਅਧਾਰ;

        ਲਈ (size_t j = 0; j < n_ਖਾਨਾ_tiles; ++j) {
            asm ਅਸਥਿਰ(HMX_CLRACC_F16());

            ਜੇ (!zero_init) {
                asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ਪੰਕਤੀ_ਅਧਾਰ), "r"(ਖਾਨਾ_ਅਧਾਰ));

            asm ਅਸਥਿਰ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            ਖਾਨਾ_ਅਧਾਰ   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ core_mma_chunk_fp16(__fp16 *ਸੀਮਿਤ c, ਅਚਲ __fp16 *ਸੀਮਿਤ a, ਅਚਲ __fp16 *ਸੀਮਿਤ b,
                          ਅਚਲ __fp16 *ਸੀਮਿਤ ਖਾਨਾ_scales, ਅਚਲ __fp16 *ਸੀਮਿਤ eye_tile,
                          uint32_t n_ਪੰਕਤੀ_tiles, uint32_t n_ਖਾਨਾ_tiles, uint32_t n_dot_tiles, bool zero_init) {
    ਜੇ (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, ਖਾਨਾ_scales, eye_tile, n_ਪੰਕਤੀ_tiles, n_ਖਾਨਾ_tiles, n_dot_tiles, zero_init);
        ਵਾਪਸ;
    }
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(n_ਖਾਨਾ_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਖਾਨਾ_scales));

    ਅਚਲ size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    ਲਈ (size_t i = 0; i < n_ਪੰਕਤੀ_tiles; ++i) {
        ਅਚਲ __fp16 *ਪੰਕਤੀ_ਅਧਾਰ = a + i * dot_tile_stride;
        __fp16 *res_ਅਧਾਰ = c + i * n_ਖਾਨਾ_tiles * HTP_MM_HMX_TILE_N_ELMS;
        ਅਚਲ __fp16 *ਖਾਨਾ_ਅਧਾਰ = b;
        __fp16 *accum_tile = res_ਅਧਾਰ;

        ਲਈ (size_t j = 0; j < n_ਖਾਨਾ_tiles; ++j) {
            ਅਚਲ __fp16 *ਖਾਨਾ_tiles = ਖਾਨਾ_ਅਧਾਰ;
            ਅਚਲ __fp16 *ਪੰਕਤੀ_tiles = ਪੰਕਤੀ_ਅਧਾਰ;

            asm ਅਸਥਿਰ(HMX_CLRACC_F16());

            ਜੇ (!zero_init) {
                asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            ਅਚਲ uint32_t n_loops = n_dot_tiles / 32;
            ਅਚਲ uint32_t rem = n_dot_tiles % 32;

            ਲਈ (uint32_t l = 0; l < n_loops; ++l) {
                asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(ਪੰਕਤੀ_tiles), "r"(ਖਾਨਾ_tiles));
                ਪੰਕਤੀ_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                ਖਾਨਾ_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            ਜੇ (rem > 0) {
                ਅਚਲ uint32_t range = 2048u * rem - 1;
                asm ਅਸਥਿਰ(HMX_ਲੱਦੋ_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(ਪੰਕਤੀ_tiles), "r"(ਖਾਨਾ_tiles));
            }

            asm ਅਸਥਿਰ(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            ਖਾਨਾ_ਅਧਾਰ += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32p

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32_ਖਾਨਾ_chunk(
    ਦਸ਼ਮਲਵ *ਸੀਮਿਤ dst,
    ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ src2,
    ਅਚਲ __fp16 *ਸੀਮਿਤ vtcm_src,
    uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
    uint32_t n_ਪੰਕਤੀਆਂ,
    uint32_t c_len,
    uint32_t ਕੁੱਲ_n_ਖਾਨੇ_ਗਿਣਤੀ,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_ਖਾਨੇ_ਗਿਣਤੀ
) {
    ਅਚਲ size_t tile_ਪੰਕਤੀ_stride = hmx_ceil_div(ਕੁੱਲ_n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) * HTP_MM_HMX_TILE_N_ELMS;

    ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);

    ਅਚਲ size_t ਹੱਦਬੰਦੀ_c         = hex_smin(c_len, dst_ਖਾਨੇ_ਗਿਣਤੀ);
    ਅਚਲ size_t ਹੱਦਬੰਦੀ_c_aligned = (ਹੱਦਬੰਦੀ_c & ~31);

    ਲਈ (size_t r = 0; r < n_ਪੰਕਤੀਆਂ; r += 2) {
        ਅਚਲ size_t r_idx0 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0;
        ਅਚਲ size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;
        ਅਚਲ size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ) / 2;  // index of the row pair within the tile
        ਅਚਲ __fp16 *ਪੰਕਤੀ_ਅਧਾਰ = vtcm_src + r0 * tile_ਪੰਕਤੀ_stride;
        ਦਸ਼ਮਲਵ *output_ਪੰਕਤੀ_ਅਧਾਰ = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        ਅਚਲ ਦਸ਼ਮਲਵ *src2_ਪੰਕਤੀ_ਅਧਾਰ = src2 ? (src2 + r * src2_stride) : ਨਿਰਾ;

        #ਹਦਾਇਤ unroll(4)
        ਲਈ (size_t c = 0; c < ਹੱਦਬੰਦੀ_c_aligned; c += HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) {
            ਅਚਲ size_t c0    = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ __fp16 *tile = ਪੰਕਤੀ_ਅਧਾਰ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ਸਦਿਸ਼ v = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) tile)[r1];
            HVX_ਸਦਿਸ਼_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_ਸਦਿਸ਼ v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ਜੇ (src2_ਪੰਕਤੀ_ਅਧਾਰ) {
                HVX_ਸਦਿਸ਼ v_src2_0 = hvx_vmemu(src2_ਪੰਕਤੀ_ਅਧਾਰ + c + 0);
                v_out0 = hvx_vec_ਜੋੜੋ_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_ਪੰਕਤੀ_ਅਧਾਰ + c + 0) = v_out0;

            ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ) {
                HVX_ਸਦਿਸ਼ v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                ਜੇ (src2_ਪੰਕਤੀ_ਅਧਾਰ) {
                    HVX_ਸਦਿਸ਼ v_src2_1 = hvx_vmemu(src2_ਪੰਕਤੀ_ਅਧਾਰ + c + src2_stride);
                    v_out1 = hvx_vec_ਜੋੜੋ_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_ਪੰਕਤੀ_ਅਧਾਰ + c + dst_stride) = v_out1;
            }
        }

        ਜੇ (ਹੱਦਬੰਦੀ_c_aligned < ਹੱਦਬੰਦੀ_c) {
            size_t c = ਹੱਦਬੰਦੀ_c_aligned;
            size_t valid_c = ਹੱਦਬੰਦੀ_c - c;
            ਅਚਲ size_t c0 = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ __fp16 *tile = ਪੰਕਤੀ_ਅਧਾਰ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ਸਦਿਸ਼ v = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) tile)[r1];
            HVX_ਸਦਿਸ਼_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_ਸਦਿਸ਼ v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ਜੇ (src2_ਪੰਕਤੀ_ਅਧਾਰ) {
                HVX_ਸਦਿਸ਼ v_src2_0 = hvx_vmemu(src2_ਪੰਕਤੀ_ਅਧਾਰ + c + 0);
                v_out0 = hvx_vec_ਜੋੜੋ_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_ਪੰਕਤੀ_ਅਧਾਰ + c, valid_c * ਆਕਾਰ(ਦਸ਼ਮਲਵ), v_out0);

            ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ) {
                HVX_ਸਦਿਸ਼ v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                ਜੇ (src2_ਪੰਕਤੀ_ਅਧਾਰ) {
                    HVX_ਸਦਿਸ਼ v_src2_1 = hvx_vmemu(src2_ਪੰਕਤੀ_ਅਧਾਰ + c + src2_stride);
                    v_out1 = hvx_vec_ਜੋੜੋ_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_ਪੰਕਤੀ_ਅਧਾਰ + c + dst_stride, valid_c * ਆਕਾਰ(ਦਸ਼ਮਲਵ), v_out1);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32(
    ਦਸ਼ਮਲਵ *ਸੀਮਿਤ dst,
    ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ src2,
    ਅਚਲ __fp16 *ਸੀਮਿਤ vtcm_src,
    uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
    uint32_t n_ਪੰਕਤੀਆਂ,
    uint32_t n_ਖਾਨੇ_ਗਿਣਤੀ,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_ਖਾਨੇ_ਗਿਣਤੀ
) {
    transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32_ਖਾਨਾ_chunk(
        dst, src2, vtcm_src, ਸ਼ੁਰੂ_ਪੰਕਤੀ, n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ, n_ਖਾਨੇ_ਗਿਣਤੀ, dst_stride, src2_stride, dst_ਖਾਨੇ_ਗਿਣਤੀ
    );
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਅਚਲ __fp16  *vtcm_src;
    ਦਸ਼ਮਲਵ         *dst;
    ਅਚਲ ਦਸ਼ਮਲਵ   *src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_ਪ੍ਰਤੀ_task;
    uint32_t       n_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_ਖਾਨੇ_ਗਿਣਤੀ;    // Actual output columns
    ਬਣਤਰ htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

ਸਥਿਰ ਖਾਲੀ transfer_activation_ਪੰਕਤੀ_pair_fp32_ਮੰਜ਼ਿਲ_fp16(
        __fp16 *ਸੀਮਿਤ vtcm_dst,
        ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ row0,
        ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ row1,
        uint32_t r,
        uint32_t k_ਖੰਡ,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

    uint32_t c = 0;
    ਲਈ (; c + 32 <= k_valid; c += 32) {
        HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
        ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0 + c);
        ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1 + c);

        HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;  // tile column index
        uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

        HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
    }
    ਜੇ (c < k_ਖੰਡ) {
        HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
        ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0 + c);
        ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem > 0 ? rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0);
        v0 = Q6_V_vmux_QVV(ਨਕਾਬ, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ਨਕਾਬ, v1, Q6_V_vzero());

        HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;  // tile column index
        uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

        HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
    }
}

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
ਸਥਿਰ ਖਾਲੀ transfer_activation_ਪੰਕਤੀ_pair_f16_ਮੰਜ਼ਿਲ_f16(__fp16 * ਸੀਮਿਤ vtcm_dst,
                                                    ਅਚਲ __fp16 * ਸੀਮਿਤ row0,
                                                    ਅਚਲ __fp16 * ਸੀਮਿਤ row1,
                                                    uint32_t r,
                                                    uint32_t k_ਖੰਡ,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

    ਅਚਲ uint32_t n_tile_ਖਾਨੇ_ਗਿਣਤੀ = k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
    __fp16 * ਸੀਮਿਤ tile_ਪੰਕਤੀ = vtcm_dst + (size_t) r0 * n_tile_ਖਾਨੇ_ਗਿਣਤੀ * HTP_MM_HMX_TILE_N_ELMS;

    ਅਚਲ HVX_UVector * pv0 = row0_valid ? (ਅਚਲ HVX_UVector *) row0 : ਨਿਰਾ;
    ਅਚਲ HVX_UVector * pv1 = row1_valid ? (ਅਚਲ HVX_UVector *) row1 : ਨਿਰਾ;

    uint32_t c = 0;
    ਲਈ (; c + 64 <= k_valid; c += 64) {
        HVX_ਸਦਿਸ਼     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_ਸਦਿਸ਼     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ * tile0 = (HVX_ਸਦਿਸ਼ *) (tile_ਪੰਕਤੀ + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_ਸਦਿਸ਼ * tile1 = (HVX_ਸਦਿਸ਼ *) (tile_ਪੰਕਤੀ + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    ਲਈ (; c < k_ਖੰਡ; c += 64) {
        HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();

        ਜੇ (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem * ਆਕਾਰ(__fp16));
            ਜੇ (pv0) {
                v0 = Q6_V_vmux_QVV(ਨਕਾਬ, pv0[c / 64], Q6_V_vzero());
            }
            ਜੇ (pv1) {
                v1 = Q6_V_vmux_QVV(ਨਕਾਬ, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ * tile0 = (HVX_ਸਦਿਸ਼ *) (tile_ਪੰਕਤੀ + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        ਜੇ (c0 + 1 < n_tile_ਖਾਨੇ_ਗਿਣਤੀ) {
            HVX_ਸਦਿਸ਼ * tile1 = (HVX_ਸਦਿਸ਼ *) (tile_ਪੰਕਤੀ + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_ਪੰਕਤੀ_pair_fp32_ਮੰਜ਼ਿਲ_fp16_ਖਾਨਾ_chunk(
        __fp16 *ਸੀਮਿਤ vtcm_dst,
        ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ row0, // offset by c_first
        ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ row1, // offset by c_first
        uint32_t r,
        uint32_t k_ਖੰਡ,
        uint32_t c_ਪਹਿਲਾ,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

    uint32_t c = 0;
    ਲਈ (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
        ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0 + c);
        ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1 + c);

        HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

        uint32_t c0       = (c_ਪਹਿਲਾ + c) / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;  // tile column index
        uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

        HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
    }
    ਜੇ (c < c_len) {
        HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
        ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0 + c);
        ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem > 0 ? rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0);
        v0 = Q6_V_vmux_QVV(ਨਕਾਬ, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(ਨਕਾਬ, v1, Q6_V_vzero());

        HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

        uint32_t c0       = (c_ਪਹਿਲਾ + c) / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;  // tile column index
        uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

        HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_fp32_ਮੰਜ਼ਿਲ_fp16_gathered(
            __fp16 *ਸੀਮਿਤ vtcm_dst,
            ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ src,
            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            uint32_t vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            uint32_t n_ਪੰਕਤੀਆਂ,
            uint32_t k_ਖੰਡ,
            ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_padded = hex_align_ਉੱਪਰ(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_tiled  = (n_ਪੰਕਤੀਆਂ / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ) * HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;

    uint32_t r = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (r = 0; r < n_ਪੰਕਤੀਆਂ_tiled; r += 2) {
        uint32_t r_idx0 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0;
        uint32_t r_idx1 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1;
        uint32_t lr = vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping0 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + r_idx0];
        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        ਅਚਲ ਦਸ਼ਮਲਵ *row0_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        ਅਚਲ ਦਸ਼ਮਲਵ *row1_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        ਲਈ (; c + 32 <= k_valid; c += 32) {
            HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);
            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
        ਜੇ (c < k_ਖੰਡ) {
            HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem > 0 ? rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0);
            v0 = Q6_V_vmux_QVV(ਨਕਾਬ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ਨਕਾਬ, v1, Q6_V_vzero());

            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
    }

    ਲਈ (; r < n_ਪੰਕਤੀਆਂ_padded; r += 2) {
        uint32_t lr = vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

        ਅਚਲ bool row0_valid = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0) < cne1;
        ਅਚਲ bool row1_valid = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1) < cne1;

        ਅਚਲ ਦਸ਼ਮਲਵ *row0_ptr = ਨਿਰਾ;
        ਅਚਲ ਦਸ਼ਮਲਵ *row1_ptr = ਨਿਰਾ;

        ਜੇ (row0_valid) {
            ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping0 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        ਜੇ (row1_valid) {
            ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        ਲਈ (; c + 32 <= k_valid; c += 32) {
            HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
            ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);

            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
        ਜੇ (c < k_ਖੰਡ) {
            HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
            ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem > 0 ? rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0);
            v0 = Q6_V_vmux_QVV(ਨਕਾਬ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ਨਕਾਬ, v1, Q6_V_vzero());

            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_fp32_ਮੰਜ਼ਿਲ_fp16_gathered_flat(
            __fp16 *ਸੀਮਿਤ vtcm_dst,
            ਅਚਲ ਦਸ਼ਮਲਵ *ਸੀਮਿਤ src,
            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            uint32_t vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            uint32_t n_ਪੰਕਤੀਆਂ,
            uint32_t k_ਖੰਡ,
            ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_padded = hex_align_ਉੱਪਰ(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_tiled  = (n_ਪੰਕਤੀਆਂ / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ) * HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;

    uint32_t r = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (r = 0; r < n_ਪੰਕਤੀਆਂ_tiled; r += 2) {
        uint32_t r_idx0 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0;
        uint32_t r_idx1 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1;
        uint32_t lr = vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping0 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + r_idx0];
        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + r_idx1];

        ਅਚਲ ਦਸ਼ਮਲਵ *row0_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + mapping0.i2 * nb12);
        ਅਚਲ ਦਸ਼ਮਲਵ *row1_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        ਲਈ (; c + 32 <= k_valid; c += 32) {
            HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);
            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
        ਜੇ (c < k_ਖੰਡ) {
            HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem > 0 ? rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0);
            v0 = Q6_V_vmux_QVV(ਨਕਾਬ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ਨਕਾਬ, v1, Q6_V_vzero());

            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
    }

    ਲਈ (; r < n_ਪੰਕਤੀਆਂ_padded; r += 2) {
        uint32_t lr = vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ;  // intra-tile row idx

        ਅਚਲ bool row0_valid = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0) < cne1;
        ਅਚਲ bool row1_valid = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1) < cne1;

        ਅਚਲ ਦਸ਼ਮਲਵ *row0_ptr = ਨਿਰਾ;
        ਅਚਲ ਦਸ਼ਮਲਵ *row1_ptr = ਨਿਰਾ;

        ਜੇ (row0_valid) {
            ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping0 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0)];
            row0_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + mapping0.i2 * nb12);
        }
        ਜੇ (row1_valid) {
            ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + (ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1)];
            row1_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        ਲਈ (; c + 32 <= k_valid; c += 32) {
            HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
            ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);

            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
        ਜੇ (c < k_ਖੰਡ) {
            HVX_ਸਦਿਸ਼ v0 = Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v1 = Q6_V_vzero();
            ਜੇ (row0_valid) v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row0_ptr + c);
            ਜੇ (row1_valid) v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(rem > 0 ? rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0);
            v0 = Q6_V_vmux_QVV(ਨਕਾਬ, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(ਨਕਾਬ, v1, Q6_V_vzero());

            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            uint32_t tile_idx = r0 * (k_ਖੰਡ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) + c0;

            HVX_ਸਦਿਸ਼ *tile = (HVX_ਸਦਿਸ਼ *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_ਬਾਹਰਲਾ;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32_scattered(
            ਦਸ਼ਮਲਵ *ਸੀਮਿਤ dst,
            ਅਚਲ __fp16 *ਸੀਮਿਤ vtcm_src,
            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            uint32_t vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            uint32_t n_ਪੰਕਤੀਆਂ,
            uint32_t n_ਖਾਨੇ_ਗਿਣਤੀ,
            ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n_ਖਾਨੇ_ਗਿਣਤੀ % HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ == 0);
    ਅਚਲ size_t tile_ਪੰਕਤੀ_stride = (n_ਖਾਨੇ_ਗਿਣਤੀ / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) * HTP_MM_HMX_TILE_N_ELMS;

    ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);

    ਲਈ (size_t r = 0; r < n_ਪੰਕਤੀਆਂ; r += 2) {
        uint32_t r_idx0 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 0;
        uint32_t r_idx1 = ਸ਼ੁਰੂ_ਪੰਕਤੀ + r + 1;
        uint32_t     lr = vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ + r;                 // vtcm-local row
        ਅਚਲ size_t r0 = (lr / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
        ਅਚਲ size_t r1 = (lr % HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ) / 2;  // index of the row pair within the tile
        ਅਚਲ __fp16 *ਪੰਕਤੀ_ਅਧਾਰ = vtcm_src + r0 * tile_ਪੰਕਤੀ_stride;

        ਜੇ (r_idx0 >= cne1) ਰੋਕੋ;

        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping0 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + r_idx0];
        ਦਸ਼ਮਲਵ *output_row0 = (ਦਸ਼ਮਲਵ *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        ਦਸ਼ਮਲਵ *output_row1 = ਨਿਰਾ;
        ਜੇ (r_idx1 < cne1) {
            ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping mapping1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a * mapping_stride + r_idx1];
            output_row1 = (ਦਸ਼ਮਲਵ *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #ਹਦਾਇਤ unroll(4)
        ਲਈ (size_t c = 0; c < (size_t)n_ਖਾਨੇ_ਗਿਣਤੀ; c += HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) {
            ਅਚਲ size_t c0 = c / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ __fp16 *tile = ਪੰਕਤੀ_ਅਧਾਰ + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_ਸਦਿਸ਼ v = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) tile)[r1];
            HVX_ਸਦਿਸ਼_Pair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            ਜੇ (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
