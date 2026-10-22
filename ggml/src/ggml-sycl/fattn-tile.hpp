#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <sycl/ext/oneapi/work_group_static.hpp>
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "fattn-common.hpp"

#అన్తర్భావ <cmath>
#అన్తర్భావ <float.h>

నామాకాశ syclex = sycl::ext::oneapi::experimental;

#నిర్వచించు GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    ఒకవేళ (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        తిరిగి ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

స్థిర constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  64,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  64,  72)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512, 32, 256, 2,  64,  64)

    తిరిగి 0;
}

స్థిర constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  2, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 3,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  2, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 3,  32, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 3,  64, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 16, 128, 3,  32, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  2, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 3,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  32, 256)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  2, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  32,  64)

    తిరిగి 0;
}

స్థిర constexpr uint32_t ggml_sycl_fattn_tile_get_config(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం cc) {
    ఒకవేళ(fast_fp16_అందుబాటు(cc))
        తిరిగి ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    లేకపోతే
        తిరిగి ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

స్థిర constexpr uint32_t ggml_sycl_fattn_tile_get_config(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
#నిర్వచితమైతే SYCL_FAST_FP16
    తిరిగి ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#లేకపోతే
    తిరిగి ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#అయితే_ముగింపు // SYCL_FAST_FP16
}

స్థిర పూర్ణాంకం ggml_sycl_fattn_tile_get_nthreads(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం cc) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

స్థిర constexpr పూర్ణాంకం ggml_sycl_fattn_tile_get_nthreads(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

స్థిర పూర్ణాంకం ggml_sycl_fattn_tile_get_occupancy(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం cc) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

స్థిర constexpr పూర్ణాంకం ggml_sycl_fattn_tile_get_occupancy(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

స్థిర పూర్ణాంకం ggml_sycl_fattn_tile_get_nbatch_fa(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం cc) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

స్థిర constexpr పూర్ణాంకం ggml_sycl_fattn_tile_get_nbatch_fa(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

స్థిర పూర్ణాంకం ggml_sycl_fattn_tile_get_nbatch_K(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం cc) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

స్థిర constexpr పూర్ణాంకం ggml_sycl_fattn_tile_get_nbatch_K(స్థిరరాశి పూర్ణాంకం DKQ, స్థిరరాశి పూర్ణాంకం DV, స్థిరరాశి పూర్ణాంకం ncols) {
    తిరిగి (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

మూస <పూర్ణాంకం warp_కొలత, పూర్ణాంకం nwarps, పూర్ణాంకం I, పూర్ణాంకం J, పూర్ణాంకం J_padding, తార్కిక oob_check>
స్థిర __dpct_inline__ శూన్య flash_attn_tile_ఎక్కించు_tile(స్థిరరాశి sycl::half2 * స్థిరరాశి __restrict__ KV,
                                                      sycl::half2 * స్థిరరాశి __restrict__ tile_KV,
                                                      స్థిరరాశి పూర్ణాంకం stride_KV,
                                                      స్థిరరాశి పూర్ణాంకం i_sup) {
    స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    constexpr పూర్ణాంకం cpy_nb = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    స్వయంచాలక load = [&] (స్థిరరాశి పూర్ణాంకం n) {
        స్థిరరాశి పూర్ణాంకం stride_j = warp_కొలత >> n;

        ఒకవేళ (stride_j == 0) {
            తిరిగి;
        }

        స్థిరరాశి పూర్ణాంకం j0_ప్రారంభం = stride_j == warp_కొలత ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        స్థిరరాశి పూర్ణాంకం j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        స్థిరరాశి పూర్ణాంకం stride_i = warp_కొలత / stride_j;

        ఒకవేళ (j0_ప్రారంభం == j0_stop) {
            తిరిగి;
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            స్థిరరాశి పూర్ణాంకం i = i0 + అంశం_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_కొలత ? 0 : అంశం_ct1.get_local_id(2) / stride_j);

            ఒకవేళ (i0 + nwarps*stride_i <= I || i < I) {
#సూచన unroll
                కొరకు (పూర్ణాంకం j0 = j0_ప్రారంభం; j0 < j0_stop; j0 += stride_j) {
                    స్థిరరాశి పూర్ణాంకం j = j0 * cpy_ne + (stride_j == warp_కొలత ? అంశం_ct1.get_local_id(2) :
                                                                         అంశం_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    స్థిరరాశి __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
                        { 0.0f, 0.0f }
                    };
                    ggml_sycl_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
                }
            }
        }
    };
    // 1: max 64*16=512 bytes, 512 half
    // 2: max 32*16=512 bytes, 256 half
    // 3: max 16*16=256 bytes, 128 half
    // 4: max  8*16=128 bytes,  64 half
    // 5: max  4*16= 64 bytes,  32 half
    // 6: max  2*16= 32 bytes,  16 half
    // 7: max  1*16= 16 bytes,   8 half
    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    ggml_sycl_unroll<7>{}(load);
}

మూస <పూర్ణాంకం warp_కొలత, పూర్ణాంకం nwarps, పూర్ణాంకం I, పూర్ణాంకం J, పూర్ణాంకం J_padding, తార్కిక oob_check>
స్థిర __dpct_inline__ శూన్య flash_attn_tile_ఎక్కించు_tile(స్థిరరాశి sycl::half2 * స్థిరరాశి __restrict__ KV,
                                                      దశాంశం * స్థిరరాశి __restrict__ tile_KV,
                                                      స్థిరరాశి పూర్ణాంకం stride_KV,
                                                      స్థిరరాశి పూర్ణాంకం i_sup) {
    constexpr పూర్ణాంకం cpy_nb = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    స్వయంచాలక load = [&] (స్థిరరాశి పూర్ణాంకం n) {
        స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
        స్థిరరాశి పూర్ణాంకం stride_j = warp_కొలత >> n;

        ఒకవేళ (stride_j == 0) {
            తిరిగి;
        }

        స్థిరరాశి పూర్ణాంకం j0_ప్రారంభం = stride_j == warp_కొలత ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        స్థిరరాశి పూర్ణాంకం j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        స్థిరరాశి పూర్ణాంకం stride_i = warp_కొలత / stride_j;

        ఒకవేళ (j0_ప్రారంభం == j0_stop) {
            తిరిగి;
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            స్థిరరాశి పూర్ణాంకం i = i0 + అంశం_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_కొలత ? 0 : అంశం_ct1.get_local_id(2) / stride_j);

            ఒకవేళ (i0 + nwarps*stride_i <= I || i < I) {
#సూచన unroll
                కొరకు (పూర్ణాంకం j0 = j0_ప్రారంభం; j0 < j0_stop; j0 += stride_j) {
                    స్థిరరాశి పూర్ణాంకం j = j0 * (cpy_ne / 2) + (stride_j == warp_కొలత ? అంశం_ct1.get_local_id(2) :
                                                                               అంశం_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    స్థిరరాశి sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<పరిమాణం(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#సూచన unroll
                    కొరకు (పూర్ణాంకం l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
                    }
                    ggml_sycl_memcpy_1<పరిమాణం(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
                }
            }
        }
    };
    // 1: max 32*16=512 bytes, 128 float
    // 2: max 16*16=256 bytes,  64 float
    // 3: max  8*16=128 bytes,  32 float
    // 4: max  4*16= 64 bytes,  16 float
    // 5: max  2*16= 32 bytes,   8 float
    static_assert(J % 8 == 0, "bad J");
    static_assert(J % cpy_ne == 0, "bad J");
    ggml_sycl_unroll<5>{}(load);
}

// Function that performs a single iteration in for the KQ matrix multiplication:
మూస <పూర్ణాంకం  warp_కొలత,
          పూర్ణాంకం  nwarps,
          పూర్ణాంకం  ncols1,
          పూర్ణాంకం  ncols2,
          పూర్ణాంకం  DKQ,
          పూర్ణాంకం  nbatch_fa,
          పూర్ణాంకం  nbatch_K,
          తార్కిక use_logit_softcap,
          తార్కిక oob_check,
          రకం_పేరు T_vec_dot>
స్థిర __dpct_inline__ శూన్య flash_attn_tile_iter_KQ(T_vec_dot * స్థిరరాశి Q_tmp,
                                                    స్థిరరాశి sycl::half2 * స్థిరరాశి __restrict__ K_h2,
                                                    T_vec_dot * స్థిరరాశి KV_tmp,
                                                    స్థిరరాశి పూర్ణాంకం         stride_K2,
                                                    స్థిరరాశి పూర్ణాంకం         k_VKQ_0,
                                                    స్థిరరాశి పూర్ణాంకం         k_VKQ_sup,
                                                    స్థిరరాశి పూర్ణాంకం         k_KQ_0,
                                                    దశాంశం *           KQ_acc) {
    స్వయంచాలక          అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    constexpr పూర్ణాంకం cpy_nb   = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    constexpr పూర్ణాంకం ncols = ncols1*ncols2;
    constexpr పూర్ణాంకం cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr పూర్ణాంకం np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_ఎక్కించు_tile<warp_కొలత, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

#నిర్వచితమైతే SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_కొలత)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#లేకపోతే
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) దశాంశం K_k[nbatch_fa/(np*warp_కొలత)][cpy_ne];
        __dpct_align__(16) దశాంశం Q_k[cpw][cpy_ne];
#అయితే_ముగింపు // SYCL_FAST_FP16

#సూచన unroll
        కొరకు (పూర్ణాంకం i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_కొలత) {
            స్థిరరాశి పూర్ణాంకం i_KQ = i_KQ_0 + (అంశం_ct1.get_local_id(1) % np) * warp_కొలత + అంశం_ct1.get_local_id(2);

#నిర్వచితమైతే SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_కొలత)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#లేకపోతే
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_కొలత)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#అయితే_ముగింపు // SYCL_FAST_FP16
        }
#సూచన unroll
        కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
            స్థిరరాశి పూర్ణాంకం jc = jc0 + (అంశం_ct1.get_local_id(1) / np) * cpw;

#నిర్వచితమైతే SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#లేకపోతే
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#అయితే_ముగింపు // SYCL_FAST_FP16
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_కొలత) {
#సూచన unroll
            కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
#సూచన unroll
                కొరకు (పూర్ణాంకం k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_కొలత)*cpw + jc0], K_k[i_KQ_0/(np*warp_కొలత)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    ఒకవేళ (k_KQ_0 + nbatch_K < DKQ) {
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
మూస <పూర్ణాంకం  warp_కొలత,
          పూర్ణాంకం  nwarps,
          పూర్ణాంకం  ncols1,
          పూర్ణాంకం  ncols2,
          పూర్ణాంకం  DKQ,
          పూర్ణాంకం  DV,
          పూర్ణాంకం  nbatch_fa,
          పూర్ణాంకం  nbatch_K,
          తార్కిక use_logit_softcap,
          తార్కిక oob_check,
          రకం_పేరు T_vec_dot,
          రకం_పేరు T_KQ,
          రకం_పేరు T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
స్థిర __dpct_inline__ శూన్య flash_attn_tile_iter(T_vec_dot * స్థిరరాశి Q_tmp,
                                                 స్థిరరాశి sycl::half2 * స్థిరరాశి __restrict__ K_h2,
                                                 స్థిరరాశి sycl::half2 * స్థిరరాశి __restrict__ V_h2,
                                                 స్థిరరాశి sycl::half * స్థిరరాశి __restrict__ ముసుగు,
                                                 స్థిరరాశి sycl::uint3 ne01,
                                                 స్థిరరాశి దశాంశం       logit_softcap,
                                                 స్థిరరాశి దశాంశం       వాలు,
                                                 T_KQ * స్థిరరాశి      KQ,
                                                 T_vec_dot * స్థిరరాశి KV_tmp,
                                                 స్థిరరాశి పూర్ణాంకం         stride_K2,
                                                 స్థిరరాశి పూర్ణాంకం         stride_V2,
                                                 స్థిరరాశి పూర్ణాంకం         stride_ముసుగు,
                                                 దశాంశం * స్థిరరాశి     KQ_max,
                                                 దశాంశం * స్థిరరాశి     KQ_కూడిక,
                                                 T_acc * స్థిరరాశి     VKQ,
                                                 స్థిరరాశి పూర్ణాంకం         k_VKQ_0,
                                                 స్థిరరాశి పూర్ణాంకం         k_VKQ_max,
                                                 స్థిరరాశి పూర్ణాంకం         నిలువు_Q_0,
                                                 దశాంశం *           KQ_max_new_shared) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    constexpr పూర్ణాంకం cpy_nb   = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    constexpr పూర్ణాంకం ncols = ncols1*ncols2;
    constexpr పూర్ణాంకం cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr పూర్ణాంకం np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr పూర్ణాంకం DVp = (DV + 2*warp_కొలత - 1) & ~(2*warp_కొలత - 1); // DV padded to multiple of 2*warp_size.

#నిర్వచితమైతే SYCL_FAST_FP16
    constexpr పూర్ణాంకం KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#లేకపోతే
    constexpr పూర్ణాంకం KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#అయితే_ముగింపు // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    స్థిరరాశి పూర్ణాంకం k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    దశాంశం KQ_max_new[cpw];
#సూచన unroll
    కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    దశాంశం KQ_acc[nbatch_fa/(np*warp_కొలత) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr పూర్ణాంకం nbatch_K_చివరి = DKQ % nbatch_K;
#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_చివరి; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_కొలత, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    ఒకవేళ (nbatch_K_చివరి > 0) {
        constexpr పూర్ణాంకం k_KQ_0 = DKQ - nbatch_K_చివరి;
        flash_attn_tile_iter_KQ<warp_కొలత, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_చివరి, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#సూచన unroll
    కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
        స్థిరరాశి పూర్ణాంకం j = fastmodulo(నిలువు_Q_0 + (jc0 + (అంశం_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#సూచన unroll
        కొరకు (పూర్ణాంకం i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_కొలత) {
            స్థిరరాశి పూర్ణాంకం i_KQ = i_KQ_0 + (అంశం_ct1.get_local_id(1) % np) * warp_కొలత + అంశం_ct1.get_local_id(2);

#ఒకవేళ defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_కొలత)*cpw + jc0] *= 4.0f;
#అయితే_ముగింపు // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            ఒకవేళ (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_కొలత)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((దశాంశం) KQ_acc[(i_KQ_0 / (np * warp_కొలత)) * cpw + jc0]);
            }

            ఒకవేళ (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_కొలత)) * cpw + jc0] +=
                    (ncols2 > 1 || ముసుగు) ? వాలు * sycl::vec<sycl::half, 1>(ముసుగు[j * stride_ముసుగు + k_VKQ_0 + i_KQ])
                                                       .convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((దశాంశం) KQ_max_new[jc0],
                               (దశాంశం) (KQ_acc[(i_KQ_0 / (np * warp_కొలత)) * cpw + jc0] + FATTN_KQ_MAX_స్థానభ్రంశం));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_కొలత>(KQ_max_new[jc0]);
    }

    ఒకవేళ constexpr (np == 1) {
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
    } లేకపోతే {
        static_assert(cpw == 1, "bad cpw");

        ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[అంశం_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(అంశం_ct1.get_local_id(1) & ~(np - 1)) + అంశం_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#సూచన unroll
    కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#నిర్వచితమైతే SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_కొలత)][KQ_cs];
#లేకపోతే
        __dpct_align__(16) దశాంశం tmp[nbatch_fa/(np*warp_కొలత)][KQ_cs];
#అయితే_ముగింపు // SYCL_FAST_FP16

#సూచన unroll
        కొరకు (పూర్ణాంకం jc1 = 0; jc1 < KQ_cs; ++jc1) {
            స్థిరరాశి పూర్ణాంకం jc = jc0 + jc1;

            స్థిరరాశి దశాంశం KQ_max_scale = sycl::native::ఘాతాంకం((దశాంశం) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            దశాంశం KQ_కూడిక_కూడు = 0.0f;
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < nbatch_fa; i0 += np*warp_కొలత) {
                స్థిరరాశి దశాంశం val =
                    !oob_check || i0 + (అంశం_ct1.get_local_id(1) % np) * warp_కొలత + అంశం_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::ఘాతాంకం((దశాంశం) (KQ_acc[(i0 / (np * warp_కొలత)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_కూడిక_కూడు += val;
                tmp[i0/(np*warp_కొలత)][jc1] = val;
            }
            KQ_కూడిక[jc] = KQ_కూడిక[jc]*KQ_max_scale + KQ_కూడిక_కూడు;

#నిర్వచితమైతే SYCL_FAST_FP16
            స్థిరరాశి sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత) {
                VKQ[jc*((DVp/2)/warp_కొలత) + i0/warp_కొలత].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_కొలత) + i0/warp_కొలత].y() *= KQ_max_scale_h2.y();
            }
#లేకపోతే
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత) {
                VKQ[jc*((DVp/2)/warp_కొలత) + i0/warp_కొలత].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_కొలత) + i0/warp_కొలత].y() *= KQ_max_scale;
            }
#అయితే_ముగింపు // SYCL_FAST_FP16
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < nbatch_fa; i0 += np*warp_కొలత) {
            స్థిరరాశి పూర్ణాంకం i = i0 + (అంశం_ct1.get_local_id(1) % np) * warp_కొలత + అంశం_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<పరిమాణం(tmp[0])>(
                KQ + (jc0 / KQ_cs + (అంశం_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_కొలత)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr పూర్ణాంకం nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#సూచన unroll
    కొరకు (పూర్ణాంకం k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_ఎక్కించు_tile<warp_కొలత, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

#నిర్వచితమైతే SYCL_FAST_FP16
#సూచన unroll
        కొరకు (పూర్ణాంకం k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_కొలత];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr పూర్ణాంకం cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_కొలత ? cpy_ne/2 : (DVp/2)/warp_కొలత;
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_కొలత],
                                                 &KV_tmp[(k1 + అంశం_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         అంశం_ct1.get_local_id(2) * cpy_ne_D]);
            }
#సూచన unroll
            కొరకు (పూర్ణాంకం jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                స్థిరరాశి పూర్ణాంకం jc_KQ = jc_VKQ_0 / KQ_cs + (అంశం_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * పరిమాణం(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + అంశం_ct1.get_local_id(1) % np) * KQ_cs);
#సూచన unroll
                కొరకు (పూర్ణాంకం jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత) {
#సూచన unroll
                కొరకు (పూర్ణాంకం jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_కొలత) + i0/warp_కొలత].x() +=
                        V_k[i0/warp_కొలత].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_కొలత) + i0/warp_కొలత].y() +=
                        V_k[i0/warp_కొలత].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#లేకపోతే
#సూచన unroll
        కొరకు (పూర్ణాంకం k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_కొలత];
            __dpct_align__(16) దశాంశం  KQ_k[cpw];

            constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < DVp/warp_కొలత ? cpy_ne : DVp/warp_కొలత;
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp; i0 += warp_కొలత*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_కొలత)], &KV_tmp[(k1 + అంశం_ct1.get_local_id(1) % np)*DV + i0 + అంశం_ct1.get_local_id(2)*cpy_ne_D]);
            }
#సూచన unroll
            కొరకు (పూర్ణాంకం jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                స్థిరరాశి పూర్ణాంకం jc_KQ = jc_VKQ_0/KQ_cs + (అంశం_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*పరిమాణం(దశాంశం)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + అంశం_ct1.get_local_id(1) % np)*KQ_cs);
            }

#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత) {
#సూచన unroll
                కొరకు (పూర్ణాంకం jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_కొలత) + i0/warp_కొలత].x() += V_k[i0/warp_కొలత].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_కొలత) + i0/warp_కొలత].y() += V_k[i0/warp_కొలత].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#అయితే_ముగింపు // SYCL_FAST_FP16
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

మూస <పూర్ణాంకం DKQ, పూర్ణాంకం DV, పూర్ణాంకం ncols1, పూర్ణాంకం ncols2, తార్కిక use_logit_softcap, పూర్ణాంకం warp_కొలత>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
స్థిర శూన్య flash_attn_tile(స్థిరరాశి అక్షరం *  Q,
                            స్థిరరాశి అక్షరం *  K,
                            స్థిరరాశి అక్షరం *  V,
                            స్థిరరాశి అక్షరం *  ముసుగు,
                            స్థిరరాశి అక్షరం *  sinks,
                            స్థిరరాశి పూర్ణాంకం *  KV_max,
                            దశాంశం *  dst,
                            sycl::float2 *  dst_meta,
                            స్థిరరాశి దశాంశం          scale,
                            స్థిరరాశి దశాంశం          max_bias,
                            స్థిరరాశి దశాంశం          m0,
                            స్థిరరాశి దశాంశం          m1,
                            స్థిరరాశి uint32_t       n_తల_log2,
                            స్థిరరాశి దశాంశం          logit_softcap,
                            స్థిరరాశి int32_t        ne00,
                            స్థిరరాశి sycl::uint3    ne01,
                            స్థిరరాశి int32_t        ne02,
                            స్థిరరాశి int32_t        ne03,
                            స్థిరరాశి int32_t        nb01,
                            స్థిరరాశి int32_t        nb02,
                            స్థిరరాశి int32_t        nb03,
                            స్థిరరాశి int32_t        ne10,
                            స్థిరరాశి int32_t        ne11,
                            స్థిరరాశి int32_t        ne12,
                            స్థిరరాశి int32_t        ne13,
                            స్థిరరాశి int32_t        nb11,
                            స్థిరరాశి int32_t        nb12,
                            స్థిరరాశి int64_t        nb13,
                            స్థిరరాశి int32_t        nb21,
                            స్థిరరాశి int32_t        nb22,
                            స్థిరరాశి int64_t        nb23,
                            స్థిరరాశి int32_t        ne31,
                            స్థిరరాశి int32_t        ne32,
                            స్థిరరాశి int32_t        ne33,
                            స్థిరరాశి int32_t        nb31,
                            స్థిరరాశి int32_t        nb32,
                            స్థిరరాశి int64_t        nb33) {
#నిర్వచితమైతే SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    ఒకవేళ ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_వాడని_VARS(Q, K, V, ముసుగు, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_తల_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        తిరిగి;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr పూర్ణాంకం ncols     = ncols1*ncols2;

    constexpr పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_కొలత;
    constexpr పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr పూర్ణాంకం nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    స్థిరరాశి పూర్ణాంకం నిలువు_Q_0 = అంశం_ct1.get_సమూహం(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    స్థిరరాశి పూర్ణాంకం           sequence  = అంశం_ct1.get_సమూహం(0) / (ne02 / ncols2);
    స్థిరరాశి పూర్ణాంకం           head0     = అంశం_ct1.get_సమూహం(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    స్థిరరాశి దశాంశం * Q_f  = (స్థిరరాశి దశాంశం *) (Q + nb03*sequence + nb02* head0);
    స్థిరరాశి sycl::half2 * K_h2      = (స్థిరరాశి sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_నిష్పత్తి));
    స్థిరరాశి sycl::half2 * V_h2 =
        (స్థిరరాశి sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_నిష్పత్తి));  // K and V have same shape

    స్థిరరాశి sycl::half * maskh = ముసుగు ? (స్థిరరాశి sycl::half *) (ముసుగు + nb33 * (sequence % ne33)) : శూన్యసూచిక;

    స్థిరరాశి పూర్ణాంకం stride_K2   = nb11 / పరిమాణం(sycl::half2);
    స్థిరరాశి పూర్ణాంకం stride_V2   = nb21 / పరిమాణం(sycl::half2);
    స్థిరరాశి పూర్ణాంకం stride_ముసుగు = nb31 / పరిమాణం(sycl::half);

    స్థిరరాశి దశాంశం వాలు = ncols2 == 1 ? get_alibi_వాలు(max_bias, head0, n_తల_log2, m0, m1) : 1.0f;

    constexpr పూర్ణాంకం cpy_nb = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    constexpr పూర్ణాంకం cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr పూర్ణాంకం np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_కొలత) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr పూర్ణాంకం DKQp = (DKQ + 2*warp_కొలత - 1) & ~(2*warp_కొలత - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr పూర్ణాంకం DVp  = (DV  + 2*warp_కొలత - 1) & ~(2*warp_కొలత - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#నిర్వచితమైతే SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_కొలత = lsm_size1 * పరిమాణం(sycl::half2) +
                                            lsm_size2 * పరిమాణం(sycl::half2) +
                                            lsm_size3 * పరిమాణం(sycl::half) +
                                            lsm_size4 * పరిమాణం(దశాంశం);

    syclex::work_సమూహం_static<అక్షరం[local_share_mem_కొలత]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    దశాంశం *KQ_max_new_shared = (దశాంశం *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_కొలత)] = {
        { 0.0f, 0.0f }
    };
#లేకపోతే
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_కొలత = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * పరిమాణం(దశాంశం);

    syclex::work_సమూహం_static<అక్షరం[local_share_mem_కొలత]> lsm;

    దశాంశం *Q_tmp = (దశాంశం *)&lsm;
    దశాంశం *KV_tmp = Q_tmp +lsm_size1;
    దశాంశం *KQ = KV_tmp+lsm_size2;
    దశాంశం *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_కొలత)] = {{0.0f, 0.0f}};


#అయితే_ముగింపు // SYCL_FAST_FP16

    దశాంశం KQ_max[cpw] = {};

#సూచన unroll
    కొరకు (పూర్ణాంకం j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    దశాంశం KQ_కూడిక[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#సూచన unroll
    కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
        స్థిరరాశి పూర్ణాంకం jc = jc0 + (అంశం_ct1.get_local_id(1) / np) * cpw;

        స్థిరరాశి పూర్ణాంకం j = jc / ncols2;
        స్థిరరాశి పూర్ణాంకం c = jc % ncols2;

        constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < DKQp/warp_కొలత ? cpy_ne : DKQp/warp_కొలత;

#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < DKQp; i0 += np*warp_కొలత*cpy_ne_D) {
            ఒకవేళ (i0 + np * warp_కొలత * cpy_ne_D <= DKQ ||
                i0 + (అంశం_ct1.get_local_id(1) % np) * (warp_కొలత * cpy_ne_D) + అంశం_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) దశాంశం tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<పరిమాణం(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / పరిమాణం(దశాంశం)) + fastmodulo(నిలువు_Q_0 + j, ne01) * (nb01 / పరిమాణం(దశాంశం)) +
                                i0 + (అంశం_ct1.get_local_id(1) % np) * (warp_కొలత * cpy_ne_D) +
                                అంశం_ct1.get_local_id(2) * cpy_ne_D]);

#సూచన unroll
                కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#నిర్వచితమైతే SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#సూచన unroll
                కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#ఒకవేళ defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#అయితే_ముగింపు // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<పరిమాణం(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (అంశం_ct1.get_local_id(1) % np) * (warp_కొలత * cpy_ne_D / 2) +
                           అంశం_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#లేకపోతే
                ggml_sycl_memcpy_1<పరిమాణం(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (అంశం_ct1.get_local_id(1) % np)*(warp_కొలత*cpy_ne_D)   + అంశం_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#అయితే_ముగింపు // SYCL_FAST_FP16
            }
        }
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    స్థిరరాశి పూర్ణాంకం k_VKQ_max = KV_max ? KV_max[sequence * అంశం_ct1.get_సమూహం_range(2) + అంశం_ct1.get_సమూహం(2)] : ne11;
    ఒకవేళ (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        పూర్ణాంకం k_VKQ_0 = అంశం_ct1.get_సమూహం(1) * nbatch_fa;
        అయితే (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr తార్కిక oob_check = అబద్ధం;
            flash_attn_tile_iter<warp_కొలత, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, వాలు, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ముసుగు, KQ_max, KQ_కూడిక, VKQ, k_VKQ_0, k_VKQ_max, నిలువు_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += అంశం_ct1.get_సమూహం_range(1) * nbatch_fa;
        }
        ఒకవేళ (k_VKQ_0 < k_VKQ_max) {
            constexpr తార్కిక oob_check = నిజం;
            flash_attn_tile_iter<warp_కొలత, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, వాలు, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ముసుగు, KQ_max, KQ_కూడిక, VKQ, k_VKQ_0, k_VKQ_max, నిలువు_Q_0,
                                            KQ_max_new_shared);
        }
    } లేకపోతే {
        // Branch without out-of-bounds checks.
        కొరకు (పూర్ణాంకం k_VKQ_0 = అంశం_ct1.get_సమూహం(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += అంశం_ct1.get_సమూహం_range(1) * nbatch_fa) {

            constexpr తార్కిక oob_check = అబద్ధం;
            flash_attn_tile_iter<warp_కొలత, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, వాలు, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ముసుగు, KQ_max, KQ_కూడిక, VKQ, k_VKQ_0, k_VKQ_max, నిలువు_Q_0,
                                            KQ_max_new_shared);
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_కూడిక[jc0] = warp_reduce_కూడిక<warp_కొలత>(KQ_కూడిక[jc0]);
    }

    ఒకవేళ constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#నిర్వచితమైతే SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#లేకపోతే
        దశాంశం * VKQ_combine    = (దశాంశం *) KV_tmp;
#అయితే_ముగింపు // SYCL_FAST_FP16

        దశాంశం * KQ_కూడిక_combine = (దశాంశం *) Q_tmp;

        ఒకవేళ (అంశం_ct1.get_local_id(1) % np != 0) {

#నిర్వచితమైతే SYCL_FAST_FP16
            constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < (DVp/2)/warp_కొలత ? cpy_ne : (DVp/2)/warp_కొలత;
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[అంశం_ct1.get_local_id(1) * (DVp / 2) + i0 + అంశం_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_కొలత]);
            }
#లేకపోతే

            constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < DVp/warp_కొలత ? cpy_ne : DVp/warp_కొలత;

#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp; i0 += warp_కొలత*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[అంశం_ct1.get_local_id(1)*DVp + i0 + అంశం_ct1.get_local_id(2)*cpy_ne_D], ((స్థిరరాశి దశాంశం *) VKQ) + i0/warp_కొలత);
            }
#అయితే_ముగింపు // SYCL_FAST_FP16

            ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
                KQ_కూడిక_combine[అంశం_ct1.get_local_id(1)] = KQ_కూడిక[0];
            }
            తిరిగి;
        }

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

#సూచన unroll
        కొరకు (పూర్ణాంకం ip = 1; ip < np; ++ip) {
#నిర్వచితమైతే SYCL_FAST_FP16
            constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < (DVp/2)/warp_కొలత ? cpy_ne : (DVp/2)/warp_కొలత;
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(అంశం_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   అంశం_ct1.get_local_id(2) * cpy_ne_D]);
#సూచన unroll
                కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_కొలత + i1] += tmp[i1];
                }
            }
#లేకపోతే
            constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < DVp/warp_కొలత ? cpy_ne : DVp/warp_కొలత;
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp; i0 += warp_కొలత*cpy_ne_D) {
                __dpct_align__(16) దశాంశం tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(అంశం_ct1.get_local_id(1) + ip)*DVp + i0 + అంశం_ct1.get_local_id(2)*cpy_ne_D]);
#సూచన unroll
                కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((దశాంశం *)VKQ)[i0/warp_కొలత + i1] += tmp[i1];
                }
            }
#అయితే_ముగింపు // SYCL_FAST_FP16

            KQ_కూడిక[0] += KQ_కూడిక_combine[అంశం_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    ఒకవేళ (sinks && అంశం_ct1.get_సమూహం(1) == 0) {
#సూచన unroll
        కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
            స్థిరరాశి పూర్ణాంకం   jc   = jc0 + (అంశం_ct1.get_local_id(1) / np) * cpw;
            స్థిరరాశి దశాంశం sink = ((స్థిరరాశి దశాంశం *) sinks)[head0 + jc % ncols2];

            దశాంశం       KQ_max_new_j = sycl::fmax((దశాంశం) KQ_max[jc0], sink);
            స్థిరరాశి దశాంశం KQ_max_scale = sycl::native::ఘాతాంకం((దశాంశం) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            స్థిరరాశి దశాంశం val = sycl::native::ఘాతాంకం((దశాంశం) (sink - KQ_max[jc0]));
            KQ_కూడిక[jc0] = KQ_కూడిక[jc0]*KQ_max_scale + val;

#నిర్వచితమైతే SYCL_FAST_FP16
            స్థిరరాశి sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత) {
                VKQ[jc0*((DVp/2)/warp_కొలత) + i0/warp_కొలత] *= KQ_max_scale_h2;
            }
#లేకపోతే
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత) {
                VKQ[jc0*((DVp/2)/warp_కొలత) + i0/warp_కొలత].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_కొలత) + i0/warp_కొలత].y() *= KQ_max_scale;
            }
#అయితే_ముగింపు // SYCL_FAST_FP16
        }
    }

    // Write back results:
#సూచన unroll
    కొరకు (పూర్ణాంకం jc0 = 0; jc0 < cpw; ++jc0) {
        స్థిరరాశి పూర్ణాంకం jc = jc0 + (అంశం_ct1.get_local_id(1) / np) * cpw;

        స్థిరరాశి పూర్ణాంకం j = jc / ncols2;
        స్థిరరాశి పూర్ణాంకం c = jc % ncols2;

        ఒకవేళ (ncols1 > 1 && నిలువు_Q_0 + j >= పూర్ణాంకం(ne01.z())) {
            తిరిగి;
        }

        స్థిరరాశి దశాంశం scale = అంశం_ct1.get_సమూహం_range(1) == 1 ? 1.0f / KQ_కూడిక[jc0] : 1.0f;

        స్థిరరాశి పూర్ణాంకం j_dst_unrolled =
            ((sequence * పూర్ణాంకం(ne01.z()) + నిలువు_Q_0 + j) * ne02 + head0 + c) * అంశం_ct1.get_సమూహం_range(1) +
            అంశం_ct1.get_సమూహం(1);

#నిర్వచితమైతే SYCL_FAST_FP16
        constexpr పూర్ణాంకం cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_కొలత ? cpy_ne/2 : (DVp/2)/warp_కొలత;
#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp/2; i0 += warp_కొలత*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#సూచన unroll
            కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_కొలత) + i0 / warp_కొలత + i1]
                              .మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            ఒకవేళ (i0 + warp_కొలత * cpy_ne_D <= DV / 2 || i0 + అంశం_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<పరిమాణం(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + అంశం_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#లేకపోతే
        constexpr పూర్ణాంకం cpy_ne_D = cpy_ne < DVp/warp_కొలత ? cpy_ne : DVp/warp_కొలత;
#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < DVp; i0 += warp_కొలత*cpy_ne_D) {
            ఒకవేళ (i0 + warp_కొలత*cpy_ne_D <= DV || i0 + అంశం_ct1.get_local_id(2)*cpy_ne_D < DV) {
#సూచన unroll
                కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_కొలత) + i0/(2*warp_కొలత) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_కొలత) + i0/(2*warp_కొలత) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + అంశం_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_కొలత) + i0/(2*warp_కొలత)]);
            }
        }
#అయితే_ముగింపు // SYCL_FAST_FP16

        ఒకవేళ (అంశం_ct1.get_సమూహం_range(1) != 1 && అంశం_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_కూడిక[jc0]);
        }
    }
#లేకపోతే
    GGML_వాడని_VARS(Q, K, V, ముసుగు, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_తల_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#అయితే_ముగింపు // SYCL_FLASH_ATTN
}

మూస <పూర్ణాంకం DKQ, పూర్ణాంకం DV, పూర్ణాంకం ncols2, తార్కిక use_logit_softcap>
స్థిర శూన్య launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * Q = dst->src[0];

    స్థిరరాశి పూర్ణాంకం id        = ggml_sycl_get_device();
    స్థిరరాశి పూర్ణాంకం cc        = ggml_sycl_info().devices[id].cc;
    స్థిరరాశి పూర్ణాంకం warp_కొలత = WARP_32_కొలత; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    ఒకవేళ (DV < 512 && Q->ne[1] < 32) {
        ఒకవేళ constexpr (ncols2 <= 32) {
            ఒకవేళ (Q->ne[1] > 16/ncols2) {
                constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 32;
                స్థిరరాశి పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc) / warp_కొలత;
                స్థిరరాశి పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc);
                launch_fattn<DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం / ncols2, ncols2, use_logit_softcap, warp_కొలత>, warp_కొలత>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, నిజం, నిజం, అబద్ధం);
                తిరిగి;
            }
        }
        ఒకవేళ constexpr (ncols2 <= 16) {
            ఒకవేళ (Q->ne[1] > 8/ncols2) {
                constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 16;
                స్థిరరాశి పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc) / warp_కొలత;
                స్థిరరాశి పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc);
                launch_fattn<DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం / ncols2, ncols2, use_logit_softcap, warp_కొలత>, warp_కొలత>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, నిజం, నిజం, అబద్ధం);
                తిరిగి;
            }
        }
        ఒకవేళ constexpr (ncols2 <= 8) {
            ఒకవేళ (Q->ne[1] > 4/ncols2) {
                constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 8;
                స్థిరరాశి పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc) / warp_కొలత;
                స్థిరరాశి పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc);
                launch_fattn<DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం / ncols2, ncols2, use_logit_softcap, warp_కొలత>, warp_కొలత>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, నిజం, నిజం, అబద్ధం);
                తిరిగి;
            }
        }
    }

    ఒకవేళ constexpr (ncols2 <= 4) {
        ఒకవేళ (Q->ne[1] > 2/ncols2) {
            constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 4;
            స్థిరరాశి పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc) / warp_కొలత;
            స్థిరరాశి పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc);
            launch_fattn<DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం / ncols2, ncols2, use_logit_softcap, warp_కొలత>, warp_కొలత>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, నిజం, నిజం, అబద్ధం);
            తిరిగి;
        }
    }

    ఒకవేళ constexpr (ncols2 <= 2) {
        constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 2;
        స్థిరరాశి పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc) / warp_కొలత;
        స్థిరరాశి పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc);
        launch_fattn<DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం / ncols2, ncols2, use_logit_softcap, warp_కొలత>, warp_కొలత>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, నిజం, నిజం, అబద్ధం);
        తిరిగి;
    }

    {
        constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = ncols2*2;
        స్థిరరాశి పూర్ణాంకం nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc) / warp_కొలత;
        స్థిరరాశి పూర్ణాంకం nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం, cc);
        launch_fattn<DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, నిలువుల_సంఖ్య_చొప్పున_ఖండం / ncols2, ncols2, use_logit_softcap, warp_కొలత>, warp_కొలత>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, నిజం, నిజం, అబద్ధం);
        తిరిగి;
    }

    GGML_ABORT("fatal error");
}

మూస <పూర్ణాంకం DKQ, పూర్ణాంకం DV, తార్కిక use_logit_softcap>
స్థిర శూన్య launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * KQV  = dst;
    స్థిరరాశి ggml_tensor * Q    = dst->src[0];
    స్థిరరాశి ggml_tensor * K    = dst->src[1];
    స్థిరరాశి ggml_tensor * ముసుగు = dst->src[3];

    దశాంశం max_bias = 0.0f;
    జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) KQV->op_params + 1, పరిమాణం(దశాంశం));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    స్థిరరాశి పూర్ణాంకం gqa_హద్దు = gqa_నిష్పత్తి <= 4 && DV <= 256 ? 16 : గరిష్ఠ_పూర్ణాంకం;
    స్థిరరాశి తార్కిక use_gqa_opt = ముసుగు && max_bias == 0.0f && Q->ne[1] <= gqa_హద్దు && K->ne[1] % FATTN_KQ_STRIDE == 0;

    ఒకవేళ constexpr (DV == 512) {
        ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }
        ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }
        ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        ఒకవేళ constexpr (DKQ == DV) {
            ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                తిరిగి;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }
    }

    ఒకవేళ constexpr (DV <= 256) {
        ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }

        ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }

        ఒకవేళ (use_gqa_opt && gqa_నిష్పత్తి % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            తిరిగి;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        తిరిగి;
    }
    GGML_ABORT("fatal error");
}

మూస <పూర్ణాంకం DKQ, పూర్ణాంకం DV>
శూన్య ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * KQV = dst;

    దశాంశం logit_softcap;
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) KQV->op_params + 2, పరిమాణం(దశాంశం));

    ఒకవేళ (logit_softcap == 0.0f) {
        constexpr తార్కిక use_logit_softcap = అబద్ధం;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } లేకపోతే {
        constexpr తార్కిక use_logit_softcap = నిజం;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

శూన్య ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#నిర్వచించు DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    మూస శూన్య ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

బాహ్య DECL_FATTN_TILE_CASE( 40,  40);
బాహ్య DECL_FATTN_TILE_CASE( 64,  64);
బాహ్య DECL_FATTN_TILE_CASE( 72,  72);
బాహ్య DECL_FATTN_TILE_CASE( 80,  80);
బాహ్య DECL_FATTN_TILE_CASE( 96,  96);
బాహ్య DECL_FATTN_TILE_CASE(112, 112);
బాహ్య DECL_FATTN_TILE_CASE(128, 128);
బాహ్య DECL_FATTN_TILE_CASE(256, 256);
బాహ్య DECL_FATTN_TILE_CASE(512, 512);
బాహ్య DECL_FATTN_TILE_CASE(576, 512);

