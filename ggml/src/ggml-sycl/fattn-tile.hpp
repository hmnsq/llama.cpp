#شامل <sycl/sycl.hpp>
#شامل <sycl/ext/oneapi/work_group_static.hpp>
#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "fattn-common.hpp"

#شامل <cmath>
#شامل <float.h>

نامخلا syclex = sycl::ext::oneapi::experimental;

#تعریف_کرو GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    اگر (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        واپس ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

ساکن constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
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

    واپس 0;
}

ساکن constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
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

    واپس 0;
}

ساکن constexpr uint32_t ggml_sycl_fattn_tile_get_config(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols, ثابت صحیح cc) {
    اگر(fast_fp16_دستیاب(cc))
        واپس ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    ورنہ
        واپس ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

ساکن constexpr uint32_t ggml_sycl_fattn_tile_get_config(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
#اگر_تعریف_شدہ SYCL_FAST_FP16
    واپس ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#ورنہ
    واپس ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#اگر_ختم // SYCL_FAST_FP16
}

ساکن صحیح ggml_sycl_fattn_tile_get_nthreads(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols, ثابت صحیح cc) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

ساکن constexpr صحیح ggml_sycl_fattn_tile_get_nthreads(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

ساکن صحیح ggml_sycl_fattn_tile_get_occupancy(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols, ثابت صحیح cc) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

ساکن constexpr صحیح ggml_sycl_fattn_tile_get_occupancy(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

ساکن صحیح ggml_sycl_fattn_tile_get_nbatch_fa(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols, ثابت صحیح cc) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

ساکن constexpr صحیح ggml_sycl_fattn_tile_get_nbatch_fa(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

ساکن صحیح ggml_sycl_fattn_tile_get_nbatch_K(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols, ثابت صحیح cc) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

ساکن constexpr صحیح ggml_sycl_fattn_tile_get_nbatch_K(ثابت صحیح DKQ, ثابت صحیح DV, ثابت صحیح ncols) {
    واپس (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

سانچہ <صحیح warp_سائز, صحیح nwarps, صحیح I, صحیح J, صحیح J_padding, منطقی oob_check>
ساکن __dpct_inline__ خالی flash_attn_tile_بھرو_tile(ثابت sycl::half2 * ثابت __restrict__ KV,
                                                      sycl::half2 * ثابت __restrict__ tile_KV,
                                                      ثابت صحیح stride_KV,
                                                      ثابت صحیح i_sup) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    constexpr صحیح cpy_nb = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    خودکار load = [&] (ثابت صحیح n) {
        ثابت صحیح stride_j = warp_سائز >> n;

        اگر (stride_j == 0) {
            واپس;
        }

        ثابت صحیح j0_آغاز = stride_j == warp_سائز ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        ثابت صحیح j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        ثابت صحیح stride_i = warp_سائز / stride_j;

        اگر (j0_آغاز == j0_stop) {
            واپس;
        }

#ہدایت unroll
        برائے (صحیح i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ثابت صحیح i = i0 + شے_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_سائز ? 0 : شے_ct1.get_local_id(2) / stride_j);

            اگر (i0 + nwarps*stride_i <= I || i < I) {
#ہدایت unroll
                برائے (صحیح j0 = j0_آغاز; j0 < j0_stop; j0 += stride_j) {
                    ثابت صحیح j = j0 * cpy_ne + (stride_j == warp_سائز ? شے_ct1.get_local_id(2) :
                                                                         شے_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    ثابت __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
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

سانچہ <صحیح warp_سائز, صحیح nwarps, صحیح I, صحیح J, صحیح J_padding, منطقی oob_check>
ساکن __dpct_inline__ خالی flash_attn_tile_بھرو_tile(ثابت sycl::half2 * ثابت __restrict__ KV,
                                                      اعشاری * ثابت __restrict__ tile_KV,
                                                      ثابت صحیح stride_KV,
                                                      ثابت صحیح i_sup) {
    constexpr صحیح cpy_nb = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    خودکار load = [&] (ثابت صحیح n) {
        خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
        ثابت صحیح stride_j = warp_سائز >> n;

        اگر (stride_j == 0) {
            واپس;
        }

        ثابت صحیح j0_آغاز = stride_j == warp_سائز ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        ثابت صحیح j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        ثابت صحیح stride_i = warp_سائز / stride_j;

        اگر (j0_آغاز == j0_stop) {
            واپس;
        }

#ہدایت unroll
        برائے (صحیح i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ثابت صحیح i = i0 + شے_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_سائز ? 0 : شے_ct1.get_local_id(2) / stride_j);

            اگر (i0 + nwarps*stride_i <= I || i < I) {
#ہدایت unroll
                برائے (صحیح j0 = j0_آغاز; j0 < j0_stop; j0 += stride_j) {
                    ثابت صحیح j = j0 * (cpy_ne / 2) + (stride_j == warp_سائز ? شے_ct1.get_local_id(2) :
                                                                               شے_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    ثابت sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<حجم(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#ہدایت unroll
                    برائے (صحیح l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
                    }
                    ggml_sycl_memcpy_1<حجم(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
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
سانچہ <صحیح  warp_سائز,
          صحیح  nwarps,
          صحیح  ncols1,
          صحیح  ncols2,
          صحیح  DKQ,
          صحیح  nbatch_fa,
          صحیح  nbatch_K,
          منطقی use_logit_softcap,
          منطقی oob_check,
          قسم_نام T_vec_dot>
ساکن __dpct_inline__ خالی flash_attn_tile_iter_KQ(T_vec_dot * ثابت Q_tmp,
                                                    ثابت sycl::half2 * ثابت __restrict__ K_h2,
                                                    T_vec_dot * ثابت KV_tmp,
                                                    ثابت صحیح         stride_K2,
                                                    ثابت صحیح         k_VKQ_0,
                                                    ثابت صحیح         k_VKQ_sup,
                                                    ثابت صحیح         k_KQ_0,
                                                    اعشاری *           KQ_acc) {
    خودکار          شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    constexpr صحیح cpy_nb   = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    constexpr صحیح ncols = ncols1*ncols2;
    constexpr صحیح cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr صحیح np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_بھرو_tile<warp_سائز, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    شے_ct1.barrier(sycl::access::fence_space::local_space);

#اگر_تعریف_شدہ SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#ہدایت unroll
    برائے (صحیح k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_سائز)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#ورنہ
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#ہدایت unroll
    برائے (صحیح k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) اعشاری K_k[nbatch_fa/(np*warp_سائز)][cpy_ne];
        __dpct_align__(16) اعشاری Q_k[cpw][cpy_ne];
#اگر_ختم // SYCL_FAST_FP16

#ہدایت unroll
        برائے (صحیح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_سائز) {
            ثابت صحیح i_KQ = i_KQ_0 + (شے_ct1.get_local_id(1) % np) * warp_سائز + شے_ct1.get_local_id(2);

#اگر_تعریف_شدہ SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_سائز)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#ورنہ
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_سائز)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#اگر_ختم // SYCL_FAST_FP16
        }
#ہدایت unroll
        برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
            ثابت صحیح jc = jc0 + (شے_ct1.get_local_id(1) / np) * cpw;

#اگر_تعریف_شدہ SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#ورنہ
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#اگر_ختم // SYCL_FAST_FP16
        }

#ہدایت unroll
        برائے (صحیح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_سائز) {
#ہدایت unroll
            برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
#ہدایت unroll
                برائے (صحیح k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_سائز)*cpw + jc0], K_k[i_KQ_0/(np*warp_سائز)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    اگر (k_KQ_0 + nbatch_K < DKQ) {
        شے_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
سانچہ <صحیح  warp_سائز,
          صحیح  nwarps,
          صحیح  ncols1,
          صحیح  ncols2,
          صحیح  DKQ,
          صحیح  DV,
          صحیح  nbatch_fa,
          صحیح  nbatch_K,
          منطقی use_logit_softcap,
          منطقی oob_check,
          قسم_نام T_vec_dot,
          قسم_نام T_KQ,
          قسم_نام T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ساکن __dpct_inline__ خالی flash_attn_tile_iter(T_vec_dot * ثابت Q_tmp,
                                                 ثابت sycl::half2 * ثابت __restrict__ K_h2,
                                                 ثابت sycl::half2 * ثابت __restrict__ V_h2,
                                                 ثابت sycl::half * ثابت __restrict__ نقاب,
                                                 ثابت sycl::uint3 ne01,
                                                 ثابت اعشاری       logit_softcap,
                                                 ثابت اعشاری       ڈھلان,
                                                 T_KQ * ثابت      KQ,
                                                 T_vec_dot * ثابت KV_tmp,
                                                 ثابت صحیح         stride_K2,
                                                 ثابت صحیح         stride_V2,
                                                 ثابت صحیح         stride_نقاب,
                                                 اعشاری * ثابت     KQ_max,
                                                 اعشاری * ثابت     KQ_مجموع,
                                                 T_acc * ثابت     VKQ,
                                                 ثابت صحیح         k_VKQ_0,
                                                 ثابت صحیح         k_VKQ_max,
                                                 ثابت صحیح         ستون_Q_0,
                                                 اعشاری *           KQ_max_new_shared) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    constexpr صحیح cpy_nb   = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    constexpr صحیح ncols = ncols1*ncols2;
    constexpr صحیح cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr صحیح np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr صحیح DVp = (DV + 2*warp_سائز - 1) & ~(2*warp_سائز - 1); // DV padded to multiple of 2*warp_size.

#اگر_تعریف_شدہ SYCL_FAST_FP16
    constexpr صحیح KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#ورنہ
    constexpr صحیح KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#اگر_ختم // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    ثابت صحیح k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    اعشاری KQ_max_new[cpw];
#ہدایت unroll
    برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    اعشاری KQ_acc[nbatch_fa/(np*warp_سائز) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr صحیح nbatch_K_آخری = DKQ % nbatch_K;
#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_آخری; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_سائز, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    اگر (nbatch_K_آخری > 0) {
        constexpr صحیح k_KQ_0 = DKQ - nbatch_K_آخری;
        flash_attn_tile_iter_KQ<warp_سائز, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_آخری, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#ہدایت unroll
    برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحیح j = fastmodulo(ستون_Q_0 + (jc0 + (شے_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#ہدایت unroll
        برائے (صحیح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_سائز) {
            ثابت صحیح i_KQ = i_KQ_0 + (شے_ct1.get_local_id(1) % np) * warp_سائز + شے_ct1.get_local_id(2);

#اگر defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_سائز)*cpw + jc0] *= 4.0f;
#اگر_ختم // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            اگر (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_سائز)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((اعشاری) KQ_acc[(i_KQ_0 / (np * warp_سائز)) * cpw + jc0]);
            }

            اگر (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_سائز)) * cpw + jc0] +=
                    (ncols2 > 1 || نقاب) ? ڈھلان * sycl::vec<sycl::half, 1>(نقاب[j * stride_نقاب + k_VKQ_0 + i_KQ])
                                                       .convert<اعشاری, sycl::rounding_منوال::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((اعشاری) KQ_max_new[jc0],
                               (اعشاری) (KQ_acc[(i_KQ_0 / (np * warp_سائز)) * cpw + jc0] + FATTN_KQ_MAX_انحراف));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_سائز>(KQ_max_new[jc0]);
    }

    اگر constexpr (np == 1) {
        شے_ct1.barrier(sycl::access::fence_space::local_space);
    } ورنہ {
        static_assert(cpw == 1, "bad cpw");

        اگر (شے_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[شے_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(شے_ct1.get_local_id(1) & ~(np - 1)) + شے_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#ہدایت unroll
    برائے (صحیح jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#اگر_تعریف_شدہ SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_سائز)][KQ_cs];
#ورنہ
        __dpct_align__(16) اعشاری tmp[nbatch_fa/(np*warp_سائز)][KQ_cs];
#اگر_ختم // SYCL_FAST_FP16

#ہدایت unroll
        برائے (صحیح jc1 = 0; jc1 < KQ_cs; ++jc1) {
            ثابت صحیح jc = jc0 + jc1;

            ثابت اعشاری KQ_max_scale = sycl::native::قوت_نما((اعشاری) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            اعشاری KQ_مجموع_جوڑو = 0.0f;
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < nbatch_fa; i0 += np*warp_سائز) {
                ثابت اعشاری val =
                    !oob_check || i0 + (شے_ct1.get_local_id(1) % np) * warp_سائز + شے_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::قوت_نما((اعشاری) (KQ_acc[(i0 / (np * warp_سائز)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_مجموع_جوڑو += val;
                tmp[i0/(np*warp_سائز)][jc1] = val;
            }
            KQ_مجموع[jc] = KQ_مجموع[jc]*KQ_max_scale + KQ_مجموع_جوڑو;

#اگر_تعریف_شدہ SYCL_FAST_FP16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز) {
                VKQ[jc*((DVp/2)/warp_سائز) + i0/warp_سائز].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_سائز) + i0/warp_سائز].y() *= KQ_max_scale_h2.y();
            }
#ورنہ
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز) {
                VKQ[jc*((DVp/2)/warp_سائز) + i0/warp_سائز].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_سائز) + i0/warp_سائز].y() *= KQ_max_scale;
            }
#اگر_ختم // SYCL_FAST_FP16
        }

#ہدایت unroll
        برائے (صحیح i0 = 0; i0 < nbatch_fa; i0 += np*warp_سائز) {
            ثابت صحیح i = i0 + (شے_ct1.get_local_id(1) % np) * warp_سائز + شے_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<حجم(tmp[0])>(
                KQ + (jc0 / KQ_cs + (شے_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_سائز)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr صحیح nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#ہدایت unroll
    برائے (صحیح k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_بھرو_tile<warp_سائز, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        شے_ct1.barrier(sycl::access::fence_space::local_space);

#اگر_تعریف_شدہ SYCL_FAST_FP16
#ہدایت unroll
        برائے (صحیح k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_سائز];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr صحیح cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_سائز ? cpy_ne/2 : (DVp/2)/warp_سائز;
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_سائز],
                                                 &KV_tmp[(k1 + شے_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         شے_ct1.get_local_id(2) * cpy_ne_D]);
            }
#ہدایت unroll
            برائے (صحیح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ثابت صحیح jc_KQ = jc_VKQ_0 / KQ_cs + (شے_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * حجم(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + شے_ct1.get_local_id(1) % np) * KQ_cs);
#ہدایت unroll
                برائے (صحیح jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز) {
#ہدایت unroll
                برائے (صحیح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائز) + i0/warp_سائز].x() +=
                        V_k[i0/warp_سائز].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائز) + i0/warp_سائز].y() +=
                        V_k[i0/warp_سائز].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#ورنہ
#ہدایت unroll
        برائے (صحیح k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_سائز];
            __dpct_align__(16) اعشاری  KQ_k[cpw];

            constexpr صحیح cpy_ne_D = cpy_ne < DVp/warp_سائز ? cpy_ne : DVp/warp_سائز;
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp; i0 += warp_سائز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_سائز)], &KV_tmp[(k1 + شے_ct1.get_local_id(1) % np)*DV + i0 + شے_ct1.get_local_id(2)*cpy_ne_D]);
            }
#ہدایت unroll
            برائے (صحیح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ثابت صحیح jc_KQ = jc_VKQ_0/KQ_cs + (شے_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*حجم(اعشاری)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + شے_ct1.get_local_id(1) % np)*KQ_cs);
            }

#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز) {
#ہدایت unroll
                برائے (صحیح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائز) + i0/warp_سائز].x() += V_k[i0/warp_سائز].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائز) + i0/warp_سائز].y() += V_k[i0/warp_سائز].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#اگر_ختم // SYCL_FAST_FP16
        شے_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

سانچہ <صحیح DKQ, صحیح DV, صحیح ncols1, صحیح ncols2, منطقی use_logit_softcap, صحیح warp_سائز>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ساکن خالی flash_attn_tile(ثابت حرف *  Q,
                            ثابت حرف *  K,
                            ثابت حرف *  V,
                            ثابت حرف *  نقاب,
                            ثابت حرف *  sinks,
                            ثابت صحیح *  KV_max,
                            اعشاری *  dst,
                            sycl::float2 *  dst_meta,
                            ثابت اعشاری          scale,
                            ثابت اعشاری          max_bias,
                            ثابت اعشاری          m0,
                            ثابت اعشاری          m1,
                            ثابت uint32_t       n_سر_log2,
                            ثابت اعشاری          logit_softcap,
                            ثابت int32_t        ne00,
                            ثابت sycl::uint3    ne01,
                            ثابت int32_t        ne02,
                            ثابت int32_t        ne03,
                            ثابت int32_t        nb01,
                            ثابت int32_t        nb02,
                            ثابت int32_t        nb03,
                            ثابت int32_t        ne10,
                            ثابت int32_t        ne11,
                            ثابت int32_t        ne12,
                            ثابت int32_t        ne13,
                            ثابت int32_t        nb11,
                            ثابت int32_t        nb12,
                            ثابت int64_t        nb13,
                            ثابت int32_t        nb21,
                            ثابت int32_t        nb22,
                            ثابت int64_t        nb23,
                            ثابت int32_t        ne31,
                            ثابت int32_t        ne32,
                            ثابت int32_t        ne33,
                            ثابت int32_t        nb31,
                            ثابت int32_t        nb32,
                            ثابت int64_t        nb33) {
#اگر_تعریف_شدہ SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    اگر ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_غیر_مستعمل_VARS(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_سر_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        واپس;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr صحیح ncols     = ncols1*ncols2;

    constexpr صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_سائز;
    constexpr صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr صحیح nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    ثابت صحیح ستون_Q_0 = شے_ct1.get_گروہ(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    ثابت صحیح           sequence  = شے_ct1.get_گروہ(0) / (ne02 / ncols2);
    ثابت صحیح           head0     = شے_ct1.get_گروہ(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    ثابت صحیح gqa_تناسب = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    ثابت اعشاری * Q_f  = (ثابت اعشاری *) (Q + nb03*sequence + nb02* head0);
    ثابت sycl::half2 * K_h2      = (ثابت sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_تناسب));
    ثابت sycl::half2 * V_h2 =
        (ثابت sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_تناسب));  // K and V have same shape

    ثابت sycl::half * maskh = نقاب ? (ثابت sycl::half *) (نقاب + nb33 * (sequence % ne33)) : خالی_اشاریہ;

    ثابت صحیح stride_K2   = nb11 / حجم(sycl::half2);
    ثابت صحیح stride_V2   = nb21 / حجم(sycl::half2);
    ثابت صحیح stride_نقاب = nb31 / حجم(sycl::half);

    ثابت اعشاری ڈھلان = ncols2 == 1 ? get_alibi_ڈھلان(max_bias, head0, n_سر_log2, m0, m1) : 1.0f;

    constexpr صحیح cpy_nb = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    constexpr صحیح cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr صحیح np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_سائز) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr صحیح DKQp = (DKQ + 2*warp_سائز - 1) & ~(2*warp_سائز - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr صحیح DVp  = (DV  + 2*warp_سائز - 1) & ~(2*warp_سائز - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#اگر_تعریف_شدہ SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_سائز = lsm_size1 * حجم(sycl::half2) +
                                            lsm_size2 * حجم(sycl::half2) +
                                            lsm_size3 * حجم(sycl::half) +
                                            lsm_size4 * حجم(اعشاری);

    syclex::work_گروہ_static<حرف[local_share_mem_سائز]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    اعشاری *KQ_max_new_shared = (اعشاری *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_سائز)] = {
        { 0.0f, 0.0f }
    };
#ورنہ
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_سائز = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * حجم(اعشاری);

    syclex::work_گروہ_static<حرف[local_share_mem_سائز]> lsm;

    اعشاری *Q_tmp = (اعشاری *)&lsm;
    اعشاری *KV_tmp = Q_tmp +lsm_size1;
    اعشاری *KQ = KV_tmp+lsm_size2;
    اعشاری *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_سائز)] = {{0.0f, 0.0f}};


#اگر_ختم // SYCL_FAST_FP16

    اعشاری KQ_max[cpw] = {};

#ہدایت unroll
    برائے (صحیح j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    اعشاری KQ_مجموع[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#ہدایت unroll
    برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحیح jc = jc0 + (شے_ct1.get_local_id(1) / np) * cpw;

        ثابت صحیح j = jc / ncols2;
        ثابت صحیح c = jc % ncols2;

        constexpr صحیح cpy_ne_D = cpy_ne < DKQp/warp_سائز ? cpy_ne : DKQp/warp_سائز;

#ہدایت unroll
        برائے (صحیح i0 = 0; i0 < DKQp; i0 += np*warp_سائز*cpy_ne_D) {
            اگر (i0 + np * warp_سائز * cpy_ne_D <= DKQ ||
                i0 + (شے_ct1.get_local_id(1) % np) * (warp_سائز * cpy_ne_D) + شے_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) اعشاری tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<حجم(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / حجم(اعشاری)) + fastmodulo(ستون_Q_0 + j, ne01) * (nb01 / حجم(اعشاری)) +
                                i0 + (شے_ct1.get_local_id(1) % np) * (warp_سائز * cpy_ne_D) +
                                شے_ct1.get_local_id(2) * cpy_ne_D]);

#ہدایت unroll
                برائے (صحیح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#اگر_تعریف_شدہ SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#ہدایت unroll
                برائے (صحیح i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#اگر defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#اگر_ختم // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<حجم(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (شے_ct1.get_local_id(1) % np) * (warp_سائز * cpy_ne_D / 2) +
                           شے_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#ورنہ
                ggml_sycl_memcpy_1<حجم(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (شے_ct1.get_local_id(1) % np)*(warp_سائز*cpy_ne_D)   + شے_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#اگر_ختم // SYCL_FAST_FP16
            }
        }
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    ثابت صحیح k_VKQ_max = KV_max ? KV_max[sequence * شے_ct1.get_گروہ_range(2) + شے_ct1.get_گروہ(2)] : ne11;
    اگر (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        صحیح k_VKQ_0 = شے_ct1.get_گروہ(1) * nbatch_fa;
        جب (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr منطقی oob_check = جھوٹ;
            flash_attn_tile_iter<warp_سائز, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ڈھلان, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_نقاب, KQ_max, KQ_مجموع, VKQ, k_VKQ_0, k_VKQ_max, ستون_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += شے_ct1.get_گروہ_range(1) * nbatch_fa;
        }
        اگر (k_VKQ_0 < k_VKQ_max) {
            constexpr منطقی oob_check = سچ;
            flash_attn_tile_iter<warp_سائز, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ڈھلان, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_نقاب, KQ_max, KQ_مجموع, VKQ, k_VKQ_0, k_VKQ_max, ستون_Q_0,
                                            KQ_max_new_shared);
        }
    } ورنہ {
        // Branch without out-of-bounds checks.
        برائے (صحیح k_VKQ_0 = شے_ct1.get_گروہ(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += شے_ct1.get_گروہ_range(1) * nbatch_fa) {

            constexpr منطقی oob_check = جھوٹ;
            flash_attn_tile_iter<warp_سائز, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ڈھلان, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_نقاب, KQ_max, KQ_مجموع, VKQ, k_VKQ_0, k_VKQ_max, ستون_Q_0,
                                            KQ_max_new_shared);
        }
    }

#ہدایت unroll
    برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_مجموع[jc0] = warp_reduce_مجموع<warp_سائز>(KQ_مجموع[jc0]);
    }

    اگر constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#اگر_تعریف_شدہ SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#ورنہ
        اعشاری * VKQ_combine    = (اعشاری *) KV_tmp;
#اگر_ختم // SYCL_FAST_FP16

        اعشاری * KQ_مجموع_combine = (اعشاری *) Q_tmp;

        اگر (شے_ct1.get_local_id(1) % np != 0) {

#اگر_تعریف_شدہ SYCL_FAST_FP16
            constexpr صحیح cpy_ne_D = cpy_ne < (DVp/2)/warp_سائز ? cpy_ne : (DVp/2)/warp_سائز;
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[شے_ct1.get_local_id(1) * (DVp / 2) + i0 + شے_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_سائز]);
            }
#ورنہ

            constexpr صحیح cpy_ne_D = cpy_ne < DVp/warp_سائز ? cpy_ne : DVp/warp_سائز;

#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp; i0 += warp_سائز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[شے_ct1.get_local_id(1)*DVp + i0 + شے_ct1.get_local_id(2)*cpy_ne_D], ((ثابت اعشاری *) VKQ) + i0/warp_سائز);
            }
#اگر_ختم // SYCL_FAST_FP16

            اگر (شے_ct1.get_local_id(2) == 0) {
                KQ_مجموع_combine[شے_ct1.get_local_id(1)] = KQ_مجموع[0];
            }
            واپس;
        }

        شے_ct1.barrier(sycl::access::fence_space::local_space);

#ہدایت unroll
        برائے (صحیح ip = 1; ip < np; ++ip) {
#اگر_تعریف_شدہ SYCL_FAST_FP16
            constexpr صحیح cpy_ne_D = cpy_ne < (DVp/2)/warp_سائز ? cpy_ne : (DVp/2)/warp_سائز;
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(شے_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   شے_ct1.get_local_id(2) * cpy_ne_D]);
#ہدایت unroll
                برائے (صحیح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_سائز + i1] += tmp[i1];
                }
            }
#ورنہ
            constexpr صحیح cpy_ne_D = cpy_ne < DVp/warp_سائز ? cpy_ne : DVp/warp_سائز;
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp; i0 += warp_سائز*cpy_ne_D) {
                __dpct_align__(16) اعشاری tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(شے_ct1.get_local_id(1) + ip)*DVp + i0 + شے_ct1.get_local_id(2)*cpy_ne_D]);
#ہدایت unroll
                برائے (صحیح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((اعشاری *)VKQ)[i0/warp_سائز + i1] += tmp[i1];
                }
            }
#اگر_ختم // SYCL_FAST_FP16

            KQ_مجموع[0] += KQ_مجموع_combine[شے_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    اگر (sinks && شے_ct1.get_گروہ(1) == 0) {
#ہدایت unroll
        برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
            ثابت صحیح   jc   = jc0 + (شے_ct1.get_local_id(1) / np) * cpw;
            ثابت اعشاری sink = ((ثابت اعشاری *) sinks)[head0 + jc % ncols2];

            اعشاری       KQ_max_new_j = sycl::fmax((اعشاری) KQ_max[jc0], sink);
            ثابت اعشاری KQ_max_scale = sycl::native::قوت_نما((اعشاری) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            ثابت اعشاری val = sycl::native::قوت_نما((اعشاری) (sink - KQ_max[jc0]));
            KQ_مجموع[jc0] = KQ_مجموع[jc0]*KQ_max_scale + val;

#اگر_تعریف_شدہ SYCL_FAST_FP16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز) {
                VKQ[jc0*((DVp/2)/warp_سائز) + i0/warp_سائز] *= KQ_max_scale_h2;
            }
#ورنہ
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز) {
                VKQ[jc0*((DVp/2)/warp_سائز) + i0/warp_سائز].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_سائز) + i0/warp_سائز].y() *= KQ_max_scale;
            }
#اگر_ختم // SYCL_FAST_FP16
        }
    }

    // Write back results:
#ہدایت unroll
    برائے (صحیح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحیح jc = jc0 + (شے_ct1.get_local_id(1) / np) * cpw;

        ثابت صحیح j = jc / ncols2;
        ثابت صحیح c = jc % ncols2;

        اگر (ncols1 > 1 && ستون_Q_0 + j >= صحیح(ne01.z())) {
            واپس;
        }

        ثابت اعشاری scale = شے_ct1.get_گروہ_range(1) == 1 ? 1.0f / KQ_مجموع[jc0] : 1.0f;

        ثابت صحیح j_dst_unrolled =
            ((sequence * صحیح(ne01.z()) + ستون_Q_0 + j) * ne02 + head0 + c) * شے_ct1.get_گروہ_range(1) +
            شے_ct1.get_گروہ(1);

#اگر_تعریف_شدہ SYCL_FAST_FP16
        constexpr صحیح cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_سائز ? cpy_ne/2 : (DVp/2)/warp_سائز;
#ہدایت unroll
        برائے (صحیح i0 = 0; i0 < DVp/2; i0 += warp_سائز*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#ہدایت unroll
            برائے (صحیح i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_سائز) + i0 / warp_سائز + i1]
                              .سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            اگر (i0 + warp_سائز * cpy_ne_D <= DV / 2 || i0 + شے_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<حجم(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + شے_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#ورنہ
        constexpr صحیح cpy_ne_D = cpy_ne < DVp/warp_سائز ? cpy_ne : DVp/warp_سائز;
#ہدایت unroll
        برائے (صحیح i0 = 0; i0 < DVp; i0 += warp_سائز*cpy_ne_D) {
            اگر (i0 + warp_سائز*cpy_ne_D <= DV || i0 + شے_ct1.get_local_id(2)*cpy_ne_D < DV) {
#ہدایت unroll
                برائے (صحیح i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_سائز) + i0/(2*warp_سائز) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_سائز) + i0/(2*warp_سائز) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + شے_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_سائز) + i0/(2*warp_سائز)]);
            }
        }
#اگر_ختم // SYCL_FAST_FP16

        اگر (شے_ct1.get_گروہ_range(1) != 1 && شے_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_مجموع[jc0]);
        }
    }
#ورنہ
    GGML_غیر_مستعمل_VARS(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_سر_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#اگر_ختم // SYCL_FLASH_ATTN
}

سانچہ <صحیح DKQ, صحیح DV, صحیح ncols2, منطقی use_logit_softcap>
ساکن خالی launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * Q = dst->src[0];

    ثابت صحیح id        = ggml_sycl_get_device();
    ثابت صحیح cc        = ggml_sycl_info().devices[id].cc;
    ثابت صحیح warp_سائز = WARP_32_سائز; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    اگر (DV < 512 && Q->ne[1] < 32) {
        اگر constexpr (ncols2 <= 32) {
            اگر (Q->ne[1] > 16/ncols2) {
                constexpr صحیح ستون_تعداد_فی_بلاک = 32;
                ثابت صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ستون_تعداد_فی_بلاک, cc) / warp_سائز;
                ثابت صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ستون_تعداد_فی_بلاک, cc);
                launch_fattn<DV, ستون_تعداد_فی_بلاک/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ستون_تعداد_فی_بلاک / ncols2, ncols2, use_logit_softcap, warp_سائز>, warp_سائز>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, جھوٹ);
                واپس;
            }
        }
        اگر constexpr (ncols2 <= 16) {
            اگر (Q->ne[1] > 8/ncols2) {
                constexpr صحیح ستون_تعداد_فی_بلاک = 16;
                ثابت صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ستون_تعداد_فی_بلاک, cc) / warp_سائز;
                ثابت صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ستون_تعداد_فی_بلاک, cc);
                launch_fattn<DV, ستون_تعداد_فی_بلاک/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ستون_تعداد_فی_بلاک / ncols2, ncols2, use_logit_softcap, warp_سائز>, warp_سائز>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, جھوٹ);
                واپس;
            }
        }
        اگر constexpr (ncols2 <= 8) {
            اگر (Q->ne[1] > 4/ncols2) {
                constexpr صحیح ستون_تعداد_فی_بلاک = 8;
                ثابت صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ستون_تعداد_فی_بلاک, cc) / warp_سائز;
                ثابت صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ستون_تعداد_فی_بلاک, cc);
                launch_fattn<DV, ستون_تعداد_فی_بلاک/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ستون_تعداد_فی_بلاک / ncols2, ncols2, use_logit_softcap, warp_سائز>, warp_سائز>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, جھوٹ);
                واپس;
            }
        }
    }

    اگر constexpr (ncols2 <= 4) {
        اگر (Q->ne[1] > 2/ncols2) {
            constexpr صحیح ستون_تعداد_فی_بلاک = 4;
            ثابت صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ستون_تعداد_فی_بلاک, cc) / warp_سائز;
            ثابت صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ستون_تعداد_فی_بلاک, cc);
            launch_fattn<DV, ستون_تعداد_فی_بلاک/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, ستون_تعداد_فی_بلاک / ncols2, ncols2, use_logit_softcap, warp_سائز>, warp_سائز>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, جھوٹ);
            واپس;
        }
    }

    اگر constexpr (ncols2 <= 2) {
        constexpr صحیح ستون_تعداد_فی_بلاک = 2;
        ثابت صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ستون_تعداد_فی_بلاک, cc) / warp_سائز;
        ثابت صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ستون_تعداد_فی_بلاک, cc);
        launch_fattn<DV, ستون_تعداد_فی_بلاک/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, ستون_تعداد_فی_بلاک / ncols2, ncols2, use_logit_softcap, warp_سائز>, warp_سائز>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, جھوٹ);
        واپس;
    }

    {
        constexpr صحیح ستون_تعداد_فی_بلاک = ncols2*2;
        ثابت صحیح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ستون_تعداد_فی_بلاک, cc) / warp_سائز;
        ثابت صحیح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ستون_تعداد_فی_بلاک, cc);
        launch_fattn<DV, ستون_تعداد_فی_بلاک/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, ستون_تعداد_فی_بلاک / ncols2, ncols2, use_logit_softcap, warp_سائز>, warp_سائز>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, جھوٹ);
        واپس;
    }

    GGML_ABORT("fatal error");
}

سانچہ <صحیح DKQ, صحیح DV, منطقی use_logit_softcap>
ساکن خالی launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV  = dst;
    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * نقاب = dst->src[3];

    اعشاری max_bias = 0.0f;
    حافظہ_نقل(&max_bias, (ثابت اعشاری *) KQV->op_params + 1, حجم(اعشاری));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    ثابت صحیح gqa_تناسب = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    ثابت صحیح gqa_حد_بندی = gqa_تناسب <= 4 && DV <= 256 ? 16 : زیادہ_سے_زیادہ_عدد;
    ثابت منطقی use_gqa_opt = نقاب && max_bias == 0.0f && Q->ne[1] <= gqa_حد_بندی && K->ne[1] % FATTN_KQ_STRIDE == 0;

    اگر constexpr (DV == 512) {
        اگر (use_gqa_opt && gqa_تناسب % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            واپس;
        }
        اگر (use_gqa_opt && gqa_تناسب % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            واپس;
        }
        اگر (use_gqa_opt && gqa_تناسب % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            واپس;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        اگر constexpr (DKQ == DV) {
            اگر (use_gqa_opt && gqa_تناسب % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                واپس;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            واپس;
        }
    }

    اگر constexpr (DV <= 256) {
        اگر (use_gqa_opt && gqa_تناسب % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            واپس;
        }

        اگر (use_gqa_opt && gqa_تناسب % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            واپس;
        }

        اگر (use_gqa_opt && gqa_تناسب % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            واپس;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        واپس;
    }
    GGML_ABORT("fatal error");
}

سانچہ <صحیح DKQ, صحیح DV>
خالی ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV = dst;

    اعشاری logit_softcap;
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) KQV->op_params + 2, حجم(اعشاری));

    اگر (logit_softcap == 0.0f) {
        constexpr منطقی use_logit_softcap = جھوٹ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } ورنہ {
        constexpr منطقی use_logit_softcap = سچ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

خالی ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#تعریف_کرو DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    سانچہ خالی ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

خارجی DECL_FATTN_TILE_CASE( 40,  40);
خارجی DECL_FATTN_TILE_CASE( 64,  64);
خارجی DECL_FATTN_TILE_CASE( 72,  72);
خارجی DECL_FATTN_TILE_CASE( 80,  80);
خارجی DECL_FATTN_TILE_CASE( 96,  96);
خارجی DECL_FATTN_TILE_CASE(112, 112);
خارجی DECL_FATTN_TILE_CASE(128, 128);
خارجی DECL_FATTN_TILE_CASE(256, 256);
خارجی DECL_FATTN_TILE_CASE(512, 512);
خارجی DECL_FATTN_TILE_CASE(576, 512);

