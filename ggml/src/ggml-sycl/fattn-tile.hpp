#ضمن <sycl/sycl.hpp>
#ضمن <sycl/ext/oneapi/work_group_static.hpp>
#ضمن "dpct/helper.hpp"
#ضمن "common.hpp"
#ضمن "fattn-common.hpp"

#ضمن <cmath>
#ضمن <float.h>

نطاق syclex = sycl::ext::oneapi::experimental;

#وصف_ڪريو GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    إذا (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        أرجع ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

ساكن constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
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

    أرجع 0;
}

ساكن constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
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

    أرجع 0;
}

ساكن constexpr uint32_t ggml_sycl_fattn_tile_get_config(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols, ثابت صحيح cc) {
    إذا(fast_fp16_موجود(cc))
        أرجع ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    وإلا
        أرجع ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

ساكن constexpr uint32_t ggml_sycl_fattn_tile_get_config(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
    أرجع ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#وإلا
    أرجع ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#جيڪڏهن_پورو // SYCL_FAST_FP16
}

ساكن صحيح ggml_sycl_fattn_tile_get_nthreads(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols, ثابت صحيح cc) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

ساكن constexpr صحيح ggml_sycl_fattn_tile_get_nthreads(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

ساكن صحيح ggml_sycl_fattn_tile_get_occupancy(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols, ثابت صحيح cc) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

ساكن constexpr صحيح ggml_sycl_fattn_tile_get_occupancy(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

ساكن صحيح ggml_sycl_fattn_tile_get_nbatch_fa(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols, ثابت صحيح cc) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

ساكن constexpr صحيح ggml_sycl_fattn_tile_get_nbatch_fa(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

ساكن صحيح ggml_sycl_fattn_tile_get_nbatch_K(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols, ثابت صحيح cc) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

ساكن constexpr صحيح ggml_sycl_fattn_tile_get_nbatch_K(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
    أرجع (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

سانچو <صحيح warp_سائيز, صحيح nwarps, صحيح I, صحيح J, صحيح J_padding, منطقي oob_check>
ساكن __dpct_inline__ فراغ flash_attn_tile_ڀريو_tile(ثابت sycl::half2 * ثابت __restrict__ KV,
                                                      sycl::half2 * ثابت __restrict__ tile_KV,
                                                      ثابت صحيح stride_KV,
                                                      ثابت صحيح i_sup) {
    خودڪار      شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_بائيٽون();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    خودڪار load = [&] (ثابت صحيح n) {
        ثابت صحيح stride_j = warp_سائيز >> n;

        إذا (stride_j == 0) {
            أرجع;
        }

        ثابت صحيح j0_شروعات = stride_j == warp_سائيز ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        ثابت صحيح j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        ثابت صحيح stride_i = warp_سائيز / stride_j;

        إذا (j0_شروعات == j0_stop) {
            أرجع;
        }

#هدايت unroll
        لكل (صحيح i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ثابت صحيح i = i0 + شي_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_سائيز ? 0 : شي_ct1.get_local_id(2) / stride_j);

            إذا (i0 + nwarps*stride_i <= I || i < I) {
#هدايت unroll
                لكل (صحيح j0 = j0_شروعات; j0 < j0_stop; j0 += stride_j) {
                    ثابت صحيح j = j0 * cpy_ne + (stride_j == warp_سائيز ? شي_ct1.get_local_id(2) :
                                                                         شي_ct1.get_local_id(2) % stride_j) *
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

سانچو <صحيح warp_سائيز, صحيح nwarps, صحيح I, صحيح J, صحيح J_padding, منطقي oob_check>
ساكن __dpct_inline__ فراغ flash_attn_tile_ڀريو_tile(ثابت sycl::half2 * ثابت __restrict__ KV,
                                                      عائم * ثابت __restrict__ tile_KV,
                                                      ثابت صحيح stride_KV,
                                                      ثابت صحيح i_sup) {
    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_بائيٽون();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    خودڪار load = [&] (ثابت صحيح n) {
        خودڪار      شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
        ثابت صحيح stride_j = warp_سائيز >> n;

        إذا (stride_j == 0) {
            أرجع;
        }

        ثابت صحيح j0_شروعات = stride_j == warp_سائيز ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        ثابت صحيح j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        ثابت صحيح stride_i = warp_سائيز / stride_j;

        إذا (j0_شروعات == j0_stop) {
            أرجع;
        }

#هدايت unroll
        لكل (صحيح i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ثابت صحيح i = i0 + شي_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_سائيز ? 0 : شي_ct1.get_local_id(2) / stride_j);

            إذا (i0 + nwarps*stride_i <= I || i < I) {
#هدايت unroll
                لكل (صحيح j0 = j0_شروعات; j0 < j0_stop; j0 += stride_j) {
                    ثابت صحيح j = j0 * (cpy_ne / 2) + (stride_j == warp_سائيز ? شي_ct1.get_local_id(2) :
                                                                               شي_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    ثابت sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<حجم(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#هدايت unroll
                    لكل (صحيح l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].سانچو convert<عائم, sycl::rounding_گهڻائي::automatic>();
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
سانچو <صحيح  warp_سائيز,
          صحيح  nwarps,
          صحيح  ncols1,
          صحيح  ncols2,
          صحيح  DKQ,
          صحيح  nbatch_fa,
          صحيح  nbatch_K,
          منطقي use_logit_softcap,
          منطقي oob_check,
          قسم_نالو T_vec_dot>
ساكن __dpct_inline__ فراغ flash_attn_tile_iter_KQ(T_vec_dot * ثابت Q_tmp,
                                                    ثابت sycl::half2 * ثابت __restrict__ K_h2,
                                                    T_vec_dot * ثابت KV_tmp,
                                                    ثابت صحيح         stride_K2,
                                                    ثابت صحيح         k_VKQ_0,
                                                    ثابت صحيح         k_VKQ_sup,
                                                    ثابت صحيح         k_KQ_0,
                                                    عائم *           KQ_acc) {
    خودڪار          شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    constexpr صحيح cpy_nb   = ggml_sycl_get_max_cpy_بائيٽون();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    constexpr صحيح ncols = ncols1*ncols2;
    constexpr صحيح cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr صحيح np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_ڀريو_tile<warp_سائيز, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    شي_ct1.barrier(sycl::access::fence_space::local_space);

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#هدايت unroll
    لكل (صحيح k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_سائيز)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#وإلا
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#هدايت unroll
    لكل (صحيح k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) عائم K_k[nbatch_fa/(np*warp_سائيز)][cpy_ne];
        __dpct_align__(16) عائم Q_k[cpw][cpy_ne];
#جيڪڏهن_پورو // SYCL_FAST_FP16

#هدايت unroll
        لكل (صحيح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_سائيز) {
            ثابت صحيح i_KQ = i_KQ_0 + (شي_ct1.get_local_id(1) % np) * warp_سائيز + شي_ct1.get_local_id(2);

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_سائيز)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#وإلا
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_سائيز)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#جيڪڏهن_پورو // SYCL_FAST_FP16
        }
#هدايت unroll
        لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
            ثابت صحيح jc = jc0 + (شي_ct1.get_local_id(1) / np) * cpw;

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#وإلا
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#جيڪڏهن_پورو // SYCL_FAST_FP16
        }

#هدايت unroll
        لكل (صحيح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_سائيز) {
#هدايت unroll
            لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
#هدايت unroll
                لكل (صحيح k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_سائيز)*cpw + jc0], K_k[i_KQ_0/(np*warp_سائيز)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    إذا (k_KQ_0 + nbatch_K < DKQ) {
        شي_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
سانچو <صحيح  warp_سائيز,
          صحيح  nwarps,
          صحيح  ncols1,
          صحيح  ncols2,
          صحيح  DKQ,
          صحيح  DV,
          صحيح  nbatch_fa,
          صحيح  nbatch_K,
          منطقي use_logit_softcap,
          منطقي oob_check,
          قسم_نالو T_vec_dot,
          قسم_نالو T_KQ,
          قسم_نالو T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ساكن __dpct_inline__ فراغ flash_attn_tile_iter(T_vec_dot * ثابت Q_tmp,
                                                 ثابت sycl::half2 * ثابت __restrict__ K_h2,
                                                 ثابت sycl::half2 * ثابت __restrict__ V_h2,
                                                 ثابت sycl::half * ثابت __restrict__ نقاب,
                                                 ثابت sycl::uint3 ne01,
                                                 ثابت عائم       logit_softcap,
                                                 ثابت عائم       لاهي,
                                                 T_KQ * ثابت      KQ,
                                                 T_vec_dot * ثابت KV_tmp,
                                                 ثابت صحيح         stride_K2,
                                                 ثابت صحيح         stride_V2,
                                                 ثابت صحيح         stride_نقاب,
                                                 عائم * ثابت     KQ_max,
                                                 عائم * ثابت     KQ_جوڙ,
                                                 T_acc * ثابت     VKQ,
                                                 ثابت صحيح         k_VKQ_0,
                                                 ثابت صحيح         k_VKQ_max,
                                                 ثابت صحيح         کانو_Q_0,
                                                 عائم *           KQ_max_new_shared) {
    خودڪار شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    constexpr صحيح cpy_nb   = ggml_sycl_get_max_cpy_بائيٽون();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    constexpr صحيح ncols = ncols1*ncols2;
    constexpr صحيح cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr صحيح np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr صحيح DVp = (DV + 2*warp_سائيز - 1) & ~(2*warp_سائيز - 1); // DV padded to multiple of 2*warp_size.

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
    constexpr صحيح KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#وإلا
    constexpr صحيح KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#جيڪڏهن_پورو // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    ثابت صحيح k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    عائم KQ_max_new[cpw];
#هدايت unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    عائم KQ_acc[nbatch_fa/(np*warp_سائيز) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr صحيح nbatch_K_آخري = DKQ % nbatch_K;
#هدايت unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_آخري; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_سائيز, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    إذا (nbatch_K_آخري > 0) {
        constexpr صحيح k_KQ_0 = DKQ - nbatch_K_آخري;
        flash_attn_tile_iter_KQ<warp_سائيز, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_آخري, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#هدايت unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحيح j = fastmodulo(کانو_Q_0 + (jc0 + (شي_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#هدايت unroll
        لكل (صحيح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_سائيز) {
            ثابت صحيح i_KQ = i_KQ_0 + (شي_ct1.get_local_id(1) % np) * warp_سائيز + شي_ct1.get_local_id(2);

#إذا defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_سائيز)*cpw + jc0] *= 4.0f;
#جيڪڏهن_پورو // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            إذا (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_سائيز)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((عائم) KQ_acc[(i_KQ_0 / (np * warp_سائيز)) * cpw + jc0]);
            }

            إذا (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_سائيز)) * cpw + jc0] +=
                    (ncols2 > 1 || نقاب) ? لاهي * sycl::vec<sycl::half, 1>(نقاب[j * stride_نقاب + k_VKQ_0 + i_KQ])
                                                       .convert<عائم, sycl::rounding_گهڻائي::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((عائم) KQ_max_new[jc0],
                               (عائم) (KQ_acc[(i_KQ_0 / (np * warp_سائيز)) * cpw + jc0] + FATTN_KQ_MAX_وٿي));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_سائيز>(KQ_max_new[jc0]);
    }

    إذا constexpr (np == 1) {
        شي_ct1.barrier(sycl::access::fence_space::local_space);
    } وإلا {
        static_assert(cpw == 1, "bad cpw");

        إذا (شي_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[شي_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        شي_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(شي_ct1.get_local_id(1) & ~(np - 1)) + شي_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#هدايت unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_سائيز)][KQ_cs];
#وإلا
        __dpct_align__(16) عائم tmp[nbatch_fa/(np*warp_سائيز)][KQ_cs];
#جيڪڏهن_پورو // SYCL_FAST_FP16

#هدايت unroll
        لكل (صحيح jc1 = 0; jc1 < KQ_cs; ++jc1) {
            ثابت صحيح jc = jc0 + jc1;

            ثابت عائم KQ_max_scale = sycl::native::قوت_نما((عائم) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            عائم KQ_جوڙ_جوڙيو = 0.0f;
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < nbatch_fa; i0 += np*warp_سائيز) {
                ثابت عائم val =
                    !oob_check || i0 + (شي_ct1.get_local_id(1) % np) * warp_سائيز + شي_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::قوت_نما((عائم) (KQ_acc[(i0 / (np * warp_سائيز)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_جوڙ_جوڙيو += val;
                tmp[i0/(np*warp_سائيز)][jc1] = val;
            }
            KQ_جوڙ[jc] = KQ_جوڙ[jc]*KQ_max_scale + KQ_جوڙ_جوڙيو;

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز) {
                VKQ[jc*((DVp/2)/warp_سائيز) + i0/warp_سائيز].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_سائيز) + i0/warp_سائيز].y() *= KQ_max_scale_h2.y();
            }
#وإلا
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز) {
                VKQ[jc*((DVp/2)/warp_سائيز) + i0/warp_سائيز].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_سائيز) + i0/warp_سائيز].y() *= KQ_max_scale;
            }
#جيڪڏهن_پورو // SYCL_FAST_FP16
        }

#هدايت unroll
        لكل (صحيح i0 = 0; i0 < nbatch_fa; i0 += np*warp_سائيز) {
            ثابت صحيح i = i0 + (شي_ct1.get_local_id(1) % np) * warp_سائيز + شي_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<حجم(tmp[0])>(
                KQ + (jc0 / KQ_cs + (شي_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_سائيز)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr صحيح nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#هدايت unroll
    لكل (صحيح k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_ڀريو_tile<warp_سائيز, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        شي_ct1.barrier(sycl::access::fence_space::local_space);

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
#هدايت unroll
        لكل (صحيح k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_سائيز];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr صحيح cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_سائيز ? cpy_ne/2 : (DVp/2)/warp_سائيز;
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_سائيز],
                                                 &KV_tmp[(k1 + شي_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         شي_ct1.get_local_id(2) * cpy_ne_D]);
            }
#هدايت unroll
            لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ثابت صحيح jc_KQ = jc_VKQ_0 / KQ_cs + (شي_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * حجم(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + شي_ct1.get_local_id(1) % np) * KQ_cs);
#هدايت unroll
                لكل (صحيح jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز) {
#هدايت unroll
                لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائيز) + i0/warp_سائيز].x() +=
                        V_k[i0/warp_سائيز].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائيز) + i0/warp_سائيز].y() +=
                        V_k[i0/warp_سائيز].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#وإلا
#هدايت unroll
        لكل (صحيح k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_سائيز];
            __dpct_align__(16) عائم  KQ_k[cpw];

            constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_سائيز ? cpy_ne : DVp/warp_سائيز;
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_سائيز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_سائيز)], &KV_tmp[(k1 + شي_ct1.get_local_id(1) % np)*DV + i0 + شي_ct1.get_local_id(2)*cpy_ne_D]);
            }
#هدايت unroll
            لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ثابت صحيح jc_KQ = jc_VKQ_0/KQ_cs + (شي_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*حجم(عائم)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + شي_ct1.get_local_id(1) % np)*KQ_cs);
            }

#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز) {
#هدايت unroll
                لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائيز) + i0/warp_سائيز].x() += V_k[i0/warp_سائيز].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_سائيز) + i0/warp_سائيز].y() += V_k[i0/warp_سائيز].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#جيڪڏهن_پورو // SYCL_FAST_FP16
        شي_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

سانچو <صحيح DKQ, صحيح DV, صحيح ncols1, صحيح ncols2, منطقي use_logit_softcap, صحيح warp_سائيز>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ساكن فراغ flash_attn_tile(ثابت حرف *  Q,
                            ثابت حرف *  K,
                            ثابت حرف *  V,
                            ثابت حرف *  نقاب,
                            ثابت حرف *  sinks,
                            ثابت صحيح *  KV_max,
                            عائم *  dst,
                            sycl::float2 *  dst_meta,
                            ثابت عائم          scale,
                            ثابت عائم          max_bias,
                            ثابت عائم          m0,
                            ثابت عائم          m1,
                            ثابت uint32_t       n_سر_log2,
                            ثابت عائم          logit_softcap,
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
#جيڪڏهن_وصف_ٿيل SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    خودڪار شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    إذا ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_اڻ_استعمال_ٿيل_VARS(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_سر_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        أرجع;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr صحيح ncols     = ncols1*ncols2;

    constexpr صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_سائيز;
    constexpr صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr صحيح nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    ثابت صحيح کانو_Q_0 = شي_ct1.get_گروهه(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    ثابت صحيح           sequence  = شي_ct1.get_گروهه(0) / (ne02 / ncols2);
    ثابت صحيح           head0     = شي_ct1.get_گروهه(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    ثابت صحيح gqa_تناسب = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    ثابت عائم * Q_f  = (ثابت عائم *) (Q + nb03*sequence + nb02* head0);
    ثابت sycl::half2 * K_h2      = (ثابت sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_تناسب));
    ثابت sycl::half2 * V_h2 =
        (ثابت sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_تناسب));  // K and V have same shape

    ثابت sycl::half * maskh = نقاب ? (ثابت sycl::half *) (نقاب + nb33 * (sequence % ne33)) : خالي_اشارو;

    ثابت صحيح stride_K2   = nb11 / حجم(sycl::half2);
    ثابت صحيح stride_V2   = nb21 / حجم(sycl::half2);
    ثابت صحيح stride_نقاب = nb31 / حجم(sycl::half);

    ثابت عائم لاهي = ncols2 == 1 ? get_alibi_لاهي(max_bias, head0, n_سر_log2, m0, m1) : 1.0f;

    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_بائيٽون();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    constexpr صحيح cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr صحيح np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_سائيز) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr صحيح DKQp = (DKQ + 2*warp_سائيز - 1) & ~(2*warp_سائيز - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr صحيح DVp  = (DV  + 2*warp_سائيز - 1) & ~(2*warp_سائيز - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_سائيز = lsm_size1 * حجم(sycl::half2) +
                                            lsm_size2 * حجم(sycl::half2) +
                                            lsm_size3 * حجم(sycl::half) +
                                            lsm_size4 * حجم(عائم);

    syclex::work_گروهه_static<حرف[local_share_mem_سائيز]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    عائم *KQ_max_new_shared = (عائم *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_سائيز)] = {
        { 0.0f, 0.0f }
    };
#وإلا
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_سائيز = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * حجم(عائم);

    syclex::work_گروهه_static<حرف[local_share_mem_سائيز]> lsm;

    عائم *Q_tmp = (عائم *)&lsm;
    عائم *KV_tmp = Q_tmp +lsm_size1;
    عائم *KQ = KV_tmp+lsm_size2;
    عائم *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_سائيز)] = {{0.0f, 0.0f}};


#جيڪڏهن_پورو // SYCL_FAST_FP16

    عائم KQ_max[cpw] = {};

#هدايت unroll
    لكل (صحيح j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    عائم KQ_جوڙ[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#هدايت unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحيح jc = jc0 + (شي_ct1.get_local_id(1) / np) * cpw;

        ثابت صحيح j = jc / ncols2;
        ثابت صحيح c = jc % ncols2;

        constexpr صحيح cpy_ne_D = cpy_ne < DKQp/warp_سائيز ? cpy_ne : DKQp/warp_سائيز;

#هدايت unroll
        لكل (صحيح i0 = 0; i0 < DKQp; i0 += np*warp_سائيز*cpy_ne_D) {
            إذا (i0 + np * warp_سائيز * cpy_ne_D <= DKQ ||
                i0 + (شي_ct1.get_local_id(1) % np) * (warp_سائيز * cpy_ne_D) + شي_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) عائم tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<حجم(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / حجم(عائم)) + fastmodulo(کانو_Q_0 + j, ne01) * (nb01 / حجم(عائم)) +
                                i0 + (شي_ct1.get_local_id(1) % np) * (warp_سائيز * cpy_ne_D) +
                                شي_ct1.get_local_id(2) * cpy_ne_D]);

#هدايت unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#هدايت unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#إذا defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#جيڪڏهن_پورو // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<حجم(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (شي_ct1.get_local_id(1) % np) * (warp_سائيز * cpy_ne_D / 2) +
                           شي_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#وإلا
                ggml_sycl_memcpy_1<حجم(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (شي_ct1.get_local_id(1) % np)*(warp_سائيز*cpy_ne_D)   + شي_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#جيڪڏهن_پورو // SYCL_FAST_FP16
            }
        }
    }

    شي_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    ثابت صحيح k_VKQ_max = KV_max ? KV_max[sequence * شي_ct1.get_گروهه_range(2) + شي_ct1.get_گروهه(2)] : ne11;
    إذا (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        صحيح k_VKQ_0 = شي_ct1.get_گروهه(1) * nbatch_fa;
        طالما (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr منطقي oob_check = خطأ;
            flash_attn_tile_iter<warp_سائيز, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, لاهي, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_نقاب, KQ_max, KQ_جوڙ, VKQ, k_VKQ_0, k_VKQ_max, کانو_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += شي_ct1.get_گروهه_range(1) * nbatch_fa;
        }
        إذا (k_VKQ_0 < k_VKQ_max) {
            constexpr منطقي oob_check = سچ;
            flash_attn_tile_iter<warp_سائيز, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, لاهي, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_نقاب, KQ_max, KQ_جوڙ, VKQ, k_VKQ_0, k_VKQ_max, کانو_Q_0,
                                            KQ_max_new_shared);
        }
    } وإلا {
        // Branch without out-of-bounds checks.
        لكل (صحيح k_VKQ_0 = شي_ct1.get_گروهه(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += شي_ct1.get_گروهه_range(1) * nbatch_fa) {

            constexpr منطقي oob_check = خطأ;
            flash_attn_tile_iter<warp_سائيز, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, لاهي, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_نقاب, KQ_max, KQ_جوڙ, VKQ, k_VKQ_0, k_VKQ_max, کانو_Q_0,
                                            KQ_max_new_shared);
        }
    }

#هدايت unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_جوڙ[jc0] = warp_reduce_جوڙ<warp_سائيز>(KQ_جوڙ[jc0]);
    }

    إذا constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#وإلا
        عائم * VKQ_combine    = (عائم *) KV_tmp;
#جيڪڏهن_پورو // SYCL_FAST_FP16

        عائم * KQ_جوڙ_combine = (عائم *) Q_tmp;

        إذا (شي_ct1.get_local_id(1) % np != 0) {

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
            constexpr صحيح cpy_ne_D = cpy_ne < (DVp/2)/warp_سائيز ? cpy_ne : (DVp/2)/warp_سائيز;
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[شي_ct1.get_local_id(1) * (DVp / 2) + i0 + شي_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_سائيز]);
            }
#وإلا

            constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_سائيز ? cpy_ne : DVp/warp_سائيز;

#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_سائيز*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[شي_ct1.get_local_id(1)*DVp + i0 + شي_ct1.get_local_id(2)*cpy_ne_D], ((ثابت عائم *) VKQ) + i0/warp_سائيز);
            }
#جيڪڏهن_پورو // SYCL_FAST_FP16

            إذا (شي_ct1.get_local_id(2) == 0) {
                KQ_جوڙ_combine[شي_ct1.get_local_id(1)] = KQ_جوڙ[0];
            }
            أرجع;
        }

        شي_ct1.barrier(sycl::access::fence_space::local_space);

#هدايت unroll
        لكل (صحيح ip = 1; ip < np; ++ip) {
#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
            constexpr صحيح cpy_ne_D = cpy_ne < (DVp/2)/warp_سائيز ? cpy_ne : (DVp/2)/warp_سائيز;
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(شي_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   شي_ct1.get_local_id(2) * cpy_ne_D]);
#هدايت unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_سائيز + i1] += tmp[i1];
                }
            }
#وإلا
            constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_سائيز ? cpy_ne : DVp/warp_سائيز;
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_سائيز*cpy_ne_D) {
                __dpct_align__(16) عائم tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(شي_ct1.get_local_id(1) + ip)*DVp + i0 + شي_ct1.get_local_id(2)*cpy_ne_D]);
#هدايت unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((عائم *)VKQ)[i0/warp_سائيز + i1] += tmp[i1];
                }
            }
#جيڪڏهن_پورو // SYCL_FAST_FP16

            KQ_جوڙ[0] += KQ_جوڙ_combine[شي_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    إذا (sinks && شي_ct1.get_گروهه(1) == 0) {
#هدايت unroll
        لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
            ثابت صحيح   jc   = jc0 + (شي_ct1.get_local_id(1) / np) * cpw;
            ثابت عائم sink = ((ثابت عائم *) sinks)[head0 + jc % ncols2];

            عائم       KQ_max_new_j = sycl::fmax((عائم) KQ_max[jc0], sink);
            ثابت عائم KQ_max_scale = sycl::native::قوت_نما((عائم) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            ثابت عائم val = sycl::native::قوت_نما((عائم) (sink - KQ_max[jc0]));
            KQ_جوڙ[jc0] = KQ_جوڙ[jc0]*KQ_max_scale + val;

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز) {
                VKQ[jc0*((DVp/2)/warp_سائيز) + i0/warp_سائيز] *= KQ_max_scale_h2;
            }
#وإلا
#هدايت unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز) {
                VKQ[jc0*((DVp/2)/warp_سائيز) + i0/warp_سائيز].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_سائيز) + i0/warp_سائيز].y() *= KQ_max_scale;
            }
#جيڪڏهن_پورو // SYCL_FAST_FP16
        }
    }

    // Write back results:
#هدايت unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحيح jc = jc0 + (شي_ct1.get_local_id(1) / np) * cpw;

        ثابت صحيح j = jc / ncols2;
        ثابت صحيح c = jc % ncols2;

        إذا (ncols1 > 1 && کانو_Q_0 + j >= صحيح(ne01.z())) {
            أرجع;
        }

        ثابت عائم scale = شي_ct1.get_گروهه_range(1) == 1 ? 1.0f / KQ_جوڙ[jc0] : 1.0f;

        ثابت صحيح j_dst_unrolled =
            ((sequence * صحيح(ne01.z()) + کانو_Q_0 + j) * ne02 + head0 + c) * شي_ct1.get_گروهه_range(1) +
            شي_ct1.get_گروهه(1);

#جيڪڏهن_وصف_ٿيل SYCL_FAST_FP16
        constexpr صحيح cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_سائيز ? cpy_ne/2 : (DVp/2)/warp_سائيز;
#هدايت unroll
        لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_سائيز*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#هدايت unroll
            لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_سائيز) + i0 / warp_سائيز + i1]
                              .سانچو convert<عائم, sycl::rounding_گهڻائي::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            إذا (i0 + warp_سائيز * cpy_ne_D <= DV / 2 || i0 + شي_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<حجم(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + شي_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#وإلا
        constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_سائيز ? cpy_ne : DVp/warp_سائيز;
#هدايت unroll
        لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_سائيز*cpy_ne_D) {
            إذا (i0 + warp_سائيز*cpy_ne_D <= DV || i0 + شي_ct1.get_local_id(2)*cpy_ne_D < DV) {
#هدايت unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_سائيز) + i0/(2*warp_سائيز) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_سائيز) + i0/(2*warp_سائيز) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + شي_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_سائيز) + i0/(2*warp_سائيز)]);
            }
        }
#جيڪڏهن_پورو // SYCL_FAST_FP16

        إذا (شي_ct1.get_گروهه_range(1) != 1 && شي_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_جوڙ[jc0]);
        }
    }
#وإلا
    GGML_اڻ_استعمال_ٿيل_VARS(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_سر_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#جيڪڏهن_پورو // SYCL_FLASH_ATTN
}

سانچو <صحيح DKQ, صحيح DV, صحيح ncols2, منطقي use_logit_softcap>
ساكن فراغ launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * Q = dst->src[0];

    ثابت صحيح id        = ggml_sycl_get_device();
    ثابت صحيح cc        = ggml_sycl_info().devices[id].cc;
    ثابت صحيح warp_سائيز = WARP_32_سائيز; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    إذا (DV < 512 && Q->ne[1] < 32) {
        إذا constexpr (ncols2 <= 32) {
            إذا (Q->ne[1] > 16/ncols2) {
                constexpr صحيح کانن_ڳڻپ_في_بلاڪ = 32;
                ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc) / warp_سائيز;
                ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc);
                launch_fattn<DV, کانن_ڳڻپ_في_بلاڪ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, کانن_ڳڻپ_في_بلاڪ / ncols2, ncols2, use_logit_softcap, warp_سائيز>, warp_سائيز>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, خطأ);
                أرجع;
            }
        }
        إذا constexpr (ncols2 <= 16) {
            إذا (Q->ne[1] > 8/ncols2) {
                constexpr صحيح کانن_ڳڻپ_في_بلاڪ = 16;
                ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc) / warp_سائيز;
                ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc);
                launch_fattn<DV, کانن_ڳڻپ_في_بلاڪ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, کانن_ڳڻپ_في_بلاڪ / ncols2, ncols2, use_logit_softcap, warp_سائيز>, warp_سائيز>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, خطأ);
                أرجع;
            }
        }
        إذا constexpr (ncols2 <= 8) {
            إذا (Q->ne[1] > 4/ncols2) {
                constexpr صحيح کانن_ڳڻپ_في_بلاڪ = 8;
                ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc) / warp_سائيز;
                ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc);
                launch_fattn<DV, کانن_ڳڻپ_في_بلاڪ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, کانن_ڳڻپ_في_بلاڪ / ncols2, ncols2, use_logit_softcap, warp_سائيز>, warp_سائيز>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, خطأ);
                أرجع;
            }
        }
    }

    إذا constexpr (ncols2 <= 4) {
        إذا (Q->ne[1] > 2/ncols2) {
            constexpr صحيح کانن_ڳڻپ_في_بلاڪ = 4;
            ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc) / warp_سائيز;
            ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc);
            launch_fattn<DV, کانن_ڳڻپ_في_بلاڪ/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, کانن_ڳڻپ_في_بلاڪ / ncols2, ncols2, use_logit_softcap, warp_سائيز>, warp_سائيز>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, خطأ);
            أرجع;
        }
    }

    إذا constexpr (ncols2 <= 2) {
        constexpr صحيح کانن_ڳڻپ_في_بلاڪ = 2;
        ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc) / warp_سائيز;
        ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc);
        launch_fattn<DV, کانن_ڳڻپ_في_بلاڪ/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, کانن_ڳڻپ_في_بلاڪ / ncols2, ncols2, use_logit_softcap, warp_سائيز>, warp_سائيز>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, خطأ);
        أرجع;
    }

    {
        constexpr صحيح کانن_ڳڻپ_في_بلاڪ = ncols2*2;
        ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc) / warp_سائيز;
        ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, کانن_ڳڻپ_في_بلاڪ, cc);
        launch_fattn<DV, کانن_ڳڻپ_في_بلاڪ/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, کانن_ڳڻپ_في_بلاڪ / ncols2, ncols2, use_logit_softcap, warp_سائيز>, warp_سائيز>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, سچ, سچ, خطأ);
        أرجع;
    }

    GGML_ABORT("fatal error");
}

سانچو <صحيح DKQ, صحيح DV, منطقي use_logit_softcap>
ساكن فراغ launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV  = dst;
    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * نقاب = dst->src[3];

    عائم max_bias = 0.0f;
    يادگيري_نقل(&max_bias, (ثابت عائم *) KQV->op_params + 1, حجم(عائم));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    ثابت صحيح gqa_تناسب = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    ثابت صحيح gqa_حد_بندي = gqa_تناسب <= 4 && DV <= 256 ? 16 : وڌ_کان_وڌ_عدد;
    ثابت منطقي use_gqa_opt = نقاب && max_bias == 0.0f && Q->ne[1] <= gqa_حد_بندي && K->ne[1] % FATTN_KQ_STRIDE == 0;

    إذا constexpr (DV == 512) {
        إذا (use_gqa_opt && gqa_تناسب % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            أرجع;
        }
        إذا (use_gqa_opt && gqa_تناسب % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            أرجع;
        }
        إذا (use_gqa_opt && gqa_تناسب % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            أرجع;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        إذا constexpr (DKQ == DV) {
            إذا (use_gqa_opt && gqa_تناسب % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                أرجع;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            أرجع;
        }
    }

    إذا constexpr (DV <= 256) {
        إذا (use_gqa_opt && gqa_تناسب % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            أرجع;
        }

        إذا (use_gqa_opt && gqa_تناسب % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            أرجع;
        }

        إذا (use_gqa_opt && gqa_تناسب % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            أرجع;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        أرجع;
    }
    GGML_ABORT("fatal error");
}

سانچو <صحيح DKQ, صحيح DV>
فراغ ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV = dst;

    عائم logit_softcap;
    يادگيري_نقل(&logit_softcap, (ثابت عائم *) KQV->op_params + 2, حجم(عائم));

    إذا (logit_softcap == 0.0f) {
        constexpr منطقي use_logit_softcap = خطأ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } وإلا {
        constexpr منطقي use_logit_softcap = سچ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

فراغ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#وصف_ڪريو DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    سانچو فراغ ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

خارجي DECL_FATTN_TILE_CASE( 40,  40);
خارجي DECL_FATTN_TILE_CASE( 64,  64);
خارجي DECL_FATTN_TILE_CASE( 72,  72);
خارجي DECL_FATTN_TILE_CASE( 80,  80);
خارجي DECL_FATTN_TILE_CASE( 96,  96);
خارجي DECL_FATTN_TILE_CASE(112, 112);
خارجي DECL_FATTN_TILE_CASE(128, 128);
خارجي DECL_FATTN_TILE_CASE(256, 256);
خارجي DECL_FATTN_TILE_CASE(512, 512);
خارجي DECL_FATTN_TILE_CASE(576, 512);

