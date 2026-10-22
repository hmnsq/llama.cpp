#समावेश <sycl/sycl.hpp>
#समावेश <sycl/ext/oneapi/work_group_static.hpp>
#समावेश "dpct/helper.hpp"
#समावेश "common.hpp"
#समावेश "fattn-common.hpp"

#समावेश <cmath>
#समावेश <float.h>

namespace syclex = sycl::ext::oneapi::experimental;

#घोषणा GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    अगर (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        वापस ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

अटल constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
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

    वापस 0;
}

अटल constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
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

    वापस 0;
}

अटल constexpr uint32_t ggml_sycl_fattn_tile_get_config(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols, स्थिर पूर्णांक cc) {
    अगर(fast_fp16_available(cc))
        वापस ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    अन्यथा
        वापस ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

अटल constexpr uint32_t ggml_sycl_fattn_tile_get_config(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
#अगर_घोषित SYCL_FAST_FP16
    वापस ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#अन्यथा
    वापस ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#पूर्ण_अगर // SYCL_FAST_FP16
}

अटल पूर्णांक ggml_sycl_fattn_tile_get_nthreads(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols, स्थिर पूर्णांक cc) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

अटल constexpr पूर्णांक ggml_sycl_fattn_tile_get_nthreads(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

अटल पूर्णांक ggml_sycl_fattn_tile_get_occupancy(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols, स्थिर पूर्णांक cc) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

अटल constexpr पूर्णांक ggml_sycl_fattn_tile_get_occupancy(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

अटल पूर्णांक ggml_sycl_fattn_tile_get_nbatch_fa(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols, स्थिर पूर्णांक cc) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

अटल constexpr पूर्णांक ggml_sycl_fattn_tile_get_nbatch_fa(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

अटल पूर्णांक ggml_sycl_fattn_tile_get_nbatch_K(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols, स्थिर पूर्णांक cc) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

अटल constexpr पूर्णांक ggml_sycl_fattn_tile_get_nbatch_K(स्थिर पूर्णांक DKQ, स्थिर पूर्णांक DV, स्थिर पूर्णांक ncols) {
    वापस (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

ढाँचा <पूर्णांक warp_आकार, पूर्णांक nwarps, पूर्णांक I, पूर्णांक J, पूर्णांक J_padding, bool oob_check>
अटल __dpct_inline__ व्योम flash_attn_tile_load_tile(स्थिर sycl::half2 * स्थिर __restrict__ KV,
                                                      sycl::half2 * स्थिर __restrict__ tile_KV,
                                                      स्थिर पूर्णांक stride_KV,
                                                      स्थिर पूर्णांक i_sup) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr पूर्णांक cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    स्वतः load = [&] (स्थिर पूर्णांक n) {
        स्थिर पूर्णांक stride_j = warp_आकार >> n;

        अगर (stride_j == 0) {
            वापस;
        }

        स्थिर पूर्णांक j0_start = stride_j == warp_आकार ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        स्थिर पूर्णांक j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        स्थिर पूर्णांक stride_i = warp_आकार / stride_j;

        अगर (j0_start == j0_stop) {
            वापस;
        }

#आशय unroll
        क्रम (पूर्णांक i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            स्थिर पूर्णांक i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_आकार ? 0 : item_ct1.get_local_id(2) / stride_j);

            अगर (i0 + nwarps*stride_i <= I || i < I) {
#आशय unroll
                क्रम (पूर्णांक j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    स्थिर पूर्णांक j = j0 * cpy_ne + (stride_j == warp_आकार ? item_ct1.get_local_id(2) :
                                                                         item_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    स्थिर __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
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

ढाँचा <पूर्णांक warp_आकार, पूर्णांक nwarps, पूर्णांक I, पूर्णांक J, पूर्णांक J_padding, bool oob_check>
अटल __dpct_inline__ व्योम flash_attn_tile_load_tile(स्थिर sycl::half2 * स्थिर __restrict__ KV,
                                                      भग्न * स्थिर __restrict__ tile_KV,
                                                      स्थिर पूर्णांक stride_KV,
                                                      स्थिर पूर्णांक i_sup) {
    constexpr पूर्णांक cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    स्वतः load = [&] (स्थिर पूर्णांक n) {
        स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        स्थिर पूर्णांक stride_j = warp_आकार >> n;

        अगर (stride_j == 0) {
            वापस;
        }

        स्थिर पूर्णांक j0_start = stride_j == warp_आकार ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        स्थिर पूर्णांक j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        स्थिर पूर्णांक stride_i = warp_आकार / stride_j;

        अगर (j0_start == j0_stop) {
            वापस;
        }

#आशय unroll
        क्रम (पूर्णांक i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            स्थिर पूर्णांक i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_आकार ? 0 : item_ct1.get_local_id(2) / stride_j);

            अगर (i0 + nwarps*stride_i <= I || i < I) {
#आशय unroll
                क्रम (पूर्णांक j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    स्थिर पूर्णांक j = j0 * (cpy_ne / 2) + (stride_j == warp_आकार ? item_ct1.get_local_id(2) :
                                                                               item_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    स्थिर sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<माप(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#आशय unroll
                    क्रम (पूर्णांक l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
                    }
                    ggml_sycl_memcpy_1<माप(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
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
ढाँचा <पूर्णांक  warp_आकार,
          पूर्णांक  nwarps,
          पूर्णांक  ncols1,
          पूर्णांक  ncols2,
          पूर्णांक  DKQ,
          पूर्णांक  nbatch_fa,
          पूर्णांक  nbatch_K,
          bool use_logit_softcap,
          bool oob_check,
          typename T_vec_dot>
अटल __dpct_inline__ व्योम flash_attn_tile_iter_KQ(T_vec_dot * स्थिर Q_tmp,
                                                    स्थिर sycl::half2 * स्थिर __restrict__ K_h2,
                                                    T_vec_dot * स्थिर KV_tmp,
                                                    स्थिर पूर्णांक         stride_K2,
                                                    स्थिर पूर्णांक         k_VKQ_0,
                                                    स्थिर पूर्णांक         k_VKQ_sup,
                                                    स्थिर पूर्णांक         k_KQ_0,
                                                    भग्न *           KQ_acc) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr पूर्णांक cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    constexpr पूर्णांक ncols = ncols1*ncols2;
    constexpr पूर्णांक cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr पूर्णांक np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_आकार, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    item_ct1.barrier(sycl::access::fence_space::local_space);

#अगर_घोषित SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#आशय unroll
    क्रम (पूर्णांक k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_आकार)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#अन्यथा
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#आशय unroll
    क्रम (पूर्णांक k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) भग्न K_k[nbatch_fa/(np*warp_आकार)][cpy_ne];
        __dpct_align__(16) भग्न Q_k[cpw][cpy_ne];
#पूर्ण_अगर // SYCL_FAST_FP16

#आशय unroll
        क्रम (पूर्णांक i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_आकार) {
            स्थिर पूर्णांक i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_आकार + item_ct1.get_local_id(2);

#अगर_घोषित SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_आकार)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#अन्यथा
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_आकार)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#पूर्ण_अगर // SYCL_FAST_FP16
        }
#आशय unroll
        क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
            स्थिर पूर्णांक jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

#अगर_घोषित SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#अन्यथा
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#पूर्ण_अगर // SYCL_FAST_FP16
        }

#आशय unroll
        क्रम (पूर्णांक i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_आकार) {
#आशय unroll
            क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
#आशय unroll
                क्रम (पूर्णांक k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_आकार)*cpw + jc0], K_k[i_KQ_0/(np*warp_आकार)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    अगर (k_KQ_0 + nbatch_K < DKQ) {
        item_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
ढाँचा <पूर्णांक  warp_आकार,
          पूर्णांक  nwarps,
          पूर्णांक  ncols1,
          पूर्णांक  ncols2,
          पूर्णांक  DKQ,
          पूर्णांक  DV,
          पूर्णांक  nbatch_fa,
          पूर्णांक  nbatch_K,
          bool use_logit_softcap,
          bool oob_check,
          typename T_vec_dot,
          typename T_KQ,
          typename T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
अटल __dpct_inline__ व्योम flash_attn_tile_iter(T_vec_dot * स्थिर Q_tmp,
                                                 स्थिर sycl::half2 * स्थिर __restrict__ K_h2,
                                                 स्थिर sycl::half2 * स्थिर __restrict__ V_h2,
                                                 स्थिर sycl::half * स्थिर __restrict__ mask,
                                                 स्थिर sycl::uint3 ne01,
                                                 स्थिर भग्न       logit_softcap,
                                                 स्थिर भग्न       slope,
                                                 T_KQ * स्थिर      KQ,
                                                 T_vec_dot * स्थिर KV_tmp,
                                                 स्थिर पूर्णांक         stride_K2,
                                                 स्थिर पूर्णांक         stride_V2,
                                                 स्थिर पूर्णांक         stride_mask,
                                                 भग्न * स्थिर     KQ_max,
                                                 भग्न * स्थिर     KQ_योग,
                                                 T_acc * स्थिर     VKQ,
                                                 स्थिर पूर्णांक         k_VKQ_0,
                                                 स्थिर पूर्णांक         k_VKQ_max,
                                                 स्थिर पूर्णांक         col_Q_0,
                                                 भग्न *           KQ_max_new_shared) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr पूर्णांक cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    constexpr पूर्णांक ncols = ncols1*ncols2;
    constexpr पूर्णांक cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr पूर्णांक np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr पूर्णांक DVp = (DV + 2*warp_आकार - 1) & ~(2*warp_आकार - 1); // DV padded to multiple of 2*warp_size.

#अगर_घोषित SYCL_FAST_FP16
    constexpr पूर्णांक KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#अन्यथा
    constexpr पूर्णांक KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#पूर्ण_अगर // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    स्थिर पूर्णांक k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    भग्न KQ_max_new[cpw];
#आशय unroll
    क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    भग्न KQ_acc[nbatch_fa/(np*warp_आकार) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr पूर्णांक nbatch_K_last = DKQ % nbatch_K;
#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_आकार, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    अगर (nbatch_K_last > 0) {
        constexpr पूर्णांक k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_आकार, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#आशय unroll
    क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
        स्थिर पूर्णांक j = fastmodulo(col_Q_0 + (jc0 + (item_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#आशय unroll
        क्रम (पूर्णांक i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_आकार) {
            स्थिर पूर्णांक i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_आकार + item_ct1.get_local_id(2);

#अगर defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_आकार)*cpw + jc0] *= 4.0f;
#पूर्ण_अगर // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            अगर (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_आकार)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((भग्न) KQ_acc[(i_KQ_0 / (np * warp_आकार)) * cpw + jc0]);
            }

            अगर (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_आकार)) * cpw + jc0] +=
                    (ncols2 > 1 || mask) ? slope * sycl::vec<sycl::half, 1>(mask[j * stride_mask + k_VKQ_0 + i_KQ])
                                                       .convert<भग्न, sycl::rounding_mode::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((भग्न) KQ_max_new[jc0],
                               (भग्न) (KQ_acc[(i_KQ_0 / (np * warp_आकार)) * cpw + jc0] + FATTN_KQ_MAX_OFFSET));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_आकार>(KQ_max_new[jc0]);
    }

    अगर constexpr (np == 1) {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } अन्यथा {
        static_assert(cpw == 1, "bad cpw");

        अगर (item_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[item_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(item_ct1.get_local_id(1) & ~(np - 1)) + item_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#आशय unroll
    क्रम (पूर्णांक jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#अगर_घोषित SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_आकार)][KQ_cs];
#अन्यथा
        __dpct_align__(16) भग्न tmp[nbatch_fa/(np*warp_आकार)][KQ_cs];
#पूर्ण_अगर // SYCL_FAST_FP16

#आशय unroll
        क्रम (पूर्णांक jc1 = 0; jc1 < KQ_cs; ++jc1) {
            स्थिर पूर्णांक jc = jc0 + jc1;

            स्थिर भग्न KQ_max_scale = sycl::native::exp((भग्न) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            भग्न KQ_योग_add = 0.0f;
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < nbatch_fa; i0 += np*warp_आकार) {
                स्थिर भग्न val =
                    !oob_check || i0 + (item_ct1.get_local_id(1) % np) * warp_आकार + item_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::exp((भग्न) (KQ_acc[(i0 / (np * warp_आकार)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_योग_add += val;
                tmp[i0/(np*warp_आकार)][jc1] = val;
            }
            KQ_योग[jc] = KQ_योग[jc]*KQ_max_scale + KQ_योग_add;

#अगर_घोषित SYCL_FAST_FP16
            स्थिर sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार) {
                VKQ[jc*((DVp/2)/warp_आकार) + i0/warp_आकार].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_आकार) + i0/warp_आकार].y() *= KQ_max_scale_h2.y();
            }
#अन्यथा
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार) {
                VKQ[jc*((DVp/2)/warp_आकार) + i0/warp_आकार].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_आकार) + i0/warp_आकार].y() *= KQ_max_scale;
            }
#पूर्ण_अगर // SYCL_FAST_FP16
        }

#आशय unroll
        क्रम (पूर्णांक i0 = 0; i0 < nbatch_fa; i0 += np*warp_आकार) {
            स्थिर पूर्णांक i = i0 + (item_ct1.get_local_id(1) % np) * warp_आकार + item_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<माप(tmp[0])>(
                KQ + (jc0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_आकार)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr पूर्णांक nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#आशय unroll
    क्रम (पूर्णांक k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_load_tile<warp_आकार, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        item_ct1.barrier(sycl::access::fence_space::local_space);

#अगर_घोषित SYCL_FAST_FP16
#आशय unroll
        क्रम (पूर्णांक k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_आकार];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr पूर्णांक cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_आकार ? cpy_ne/2 : (DVp/2)/warp_आकार;
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_आकार],
                                                 &KV_tmp[(k1 + item_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         item_ct1.get_local_id(2) * cpy_ne_D]);
            }
#आशय unroll
            क्रम (पूर्णांक jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                स्थिर पूर्णांक jc_KQ = jc_VKQ_0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * माप(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np) * KQ_cs);
#आशय unroll
                क्रम (पूर्णांक jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार) {
#आशय unroll
                क्रम (पूर्णांक jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_आकार) + i0/warp_आकार].x() +=
                        V_k[i0/warp_आकार].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_आकार) + i0/warp_आकार].y() +=
                        V_k[i0/warp_आकार].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#अन्यथा
#आशय unroll
        क्रम (पूर्णांक k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_आकार];
            __dpct_align__(16) भग्न  KQ_k[cpw];

            constexpr पूर्णांक cpy_ne_D = cpy_ne < DVp/warp_आकार ? cpy_ne : DVp/warp_आकार;
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp; i0 += warp_आकार*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_आकार)], &KV_tmp[(k1 + item_ct1.get_local_id(1) % np)*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
            }
#आशय unroll
            क्रम (पूर्णांक jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                स्थिर पूर्णांक jc_KQ = jc_VKQ_0/KQ_cs + (item_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*माप(भग्न)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np)*KQ_cs);
            }

#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार) {
#आशय unroll
                क्रम (पूर्णांक jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_आकार) + i0/warp_आकार].x() += V_k[i0/warp_आकार].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_आकार) + i0/warp_आकार].y() += V_k[i0/warp_आकार].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#पूर्ण_अगर // SYCL_FAST_FP16
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

ढाँचा <पूर्णांक DKQ, पूर्णांक DV, पूर्णांक ncols1, पूर्णांक ncols2, bool use_logit_softcap, पूर्णांक warp_आकार>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
अटल व्योम flash_attn_tile(स्थिर अक्षर *  Q,
                            स्थिर अक्षर *  K,
                            स्थिर अक्षर *  V,
                            स्थिर अक्षर *  mask,
                            स्थिर अक्षर *  sinks,
                            स्थिर पूर्णांक *  KV_max,
                            भग्न *  dst,
                            sycl::float2 *  dst_meta,
                            स्थिर भग्न          scale,
                            स्थिर भग्न          max_bias,
                            स्थिर भग्न          m0,
                            स्थिर भग्न          m1,
                            स्थिर uint32_t       n_शीर्ष_log2,
                            स्थिर भग्न          logit_softcap,
                            स्थिर int32_t        ne00,
                            स्थिर sycl::uint3    ne01,
                            स्थिर int32_t        ne02,
                            स्थिर int32_t        ne03,
                            स्थिर int32_t        nb01,
                            स्थिर int32_t        nb02,
                            स्थिर int32_t        nb03,
                            स्थिर int32_t        ne10,
                            स्थिर int32_t        ne11,
                            स्थिर int32_t        ne12,
                            स्थिर int32_t        ne13,
                            स्थिर int32_t        nb11,
                            स्थिर int32_t        nb12,
                            स्थिर int64_t        nb13,
                            स्थिर int32_t        nb21,
                            स्थिर int32_t        nb22,
                            स्थिर int64_t        nb23,
                            स्थिर int32_t        ne31,
                            स्थिर int32_t        ne32,
                            स्थिर int32_t        ne33,
                            स्थिर int32_t        nb31,
                            स्थिर int32_t        nb32,
                            स्थिर int64_t        nb33) {
#अगर_घोषित SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अगर ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_शीर्ष_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        वापस;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr पूर्णांक ncols     = ncols1*ncols2;

    constexpr पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_आकार;
    constexpr पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr पूर्णांक nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    स्थिर पूर्णांक col_Q_0 = item_ct1.get_group(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    स्थिर पूर्णांक           sequence  = item_ct1.get_group(0) / (ne02 / ncols2);
    स्थिर पूर्णांक           head0     = item_ct1.get_group(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    स्थिर पूर्णांक gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    स्थिर भग्न * Q_f  = (स्थिर भग्न *) (Q + nb03*sequence + nb02* head0);
    स्थिर sycl::half2 * K_h2      = (स्थिर sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_ratio));
    स्थिर sycl::half2 * V_h2 =
        (स्थिर sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_ratio));  // K and V have same shape

    स्थिर sycl::half * maskh = mask ? (स्थिर sycl::half *) (mask + nb33 * (sequence % ne33)) : nullptr;

    स्थिर पूर्णांक stride_K2   = nb11 / माप(sycl::half2);
    स्थिर पूर्णांक stride_V2   = nb21 / माप(sycl::half2);
    स्थिर पूर्णांक stride_mask = nb31 / माप(sycl::half);

    स्थिर भग्न slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_शीर्ष_log2, m0, m1) : 1.0f;

    constexpr पूर्णांक cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    constexpr पूर्णांक cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr पूर्णांक np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_आकार) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr पूर्णांक DKQp = (DKQ + 2*warp_आकार - 1) & ~(2*warp_आकार - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr पूर्णांक DVp  = (DV  + 2*warp_आकार - 1) & ~(2*warp_आकार - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#अगर_घोषित SYCL_FAST_FP16
    constexpr माप_प्रकार lsm_size1 = ncols * DKQ/2 ;
    constexpr माप_प्रकार lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr माप_प्रकार lsm_size3 = ncols * nbatch_fa;
    constexpr माप_प्रकार lsm_size4 = nwarps;

    constexpr माप_प्रकार local_share_mem_आकार = lsm_size1 * माप(sycl::half2) +
                                            lsm_size2 * माप(sycl::half2) +
                                            lsm_size3 * माप(sycl::half) +
                                            lsm_size4 * माप(भग्न);

    syclex::work_group_static<अक्षर[local_share_mem_आकार]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    भग्न *KQ_max_new_shared = (भग्न *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_आकार)] = {
        { 0.0f, 0.0f }
    };
#अन्यथा
    constexpr माप_प्रकार lsm_size1 = ncols * DKQ ;
    constexpr माप_प्रकार lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr माप_प्रकार lsm_size3 = ncols * nbatch_fa;
    constexpr माप_प्रकार lsm_size4 = nwarps;

    constexpr माप_प्रकार local_share_mem_आकार = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * माप(भग्न);

    syclex::work_group_static<अक्षर[local_share_mem_आकार]> lsm;

    भग्न *Q_tmp = (भग्न *)&lsm;
    भग्न *KV_tmp = Q_tmp +lsm_size1;
    भग्न *KQ = KV_tmp+lsm_size2;
    भग्न *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_आकार)] = {{0.0f, 0.0f}};


#पूर्ण_अगर // SYCL_FAST_FP16

    भग्न KQ_max[cpw] = {};

#आशय unroll
    क्रम (पूर्णांक j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -भग्न_उच्च/2.0f;
    }
    भग्न KQ_योग[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#आशय unroll
    क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
        स्थिर पूर्णांक jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        स्थिर पूर्णांक j = jc / ncols2;
        स्थिर पूर्णांक c = jc % ncols2;

        constexpr पूर्णांक cpy_ne_D = cpy_ne < DKQp/warp_आकार ? cpy_ne : DKQp/warp_आकार;

#आशय unroll
        क्रम (पूर्णांक i0 = 0; i0 < DKQp; i0 += np*warp_आकार*cpy_ne_D) {
            अगर (i0 + np * warp_आकार * cpy_ne_D <= DKQ ||
                i0 + (item_ct1.get_local_id(1) % np) * (warp_आकार * cpy_ne_D) + item_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) भग्न tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<माप(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / माप(भग्न)) + fastmodulo(col_Q_0 + j, ne01) * (nb01 / माप(भग्न)) +
                                i0 + (item_ct1.get_local_id(1) % np) * (warp_आकार * cpy_ne_D) +
                                item_ct1.get_local_id(2) * cpy_ne_D]);

#आशय unroll
                क्रम (पूर्णांक i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#अगर_घोषित SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#आशय unroll
                क्रम (पूर्णांक i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#अगर defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#पूर्ण_अगर // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<माप(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (item_ct1.get_local_id(1) % np) * (warp_आकार * cpy_ne_D / 2) +
                           item_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#अन्यथा
                ggml_sycl_memcpy_1<माप(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (item_ct1.get_local_id(1) % np)*(warp_आकार*cpy_ne_D)   + item_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#पूर्ण_अगर // SYCL_FAST_FP16
            }
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    स्थिर पूर्णांक k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    अगर (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        पूर्णांक k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa;
        जबतक (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_आकार, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_योग, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa;
        }
        अगर (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_iter<warp_आकार, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_योग, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    } अन्यथा {
        // Branch without out-of-bounds checks.
        क्रम (पूर्णांक k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa) {

            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_आकार, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_योग, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    }

#आशय unroll
    क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_योग[jc0] = warp_reduce_योग<warp_आकार>(KQ_योग[jc0]);
    }

    अगर constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#अगर_घोषित SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#अन्यथा
        भग्न * VKQ_combine    = (भग्न *) KV_tmp;
#पूर्ण_अगर // SYCL_FAST_FP16

        भग्न * KQ_योग_combine = (भग्न *) Q_tmp;

        अगर (item_ct1.get_local_id(1) % np != 0) {

#अगर_घोषित SYCL_FAST_FP16
            constexpr पूर्णांक cpy_ne_D = cpy_ne < (DVp/2)/warp_आकार ? cpy_ne : (DVp/2)/warp_आकार;
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[item_ct1.get_local_id(1) * (DVp / 2) + i0 + item_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_आकार]);
            }
#अन्यथा

            constexpr पूर्णांक cpy_ne_D = cpy_ne < DVp/warp_आकार ? cpy_ne : DVp/warp_आकार;

#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp; i0 += warp_आकार*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[item_ct1.get_local_id(1)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D], ((स्थिर भग्न *) VKQ) + i0/warp_आकार);
            }
#पूर्ण_अगर // SYCL_FAST_FP16

            अगर (item_ct1.get_local_id(2) == 0) {
                KQ_योग_combine[item_ct1.get_local_id(1)] = KQ_योग[0];
            }
            वापस;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

#आशय unroll
        क्रम (पूर्णांक ip = 1; ip < np; ++ip) {
#अगर_घोषित SYCL_FAST_FP16
            constexpr पूर्णांक cpy_ne_D = cpy_ne < (DVp/2)/warp_आकार ? cpy_ne : (DVp/2)/warp_आकार;
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   item_ct1.get_local_id(2) * cpy_ne_D]);
#आशय unroll
                क्रम (पूर्णांक i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_आकार + i1] += tmp[i1];
                }
            }
#अन्यथा
            constexpr पूर्णांक cpy_ne_D = cpy_ne < DVp/warp_आकार ? cpy_ne : DVp/warp_आकार;
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp; i0 += warp_आकार*cpy_ne_D) {
                __dpct_align__(16) भग्न tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
#आशय unroll
                क्रम (पूर्णांक i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((भग्न *)VKQ)[i0/warp_आकार + i1] += tmp[i1];
                }
            }
#पूर्ण_अगर // SYCL_FAST_FP16

            KQ_योग[0] += KQ_योग_combine[item_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    अगर (sinks && item_ct1.get_group(1) == 0) {
#आशय unroll
        क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
            स्थिर पूर्णांक   jc   = jc0 + (item_ct1.get_local_id(1) / np) * cpw;
            स्थिर भग्न sink = ((स्थिर भग्न *) sinks)[head0 + jc % ncols2];

            भग्न       KQ_max_new_j = sycl::fmax((भग्न) KQ_max[jc0], sink);
            स्थिर भग्न KQ_max_scale = sycl::native::exp((भग्न) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            स्थिर भग्न val = sycl::native::exp((भग्न) (sink - KQ_max[jc0]));
            KQ_योग[jc0] = KQ_योग[jc0]*KQ_max_scale + val;

#अगर_घोषित SYCL_FAST_FP16
            स्थिर sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार) {
                VKQ[jc0*((DVp/2)/warp_आकार) + i0/warp_आकार] *= KQ_max_scale_h2;
            }
#अन्यथा
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार) {
                VKQ[jc0*((DVp/2)/warp_आकार) + i0/warp_आकार].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_आकार) + i0/warp_आकार].y() *= KQ_max_scale;
            }
#पूर्ण_अगर // SYCL_FAST_FP16
        }
    }

    // Write back results:
#आशय unroll
    क्रम (पूर्णांक jc0 = 0; jc0 < cpw; ++jc0) {
        स्थिर पूर्णांक jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        स्थिर पूर्णांक j = jc / ncols2;
        स्थिर पूर्णांक c = jc % ncols2;

        अगर (ncols1 > 1 && col_Q_0 + j >= पूर्णांक(ne01.z())) {
            वापस;
        }

        स्थिर भग्न scale = item_ct1.get_group_range(1) == 1 ? 1.0f / KQ_योग[jc0] : 1.0f;

        स्थिर पूर्णांक j_dst_unrolled =
            ((sequence * पूर्णांक(ne01.z()) + col_Q_0 + j) * ne02 + head0 + c) * item_ct1.get_group_range(1) +
            item_ct1.get_group(1);

#अगर_घोषित SYCL_FAST_FP16
        constexpr पूर्णांक cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_आकार ? cpy_ne/2 : (DVp/2)/warp_आकार;
#आशय unroll
        क्रम (पूर्णांक i0 = 0; i0 < DVp/2; i0 += warp_आकार*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#आशय unroll
            क्रम (पूर्णांक i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_आकार) + i0 / warp_आकार + i1]
                              .ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            अगर (i0 + warp_आकार * cpy_ne_D <= DV / 2 || i0 + item_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<माप(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + item_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#अन्यथा
        constexpr पूर्णांक cpy_ne_D = cpy_ne < DVp/warp_आकार ? cpy_ne : DVp/warp_आकार;
#आशय unroll
        क्रम (पूर्णांक i0 = 0; i0 < DVp; i0 += warp_आकार*cpy_ne_D) {
            अगर (i0 + warp_आकार*cpy_ne_D <= DV || i0 + item_ct1.get_local_id(2)*cpy_ne_D < DV) {
#आशय unroll
                क्रम (पूर्णांक i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_आकार) + i0/(2*warp_आकार) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_आकार) + i0/(2*warp_आकार) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_आकार) + i0/(2*warp_आकार)]);
            }
        }
#पूर्ण_अगर // SYCL_FAST_FP16

        अगर (item_ct1.get_group_range(1) != 1 && item_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_योग[jc0]);
        }
    }
#अन्यथा
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_शीर्ष_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#पूर्ण_अगर // SYCL_FLASH_ATTN
}

ढाँचा <पूर्णांक DKQ, पूर्णांक DV, पूर्णांक ncols2, bool use_logit_softcap>
अटल व्योम launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * Q = dst->src[0];

    स्थिर पूर्णांक id        = ggml_sycl_get_device();
    स्थिर पूर्णांक cc        = ggml_sycl_info().devices[id].cc;
    स्थिर पूर्णांक warp_आकार = WARP_32_आकार; //can't support WARP_16_SIZE

    constexpr माप_प्रकार nbytes_shared = 0;

    अगर (DV < 512 && Q->ne[1] < 32) {
        अगर constexpr (ncols2 <= 32) {
            अगर (Q->ne[1] > 16/ncols2) {
                constexpr पूर्णांक cols_per_block = 32;
                स्थिर पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_आकार;
                स्थिर पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_आकार>, warp_आकार>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
                वापस;
            }
        }
        अगर constexpr (ncols2 <= 16) {
            अगर (Q->ne[1] > 8/ncols2) {
                constexpr पूर्णांक cols_per_block = 16;
                स्थिर पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_आकार;
                स्थिर पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_आकार>, warp_आकार>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
                वापस;
            }
        }
        अगर constexpr (ncols2 <= 8) {
            अगर (Q->ne[1] > 4/ncols2) {
                constexpr पूर्णांक cols_per_block = 8;
                स्थिर पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_आकार;
                स्थिर पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_आकार>, warp_आकार>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
                वापस;
            }
        }
    }

    अगर constexpr (ncols2 <= 4) {
        अगर (Q->ne[1] > 2/ncols2) {
            constexpr पूर्णांक cols_per_block = 4;
            स्थिर पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_आकार;
            स्थिर पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            launch_fattn<DV, cols_per_block/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_आकार>, warp_आकार>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
            वापस;
        }
    }

    अगर constexpr (ncols2 <= 2) {
        constexpr पूर्णांक cols_per_block = 2;
        स्थिर पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_आकार;
        स्थिर पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_आकार>, warp_आकार>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
        वापस;
    }

    {
        constexpr पूर्णांक cols_per_block = ncols2*2;
        स्थिर पूर्णांक nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_आकार;
        स्थिर पूर्णांक nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_आकार>, warp_आकार>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
        वापस;
    }

    GGML_ABORT("fatal error");
}

ढाँचा <पूर्णांक DKQ, पूर्णांक DV, bool use_logit_softcap>
अटल व्योम launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * KQV  = dst;
    स्थिर ggml_tensor * Q    = dst->src[0];
    स्थिर ggml_tensor * K    = dst->src[1];
    स्थिर ggml_tensor * mask = dst->src[3];

    भग्न max_bias = 0.0f;
    स_नकल(&max_bias, (स्थिर भग्न *) KQV->op_params + 1, माप(भग्न));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    स्थिर पूर्णांक gqa_ratio = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    स्थिर पूर्णांक gqa_limit = gqa_ratio <= 4 && DV <= 256 ? 16 : पूर्णांक_उच्च;
    स्थिर bool use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    अगर constexpr (DV == 512) {
        अगर (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            वापस;
        }
        अगर (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            वापस;
        }
        अगर (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            वापस;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        अगर constexpr (DKQ == DV) {
            अगर (use_gqa_opt && gqa_ratio % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                वापस;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            वापस;
        }
    }

    अगर constexpr (DV <= 256) {
        अगर (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            वापस;
        }

        अगर (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            वापस;
        }

        अगर (use_gqa_opt && gqa_ratio % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            वापस;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        वापस;
    }
    GGML_ABORT("fatal error");
}

ढाँचा <पूर्णांक DKQ, पूर्णांक DV>
व्योम ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * KQV = dst;

    भग्न logit_softcap;
    स_नकल(&logit_softcap, (स्थिर भग्न *) KQV->op_params + 2, माप(भग्न));

    अगर (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } अन्यथा {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

व्योम ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#घोषणा DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    ढाँचा व्योम ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

बाह्य DECL_FATTN_TILE_CASE( 40,  40);
बाह्य DECL_FATTN_TILE_CASE( 64,  64);
बाह्य DECL_FATTN_TILE_CASE( 72,  72);
बाह्य DECL_FATTN_TILE_CASE( 80,  80);
बाह्य DECL_FATTN_TILE_CASE( 96,  96);
बाह्य DECL_FATTN_TILE_CASE(112, 112);
बाह्य DECL_FATTN_TILE_CASE(128, 128);
बाह्य DECL_FATTN_TILE_CASE(256, 256);
बाह्य DECL_FATTN_TILE_CASE(512, 512);
बाह्य DECL_FATTN_TILE_CASE(576, 512);

