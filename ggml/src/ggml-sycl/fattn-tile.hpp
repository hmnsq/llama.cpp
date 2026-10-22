#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/work_group_static.hpp>
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-common.hpp"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <float.h>

ನಾಮಾಕಾಶ syclex = sycl::ext::oneapi::experimental;

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    ಆದರೆ (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        ಹಿಂತಿರುಗಿ ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

ಸ್ಥಿರ constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
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

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
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

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ constexpr uint32_t ggml_sycl_fattn_tile_get_config(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc) {
    ಆದರೆ(fast_fp16_ಲಭ್ಯ(cc))
        ಹಿಂತಿರುಗಿ ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

ಸ್ಥಿರ constexpr uint32_t ggml_sycl_fattn_tile_get_config(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
    ಹಿಂತಿರುಗಿ ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_nthreads(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_nthreads(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_occupancy(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_occupancy(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_nbatch_fa(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_nbatch_fa(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_nbatch_K(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_tile_get_nbatch_K(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols) {
    ಹಿಂತಿರುಗಿ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ nwarps, ಪೂರ್ಣಾಂಕ I, ಪೂರ್ಣಾಂಕ J, ಪೂರ್ಣಾಂಕ J_padding, ತಾರ್ಕಿಕ oob_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ flash_attn_tile_ತುಂಬು_tile(ಸ್ಥಿರಾಂಕ sycl::half2 * ಸ್ಥಿರಾಂಕ __restrict__ KV,
                                                      sycl::half2 * ಸ್ಥಿರಾಂಕ __restrict__ tile_KV,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_KV,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_sup) {
    ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    constexpr ಪೂರ್ಣಾಂಕ cpy_nb = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    ಸ್ವಯಂಚಾಲಿತ load = [&] (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_j = warp_ಗಾತ್ರ >> n;

        ಆದರೆ (stride_j == 0) {
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j0_ಆರಂಭ = stride_j == warp_ಗಾತ್ರ ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_i = warp_ಗಾತ್ರ / stride_j;

        ಆದರೆ (j0_ಆರಂಭ == j0_stop) {
            ಹಿಂತಿರುಗಿ;
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + ಅಂಶ_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_ಗಾತ್ರ ? 0 : ಅಂಶ_ct1.get_local_id(2) / stride_j);

            ಆದರೆ (i0 + nwarps*stride_i <= I || i < I) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j0 = j0_ಆರಂಭ; j0 < j0_stop; j0 += stride_j) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = j0 * cpy_ne + (stride_j == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) :
                                                                         ಅಂಶ_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    ಸ್ಥಿರಾಂಕ __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
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

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ nwarps, ಪೂರ್ಣಾಂಕ I, ಪೂರ್ಣಾಂಕ J, ಪೂರ್ಣಾಂಕ J_padding, ತಾರ್ಕಿಕ oob_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ flash_attn_tile_ತುಂಬು_tile(ಸ್ಥಿರಾಂಕ sycl::half2 * ಸ್ಥಿರಾಂಕ __restrict__ KV,
                                                      ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ __restrict__ tile_KV,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_KV,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_sup) {
    constexpr ಪೂರ್ಣಾಂಕ cpy_nb = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    ಸ್ವಯಂಚಾಲಿತ load = [&] (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n) {
        ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_j = warp_ಗಾತ್ರ >> n;

        ಆದರೆ (stride_j == 0) {
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j0_ಆರಂಭ = stride_j == warp_ಗಾತ್ರ ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_i = warp_ಗಾತ್ರ / stride_j;

        ಆದರೆ (j0_ಆರಂಭ == j0_stop) {
            ಹಿಂತಿರುಗಿ;
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + ಅಂಶ_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_ಗಾತ್ರ ? 0 : ಅಂಶ_ct1.get_local_id(2) / stride_j);

            ಆದರೆ (i0 + nwarps*stride_i <= I || i < I) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j0 = j0_ಆರಂಭ; j0 < j0_stop; j0 += stride_j) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = j0 * (cpy_ne / 2) + (stride_j == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) :
                                                                               ಅಂಶ_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    ಸ್ಥಿರಾಂಕ sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
                    }
                    ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
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
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ  warp_ಗಾತ್ರ,
          ಪೂರ್ಣಾಂಕ  nwarps,
          ಪೂರ್ಣಾಂಕ  ncols1,
          ಪೂರ್ಣಾಂಕ  ncols2,
          ಪೂರ್ಣಾಂಕ  DKQ,
          ಪೂರ್ಣಾಂಕ  nbatch_fa,
          ಪೂರ್ಣಾಂಕ  nbatch_K,
          ತಾರ್ಕಿಕ use_logit_softcap,
          ತಾರ್ಕಿಕ oob_check,
          ಬಗೆ_ಹೆಸರು T_vec_dot>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ flash_attn_tile_iter_KQ(T_vec_dot * ಸ್ಥಿರಾಂಕ Q_tmp,
                                                    ಸ್ಥಿರಾಂಕ sycl::half2 * ಸ್ಥಿರಾಂಕ __restrict__ K_h2,
                                                    T_vec_dot * ಸ್ಥಿರಾಂಕ KV_tmp,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         stride_K2,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         k_VKQ_0,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         k_VKQ_sup,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         k_KQ_0,
                                                    ದಶಮಾಂಶ *           KQ_acc) {
    ಸ್ವಯಂಚಾಲಿತ          ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    constexpr ಪೂರ್ಣಾಂಕ cpy_nb   = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    constexpr ಪೂರ್ಣಾಂಕ ncols = ncols1*ncols2;
    constexpr ಪೂರ್ಣಾಂಕ cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr ಪೂರ್ಣಾಂಕ np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_ತುಂಬು_tile<warp_ಗಾತ್ರ, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_ಗಾತ್ರ)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#ಇಲ್ಲದಿದ್ದರೆ
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) ದಶಮಾಂಶ K_k[nbatch_fa/(np*warp_ಗಾತ್ರ)][cpy_ne];
        __dpct_align__(16) ದಶಮಾಂಶ Q_k[cpw][cpy_ne];
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_KQ = i_KQ_0 + (ಅಂಶ_ct1.get_local_id(1) % np) * warp_ಗಾತ್ರ + ಅಂಶ_ct1.get_local_id(2);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_ಗಾತ್ರ)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#ಇಲ್ಲದಿದ್ದರೆ
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_ಗಾತ್ರ)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
        }
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc = jc0 + (ಅಂಶ_ct1.get_local_id(1) / np) * cpw;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#ಇಲ್ಲದಿದ್ದರೆ
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_ಗಾತ್ರ) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_ಗಾತ್ರ)*cpw + jc0], K_k[i_KQ_0/(np*warp_ಗಾತ್ರ)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    ಆದರೆ (k_KQ_0 + nbatch_K < DKQ) {
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ  warp_ಗಾತ್ರ,
          ಪೂರ್ಣಾಂಕ  nwarps,
          ಪೂರ್ಣಾಂಕ  ncols1,
          ಪೂರ್ಣಾಂಕ  ncols2,
          ಪೂರ್ಣಾಂಕ  DKQ,
          ಪೂರ್ಣಾಂಕ  DV,
          ಪೂರ್ಣಾಂಕ  nbatch_fa,
          ಪೂರ್ಣಾಂಕ  nbatch_K,
          ತಾರ್ಕಿಕ use_logit_softcap,
          ತಾರ್ಕಿಕ oob_check,
          ಬಗೆ_ಹೆಸರು T_vec_dot,
          ಬಗೆ_ಹೆಸರು T_KQ,
          ಬಗೆ_ಹೆಸರು T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ flash_attn_tile_iter(T_vec_dot * ಸ್ಥಿರಾಂಕ Q_tmp,
                                                 ಸ್ಥಿರಾಂಕ sycl::half2 * ಸ್ಥಿರಾಂಕ __restrict__ K_h2,
                                                 ಸ್ಥಿರಾಂಕ sycl::half2 * ಸ್ಥಿರಾಂಕ __restrict__ V_h2,
                                                 ಸ್ಥಿರಾಂಕ sycl::half * ಸ್ಥಿರಾಂಕ __restrict__ ಮುಖವಾಡ,
                                                 ಸ್ಥಿರಾಂಕ sycl::uint3 ne01,
                                                 ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ       logit_softcap,
                                                 ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ       ಇಳಿಜಾರು,
                                                 T_KQ * ಸ್ಥಿರಾಂಕ      KQ,
                                                 T_vec_dot * ಸ್ಥಿರಾಂಕ KV_tmp,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         stride_K2,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         stride_V2,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         stride_ಮುಖವಾಡ,
                                                 ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ     KQ_max,
                                                 ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ     KQ_ಮೊತ್ತ,
                                                 T_acc * ಸ್ಥಿರಾಂಕ     VKQ,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         k_VKQ_0,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         k_VKQ_max,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         ಲಂಬಸಾಲು_Q_0,
                                                 ದಶಮಾಂಶ *           KQ_max_new_shared) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    constexpr ಪೂರ್ಣಾಂಕ cpy_nb   = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    constexpr ಪೂರ್ಣಾಂಕ ncols = ncols1*ncols2;
    constexpr ಪೂರ್ಣಾಂಕ cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr ಪೂರ್ಣಾಂಕ np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr ಪೂರ್ಣಾಂಕ DVp = (DV + 2*warp_ಗಾತ್ರ - 1) & ~(2*warp_ಗಾತ್ರ - 1); // DV padded to multiple of 2*warp_size.

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
    constexpr ಪೂರ್ಣಾಂಕ KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#ಇಲ್ಲದಿದ್ದರೆ
    constexpr ಪೂರ್ಣಾಂಕ KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    ದಶಮಾಂಶ KQ_max_new[cpw];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    ದಶಮಾಂಶ KQ_acc[nbatch_fa/(np*warp_ಗಾತ್ರ) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr ಪೂರ್ಣಾಂಕ nbatch_K_ಕೊನೆಯ = DKQ % nbatch_K;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_ಕೊನೆಯ; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_ಗಾತ್ರ, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    ಆದರೆ (nbatch_K_ಕೊನೆಯ > 0) {
        constexpr ಪೂರ್ಣಾಂಕ k_KQ_0 = DKQ - nbatch_K_ಕೊನೆಯ;
        flash_attn_tile_iter_KQ<warp_ಗಾತ್ರ, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_ಕೊನೆಯ, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = fastmodulo(ಲಂಬಸಾಲು_Q_0 + (jc0 + (ಅಂಶ_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_KQ = i_KQ_0 + (ಅಂಶ_ct1.get_local_id(1) % np) * warp_ಗಾತ್ರ + ಅಂಶ_ct1.get_local_id(2);

#ಆದರೆ defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_ಗಾತ್ರ)*cpw + jc0] *= 4.0f;
#ಆದರೆ_ಅಂತ್ಯ // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            ಆದರೆ (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_ಗಾತ್ರ)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((ದಶಮಾಂಶ) KQ_acc[(i_KQ_0 / (np * warp_ಗಾತ್ರ)) * cpw + jc0]);
            }

            ಆದರೆ (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_ಗಾತ್ರ)) * cpw + jc0] +=
                    (ncols2 > 1 || ಮುಖವಾಡ) ? ಇಳಿಜಾರು * sycl::vec<sycl::half, 1>(ಮುಖವಾಡ[j * stride_ಮುಖವಾಡ + k_VKQ_0 + i_KQ])
                                                       .convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((ದಶಮಾಂಶ) KQ_max_new[jc0],
                               (ದಶಮಾಂಶ) (KQ_acc[(i_KQ_0 / (np * warp_ಗಾತ್ರ)) * cpw + jc0] + FATTN_KQ_MAX_ಸ್ಥಾನಾಂತರ));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_ಗಾತ್ರ>(KQ_max_new[jc0]);
    }

    ಆದರೆ constexpr (np == 1) {
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(cpw == 1, "bad cpw");

        ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[ಅಂಶ_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(ಅಂಶ_ct1.get_local_id(1) & ~(np - 1)) + ಅಂಶ_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_ಗಾತ್ರ)][KQ_cs];
#ಇಲ್ಲದಿದ್ದರೆ
        __dpct_align__(16) ದಶಮಾಂಶ tmp[nbatch_fa/(np*warp_ಗಾತ್ರ)][KQ_cs];
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc1 = 0; jc1 < KQ_cs; ++jc1) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc = jc0 + jc1;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ KQ_max_scale = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            ದಶಮಾಂಶ KQ_ಮೊತ್ತ_ಕೂಡಿಸು = 0.0f;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nbatch_fa; i0 += np*warp_ಗಾತ್ರ) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val =
                    !oob_check || i0 + (ಅಂಶ_ct1.get_local_id(1) % np) * warp_ಗಾತ್ರ + ಅಂಶ_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_acc[(i0 / (np * warp_ಗಾತ್ರ)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_ಮೊತ್ತ_ಕೂಡಿಸು += val;
                tmp[i0/(np*warp_ಗಾತ್ರ)][jc1] = val;
            }
            KQ_ಮೊತ್ತ[jc] = KQ_ಮೊತ್ತ[jc]*KQ_max_scale + KQ_ಮೊತ್ತ_ಕೂಡಿಸು;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
            ಸ್ಥಿರಾಂಕ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ) {
                VKQ[jc*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].y() *= KQ_max_scale_h2.y();
            }
#ಇಲ್ಲದಿದ್ದರೆ
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ) {
                VKQ[jc*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].y() *= KQ_max_scale;
            }
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nbatch_fa; i0 += np*warp_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + (ಅಂಶ_ct1.get_local_id(1) % np) * warp_ಗಾತ್ರ + ಅಂಶ_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp[0])>(
                KQ + (jc0 / KQ_cs + (ಅಂಶ_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_ಗಾತ್ರ)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr ಪೂರ್ಣಾಂಕ nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_ತುಂಬು_tile<warp_ಗಾತ್ರ, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_ಗಾತ್ರ];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_ಗಾತ್ರ ? cpy_ne/2 : (DVp/2)/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_ಗಾತ್ರ],
                                                 &KV_tmp[(k1 + ಅಂಶ_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         ಅಂಶ_ct1.get_local_id(2) * cpy_ne_D]);
            }
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc_KQ = jc_VKQ_0 / KQ_cs + (ಅಂಶ_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * ಗಾತ್ರ(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + ಅಂಶ_ct1.get_local_id(1) % np) * KQ_cs);
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].x() +=
                        V_k[i0/warp_ಗಾತ್ರ].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].y() +=
                        V_k[i0/warp_ಗಾತ್ರ].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#ಇಲ್ಲದಿದ್ದರೆ
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_ಗಾತ್ರ];
            __dpct_align__(16) ದಶಮಾಂಶ  KQ_k[cpw];

            constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < DVp/warp_ಗಾತ್ರ ? cpy_ne : DVp/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_ಗಾತ್ರ)], &KV_tmp[(k1 + ಅಂಶ_ct1.get_local_id(1) % np)*DV + i0 + ಅಂಶ_ct1.get_local_id(2)*cpy_ne_D]);
            }
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc_KQ = jc_VKQ_0/KQ_cs + (ಅಂಶ_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*ಗಾತ್ರ(ದಶಮಾಂಶ)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + ಅಂಶ_ct1.get_local_id(1) % np)*KQ_cs);
            }

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].x() += V_k[i0/warp_ಗಾತ್ರ].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].y() += V_k[i0/warp_ಗಾತ್ರ].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DKQ, ಪೂರ್ಣಾಂಕ DV, ಪೂರ್ಣಾಂಕ ncols1, ಪೂರ್ಣಾಂಕ ncols2, ತಾರ್ಕಿಕ use_logit_softcap, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_tile(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  Q,
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  K,
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  V,
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  ಮುಖವಾಡ,
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  sinks,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *  KV_max,
                            ದಶಮಾಂಶ *  dst,
                            sycl::float2 *  dst_meta,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ          scale,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ          max_bias,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ          m0,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ          m1,
                            ಸ್ಥಿರಾಂಕ uint32_t       n_ತಲೆ_log2,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ          logit_softcap,
                            ಸ್ಥಿರಾಂಕ int32_t        ne00,
                            ಸ್ಥಿರಾಂಕ sycl::uint3    ne01,
                            ಸ್ಥಿರಾಂಕ int32_t        ne02,
                            ಸ್ಥಿರಾಂಕ int32_t        ne03,
                            ಸ್ಥಿರಾಂಕ int32_t        nb01,
                            ಸ್ಥಿರಾಂಕ int32_t        nb02,
                            ಸ್ಥಿರಾಂಕ int32_t        nb03,
                            ಸ್ಥಿರಾಂಕ int32_t        ne10,
                            ಸ್ಥಿರಾಂಕ int32_t        ne11,
                            ಸ್ಥಿರಾಂಕ int32_t        ne12,
                            ಸ್ಥಿರಾಂಕ int32_t        ne13,
                            ಸ್ಥಿರಾಂಕ int32_t        nb11,
                            ಸ್ಥಿರಾಂಕ int32_t        nb12,
                            ಸ್ಥಿರಾಂಕ int64_t        nb13,
                            ಸ್ಥಿರಾಂಕ int32_t        nb21,
                            ಸ್ಥಿರಾಂಕ int32_t        nb22,
                            ಸ್ಥಿರಾಂಕ int64_t        nb23,
                            ಸ್ಥಿರಾಂಕ int32_t        ne31,
                            ಸ್ಥಿರಾಂಕ int32_t        ne32,
                            ಸ್ಥಿರಾಂಕ int32_t        ne33,
                            ಸ್ಥಿರಾಂಕ int32_t        nb31,
                            ಸ್ಥಿರಾಂಕ int32_t        nb32,
                            ಸ್ಥಿರಾಂಕ int64_t        nb33) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಆದರೆ ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_ಬಳಸದ_VARS(Q, K, V, ಮುಖವಾಡ, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_ತಲೆ_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ಹಿಂತಿರುಗಿ;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr ಪೂರ್ಣಾಂಕ ncols     = ncols1*ncols2;

    constexpr ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_ಗಾತ್ರ;
    constexpr ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr ಪೂರ್ಣಾಂಕ nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_Q_0 = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           sequence  = ಅಂಶ_ct1.get_ಗುಂಪು(0) / (ne02 / ncols2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           head0     = ಅಂಶ_ct1.get_ಗುಂಪು(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * Q_f  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (Q + nb03*sequence + nb02* head0);
    ಸ್ಥಿರಾಂಕ sycl::half2 * K_h2      = (ಸ್ಥಿರಾಂಕ sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_ಅನುಪಾತ));
    ಸ್ಥಿರಾಂಕ sycl::half2 * V_h2 =
        (ಸ್ಥಿರಾಂಕ sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_ಅನುಪಾತ));  // K and V have same shape

    ಸ್ಥಿರಾಂಕ sycl::half * maskh = ಮುಖವಾಡ ? (ಸ್ಥಿರಾಂಕ sycl::half *) (ಮುಖವಾಡ + nb33 * (sequence % ne33)) : ಶೂನ್ಯಸೂಚಕ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_K2   = nb11 / ಗಾತ್ರ(sycl::half2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_V2   = nb21 / ಗಾತ್ರ(sycl::half2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಮುಖವಾಡ = nb31 / ಗಾತ್ರ(sycl::half);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಇಳಿಜಾರು = ncols2 == 1 ? get_alibi_ಇಳಿಜಾರು(max_bias, head0, n_ತಲೆ_log2, m0, m1) : 1.0f;

    constexpr ಪೂರ್ಣಾಂಕ cpy_nb = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    constexpr ಪೂರ್ಣಾಂಕ cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr ಪೂರ್ಣಾಂಕ np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_ಗಾತ್ರ) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr ಪೂರ್ಣಾಂಕ DKQp = (DKQ + 2*warp_ಗಾತ್ರ - 1) & ~(2*warp_ಗಾತ್ರ - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr ಪೂರ್ಣಾಂಕ DVp  = (DV  + 2*warp_ಗಾತ್ರ - 1) & ~(2*warp_ಗಾತ್ರ - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_ಗಾತ್ರ = lsm_size1 * ಗಾತ್ರ(sycl::half2) +
                                            lsm_size2 * ಗಾತ್ರ(sycl::half2) +
                                            lsm_size3 * ಗಾತ್ರ(sycl::half) +
                                            lsm_size4 * ಗಾತ್ರ(ದಶಮಾಂಶ);

    syclex::work_ಗುಂಪು_static<ಅಕ್ಷರ[local_share_mem_ಗಾತ್ರ]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    ದಶಮಾಂಶ *KQ_max_new_shared = (ದಶಮಾಂಶ *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_ಗಾತ್ರ)] = {
        { 0.0f, 0.0f }
    };
#ಇಲ್ಲದಿದ್ದರೆ
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_ಗಾತ್ರ = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * ಗಾತ್ರ(ದಶಮಾಂಶ);

    syclex::work_ಗುಂಪು_static<ಅಕ್ಷರ[local_share_mem_ಗಾತ್ರ]> lsm;

    ದಶಮಾಂಶ *Q_tmp = (ದಶಮಾಂಶ *)&lsm;
    ದಶಮಾಂಶ *KV_tmp = Q_tmp +lsm_size1;
    ದಶಮಾಂಶ *KQ = KV_tmp+lsm_size2;
    ದಶಮಾಂಶ *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_ಗಾತ್ರ)] = {{0.0f, 0.0f}};


#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

    ದಶಮಾಂಶ KQ_max[cpw] = {};

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    ದಶಮಾಂಶ KQ_ಮೊತ್ತ[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc = jc0 + (ಅಂಶ_ct1.get_local_id(1) / np) * cpw;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = jc / ncols2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = jc % ncols2;

        constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < DKQp/warp_ಗಾತ್ರ ? cpy_ne : DKQp/warp_ಗಾತ್ರ;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DKQp; i0 += np*warp_ಗಾತ್ರ*cpy_ne_D) {
            ಆದರೆ (i0 + np * warp_ಗಾತ್ರ * cpy_ne_D <= DKQ ||
                i0 + (ಅಂಶ_ct1.get_local_id(1) % np) * (warp_ಗಾತ್ರ * cpy_ne_D) + ಅಂಶ_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) ದಶಮಾಂಶ tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / ಗಾತ್ರ(ದಶಮಾಂಶ)) + fastmodulo(ಲಂಬಸಾಲು_Q_0 + j, ne01) * (nb01 / ಗಾತ್ರ(ದಶಮಾಂಶ)) +
                                i0 + (ಅಂಶ_ct1.get_local_id(1) % np) * (warp_ಗಾತ್ರ * cpy_ne_D) +
                                ಅಂಶ_ct1.get_local_id(2) * cpy_ne_D]);

#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#ಆದರೆ defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#ಆದರೆ_ಅಂತ್ಯ // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (ಅಂಶ_ct1.get_local_id(1) % np) * (warp_ಗಾತ್ರ * cpy_ne_D / 2) +
                           ಅಂಶ_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#ಇಲ್ಲದಿದ್ದರೆ
                ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (ಅಂಶ_ct1.get_local_id(1) % np)*(warp_ಗಾತ್ರ*cpy_ne_D)   + ಅಂಶ_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
            }
        }
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_VKQ_max = KV_max ? KV_max[sequence * ಅಂಶ_ct1.get_ಗುಂಪು_range(2) + ಅಂಶ_ct1.get_ಗುಂಪು(2)] : ne11;
    ಆದರೆ (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        ಪೂರ್ಣಾಂಕ k_VKQ_0 = ಅಂಶ_ct1.get_ಗುಂಪು(1) * nbatch_fa;
        ಆಗ (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr ತಾರ್ಕಿಕ oob_check = ಸುಳ್ಳು;
            flash_attn_tile_iter<warp_ಗಾತ್ರ, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ಇಳಿಜಾರು, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ಮುಖವಾಡ, KQ_max, KQ_ಮೊತ್ತ, VKQ, k_VKQ_0, k_VKQ_max, ಲಂಬಸಾಲು_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += ಅಂಶ_ct1.get_ಗುಂಪು_range(1) * nbatch_fa;
        }
        ಆದರೆ (k_VKQ_0 < k_VKQ_max) {
            constexpr ತಾರ್ಕಿಕ oob_check = ನಿಜ;
            flash_attn_tile_iter<warp_ಗಾತ್ರ, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ಇಳಿಜಾರು, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ಮುಖವಾಡ, KQ_max, KQ_ಮೊತ್ತ, VKQ, k_VKQ_0, k_VKQ_max, ಲಂಬಸಾಲು_Q_0,
                                            KQ_max_new_shared);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Branch without out-of-bounds checks.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_VKQ_0 = ಅಂಶ_ct1.get_ಗುಂಪು(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += ಅಂಶ_ct1.get_ಗುಂಪು_range(1) * nbatch_fa) {

            constexpr ತಾರ್ಕಿಕ oob_check = ಸುಳ್ಳು;
            flash_attn_tile_iter<warp_ಗಾತ್ರ, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ಇಳಿಜಾರು, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ಮುಖವಾಡ, KQ_max, KQ_ಮೊತ್ತ, VKQ, k_VKQ_0, k_VKQ_max, ಲಂಬಸಾಲು_Q_0,
                                            KQ_max_new_shared);
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_ಮೊತ್ತ[jc0] = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(KQ_ಮೊತ್ತ[jc0]);
    }

    ಆದರೆ constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#ಇಲ್ಲದಿದ್ದರೆ
        ದಶಮಾಂಶ * VKQ_combine    = (ದಶಮಾಂಶ *) KV_tmp;
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

        ದಶಮಾಂಶ * KQ_ಮೊತ್ತ_combine = (ದಶಮಾಂಶ *) Q_tmp;

        ಆದರೆ (ಅಂಶ_ct1.get_local_id(1) % np != 0) {

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
            constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < (DVp/2)/warp_ಗಾತ್ರ ? cpy_ne : (DVp/2)/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[ಅಂಶ_ct1.get_local_id(1) * (DVp / 2) + i0 + ಅಂಶ_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_ಗಾತ್ರ]);
            }
#ಇಲ್ಲದಿದ್ದರೆ

            constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < DVp/warp_ಗಾತ್ರ ? cpy_ne : DVp/warp_ಗಾತ್ರ;

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[ಅಂಶ_ct1.get_local_id(1)*DVp + i0 + ಅಂಶ_ct1.get_local_id(2)*cpy_ne_D], ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) VKQ) + i0/warp_ಗಾತ್ರ);
            }
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

            ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
                KQ_ಮೊತ್ತ_combine[ಅಂಶ_ct1.get_local_id(1)] = KQ_ಮೊತ್ತ[0];
            }
            ಹಿಂತಿರುಗಿ;
        }

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ip = 1; ip < np; ++ip) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
            constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < (DVp/2)/warp_ಗಾತ್ರ ? cpy_ne : (DVp/2)/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(ಅಂಶ_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   ಅಂಶ_ct1.get_local_id(2) * cpy_ne_D]);
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_ಗಾತ್ರ + i1] += tmp[i1];
                }
            }
#ಇಲ್ಲದಿದ್ದರೆ
            constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < DVp/warp_ಗಾತ್ರ ? cpy_ne : DVp/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
                __dpct_align__(16) ದಶಮಾಂಶ tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(ಅಂಶ_ct1.get_local_id(1) + ip)*DVp + i0 + ಅಂಶ_ct1.get_local_id(2)*cpy_ne_D]);
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((ದಶಮಾಂಶ *)VKQ)[i0/warp_ಗಾತ್ರ + i1] += tmp[i1];
                }
            }
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

            KQ_ಮೊತ್ತ[0] += KQ_ಮೊತ್ತ_combine[ಅಂಶ_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    ಆದರೆ (sinks && ಅಂಶ_ct1.get_ಗುಂಪು(1) == 0) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   jc   = jc0 + (ಅಂಶ_ct1.get_local_id(1) / np) * cpw;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sink = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) sinks)[head0 + jc % ncols2];

            ದಶಮಾಂಶ       KQ_max_new_j = sycl::fmax((ದಶಮಾಂಶ) KQ_max[jc0], sink);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ KQ_max_scale = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (sink - KQ_max[jc0]));
            KQ_ಮೊತ್ತ[jc0] = KQ_ಮೊತ್ತ[jc0]*KQ_max_scale + val;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
            ಸ್ಥಿರಾಂಕ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ) {
                VKQ[jc0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ] *= KQ_max_scale_h2;
            }
#ಇಲ್ಲದಿದ್ದರೆ
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ) {
                VKQ[jc0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_ಗಾತ್ರ) + i0/warp_ಗಾತ್ರ].y() *= KQ_max_scale;
            }
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16
        }
    }

    // Write back results:
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ jc0 = 0; jc0 < cpw; ++jc0) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc = jc0 + (ಅಂಶ_ct1.get_local_id(1) / np) * cpw;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = jc / ncols2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = jc % ncols2;

        ಆದರೆ (ncols1 > 1 && ಲಂಬಸಾಲು_Q_0 + j >= ಪೂರ್ಣಾಂಕ(ne01.z())) {
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = ಅಂಶ_ct1.get_ಗುಂಪು_range(1) == 1 ? 1.0f / KQ_ಮೊತ್ತ[jc0] : 1.0f;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j_dst_unrolled =
            ((sequence * ಪೂರ್ಣಾಂಕ(ne01.z()) + ಲಂಬಸಾಲು_Q_0 + j) * ne02 + head0 + c) * ಅಂಶ_ct1.get_ಗುಂಪು_range(1) +
            ಅಂಶ_ct1.get_ಗುಂಪು(1);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FAST_FP16
        constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_ಗಾತ್ರ ? cpy_ne/2 : (DVp/2)/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp/2; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_ಗಾತ್ರ) + i0 / warp_ಗಾತ್ರ + i1]
                              .ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            ಆದರೆ (i0 + warp_ಗಾತ್ರ * cpy_ne_D <= DV / 2 || i0 + ಅಂಶ_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + ಅಂಶ_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#ಇಲ್ಲದಿದ್ದರೆ
        constexpr ಪೂರ್ಣಾಂಕ cpy_ne_D = cpy_ne < DVp/warp_ಗಾತ್ರ ? cpy_ne : DVp/warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DVp; i0 += warp_ಗಾತ್ರ*cpy_ne_D) {
            ಆದರೆ (i0 + warp_ಗಾತ್ರ*cpy_ne_D <= DV || i0 + ಅಂಶ_ct1.get_local_id(2)*cpy_ne_D < DV) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_ಗಾತ್ರ) + i0/(2*warp_ಗಾತ್ರ) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_ಗಾತ್ರ) + i0/(2*warp_ಗಾತ್ರ) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + ಅಂಶ_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_ಗಾತ್ರ) + i0/(2*warp_ಗಾತ್ರ)]);
            }
        }
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FAST_FP16

        ಆದರೆ (ಅಂಶ_ct1.get_ಗುಂಪು_range(1) != 1 && ಅಂಶ_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_ಮೊತ್ತ[jc0]);
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ_VARS(Q, K, V, ಮುಖವಾಡ, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_ತಲೆ_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#ಆದರೆ_ಅಂತ್ಯ // SYCL_FLASH_ATTN
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DKQ, ಪೂರ್ಣಾಂಕ DV, ಪೂರ್ಣಾಂಕ ncols2, ತಾರ್ಕಿಕ use_logit_softcap>
ಸ್ಥಿರ ಶೂನ್ಯ launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q = dst->src[0];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id        = ggml_sycl_get_device();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc        = ggml_sycl_info().devices[id].cc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ = WARP_32_ಗಾತ್ರ; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    ಆದರೆ (DV < 512 && Q->ne[1] < 32) {
        ಆದರೆ constexpr (ncols2 <= 32) {
            ಆದರೆ (Q->ne[1] > 16/ncols2) {
                constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 32;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc) / warp_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc);
                launch_fattn<DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ / ncols2, ncols2, use_logit_softcap, warp_ಗಾತ್ರ>, warp_ಗಾತ್ರ>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
                ಹಿಂತಿರುಗಿ;
            }
        }
        ಆದರೆ constexpr (ncols2 <= 16) {
            ಆದರೆ (Q->ne[1] > 8/ncols2) {
                constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 16;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc) / warp_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc);
                launch_fattn<DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ / ncols2, ncols2, use_logit_softcap, warp_ಗಾತ್ರ>, warp_ಗಾತ್ರ>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
                ಹಿಂತಿರುಗಿ;
            }
        }
        ಆದರೆ constexpr (ncols2 <= 8) {
            ಆದರೆ (Q->ne[1] > 4/ncols2) {
                constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 8;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc) / warp_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc);
                launch_fattn<DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ / ncols2, ncols2, use_logit_softcap, warp_ಗಾತ್ರ>, warp_ಗಾತ್ರ>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
                ಹಿಂತಿರುಗಿ;
            }
        }
    }

    ಆದರೆ constexpr (ncols2 <= 4) {
        ಆದರೆ (Q->ne[1] > 2/ncols2) {
            constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 4;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc) / warp_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc);
            launch_fattn<DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ / ncols2, ncols2, use_logit_softcap, warp_ಗಾತ್ರ>, warp_ಗಾತ್ರ>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
            ಹಿಂತಿರುಗಿ;
        }
    }

    ಆದರೆ constexpr (ncols2 <= 2) {
        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc) / warp_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc);
        launch_fattn<DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ / ncols2, ncols2, use_logit_softcap, warp_ಗಾತ್ರ>, warp_ಗಾತ್ರ>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
        ಹಿಂತಿರುಗಿ;
    }

    {
        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = ncols2*2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc) / warp_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, cc);
        launch_fattn<DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ / ncols2, ncols2, use_logit_softcap, warp_ಗಾತ್ರ>, warp_ಗಾತ್ರ>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ABORT("fatal error");
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DKQ, ಪೂರ್ಣಾಂಕ DV, ತಾರ್ಕಿಕ use_logit_softcap>
ಸ್ಥಿರ ಶೂನ್ಯ launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * KQV  = dst;
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q    = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = dst->src[3];

    ದಶಮಾಂಶ max_bias = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಎಲ್ಲೆ = gqa_ಅನುಪಾತ <= 4 && DV <= 256 ? 16 : ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_gqa_opt = ಮುಖವಾಡ && max_bias == 0.0f && Q->ne[1] <= gqa_ಎಲ್ಲೆ && K->ne[1] % FATTN_KQ_STRIDE == 0;

    ಆದರೆ constexpr (DV == 512) {
        ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }
        ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }
        ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        ಆದರೆ constexpr (DKQ == DV) {
            ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                ಹಿಂತಿರುಗಿ;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }
    }

    ಆದರೆ constexpr (DV <= 256) {
        ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (use_gqa_opt && gqa_ಅನುಪಾತ % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            ಹಿಂತಿರುಗಿ;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        ಹಿಂತಿರುಗಿ;
    }
    GGML_ABORT("fatal error");
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DKQ, ಪೂರ್ಣಾಂಕ DV>
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * KQV = dst;

    ದಶಮಾಂಶ logit_softcap;
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logit_softcap == 0.0f) {
        constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಸುಳ್ಳು;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        constexpr ತಾರ್ಕಿಕ use_logit_softcap = ನಿಜ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

ಶೂನ್ಯ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ವ್ಯಾಖ್ಯಾನಿಸು DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    ಮಾದರಿ ಶೂನ್ಯ ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

ಬಾಹ್ಯ DECL_FATTN_TILE_CASE( 40,  40);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE( 64,  64);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE( 72,  72);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE( 80,  80);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE( 96,  96);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE(112, 112);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE(128, 128);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE(256, 256);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE(512, 512);
ಬಾಹ್ಯ DECL_FATTN_TILE_CASE(576, 512);

