#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/work_group_static.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-common.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>

ਨਾਮਾਕਾਸ਼ syclex = sycl::ext::oneapi::experimental;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    ਜੇ (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        ਵਾਪਸ ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

ਸਥਿਰ constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
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

    ਵਾਪਸ 0;
}

ਸਥਿਰ constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
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

    ਵਾਪਸ 0;
}

ਸਥਿਰ constexpr uint32_t ggml_sycl_fattn_tile_get_config(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ cc) {
    ਜੇ(fast_fp16_ਉਪਲਬਧ(cc))
        ਵਾਪਸ ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    ਨਹੀਂਤਾਂ
        ਵਾਪਸ ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

ਸਥਿਰ constexpr uint32_t ggml_sycl_fattn_tile_get_config(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
    ਵਾਪਸ ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
}

ਸਥਿਰ ਪੂਰਨ ggml_sycl_fattn_tile_get_nthreads(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ cc) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

ਸਥਿਰ constexpr ਪੂਰਨ ggml_sycl_fattn_tile_get_nthreads(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

ਸਥਿਰ ਪੂਰਨ ggml_sycl_fattn_tile_get_occupancy(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ cc) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

ਸਥਿਰ constexpr ਪੂਰਨ ggml_sycl_fattn_tile_get_occupancy(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

ਸਥਿਰ ਪੂਰਨ ggml_sycl_fattn_tile_get_nbatch_fa(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ cc) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

ਸਥਿਰ constexpr ਪੂਰਨ ggml_sycl_fattn_tile_get_nbatch_fa(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

ਸਥਿਰ ਪੂਰਨ ggml_sycl_fattn_tile_get_nbatch_K(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ cc) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

ਸਥਿਰ constexpr ਪੂਰਨ ggml_sycl_fattn_tile_get_nbatch_K(ਅਚਲ ਪੂਰਨ DKQ, ਅਚਲ ਪੂਰਨ DV, ਅਚਲ ਪੂਰਨ ncols) {
    ਵਾਪਸ (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

ਸਾਂਚਾ <ਪੂਰਨ warp_ਮਾਪ, ਪੂਰਨ nwarps, ਪੂਰਨ I, ਪੂਰਨ J, ਪੂਰਨ J_padding, ਤਾਰਕਿਕ oob_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ flash_attn_tile_ਲੱਦੋ_tile(ਅਚਲ sycl::half2 * ਅਚਲ __restrict__ KV,
                                                      sycl::half2 * ਅਚਲ __restrict__ tile_KV,
                                                      ਅਚਲ ਪੂਰਨ stride_KV,
                                                      ਅਚਲ ਪੂਰਨ i_sup) {
    ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    constexpr ਪੂਰਨ cpy_nb = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    ਸਵੈਚਾਲਿਤ load = [&] (ਅਚਲ ਪੂਰਨ n) {
        ਅਚਲ ਪੂਰਨ stride_j = warp_ਮਾਪ >> n;

        ਜੇ (stride_j == 0) {
            ਵਾਪਸ;
        }

        ਅਚਲ ਪੂਰਨ j0_ਸ਼ੁਰੂ = stride_j == warp_ਮਾਪ ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        ਅਚਲ ਪੂਰਨ j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        ਅਚਲ ਪੂਰਨ stride_i = warp_ਮਾਪ / stride_j;

        ਜੇ (j0_ਸ਼ੁਰੂ == j0_stop) {
            ਵਾਪਸ;
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ਅਚਲ ਪੂਰਨ i = i0 + ਵਸਤੂ_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_ਮਾਪ ? 0 : ਵਸਤੂ_ct1.get_local_id(2) / stride_j);

            ਜੇ (i0 + nwarps*stride_i <= I || i < I) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ j0 = j0_ਸ਼ੁਰੂ; j0 < j0_stop; j0 += stride_j) {
                    ਅਚਲ ਪੂਰਨ j = j0 * cpy_ne + (stride_j == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) :
                                                                         ਵਸਤੂ_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    ਅਚਲ __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
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

ਸਾਂਚਾ <ਪੂਰਨ warp_ਮਾਪ, ਪੂਰਨ nwarps, ਪੂਰਨ I, ਪੂਰਨ J, ਪੂਰਨ J_padding, ਤਾਰਕਿਕ oob_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ flash_attn_tile_ਲੱਦੋ_tile(ਅਚਲ sycl::half2 * ਅਚਲ __restrict__ KV,
                                                      ਦਸ਼ਮਲਵ * ਅਚਲ __restrict__ tile_KV,
                                                      ਅਚਲ ਪੂਰਨ stride_KV,
                                                      ਅਚਲ ਪੂਰਨ i_sup) {
    constexpr ਪੂਰਨ cpy_nb = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    ਸਵੈਚਾਲਿਤ load = [&] (ਅਚਲ ਪੂਰਨ n) {
        ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
        ਅਚਲ ਪੂਰਨ stride_j = warp_ਮਾਪ >> n;

        ਜੇ (stride_j == 0) {
            ਵਾਪਸ;
        }

        ਅਚਲ ਪੂਰਨ j0_ਸ਼ੁਰੂ = stride_j == warp_ਮਾਪ ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        ਅਚਲ ਪੂਰਨ j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        ਅਚਲ ਪੂਰਨ stride_i = warp_ਮਾਪ / stride_j;

        ਜੇ (j0_ਸ਼ੁਰੂ == j0_stop) {
            ਵਾਪਸ;
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ਅਚਲ ਪੂਰਨ i = i0 + ਵਸਤੂ_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_ਮਾਪ ? 0 : ਵਸਤੂ_ct1.get_local_id(2) / stride_j);

            ਜੇ (i0 + nwarps*stride_i <= I || i < I) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ j0 = j0_ਸ਼ੁਰੂ; j0 < j0_stop; j0 += stride_j) {
                    ਅਚਲ ਪੂਰਨ j = j0 * (cpy_ne / 2) + (stride_j == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) :
                                                                               ਵਸਤੂ_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    ਅਚਲ sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<ਆਕਾਰ(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
                    }
                    ggml_sycl_memcpy_1<ਆਕਾਰ(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
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
ਸਾਂਚਾ <ਪੂਰਨ  warp_ਮਾਪ,
          ਪੂਰਨ  nwarps,
          ਪੂਰਨ  ncols1,
          ਪੂਰਨ  ncols2,
          ਪੂਰਨ  DKQ,
          ਪੂਰਨ  nbatch_fa,
          ਪੂਰਨ  nbatch_K,
          ਤਾਰਕਿਕ use_logit_softcap,
          ਤਾਰਕਿਕ oob_check,
          ਕਿਸਮ_ਨਾਂ T_vec_dot>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ flash_attn_tile_iter_KQ(T_vec_dot * ਅਚਲ Q_tmp,
                                                    ਅਚਲ sycl::half2 * ਅਚਲ __restrict__ K_h2,
                                                    T_vec_dot * ਅਚਲ KV_tmp,
                                                    ਅਚਲ ਪੂਰਨ         stride_K2,
                                                    ਅਚਲ ਪੂਰਨ         k_VKQ_0,
                                                    ਅਚਲ ਪੂਰਨ         k_VKQ_sup,
                                                    ਅਚਲ ਪੂਰਨ         k_KQ_0,
                                                    ਦਸ਼ਮਲਵ *           KQ_acc) {
    ਸਵੈਚਾਲਿਤ          ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    constexpr ਪੂਰਨ cpy_nb   = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    constexpr ਪੂਰਨ ncols = ncols1*ncols2;
    constexpr ਪੂਰਨ cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr ਪੂਰਨ np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_ਲੱਦੋ_tile<warp_ਮਾਪ, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_ਮਾਪ)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#ਨਹੀਂਤਾਂ
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) ਦਸ਼ਮਲਵ K_k[nbatch_fa/(np*warp_ਮਾਪ)][cpy_ne];
        __dpct_align__(16) ਦਸ਼ਮਲਵ Q_k[cpw][cpy_ne];
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ i_KQ = i_KQ_0 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * warp_ਮਾਪ + ਵਸਤੂ_ct1.get_local_id(2);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_ਮਾਪ)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#ਨਹੀਂਤਾਂ
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_ਮਾਪ)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
        }
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
            ਅਚਲ ਪੂਰਨ jc = jc0 + (ਵਸਤੂ_ct1.get_local_id(1) / np) * cpw;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#ਨਹੀਂਤਾਂ
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_ਮਾਪ) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_ਮਾਪ)*cpw + jc0], K_k[i_KQ_0/(np*warp_ਮਾਪ)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    ਜੇ (k_KQ_0 + nbatch_K < DKQ) {
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
ਸਾਂਚਾ <ਪੂਰਨ  warp_ਮਾਪ,
          ਪੂਰਨ  nwarps,
          ਪੂਰਨ  ncols1,
          ਪੂਰਨ  ncols2,
          ਪੂਰਨ  DKQ,
          ਪੂਰਨ  DV,
          ਪੂਰਨ  nbatch_fa,
          ਪੂਰਨ  nbatch_K,
          ਤਾਰਕਿਕ use_logit_softcap,
          ਤਾਰਕਿਕ oob_check,
          ਕਿਸਮ_ਨਾਂ T_vec_dot,
          ਕਿਸਮ_ਨਾਂ T_KQ,
          ਕਿਸਮ_ਨਾਂ T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ flash_attn_tile_iter(T_vec_dot * ਅਚਲ Q_tmp,
                                                 ਅਚਲ sycl::half2 * ਅਚਲ __restrict__ K_h2,
                                                 ਅਚਲ sycl::half2 * ਅਚਲ __restrict__ V_h2,
                                                 ਅਚਲ sycl::half * ਅਚਲ __restrict__ ਨਕਾਬ,
                                                 ਅਚਲ sycl::uint3 ne01,
                                                 ਅਚਲ ਦਸ਼ਮਲਵ       logit_softcap,
                                                 ਅਚਲ ਦਸ਼ਮਲਵ       ਢਲਾਨ,
                                                 T_KQ * ਅਚਲ      KQ,
                                                 T_vec_dot * ਅਚਲ KV_tmp,
                                                 ਅਚਲ ਪੂਰਨ         stride_K2,
                                                 ਅਚਲ ਪੂਰਨ         stride_V2,
                                                 ਅਚਲ ਪੂਰਨ         stride_ਨਕਾਬ,
                                                 ਦਸ਼ਮਲਵ * ਅਚਲ     KQ_max,
                                                 ਦਸ਼ਮਲਵ * ਅਚਲ     KQ_ਜੋੜ,
                                                 T_acc * ਅਚਲ     VKQ,
                                                 ਅਚਲ ਪੂਰਨ         k_VKQ_0,
                                                 ਅਚਲ ਪੂਰਨ         k_VKQ_max,
                                                 ਅਚਲ ਪੂਰਨ         ਖਾਨਾ_Q_0,
                                                 ਦਸ਼ਮਲਵ *           KQ_max_new_shared) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    constexpr ਪੂਰਨ cpy_nb   = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    constexpr ਪੂਰਨ ncols = ncols1*ncols2;
    constexpr ਪੂਰਨ cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr ਪੂਰਨ np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr ਪੂਰਨ DVp = (DV + 2*warp_ਮਾਪ - 1) & ~(2*warp_ਮਾਪ - 1); // DV padded to multiple of 2*warp_size.

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
    constexpr ਪੂਰਨ KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#ਨਹੀਂਤਾਂ
    constexpr ਪੂਰਨ KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    ਅਚਲ ਪੂਰਨ k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    ਦਸ਼ਮਲਵ KQ_max_new[cpw];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    ਦਸ਼ਮਲਵ KQ_acc[nbatch_fa/(np*warp_ਮਾਪ) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr ਪੂਰਨ nbatch_K_ਆਖਰੀ = DKQ % nbatch_K;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_ਆਖਰੀ; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_ਮਾਪ, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    ਜੇ (nbatch_K_ਆਖਰੀ > 0) {
        constexpr ਪੂਰਨ k_KQ_0 = DKQ - nbatch_K_ਆਖਰੀ;
        flash_attn_tile_iter_KQ<warp_ਮਾਪ, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_ਆਖਰੀ, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
        ਅਚਲ ਪੂਰਨ j = fastmodulo(ਖਾਨਾ_Q_0 + (jc0 + (ਵਸਤੂ_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ i_KQ = i_KQ_0 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * warp_ਮਾਪ + ਵਸਤੂ_ct1.get_local_id(2);

#ਜੇ defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_ਮਾਪ)*cpw + jc0] *= 4.0f;
#ਜੇ_ਅੰਤ // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            ਜੇ (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_ਮਾਪ)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((ਦਸ਼ਮਲਵ) KQ_acc[(i_KQ_0 / (np * warp_ਮਾਪ)) * cpw + jc0]);
            }

            ਜੇ (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_ਮਾਪ)) * cpw + jc0] +=
                    (ncols2 > 1 || ਨਕਾਬ) ? ਢਲਾਨ * sycl::vec<sycl::half, 1>(ਨਕਾਬ[j * stride_ਨਕਾਬ + k_VKQ_0 + i_KQ])
                                                       .convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((ਦਸ਼ਮਲਵ) KQ_max_new[jc0],
                               (ਦਸ਼ਮਲਵ) (KQ_acc[(i_KQ_0 / (np * warp_ਮਾਪ)) * cpw + jc0] + FATTN_KQ_MAX_ਵਿੱਥ));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_ਮਾਪ>(KQ_max_new[jc0]);
    }

    ਜੇ constexpr (np == 1) {
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
    } ਨਹੀਂਤਾਂ {
        static_assert(cpw == 1, "bad cpw");

        ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[ਵਸਤੂ_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(ਵਸਤੂ_ct1.get_local_id(1) & ~(np - 1)) + ਵਸਤੂ_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_ਮਾਪ)][KQ_cs];
#ਨਹੀਂਤਾਂ
        __dpct_align__(16) ਦਸ਼ਮਲਵ tmp[nbatch_fa/(np*warp_ਮਾਪ)][KQ_cs];
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ jc1 = 0; jc1 < KQ_cs; ++jc1) {
            ਅਚਲ ਪੂਰਨ jc = jc0 + jc1;

            ਅਚਲ ਦਸ਼ਮਲਵ KQ_max_scale = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            ਦਸ਼ਮਲਵ KQ_ਜੋੜ_ਜੋੜੋ = 0.0f;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < nbatch_fa; i0 += np*warp_ਮਾਪ) {
                ਅਚਲ ਦਸ਼ਮਲਵ val =
                    !oob_check || i0 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * warp_ਮਾਪ + ਵਸਤੂ_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_acc[(i0 / (np * warp_ਮਾਪ)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_ਜੋੜ_ਜੋੜੋ += val;
                tmp[i0/(np*warp_ਮਾਪ)][jc1] = val;
            }
            KQ_ਜੋੜ[jc] = KQ_ਜੋੜ[jc]*KQ_max_scale + KQ_ਜੋੜ_ਜੋੜੋ;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
            ਅਚਲ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ) {
                VKQ[jc*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].y() *= KQ_max_scale_h2.y();
            }
#ਨਹੀਂਤਾਂ
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ) {
                VKQ[jc*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].y() *= KQ_max_scale;
            }
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < nbatch_fa; i0 += np*warp_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ i = i0 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * warp_ਮਾਪ + ਵਸਤੂ_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<ਆਕਾਰ(tmp[0])>(
                KQ + (jc0 / KQ_cs + (ਵਸਤੂ_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_ਮਾਪ)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr ਪੂਰਨ nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_ਲੱਦੋ_tile<warp_ਮਾਪ, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_ਮਾਪ];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr ਪੂਰਨ cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_ਮਾਪ ? cpy_ne/2 : (DVp/2)/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_ਮਾਪ],
                                                 &KV_tmp[(k1 + ਵਸਤੂ_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         ਵਸਤੂ_ct1.get_local_id(2) * cpy_ne_D]);
            }
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ਅਚਲ ਪੂਰਨ jc_KQ = jc_VKQ_0 / KQ_cs + (ਵਸਤੂ_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * ਆਕਾਰ(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + ਵਸਤੂ_ct1.get_local_id(1) % np) * KQ_cs);
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].x() +=
                        V_k[i0/warp_ਮਾਪ].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].y() +=
                        V_k[i0/warp_ਮਾਪ].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#ਨਹੀਂਤਾਂ
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_ਮਾਪ];
            __dpct_align__(16) ਦਸ਼ਮਲਵ  KQ_k[cpw];

            constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < DVp/warp_ਮਾਪ ? cpy_ne : DVp/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp; i0 += warp_ਮਾਪ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_ਮਾਪ)], &KV_tmp[(k1 + ਵਸਤੂ_ct1.get_local_id(1) % np)*DV + i0 + ਵਸਤੂ_ct1.get_local_id(2)*cpy_ne_D]);
            }
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ਅਚਲ ਪੂਰਨ jc_KQ = jc_VKQ_0/KQ_cs + (ਵਸਤੂ_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*ਆਕਾਰ(ਦਸ਼ਮਲਵ)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + ਵਸਤੂ_ct1.get_local_id(1) % np)*KQ_cs);
            }

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].x() += V_k[i0/warp_ਮਾਪ].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].y() += V_k[i0/warp_ਮਾਪ].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

ਸਾਂਚਾ <ਪੂਰਨ DKQ, ਪੂਰਨ DV, ਪੂਰਨ ncols1, ਪੂਰਨ ncols2, ਤਾਰਕਿਕ use_logit_softcap, ਪੂਰਨ warp_ਮਾਪ>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ਸਥਿਰ ਖਾਲੀ flash_attn_tile(ਅਚਲ ਅੱਖਰ *  Q,
                            ਅਚਲ ਅੱਖਰ *  K,
                            ਅਚਲ ਅੱਖਰ *  V,
                            ਅਚਲ ਅੱਖਰ *  ਨਕਾਬ,
                            ਅਚਲ ਅੱਖਰ *  sinks,
                            ਅਚਲ ਪੂਰਨ *  KV_max,
                            ਦਸ਼ਮਲਵ *  dst,
                            sycl::float2 *  dst_meta,
                            ਅਚਲ ਦਸ਼ਮਲਵ          scale,
                            ਅਚਲ ਦਸ਼ਮਲਵ          max_bias,
                            ਅਚਲ ਦਸ਼ਮਲਵ          m0,
                            ਅਚਲ ਦਸ਼ਮਲਵ          m1,
                            ਅਚਲ uint32_t       n_ਸਿਰ_log2,
                            ਅਚਲ ਦਸ਼ਮਲਵ          logit_softcap,
                            ਅਚਲ int32_t        ne00,
                            ਅਚਲ sycl::uint3    ne01,
                            ਅਚਲ int32_t        ne02,
                            ਅਚਲ int32_t        ne03,
                            ਅਚਲ int32_t        nb01,
                            ਅਚਲ int32_t        nb02,
                            ਅਚਲ int32_t        nb03,
                            ਅਚਲ int32_t        ne10,
                            ਅਚਲ int32_t        ne11,
                            ਅਚਲ int32_t        ne12,
                            ਅਚਲ int32_t        ne13,
                            ਅਚਲ int32_t        nb11,
                            ਅਚਲ int32_t        nb12,
                            ਅਚਲ int64_t        nb13,
                            ਅਚਲ int32_t        nb21,
                            ਅਚਲ int32_t        nb22,
                            ਅਚਲ int64_t        nb23,
                            ਅਚਲ int32_t        ne31,
                            ਅਚਲ int32_t        ne32,
                            ਅਚਲ int32_t        ne33,
                            ਅਚਲ int32_t        nb31,
                            ਅਚਲ int32_t        nb32,
                            ਅਚਲ int64_t        nb33) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਜੇ ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_ਅਣਵਰਤਿਆ_VARS(Q, K, V, ਨਕਾਬ, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_ਸਿਰ_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ਵਾਪਸ;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr ਪੂਰਨ ncols     = ncols1*ncols2;

    constexpr ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_ਮਾਪ;
    constexpr ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr ਪੂਰਨ nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    ਅਚਲ ਪੂਰਨ ਖਾਨਾ_Q_0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    ਅਚਲ ਪੂਰਨ           sequence  = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) / (ne02 / ncols2);
    ਅਚਲ ਪੂਰਨ           head0     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    ਅਚਲ ਦਸ਼ਮਲਵ * Q_f  = (ਅਚਲ ਦਸ਼ਮਲਵ *) (Q + nb03*sequence + nb02* head0);
    ਅਚਲ sycl::half2 * K_h2      = (ਅਚਲ sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_ਅਨੁਪਾਤ));
    ਅਚਲ sycl::half2 * V_h2 =
        (ਅਚਲ sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_ਅਨੁਪਾਤ));  // K and V have same shape

    ਅਚਲ sycl::half * maskh = ਨਕਾਬ ? (ਅਚਲ sycl::half *) (ਨਕਾਬ + nb33 * (sequence % ne33)) : ਖਾਲੀ_ਸੂਚਕ;

    ਅਚਲ ਪੂਰਨ stride_K2   = nb11 / ਆਕਾਰ(sycl::half2);
    ਅਚਲ ਪੂਰਨ stride_V2   = nb21 / ਆਕਾਰ(sycl::half2);
    ਅਚਲ ਪੂਰਨ stride_ਨਕਾਬ = nb31 / ਆਕਾਰ(sycl::half);

    ਅਚਲ ਦਸ਼ਮਲਵ ਢਲਾਨ = ncols2 == 1 ? get_alibi_ਢਲਾਨ(max_bias, head0, n_ਸਿਰ_log2, m0, m1) : 1.0f;

    constexpr ਪੂਰਨ cpy_nb = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    constexpr ਪੂਰਨ cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr ਪੂਰਨ np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_ਮਾਪ) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr ਪੂਰਨ DKQp = (DKQ + 2*warp_ਮਾਪ - 1) & ~(2*warp_ਮਾਪ - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr ਪੂਰਨ DVp  = (DV  + 2*warp_ਮਾਪ - 1) & ~(2*warp_ਮਾਪ - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_ਮਾਪ = lsm_size1 * ਆਕਾਰ(sycl::half2) +
                                            lsm_size2 * ਆਕਾਰ(sycl::half2) +
                                            lsm_size3 * ਆਕਾਰ(sycl::half) +
                                            lsm_size4 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    syclex::work_ਸਮੂਹ_static<ਅੱਖਰ[local_share_mem_ਮਾਪ]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    ਦਸ਼ਮਲਵ *KQ_max_new_shared = (ਦਸ਼ਮਲਵ *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_ਮਾਪ)] = {
        { 0.0f, 0.0f }
    };
#ਨਹੀਂਤਾਂ
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_ਮਾਪ = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    syclex::work_ਸਮੂਹ_static<ਅੱਖਰ[local_share_mem_ਮਾਪ]> lsm;

    ਦਸ਼ਮਲਵ *Q_tmp = (ਦਸ਼ਮਲਵ *)&lsm;
    ਦਸ਼ਮਲਵ *KV_tmp = Q_tmp +lsm_size1;
    ਦਸ਼ਮਲਵ *KQ = KV_tmp+lsm_size2;
    ਦਸ਼ਮਲਵ *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_ਮਾਪ)] = {{0.0f, 0.0f}};


#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

    ਦਸ਼ਮਲਵ KQ_max[cpw] = {};

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    ਦਸ਼ਮਲਵ KQ_ਜੋੜ[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
        ਅਚਲ ਪੂਰਨ jc = jc0 + (ਵਸਤੂ_ct1.get_local_id(1) / np) * cpw;

        ਅਚਲ ਪੂਰਨ j = jc / ncols2;
        ਅਚਲ ਪੂਰਨ c = jc % ncols2;

        constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < DKQp/warp_ਮਾਪ ? cpy_ne : DKQp/warp_ਮਾਪ;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < DKQp; i0 += np*warp_ਮਾਪ*cpy_ne_D) {
            ਜੇ (i0 + np * warp_ਮਾਪ * cpy_ne_D <= DKQ ||
                i0 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * (warp_ਮਾਪ * cpy_ne_D) + ਵਸਤੂ_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) ਦਸ਼ਮਲਵ tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<ਆਕਾਰ(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + fastmodulo(ਖਾਨਾ_Q_0 + j, ne01) * (nb01 / ਆਕਾਰ(ਦਸ਼ਮਲਵ)) +
                                i0 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * (warp_ਮਾਪ * cpy_ne_D) +
                                ਵਸਤੂ_ct1.get_local_id(2) * cpy_ne_D]);

#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#ਜੇ defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#ਜੇ_ਅੰਤ // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<ਆਕਾਰ(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (ਵਸਤੂ_ct1.get_local_id(1) % np) * (warp_ਮਾਪ * cpy_ne_D / 2) +
                           ਵਸਤੂ_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#ਨਹੀਂਤਾਂ
                ggml_sycl_memcpy_1<ਆਕਾਰ(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (ਵਸਤੂ_ct1.get_local_id(1) % np)*(warp_ਮਾਪ*cpy_ne_D)   + ਵਸਤੂ_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
            }
        }
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    ਅਚਲ ਪੂਰਨ k_VKQ_max = KV_max ? KV_max[sequence * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2) + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2)] : ne11;
    ਜੇ (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        ਪੂਰਨ k_VKQ_0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * nbatch_fa;
        ਜਦੋਂ (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr ਤਾਰਕਿਕ oob_check = ਝੂਠ;
            flash_attn_tile_iter<warp_ਮਾਪ, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ਢਲਾਨ, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ਨਕਾਬ, KQ_max, KQ_ਜੋੜ, VKQ, k_VKQ_0, k_VKQ_max, ਖਾਨਾ_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) * nbatch_fa;
        }
        ਜੇ (k_VKQ_0 < k_VKQ_max) {
            constexpr ਤਾਰਕਿਕ oob_check = ਸੱਚ;
            flash_attn_tile_iter<warp_ਮਾਪ, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ਢਲਾਨ, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ਨਕਾਬ, KQ_max, KQ_ਜੋੜ, VKQ, k_VKQ_0, k_VKQ_max, ਖਾਨਾ_Q_0,
                                            KQ_max_new_shared);
        }
    } ਨਹੀਂਤਾਂ {
        // Branch without out-of-bounds checks.
        ਲਈ (ਪੂਰਨ k_VKQ_0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) * nbatch_fa) {

            constexpr ਤਾਰਕਿਕ oob_check = ਝੂਠ;
            flash_attn_tile_iter<warp_ਮਾਪ, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, ਢਲਾਨ, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_ਨਕਾਬ, KQ_max, KQ_ਜੋੜ, VKQ, k_VKQ_0, k_VKQ_max, ਖਾਨਾ_Q_0,
                                            KQ_max_new_shared);
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_ਜੋੜ[jc0] = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(KQ_ਜੋੜ[jc0]);
    }

    ਜੇ constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#ਨਹੀਂਤਾਂ
        ਦਸ਼ਮਲਵ * VKQ_combine    = (ਦਸ਼ਮਲਵ *) KV_tmp;
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

        ਦਸ਼ਮਲਵ * KQ_ਜੋੜ_combine = (ਦਸ਼ਮਲਵ *) Q_tmp;

        ਜੇ (ਵਸਤੂ_ct1.get_local_id(1) % np != 0) {

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
            constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < (DVp/2)/warp_ਮਾਪ ? cpy_ne : (DVp/2)/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[ਵਸਤੂ_ct1.get_local_id(1) * (DVp / 2) + i0 + ਵਸਤੂ_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_ਮਾਪ]);
            }
#ਨਹੀਂਤਾਂ

            constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < DVp/warp_ਮਾਪ ? cpy_ne : DVp/warp_ਮਾਪ;

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp; i0 += warp_ਮਾਪ*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[ਵਸਤੂ_ct1.get_local_id(1)*DVp + i0 + ਵਸਤੂ_ct1.get_local_id(2)*cpy_ne_D], ((ਅਚਲ ਦਸ਼ਮਲਵ *) VKQ) + i0/warp_ਮਾਪ);
            }
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

            ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
                KQ_ਜੋੜ_combine[ਵਸਤੂ_ct1.get_local_id(1)] = KQ_ਜੋੜ[0];
            }
            ਵਾਪਸ;
        }

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ip = 1; ip < np; ++ip) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
            constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < (DVp/2)/warp_ਮਾਪ ? cpy_ne : (DVp/2)/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(ਵਸਤੂ_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   ਵਸਤੂ_ct1.get_local_id(2) * cpy_ne_D]);
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_ਮਾਪ + i1] += tmp[i1];
                }
            }
#ਨਹੀਂਤਾਂ
            constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < DVp/warp_ਮਾਪ ? cpy_ne : DVp/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp; i0 += warp_ਮਾਪ*cpy_ne_D) {
                __dpct_align__(16) ਦਸ਼ਮਲਵ tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(ਵਸਤੂ_ct1.get_local_id(1) + ip)*DVp + i0 + ਵਸਤੂ_ct1.get_local_id(2)*cpy_ne_D]);
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((ਦਸ਼ਮਲਵ *)VKQ)[i0/warp_ਮਾਪ + i1] += tmp[i1];
                }
            }
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

            KQ_ਜੋੜ[0] += KQ_ਜੋੜ_combine[ਵਸਤੂ_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    ਜੇ (sinks && ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) == 0) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
            ਅਚਲ ਪੂਰਨ   jc   = jc0 + (ਵਸਤੂ_ct1.get_local_id(1) / np) * cpw;
            ਅਚਲ ਦਸ਼ਮਲਵ sink = ((ਅਚਲ ਦਸ਼ਮਲਵ *) sinks)[head0 + jc % ncols2];

            ਦਸ਼ਮਲਵ       KQ_max_new_j = sycl::fmax((ਦਸ਼ਮਲਵ) KQ_max[jc0], sink);
            ਅਚਲ ਦਸ਼ਮਲਵ KQ_max_scale = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            ਅਚਲ ਦਸ਼ਮਲਵ val = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (sink - KQ_max[jc0]));
            KQ_ਜੋੜ[jc0] = KQ_ਜੋੜ[jc0]*KQ_max_scale + val;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
            ਅਚਲ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ) {
                VKQ[jc0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ] *= KQ_max_scale_h2;
            }
#ਨਹੀਂਤਾਂ
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ) {
                VKQ[jc0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_ਮਾਪ) + i0/warp_ਮਾਪ].y() *= KQ_max_scale;
            }
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16
        }
    }

    // Write back results:
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ jc0 = 0; jc0 < cpw; ++jc0) {
        ਅਚਲ ਪੂਰਨ jc = jc0 + (ਵਸਤੂ_ct1.get_local_id(1) / np) * cpw;

        ਅਚਲ ਪੂਰਨ j = jc / ncols2;
        ਅਚਲ ਪੂਰਨ c = jc % ncols2;

        ਜੇ (ncols1 > 1 && ਖਾਨਾ_Q_0 + j >= ਪੂਰਨ(ne01.z())) {
            ਵਾਪਸ;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ scale = ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) == 1 ? 1.0f / KQ_ਜੋੜ[jc0] : 1.0f;

        ਅਚਲ ਪੂਰਨ j_dst_unrolled =
            ((sequence * ਪੂਰਨ(ne01.z()) + ਖਾਨਾ_Q_0 + j) * ne02 + head0 + c) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) +
            ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FAST_FP16
        constexpr ਪੂਰਨ cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_ਮਾਪ ? cpy_ne/2 : (DVp/2)/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp/2; i0 += warp_ਮਾਪ*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_ਮਾਪ) + i0 / warp_ਮਾਪ + i1]
                              .ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            ਜੇ (i0 + warp_ਮਾਪ * cpy_ne_D <= DV / 2 || i0 + ਵਸਤੂ_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<ਆਕਾਰ(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + ਵਸਤੂ_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#ਨਹੀਂਤਾਂ
        constexpr ਪੂਰਨ cpy_ne_D = cpy_ne < DVp/warp_ਮਾਪ ? cpy_ne : DVp/warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < DVp; i0 += warp_ਮਾਪ*cpy_ne_D) {
            ਜੇ (i0 + warp_ਮਾਪ*cpy_ne_D <= DV || i0 + ਵਸਤੂ_ct1.get_local_id(2)*cpy_ne_D < DV) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_ਮਾਪ) + i0/(2*warp_ਮਾਪ) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_ਮਾਪ) + i0/(2*warp_ਮਾਪ) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + ਵਸਤੂ_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_ਮਾਪ) + i0/(2*warp_ਮਾਪ)]);
            }
        }
#ਜੇ_ਅੰਤ // SYCL_FAST_FP16

        ਜੇ (ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) != 1 && ਵਸਤੂ_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_ਜੋੜ[jc0]);
        }
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ_VARS(Q, K, V, ਨਕਾਬ, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_ਸਿਰ_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#ਜੇ_ਅੰਤ // SYCL_FLASH_ATTN
}

ਸਾਂਚਾ <ਪੂਰਨ DKQ, ਪੂਰਨ DV, ਪੂਰਨ ncols2, ਤਾਰਕਿਕ use_logit_softcap>
ਸਥਿਰ ਖਾਲੀ launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * Q = dst->src[0];

    ਅਚਲ ਪੂਰਨ id        = ggml_sycl_get_device();
    ਅਚਲ ਪੂਰਨ cc        = ggml_sycl_info().devices[id].cc;
    ਅਚਲ ਪੂਰਨ warp_ਮਾਪ = WARP_32_ਮਾਪ; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    ਜੇ (DV < 512 && Q->ne[1] < 32) {
        ਜੇ constexpr (ncols2 <= 32) {
            ਜੇ (Q->ne[1] > 16/ncols2) {
                constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 32;
                ਅਚਲ ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc) / warp_ਮਾਪ;
                ਅਚਲ ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc);
                launch_fattn<DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ / ncols2, ncols2, use_logit_softcap, warp_ਮਾਪ>, warp_ਮਾਪ>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ਸੱਚ, ਸੱਚ, ਝੂਠ);
                ਵਾਪਸ;
            }
        }
        ਜੇ constexpr (ncols2 <= 16) {
            ਜੇ (Q->ne[1] > 8/ncols2) {
                constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 16;
                ਅਚਲ ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc) / warp_ਮਾਪ;
                ਅਚਲ ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc);
                launch_fattn<DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ / ncols2, ncols2, use_logit_softcap, warp_ਮਾਪ>, warp_ਮਾਪ>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ਸੱਚ, ਸੱਚ, ਝੂਠ);
                ਵਾਪਸ;
            }
        }
        ਜੇ constexpr (ncols2 <= 8) {
            ਜੇ (Q->ne[1] > 4/ncols2) {
                constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 8;
                ਅਚਲ ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc) / warp_ਮਾਪ;
                ਅਚਲ ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc);
                launch_fattn<DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ / ncols2, ncols2, use_logit_softcap, warp_ਮਾਪ>, warp_ਮਾਪ>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ਸੱਚ, ਸੱਚ, ਝੂਠ);
                ਵਾਪਸ;
            }
        }
    }

    ਜੇ constexpr (ncols2 <= 4) {
        ਜੇ (Q->ne[1] > 2/ncols2) {
            constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 4;
            ਅਚਲ ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc) / warp_ਮਾਪ;
            ਅਚਲ ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc);
            launch_fattn<DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ / ncols2, ncols2, use_logit_softcap, warp_ਮਾਪ>, warp_ਮਾਪ>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ਸੱਚ, ਸੱਚ, ਝੂਠ);
            ਵਾਪਸ;
        }
    }

    ਜੇ constexpr (ncols2 <= 2) {
        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 2;
        ਅਚਲ ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc) / warp_ਮਾਪ;
        ਅਚਲ ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc);
        launch_fattn<DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ / ncols2, ncols2, use_logit_softcap, warp_ਮਾਪ>, warp_ਮਾਪ>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ਸੱਚ, ਸੱਚ, ਝੂਠ);
        ਵਾਪਸ;
    }

    {
        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = ncols2*2;
        ਅਚਲ ਪੂਰਨ nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc) / warp_ਮਾਪ;
        ਅਚਲ ਪੂਰਨ nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, cc);
        launch_fattn<DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ / ncols2, ncols2, use_logit_softcap, warp_ਮਾਪ>, warp_ਮਾਪ>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, ਸੱਚ, ਸੱਚ, ਝੂਠ);
        ਵਾਪਸ;
    }

    GGML_ABORT("fatal error");
}

ਸਾਂਚਾ <ਪੂਰਨ DKQ, ਪੂਰਨ DV, ਤਾਰਕਿਕ use_logit_softcap>
ਸਥਿਰ ਖਾਲੀ launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * KQV  = dst;
    ਅਚਲ ggml_tensor * Q    = dst->src[0];
    ਅਚਲ ggml_tensor * K    = dst->src[1];
    ਅਚਲ ggml_tensor * ਨਕਾਬ = dst->src[3];

    ਦਸ਼ਮਲਵ max_bias = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    ਅਚਲ ਪੂਰਨ gqa_ਹੱਦਬੰਦੀ = gqa_ਅਨੁਪਾਤ <= 4 && DV <= 256 ? 16 : ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ;
    ਅਚਲ ਤਾਰਕਿਕ use_gqa_opt = ਨਕਾਬ && max_bias == 0.0f && Q->ne[1] <= gqa_ਹੱਦਬੰਦੀ && K->ne[1] % FATTN_KQ_STRIDE == 0;

    ਜੇ constexpr (DV == 512) {
        ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }
        ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }
        ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        ਜੇ constexpr (DKQ == DV) {
            ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                ਵਾਪਸ;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }
    }

    ਜੇ constexpr (DV <= 256) {
        ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }

        ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }

        ਜੇ (use_gqa_opt && gqa_ਅਨੁਪਾਤ % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            ਵਾਪਸ;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        ਵਾਪਸ;
    }
    GGML_ABORT("fatal error");
}

ਸਾਂਚਾ <ਪੂਰਨ DKQ, ਪੂਰਨ DV>
ਖਾਲੀ ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * KQV = dst;

    ਦਸ਼ਮਲਵ logit_softcap;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logit_softcap == 0.0f) {
        constexpr ਤਾਰਕਿਕ use_logit_softcap = ਝੂਠ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } ਨਹੀਂਤਾਂ {
        constexpr ਤਾਰਕਿਕ use_logit_softcap = ਸੱਚ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

ਖਾਲੀ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    ਸਾਂਚਾ ਖਾਲੀ ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

ਬਾਹਰੀ DECL_FATTN_TILE_CASE( 40,  40);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE( 64,  64);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE( 72,  72);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE( 80,  80);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE( 96,  96);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE(112, 112);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE(128, 128);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE(256, 256);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE(512, 512);
ਬਾਹਰੀ DECL_FATTN_TILE_CASE(576, 512);

