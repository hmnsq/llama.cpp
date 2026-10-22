#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব <sycl/ext/oneapi/work_group_static.hpp>
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "fattn-common.hpp"

#অন্তর্ভাব <cmath>
#অন্তর্ভাব <float.h>

নামাকাশ syclex = sycl::ext::oneapi::experimental;

#define GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    যদি (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        ফেরত ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

স্থির constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
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

    ফেরত 0;
}

স্থির constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
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

    ফেরত 0;
}

স্থির constexpr uint32_t ggml_sycl_fattn_tile_get_config(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা cc) {
    যদি(fast_fp16_available(cc))
        ফেরত ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    অন্যথা
        ফেরত ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

স্থির constexpr uint32_t ggml_sycl_fattn_tile_get_config(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
#ifdef SYCL_FAST_FP16
    ফেরত ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#অন্যথা
    ফেরত ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#endif // SYCL_FAST_FP16
}

স্থির পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_nthreads(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা cc) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

স্থির constexpr পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_nthreads(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

স্থির পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_occupancy(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা cc) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

স্থির constexpr পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_occupancy(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

স্থির পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_nbatch_fa(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা cc) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

স্থির constexpr পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_nbatch_fa(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

স্থির পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_nbatch_K(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা cc) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

স্থির constexpr পূর্ণসংখ্যা ggml_sycl_fattn_tile_get_nbatch_K(ধ্রুবক পূর্ণসংখ্যা DKQ, ধ্রুবক পূর্ণসংখ্যা DV, ধ্রুবক পূর্ণসংখ্যা ncols) {
    ফেরত (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

template <পূর্ণসংখ্যা warp_আকার, পূর্ণসংখ্যা nwarps, পূর্ণসংখ্যা I, পূর্ণসংখ্যা J, পূর্ণসংখ্যা J_padding, যৌক্তিক oob_check>
স্থির __dpct_inline__ শূন্য flash_attn_tile_load_tile(ধ্রুবক sycl::half2 * ধ্রুবক __restrict__ KV,
                                                      sycl::half2 * ধ্রুবক __restrict__ tile_KV,
                                                      ধ্রুবক পূর্ণসংখ্যা stride_KV,
                                                      ধ্রুবক পূর্ণসংখ্যা i_sup) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr পূর্ণসংখ্যা cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    auto load = [&] (ধ্রুবক পূর্ণসংখ্যা n) {
        ধ্রুবক পূর্ণসংখ্যা stride_j = warp_আকার >> n;

        যদি (stride_j == 0) {
            ফেরত;
        }

        ধ্রুবক পূর্ণসংখ্যা j0_start = stride_j == warp_আকার ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        ধ্রুবক পূর্ণসংখ্যা j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        ধ্রুবক পূর্ণসংখ্যা stride_i = warp_আকার / stride_j;

        যদি (j0_start == j0_stop) {
            ফেরত;
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ধ্রুবক পূর্ণসংখ্যা i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_আকার ? 0 : item_ct1.get_local_id(2) / stride_j);

            যদি (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    ধ্রুবক পূর্ণসংখ্যা j = j0 * cpy_ne + (stride_j == warp_আকার ? item_ct1.get_local_id(2) :
                                                                         item_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    ধ্রুবক __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
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

template <পূর্ণসংখ্যা warp_আকার, পূর্ণসংখ্যা nwarps, পূর্ণসংখ্যা I, পূর্ণসংখ্যা J, পূর্ণসংখ্যা J_padding, যৌক্তিক oob_check>
স্থির __dpct_inline__ শূন্য flash_attn_tile_load_tile(ধ্রুবক sycl::half2 * ধ্রুবক __restrict__ KV,
                                                      দশমিক * ধ্রুবক __restrict__ tile_KV,
                                                      ধ্রুবক পূর্ণসংখ্যা stride_KV,
                                                      ধ্রুবক পূর্ণসংখ্যা i_sup) {
    constexpr পূর্ণসংখ্যা cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    auto load = [&] (ধ্রুবক পূর্ণসংখ্যা n) {
        auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        ধ্রুবক পূর্ণসংখ্যা stride_j = warp_আকার >> n;

        যদি (stride_j == 0) {
            ফেরত;
        }

        ধ্রুবক পূর্ণসংখ্যা j0_start = stride_j == warp_আকার ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        ধ্রুবক পূর্ণসংখ্যা j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        ধ্রুবক পূর্ণসংখ্যা stride_i = warp_আকার / stride_j;

        যদি (j0_start == j0_stop) {
            ফেরত;
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ধ্রুবক পূর্ণসংখ্যা i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_আকার ? 0 : item_ct1.get_local_id(2) / stride_j);

            যদি (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    ধ্রুবক পূর্ণসংখ্যা j = j0 * (cpy_ne / 2) + (stride_j == warp_আকার ? item_ct1.get_local_id(2) :
                                                                               item_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    ধ্রুবক sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<আকার(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#pragma unroll
                    জন্য (পূর্ণসংখ্যা l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].template convert<দশমিক, sycl::rounding_mode::automatic>();
                    }
                    ggml_sycl_memcpy_1<আকার(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
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
template <পূর্ণসংখ্যা  warp_আকার,
          পূর্ণসংখ্যা  nwarps,
          পূর্ণসংখ্যা  ncols1,
          পূর্ণসংখ্যা  ncols2,
          পূর্ণসংখ্যা  DKQ,
          পূর্ণসংখ্যা  nbatch_fa,
          পূর্ণসংখ্যা  nbatch_K,
          যৌক্তিক use_logit_softcap,
          যৌক্তিক oob_check,
          typename T_vec_dot>
স্থির __dpct_inline__ শূন্য flash_attn_tile_iter_KQ(T_vec_dot * ধ্রুবক Q_tmp,
                                                    ধ্রুবক sycl::half2 * ধ্রুবক __restrict__ K_h2,
                                                    T_vec_dot * ধ্রুবক KV_tmp,
                                                    ধ্রুবক পূর্ণসংখ্যা         stride_K2,
                                                    ধ্রুবক পূর্ণসংখ্যা         k_VKQ_0,
                                                    ধ্রুবক পূর্ণসংখ্যা         k_VKQ_sup,
                                                    ধ্রুবক পূর্ণসংখ্যা         k_KQ_0,
                                                    দশমিক *           KQ_acc) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr পূর্ণসংখ্যা cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    constexpr পূর্ণসংখ্যা ncols = ncols1*ncols2;
    constexpr পূর্ণসংখ্যা cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr পূর্ণসংখ্যা np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_আকার, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_আকার)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#অন্যথা
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) দশমিক K_k[nbatch_fa/(np*warp_আকার)][cpy_ne];
        __dpct_align__(16) দশমিক Q_k[cpw][cpy_ne];
#endif // SYCL_FAST_FP16

#pragma unroll
        জন্য (পূর্ণসংখ্যা i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_আকার) {
            ধ্রুবক পূর্ণসংখ্যা i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_আকার + item_ct1.get_local_id(2);

#ifdef SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_আকার)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#অন্যথা
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_আকার)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // SYCL_FAST_FP16
        }
#pragma unroll
        জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
            ধ্রুবক পূর্ণসংখ্যা jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

#ifdef SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#অন্যথা
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // SYCL_FAST_FP16
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_আকার) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_আকার)*cpw + jc0], K_k[i_KQ_0/(np*warp_আকার)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    যদি (k_KQ_0 + nbatch_K < DKQ) {
        item_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <পূর্ণসংখ্যা  warp_আকার,
          পূর্ণসংখ্যা  nwarps,
          পূর্ণসংখ্যা  ncols1,
          পূর্ণসংখ্যা  ncols2,
          পূর্ণসংখ্যা  DKQ,
          পূর্ণসংখ্যা  DV,
          পূর্ণসংখ্যা  nbatch_fa,
          পূর্ণসংখ্যা  nbatch_K,
          যৌক্তিক use_logit_softcap,
          যৌক্তিক oob_check,
          typename T_vec_dot,
          typename T_KQ,
          typename T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
স্থির __dpct_inline__ শূন্য flash_attn_tile_iter(T_vec_dot * ধ্রুবক Q_tmp,
                                                 ধ্রুবক sycl::half2 * ধ্রুবক __restrict__ K_h2,
                                                 ধ্রুবক sycl::half2 * ধ্রুবক __restrict__ V_h2,
                                                 ধ্রুবক sycl::half * ধ্রুবক __restrict__ mask,
                                                 ধ্রুবক sycl::uint3 ne01,
                                                 ধ্রুবক দশমিক       logit_softcap,
                                                 ধ্রুবক দশমিক       slope,
                                                 T_KQ * ধ্রুবক      KQ,
                                                 T_vec_dot * ধ্রুবক KV_tmp,
                                                 ধ্রুবক পূর্ণসংখ্যা         stride_K2,
                                                 ধ্রুবক পূর্ণসংখ্যা         stride_V2,
                                                 ধ্রুবক পূর্ণসংখ্যা         stride_mask,
                                                 দশমিক * ধ্রুবক     KQ_max,
                                                 দশমিক * ধ্রুবক     KQ_যোগ,
                                                 T_acc * ধ্রুবক     VKQ,
                                                 ধ্রুবক পূর্ণসংখ্যা         k_VKQ_0,
                                                 ধ্রুবক পূর্ণসংখ্যা         k_VKQ_max,
                                                 ধ্রুবক পূর্ণসংখ্যা         col_Q_0,
                                                 দশমিক *           KQ_max_new_shared) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr পূর্ণসংখ্যা cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    constexpr পূর্ণসংখ্যা ncols = ncols1*ncols2;
    constexpr পূর্ণসংখ্যা cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr পূর্ণসংখ্যা np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr পূর্ণসংখ্যা DVp = (DV + 2*warp_আকার - 1) & ~(2*warp_আকার - 1); // DV padded to multiple of 2*warp_size.

#ifdef SYCL_FAST_FP16
    constexpr পূর্ণসংখ্যা KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#অন্যথা
    constexpr পূর্ণসংখ্যা KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    ধ্রুবক পূর্ণসংখ্যা k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    দশমিক KQ_max_new[cpw];
#pragma unroll
    জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    দশমিক KQ_acc[nbatch_fa/(np*warp_আকার) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr পূর্ণসংখ্যা nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_আকার, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    যদি (nbatch_K_last > 0) {
        constexpr পূর্ণসংখ্যা k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_আকার, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
        ধ্রুবক পূর্ণসংখ্যা j = fastmodulo(col_Q_0 + (jc0 + (item_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#pragma unroll
        জন্য (পূর্ণসংখ্যা i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_আকার) {
            ধ্রুবক পূর্ণসংখ্যা i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_আকার + item_ct1.get_local_id(2);

#যদি defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_আকার)*cpw + jc0] *= 4.0f;
#endif // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            যদি (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_আকার)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((দশমিক) KQ_acc[(i_KQ_0 / (np * warp_আকার)) * cpw + jc0]);
            }

            যদি (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_আকার)) * cpw + jc0] +=
                    (ncols2 > 1 || mask) ? slope * sycl::vec<sycl::half, 1>(mask[j * stride_mask + k_VKQ_0 + i_KQ])
                                                       .convert<দশমিক, sycl::rounding_mode::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((দশমিক) KQ_max_new[jc0],
                               (দশমিক) (KQ_acc[(i_KQ_0 / (np * warp_আকার)) * cpw + jc0] + FATTN_KQ_MAX_OFFSET));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_আকার>(KQ_max_new[jc0]);
    }

    যদি constexpr (np == 1) {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } অন্যথা {
        static_assert(cpw == 1, "bad cpw");

        যদি (item_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[item_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(item_ct1.get_local_id(1) & ~(np - 1)) + item_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_আকার)][KQ_cs];
#অন্যথা
        __dpct_align__(16) দশমিক tmp[nbatch_fa/(np*warp_আকার)][KQ_cs];
#endif // SYCL_FAST_FP16

#pragma unroll
        জন্য (পূর্ণসংখ্যা jc1 = 0; jc1 < KQ_cs; ++jc1) {
            ধ্রুবক পূর্ণসংখ্যা jc = jc0 + jc1;

            ধ্রুবক দশমিক KQ_max_scale = sycl::native::exp((দশমিক) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            দশমিক KQ_যোগ_add = 0.0f;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < nbatch_fa; i0 += np*warp_আকার) {
                ধ্রুবক দশমিক val =
                    !oob_check || i0 + (item_ct1.get_local_id(1) % np) * warp_আকার + item_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::exp((দশমিক) (KQ_acc[(i0 / (np * warp_আকার)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_যোগ_add += val;
                tmp[i0/(np*warp_আকার)][jc1] = val;
            }
            KQ_যোগ[jc] = KQ_যোগ[jc]*KQ_max_scale + KQ_যোগ_add;

#ifdef SYCL_FAST_FP16
            ধ্রুবক sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার) {
                VKQ[jc*((DVp/2)/warp_আকার) + i0/warp_আকার].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_আকার) + i0/warp_আকার].y() *= KQ_max_scale_h2.y();
            }
#অন্যথা
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার) {
                VKQ[jc*((DVp/2)/warp_আকার) + i0/warp_আকার].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_আকার) + i0/warp_আকার].y() *= KQ_max_scale;
            }
#endif // SYCL_FAST_FP16
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < nbatch_fa; i0 += np*warp_আকার) {
            ধ্রুবক পূর্ণসংখ্যা i = i0 + (item_ct1.get_local_id(1) % np) * warp_আকার + item_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<আকার(tmp[0])>(
                KQ + (jc0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_আকার)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr পূর্ণসংখ্যা nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    জন্য (পূর্ণসংখ্যা k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_load_tile<warp_আকার, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef SYCL_FAST_FP16
#pragma unroll
        জন্য (পূর্ণসংখ্যা k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_আকার];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_আকার ? cpy_ne/2 : (DVp/2)/warp_আকার;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_আকার],
                                                 &KV_tmp[(k1 + item_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         item_ct1.get_local_id(2) * cpy_ne_D]);
            }
#pragma unroll
            জন্য (পূর্ণসংখ্যা jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ধ্রুবক পূর্ণসংখ্যা jc_KQ = jc_VKQ_0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * আকার(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np) * KQ_cs);
#pragma unroll
                জন্য (পূর্ণসংখ্যা jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_আকার) + i0/warp_আকার].x() +=
                        V_k[i0/warp_আকার].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_আকার) + i0/warp_আকার].y() +=
                        V_k[i0/warp_আকার].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#অন্যথা
#pragma unroll
        জন্য (পূর্ণসংখ্যা k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_আকার];
            __dpct_align__(16) দশমিক  KQ_k[cpw];

            constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < DVp/warp_আকার ? cpy_ne : DVp/warp_আকার;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp; i0 += warp_আকার*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_আকার)], &KV_tmp[(k1 + item_ct1.get_local_id(1) % np)*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
            }
#pragma unroll
            জন্য (পূর্ণসংখ্যা jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ধ্রুবক পূর্ণসংখ্যা jc_KQ = jc_VKQ_0/KQ_cs + (item_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*আকার(দশমিক)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np)*KQ_cs);
            }

#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_আকার) + i0/warp_আকার].x() += V_k[i0/warp_আকার].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_আকার) + i0/warp_আকার].y() += V_k[i0/warp_আকার].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#endif // SYCL_FAST_FP16
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

template <পূর্ণসংখ্যা DKQ, পূর্ণসংখ্যা DV, পূর্ণসংখ্যা ncols1, পূর্ণসংখ্যা ncols2, যৌক্তিক use_logit_softcap, পূর্ণসংখ্যা warp_আকার>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
স্থির শূন্য flash_attn_tile(ধ্রুবক অক্ষর *  Q,
                            ধ্রুবক অক্ষর *  K,
                            ধ্রুবক অক্ষর *  V,
                            ধ্রুবক অক্ষর *  mask,
                            ধ্রুবক অক্ষর *  sinks,
                            ধ্রুবক পূর্ণসংখ্যা *  KV_max,
                            দশমিক *  dst,
                            sycl::float2 *  dst_meta,
                            ধ্রুবক দশমিক          scale,
                            ধ্রুবক দশমিক          max_bias,
                            ধ্রুবক দশমিক          m0,
                            ধ্রুবক দশমিক          m1,
                            ধ্রুবক uint32_t       n_শীর্ষ_log2,
                            ধ্রুবক দশমিক          logit_softcap,
                            ধ্রুবক int32_t        ne00,
                            ধ্রুবক sycl::uint3    ne01,
                            ধ্রুবক int32_t        ne02,
                            ধ্রুবক int32_t        ne03,
                            ধ্রুবক int32_t        nb01,
                            ধ্রুবক int32_t        nb02,
                            ধ্রুবক int32_t        nb03,
                            ধ্রুবক int32_t        ne10,
                            ধ্রুবক int32_t        ne11,
                            ধ্রুবক int32_t        ne12,
                            ধ্রুবক int32_t        ne13,
                            ধ্রুবক int32_t        nb11,
                            ধ্রুবক int32_t        nb12,
                            ধ্রুবক int64_t        nb13,
                            ধ্রুবক int32_t        nb21,
                            ধ্রুবক int32_t        nb22,
                            ধ্রুবক int64_t        nb23,
                            ধ্রুবক int32_t        ne31,
                            ধ্রুবক int32_t        ne32,
                            ধ্রুবক int32_t        ne33,
                            ধ্রুবক int32_t        nb31,
                            ধ্রুবক int32_t        nb32,
                            ধ্রুবক int64_t        nb33) {
#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    যদি ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_শীর্ষ_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ফেরত;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr পূর্ণসংখ্যা ncols     = ncols1*ncols2;

    constexpr পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_আকার;
    constexpr পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr পূর্ণসংখ্যা nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    ধ্রুবক পূর্ণসংখ্যা col_Q_0 = item_ct1.get_group(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    ধ্রুবক পূর্ণসংখ্যা           sequence  = item_ct1.get_group(0) / (ne02 / ncols2);
    ধ্রুবক পূর্ণসংখ্যা           head0     = item_ct1.get_group(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    ধ্রুবক পূর্ণসংখ্যা gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    ধ্রুবক দশমিক * Q_f  = (ধ্রুবক দশমিক *) (Q + nb03*sequence + nb02* head0);
    ধ্রুবক sycl::half2 * K_h2      = (ধ্রুবক sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_ratio));
    ধ্রুবক sycl::half2 * V_h2 =
        (ধ্রুবক sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_ratio));  // K and V have same shape

    ধ্রুবক sycl::half * maskh = mask ? (ধ্রুবক sycl::half *) (mask + nb33 * (sequence % ne33)) : nullptr;

    ধ্রুবক পূর্ণসংখ্যা stride_K2   = nb11 / আকার(sycl::half2);
    ধ্রুবক পূর্ণসংখ্যা stride_V2   = nb21 / আকার(sycl::half2);
    ধ্রুবক পূর্ণসংখ্যা stride_mask = nb31 / আকার(sycl::half);

    ধ্রুবক দশমিক slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_শীর্ষ_log2, m0, m1) : 1.0f;

    constexpr পূর্ণসংখ্যা cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    constexpr পূর্ণসংখ্যা cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr পূর্ণসংখ্যা np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_আকার) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr পূর্ণসংখ্যা DKQp = (DKQ + 2*warp_আকার - 1) & ~(2*warp_আকার - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr পূর্ণসংখ্যা DVp  = (DV  + 2*warp_আকার - 1) & ~(2*warp_আকার - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#ifdef SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_আকার = lsm_size1 * আকার(sycl::half2) +
                                            lsm_size2 * আকার(sycl::half2) +
                                            lsm_size3 * আকার(sycl::half) +
                                            lsm_size4 * আকার(দশমিক);

    syclex::work_group_static<অক্ষর[local_share_mem_আকার]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    দশমিক *KQ_max_new_shared = (দশমিক *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_আকার)] = {
        { 0.0f, 0.0f }
    };
#অন্যথা
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_আকার = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * আকার(দশমিক);

    syclex::work_group_static<অক্ষর[local_share_mem_আকার]> lsm;

    দশমিক *Q_tmp = (দশমিক *)&lsm;
    দশমিক *KV_tmp = Q_tmp +lsm_size1;
    দশমিক *KQ = KV_tmp+lsm_size2;
    দশমিক *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_আকার)] = {{0.0f, 0.0f}};


#endif // SYCL_FAST_FP16

    দশমিক KQ_max[cpw] = {};

#pragma unroll
    জন্য (পূর্ণসংখ্যা j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    দশমিক KQ_যোগ[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
        ধ্রুবক পূর্ণসংখ্যা jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        ধ্রুবক পূর্ণসংখ্যা j = jc / ncols2;
        ধ্রুবক পূর্ণসংখ্যা c = jc % ncols2;

        constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < DKQp/warp_আকার ? cpy_ne : DKQp/warp_আকার;

#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DKQp; i0 += np*warp_আকার*cpy_ne_D) {
            যদি (i0 + np * warp_আকার * cpy_ne_D <= DKQ ||
                i0 + (item_ct1.get_local_id(1) % np) * (warp_আকার * cpy_ne_D) + item_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) দশমিক tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<আকার(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / আকার(দশমিক)) + fastmodulo(col_Q_0 + j, ne01) * (nb01 / আকার(দশমিক)) +
                                i0 + (item_ct1.get_local_id(1) % np) * (warp_আকার * cpy_ne_D) +
                                item_ct1.get_local_id(2) * cpy_ne_D]);

#pragma unroll
                জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#pragma unroll
                জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#যদি defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#endif // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<আকার(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (item_ct1.get_local_id(1) % np) * (warp_আকার * cpy_ne_D / 2) +
                           item_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#অন্যথা
                ggml_sycl_memcpy_1<আকার(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (item_ct1.get_local_id(1) % np)*(warp_আকার*cpy_ne_D)   + item_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#endif // SYCL_FAST_FP16
            }
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    ধ্রুবক পূর্ণসংখ্যা k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    যদি (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        পূর্ণসংখ্যা k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa;
        যখন (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr যৌক্তিক oob_check = মিথ্যা;
            flash_attn_tile_iter<warp_আকার, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_যোগ, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa;
        }
        যদি (k_VKQ_0 < k_VKQ_max) {
            constexpr যৌক্তিক oob_check = সত্য;
            flash_attn_tile_iter<warp_আকার, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_যোগ, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    } অন্যথা {
        // Branch without out-of-bounds checks.
        জন্য (পূর্ণসংখ্যা k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa) {

            constexpr যৌক্তিক oob_check = মিথ্যা;
            flash_attn_tile_iter<warp_আকার, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_যোগ, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_যোগ[jc0] = warp_reduce_যোগ<warp_আকার>(KQ_যোগ[jc0]);
    }

    যদি constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#অন্যথা
        দশমিক * VKQ_combine    = (দশমিক *) KV_tmp;
#endif // SYCL_FAST_FP16

        দশমিক * KQ_যোগ_combine = (দশমিক *) Q_tmp;

        যদি (item_ct1.get_local_id(1) % np != 0) {

#ifdef SYCL_FAST_FP16
            constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < (DVp/2)/warp_আকার ? cpy_ne : (DVp/2)/warp_আকার;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[item_ct1.get_local_id(1) * (DVp / 2) + i0 + item_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_আকার]);
            }
#অন্যথা

            constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < DVp/warp_আকার ? cpy_ne : DVp/warp_আকার;

#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp; i0 += warp_আকার*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[item_ct1.get_local_id(1)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D], ((ধ্রুবক দশমিক *) VKQ) + i0/warp_আকার);
            }
#endif // SYCL_FAST_FP16

            যদি (item_ct1.get_local_id(2) == 0) {
                KQ_যোগ_combine[item_ct1.get_local_id(1)] = KQ_যোগ[0];
            }
            ফেরত;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        জন্য (পূর্ণসংখ্যা ip = 1; ip < np; ++ip) {
#ifdef SYCL_FAST_FP16
            constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < (DVp/2)/warp_আকার ? cpy_ne : (DVp/2)/warp_আকার;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   item_ct1.get_local_id(2) * cpy_ne_D]);
#pragma unroll
                জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_আকার + i1] += tmp[i1];
                }
            }
#অন্যথা
            constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < DVp/warp_আকার ? cpy_ne : DVp/warp_আকার;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp; i0 += warp_আকার*cpy_ne_D) {
                __dpct_align__(16) দশমিক tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
#pragma unroll
                জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((দশমিক *)VKQ)[i0/warp_আকার + i1] += tmp[i1];
                }
            }
#endif // SYCL_FAST_FP16

            KQ_যোগ[0] += KQ_যোগ_combine[item_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    যদি (sinks && item_ct1.get_group(1) == 0) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
            ধ্রুবক পূর্ণসংখ্যা   jc   = jc0 + (item_ct1.get_local_id(1) / np) * cpw;
            ধ্রুবক দশমিক sink = ((ধ্রুবক দশমিক *) sinks)[head0 + jc % ncols2];

            দশমিক       KQ_max_new_j = sycl::fmax((দশমিক) KQ_max[jc0], sink);
            ধ্রুবক দশমিক KQ_max_scale = sycl::native::exp((দশমিক) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            ধ্রুবক দশমিক val = sycl::native::exp((দশমিক) (sink - KQ_max[jc0]));
            KQ_যোগ[jc0] = KQ_যোগ[jc0]*KQ_max_scale + val;

#ifdef SYCL_FAST_FP16
            ধ্রুবক sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার) {
                VKQ[jc0*((DVp/2)/warp_আকার) + i0/warp_আকার] *= KQ_max_scale_h2;
            }
#অন্যথা
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার) {
                VKQ[jc0*((DVp/2)/warp_আকার) + i0/warp_আকার].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_আকার) + i0/warp_আকার].y() *= KQ_max_scale;
            }
#endif // SYCL_FAST_FP16
        }
    }

    // Write back results:
#pragma unroll
    জন্য (পূর্ণসংখ্যা jc0 = 0; jc0 < cpw; ++jc0) {
        ধ্রুবক পূর্ণসংখ্যা jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        ধ্রুবক পূর্ণসংখ্যা j = jc / ncols2;
        ধ্রুবক পূর্ণসংখ্যা c = jc % ncols2;

        যদি (ncols1 > 1 && col_Q_0 + j >= পূর্ণসংখ্যা(ne01.z())) {
            ফেরত;
        }

        ধ্রুবক দশমিক scale = item_ct1.get_group_range(1) == 1 ? 1.0f / KQ_যোগ[jc0] : 1.0f;

        ধ্রুবক পূর্ণসংখ্যা j_dst_unrolled =
            ((sequence * পূর্ণসংখ্যা(ne01.z()) + col_Q_0 + j) * ne02 + head0 + c) * item_ct1.get_group_range(1) +
            item_ct1.get_group(1);

#ifdef SYCL_FAST_FP16
        constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_আকার ? cpy_ne/2 : (DVp/2)/warp_আকার;
#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp/2; i0 += warp_আকার*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#pragma unroll
            জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_আকার) + i0 / warp_আকার + i1]
                              .template convert<দশমিক, sycl::rounding_mode::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            যদি (i0 + warp_আকার * cpy_ne_D <= DV / 2 || i0 + item_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<আকার(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + item_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#অন্যথা
        constexpr পূর্ণসংখ্যা cpy_ne_D = cpy_ne < DVp/warp_আকার ? cpy_ne : DVp/warp_আকার;
#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DVp; i0 += warp_আকার*cpy_ne_D) {
            যদি (i0 + warp_আকার*cpy_ne_D <= DV || i0 + item_ct1.get_local_id(2)*cpy_ne_D < DV) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_আকার) + i0/(2*warp_আকার) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_আকার) + i0/(2*warp_আকার) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_আকার) + i0/(2*warp_আকার)]);
            }
        }
#endif // SYCL_FAST_FP16

        যদি (item_ct1.get_group_range(1) != 1 && item_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_যোগ[jc0]);
        }
    }
#অন্যথা
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_শীর্ষ_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#endif // SYCL_FLASH_ATTN
}

template <পূর্ণসংখ্যা DKQ, পূর্ণসংখ্যা DV, পূর্ণসংখ্যা ncols2, যৌক্তিক use_logit_softcap>
স্থির শূন্য launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * Q = dst->src[0];

    ধ্রুবক পূর্ণসংখ্যা id        = ggml_sycl_get_device();
    ধ্রুবক পূর্ণসংখ্যা cc        = ggml_sycl_info().devices[id].cc;
    ধ্রুবক পূর্ণসংখ্যা warp_আকার = WARP_32_আকার; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    যদি (DV < 512 && Q->ne[1] < 32) {
        যদি constexpr (ncols2 <= 32) {
            যদি (Q->ne[1] > 16/ncols2) {
                constexpr পূর্ণসংখ্যা cols_per_block = 32;
                ধ্রুবক পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_আকার;
                ধ্রুবক পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_আকার>, warp_আকার>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, সত্য, সত্য, মিথ্যা);
                ফেরত;
            }
        }
        যদি constexpr (ncols2 <= 16) {
            যদি (Q->ne[1] > 8/ncols2) {
                constexpr পূর্ণসংখ্যা cols_per_block = 16;
                ধ্রুবক পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_আকার;
                ধ্রুবক পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_আকার>, warp_আকার>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, সত্য, সত্য, মিথ্যা);
                ফেরত;
            }
        }
        যদি constexpr (ncols2 <= 8) {
            যদি (Q->ne[1] > 4/ncols2) {
                constexpr পূর্ণসংখ্যা cols_per_block = 8;
                ধ্রুবক পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_আকার;
                ধ্রুবক পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_আকার>, warp_আকার>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, সত্য, সত্য, মিথ্যা);
                ফেরত;
            }
        }
    }

    যদি constexpr (ncols2 <= 4) {
        যদি (Q->ne[1] > 2/ncols2) {
            constexpr পূর্ণসংখ্যা cols_per_block = 4;
            ধ্রুবক পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_আকার;
            ধ্রুবক পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            launch_fattn<DV, cols_per_block/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_আকার>, warp_আকার>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, সত্য, সত্য, মিথ্যা);
            ফেরত;
        }
    }

    যদি constexpr (ncols2 <= 2) {
        constexpr পূর্ণসংখ্যা cols_per_block = 2;
        ধ্রুবক পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_আকার;
        ধ্রুবক পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_আকার>, warp_আকার>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, সত্য, সত্য, মিথ্যা);
        ফেরত;
    }

    {
        constexpr পূর্ণসংখ্যা cols_per_block = ncols2*2;
        ধ্রুবক পূর্ণসংখ্যা nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_আকার;
        ধ্রুবক পূর্ণসংখ্যা nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_আকার>, warp_আকার>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, সত্য, সত্য, মিথ্যা);
        ফেরত;
    }

    GGML_ABORT("fatal error");
}

template <পূর্ণসংখ্যা DKQ, পূর্ণসংখ্যা DV, যৌক্তিক use_logit_softcap>
স্থির শূন্য launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * KQV  = dst;
    ধ্রুবক ggml_tensor * Q    = dst->src[0];
    ধ্রুবক ggml_tensor * K    = dst->src[1];
    ধ্রুবক ggml_tensor * mask = dst->src[3];

    দশমিক max_bias = 0.0f;
    memcpy(&max_bias, (ধ্রুবক দশমিক *) KQV->op_params + 1, আকার(দশমিক));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    ধ্রুবক পূর্ণসংখ্যা gqa_ratio = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    ধ্রুবক পূর্ণসংখ্যা gqa_limit = gqa_ratio <= 4 && DV <= 256 ? 16 : INT_MAX;
    ধ্রুবক যৌক্তিক use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    যদি constexpr (DV == 512) {
        যদি (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            ফেরত;
        }
        যদি (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            ফেরত;
        }
        যদি (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            ফেরত;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        যদি constexpr (DKQ == DV) {
            যদি (use_gqa_opt && gqa_ratio % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                ফেরত;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            ফেরত;
        }
    }

    যদি constexpr (DV <= 256) {
        যদি (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            ফেরত;
        }

        যদি (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            ফেরত;
        }

        যদি (use_gqa_opt && gqa_ratio % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            ফেরত;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        ফেরত;
    }
    GGML_ABORT("fatal error");
}

template <পূর্ণসংখ্যা DKQ, পূর্ণসংখ্যা DV>
শূন্য ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * KQV = dst;

    দশমিক logit_softcap;
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *) KQV->op_params + 2, আকার(দশমিক));

    যদি (logit_softcap == 0.0f) {
        constexpr যৌক্তিক use_logit_softcap = মিথ্যা;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } অন্যথা {
        constexpr যৌক্তিক use_logit_softcap = সত্য;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

শূন্য ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    template শূন্য ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

বাহ্যিক DECL_FATTN_TILE_CASE( 40,  40);
বাহ্যিক DECL_FATTN_TILE_CASE( 64,  64);
বাহ্যিক DECL_FATTN_TILE_CASE( 72,  72);
বাহ্যিক DECL_FATTN_TILE_CASE( 80,  80);
বাহ্যিক DECL_FATTN_TILE_CASE( 96,  96);
বাহ্যিক DECL_FATTN_TILE_CASE(112, 112);
বাহ্যিক DECL_FATTN_TILE_CASE(128, 128);
বাহ্যিক DECL_FATTN_TILE_CASE(256, 256);
বাহ্যিক DECL_FATTN_TILE_CASE(512, 512);
বাহ্যিক DECL_FATTN_TILE_CASE(576, 512);

