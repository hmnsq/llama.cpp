#ضمن <sycl/sycl.hpp>
#ضمن <sycl/ext/oneapi/work_group_static.hpp>
#ضمن "dpct/helper.hpp"
#ضمن "common.hpp"
#ضمن "fattn-common.hpp"

#ضمن <cmath>
#ضمن <float.h>

نطاق syclex = sycl::ext::oneapi::experimental;

#define GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    𐤀𐤌 (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
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
    𐤀𐤌(fast_fp16_available(cc))
        أرجع ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    وإلا
        أرجع ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

ساكن constexpr uint32_t ggml_sycl_fattn_tile_get_config(ثابت صحيح DKQ, ثابت صحيح DV, ثابت صحيح ncols) {
#ifdef SYCL_FAST_FP16
    أرجع ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#وإلا
    أرجع ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#endif // SYCL_FAST_FP16
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

template <صحيح warp_size, صحيح nwarps, صحيح I, صحيح J, صحيح J_padding, منطقي oob_check>
ساكن __dpct_inline__ فراغ flash_attn_tile_load_tile(ثابت sycl::half2 * ثابت __restrict__ KV,
                                                      sycl::half2 * ثابت __restrict__ tile_KV,
                                                      ثابت صحيح stride_KV,
                                                      ثابت صحيح i_sup) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    auto load = [&] (ثابت صحيح n) {
        ثابت صحيح stride_j = warp_size >> n;

        𐤀𐤌 (stride_j == 0) {
            أرجع;
        }

        ثابت صحيح j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        ثابت صحيح j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        ثابت صحيح stride_i = warp_size / stride_j;

        𐤀𐤌 (j0_start == j0_stop) {
            أرجع;
        }

#pragma unroll
        لكل (صحيح i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ثابت صحيح i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_size ? 0 : item_ct1.get_local_id(2) / stride_j);

            𐤀𐤌 (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                لكل (صحيح j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    ثابت صحيح j = j0 * cpy_ne + (stride_j == warp_size ? item_ct1.get_local_id(2) :
                                                                         item_ct1.get_local_id(2) % stride_j) *
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

template <صحيح warp_size, صحيح nwarps, صحيح I, صحيح J, صحيح J_padding, منطقي oob_check>
ساكن __dpct_inline__ فراغ flash_attn_tile_load_tile(ثابت sycl::half2 * ثابت __restrict__ KV,
                                                      عائم * ثابت __restrict__ tile_KV,
                                                      ثابت صحيح stride_KV,
                                                      ثابت صحيح i_sup) {
    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    auto load = [&] (ثابت صحيح n) {
        auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        ثابت صحيح stride_j = warp_size >> n;

        𐤀𐤌 (stride_j == 0) {
            أرجع;
        }

        ثابت صحيح j0_start = stride_j == warp_size ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        ثابت صحيح j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        ثابت صحيح stride_i = warp_size / stride_j;

        𐤀𐤌 (j0_start == j0_stop) {
            أرجع;
        }

#pragma unroll
        لكل (صحيح i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            ثابت صحيح i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_size ? 0 : item_ct1.get_local_id(2) / stride_j);

            𐤀𐤌 (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                لكل (صحيح j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    ثابت صحيح j = j0 * (cpy_ne / 2) + (stride_j == warp_size ? item_ct1.get_local_id(2) :
                                                                               item_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    ثابت sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<حجم(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#pragma unroll
                    لكل (صحيح l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].template convert<عائم, sycl::rounding_mode::automatic>();
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
template <صحيح  warp_size,
          صحيح  nwarps,
          صحيح  ncols1,
          صحيح  ncols2,
          صحيح  DKQ,
          صحيح  nbatch_fa,
          صحيح  nbatch_K,
          منطقي use_logit_softcap,
          منطقي oob_check,
          typename T_vec_dot>
ساكن __dpct_inline__ فراغ flash_attn_tile_iter_KQ(T_vec_dot * ثابت Q_tmp,
                                                    ثابت sycl::half2 * ثابت __restrict__ K_h2,
                                                    T_vec_dot * ثابت KV_tmp,
                                                    ثابت صحيح         stride_K2,
                                                    ثابت صحيح         k_VKQ_0,
                                                    ثابت صحيح         k_VKQ_sup,
                                                    ثابت صحيح         k_KQ_0,
                                                    عائم *           KQ_acc) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr صحيح cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    constexpr صحيح ncols = ncols1*ncols2;
    constexpr صحيح cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr صحيح np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    لكل (صحيح k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_size)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#وإلا
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    لكل (صحيح k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) عائم K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __dpct_align__(16) عائم Q_k[cpw][cpy_ne];
#endif // SYCL_FAST_FP16

#pragma unroll
        لكل (صحيح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            ثابت صحيح i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2);

#ifdef SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#وإلا
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // SYCL_FAST_FP16
        }
#pragma unroll
        لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
            ثابت صحيح jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

#ifdef SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#وإلا
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // SYCL_FAST_FP16
        }

#pragma unroll
        لكل (صحيح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                لكل (صحيح k = 0; k < cpy_ne; ++k) {
                    ggml_sycl_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    𐤀𐤌 (k_KQ_0 + nbatch_K < DKQ) {
        item_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <صحيح  warp_size,
          صحيح  nwarps,
          صحيح  ncols1,
          صحيح  ncols2,
          صحيح  DKQ,
          صحيح  DV,
          صحيح  nbatch_fa,
          صحيح  nbatch_K,
          منطقي use_logit_softcap,
          منطقي oob_check,
          typename T_vec_dot,
          typename T_KQ,
          typename T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ساكن __dpct_inline__ فراغ flash_attn_tile_iter(T_vec_dot * ثابت Q_tmp,
                                                 ثابت sycl::half2 * ثابت __restrict__ K_h2,
                                                 ثابت sycl::half2 * ثابت __restrict__ V_h2,
                                                 ثابت sycl::half * ثابت __restrict__ mask,
                                                 ثابت sycl::uint3 ne01,
                                                 ثابت عائم       logit_softcap,
                                                 ثابت عائم       slope,
                                                 T_KQ * ثابت      KQ,
                                                 T_vec_dot * ثابت KV_tmp,
                                                 ثابت صحيح         stride_K2,
                                                 ثابت صحيح         stride_V2,
                                                 ثابت صحيح         stride_mask,
                                                 عائم * ثابت     KQ_max,
                                                 عائم * ثابت     KQ_sum,
                                                 T_acc * ثابت     VKQ,
                                                 ثابت صحيح         k_VKQ_0,
                                                 ثابت صحيح         k_VKQ_max,
                                                 ثابت صحيح         col_Q_0,
                                                 عائم *           KQ_max_new_shared) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr صحيح cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    constexpr صحيح ncols = ncols1*ncols2;
    constexpr صحيح cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr صحيح np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr صحيح DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

#ifdef SYCL_FAST_FP16
    constexpr صحيح KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#وإلا
    constexpr صحيح KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    ثابت صحيح k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    عائم KQ_max_new[cpw];
#pragma unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    عائم KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr صحيح nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    𐤀𐤌 (nbatch_K_last > 0) {
        constexpr صحيح k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحيح j = fastmodulo(col_Q_0 + (jc0 + (item_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#pragma unroll
        لكل (صحيح i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            ثابت صحيح i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2);

#𐤀𐤌 defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            𐤀𐤌 (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((عائم) KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0]);
            }

            𐤀𐤌 (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0] +=
                    (ncols2 > 1 || mask) ? slope * sycl::vec<sycl::half, 1>(mask[j * stride_mask + k_VKQ_0 + i_KQ])
                                                       .convert<عائم, sycl::rounding_mode::automatic>()[0] :
                                           0.0f;

                KQ_max_new[jc0] =
                    sycl::fmax((عائم) KQ_max_new[jc0],
                               (عائم) (KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0] + FATTN_KQ_MAX_OFFSET));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
    }

    𐤀𐤌 constexpr (np == 1) {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } وإلا {
        static_assert(cpw == 1, "bad cpw");

        𐤀𐤌 (item_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[item_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(item_ct1.get_local_id(1) & ~(np - 1)) + item_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_size)][KQ_cs];
#وإلا
        __dpct_align__(16) عائم tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#endif // SYCL_FAST_FP16

#pragma unroll
        لكل (صحيح jc1 = 0; jc1 < KQ_cs; ++jc1) {
            ثابت صحيح jc = jc0 + jc1;

            ثابت عائم KQ_max_scale = sycl::native::exp((عائم) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            عائم KQ_sum_add = 0.0f;
#pragma unroll
            لكل (صحيح i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                ثابت عائم val =
                    !oob_check || i0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        sycl::native::exp((عائم) (KQ_acc[(i0 / (np * warp_size)) * cpw + jc] - KQ_max[jc])) :
                        0.0f;
                KQ_sum_add += val;
                tmp[i0/(np*warp_size)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

#ifdef SYCL_FAST_FP16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y() *= KQ_max_scale_h2.y();
            }
#وإلا
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y() *= KQ_max_scale;
            }
#endif // SYCL_FAST_FP16
        }

#pragma unroll
        لكل (صحيح i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
            ثابت صحيح i = i0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<حجم(tmp[0])>(
                KQ + (jc0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_size)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr صحيح nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    لكل (صحيح k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef SYCL_FAST_FP16
#pragma unroll
        لكل (صحيح k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_size];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr صحيح cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_size],
                                                 &KV_tmp[(k1 + item_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         item_ct1.get_local_id(2) * cpy_ne_D]);
            }
#pragma unroll
            لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ثابت صحيح jc_KQ = jc_VKQ_0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * حجم(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np) * KQ_cs);
#pragma unroll
                لكل (صحيح jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x() +=
                        V_k[i0/warp_size].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y() +=
                        V_k[i0/warp_size].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#وإلا
#pragma unroll
        لكل (صحيح k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_size];
            __dpct_align__(16) عائم  KQ_k[cpw];

            constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + item_ct1.get_local_id(1) % np)*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
            }
#pragma unroll
            لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                ثابت صحيح jc_KQ = jc_VKQ_0/KQ_cs + (item_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*حجم(عائم)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np)*KQ_cs);
            }

#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                لكل (صحيح jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x() += V_k[i0/warp_size].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y() += V_k[i0/warp_size].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#endif // SYCL_FAST_FP16
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

template <صحيح DKQ, صحيح DV, صحيح ncols1, صحيح ncols2, منطقي use_logit_softcap, صحيح warp_size>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
ساكن فراغ flash_attn_tile(ثابت حرف *  Q,
                            ثابت حرف *  K,
                            ثابت حرف *  V,
                            ثابت حرف *  mask,
                            ثابت حرف *  sinks,
                            ثابت صحيح *  KV_max,
                            عائم *  dst,
                            sycl::float2 *  dst_meta,
                            ثابت عائم          scale,
                            ثابت عائم          max_bias,
                            ثابت عائم          m0,
                            ثابت عائم          m1,
                            ثابت uint32_t       n_head_log2,
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
#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    𐤀𐤌 ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
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

    constexpr صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr صحيح nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    ثابت صحيح col_Q_0 = item_ct1.get_group(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    ثابت صحيح           sequence  = item_ct1.get_group(0) / (ne02 / ncols2);
    ثابت صحيح           head0     = item_ct1.get_group(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    ثابت صحيح gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    ثابت عائم * Q_f  = (ثابت عائم *) (Q + nb03*sequence + nb02* head0);
    ثابت sycl::half2 * K_h2      = (ثابت sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_ratio));
    ثابت sycl::half2 * V_h2 =
        (ثابت sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_ratio));  // K and V have same shape

    ثابت sycl::half * maskh = mask ? (ثابت sycl::half *) (mask + nb33 * (sequence % ne33)) : nullptr;

    ثابت صحيح stride_K2   = nb11 / حجم(sycl::half2);
    ثابت صحيح stride_V2   = nb21 / حجم(sycl::half2);
    ثابت صحيح stride_mask = nb31 / حجم(sycl::half);

    ثابت عائم slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    constexpr صحيح cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr صحيح np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr صحيح DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr صحيح DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

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

    constexpr size_t local_share_mem_size = lsm_size1 * حجم(sycl::half2) +
                                            lsm_size2 * حجم(sycl::half2) +
                                            lsm_size3 * حجم(sycl::half) +
                                            lsm_size4 * حجم(عائم);

    syclex::work_group_static<حرف[local_share_mem_size]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    عائم *KQ_max_new_shared = (عائم *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_size)] = {
        { 0.0f, 0.0f }
    };
#وإلا
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * حجم(عائم);

    syclex::work_group_static<حرف[local_share_mem_size]> lsm;

    عائم *Q_tmp = (عائم *)&lsm;
    عائم *KV_tmp = Q_tmp +lsm_size1;
    عائم *KQ = KV_tmp+lsm_size2;
    عائم *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};


#endif // SYCL_FAST_FP16

    عائم KQ_max[cpw] = {};

#pragma unroll
    لكل (صحيح j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    عائم KQ_sum[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحيح jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        ثابت صحيح j = jc / ncols2;
        ثابت صحيح c = jc % ncols2;

        constexpr صحيح cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        لكل (صحيح i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            𐤀𐤌 (i0 + np * warp_size * cpy_ne_D <= DKQ ||
                i0 + (item_ct1.get_local_id(1) % np) * (warp_size * cpy_ne_D) + item_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) عائم tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<حجم(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / حجم(عائم)) + fastmodulo(col_Q_0 + j, ne01) * (nb01 / حجم(عائم)) +
                                i0 + (item_ct1.get_local_id(1) % np) * (warp_size * cpy_ne_D) +
                                item_ct1.get_local_id(2) * cpy_ne_D]);

#pragma unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#pragma unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#𐤀𐤌 defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#endif // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<حجم(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (item_ct1.get_local_id(1) % np) * (warp_size * cpy_ne_D / 2) +
                           item_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#وإلا
                ggml_sycl_memcpy_1<حجم(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (item_ct1.get_local_id(1) % np)*(warp_size*cpy_ne_D)   + item_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#endif // SYCL_FAST_FP16
            }
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    ثابت صحيح k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    𐤀𐤌 (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        صحيح k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa;
        طالما (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr منطقي oob_check = خطأ;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa;
        }
        𐤀𐤌 (k_VKQ_0 < k_VKQ_max) {
            constexpr منطقي oob_check = true;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    } وإلا {
        // Branch without out-of-bounds checks.
        لكل (صحيح k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa) {

            constexpr منطقي oob_check = خطأ;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    }

#pragma unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    𐤀𐤌 constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#وإلا
        عائم * VKQ_combine    = (عائم *) KV_tmp;
#endif // SYCL_FAST_FP16

        عائم * KQ_sum_combine = (عائم *) Q_tmp;

        𐤀𐤌 (item_ct1.get_local_id(1) % np != 0) {

#ifdef SYCL_FAST_FP16
            constexpr صحيح cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[item_ct1.get_local_id(1) * (DVp / 2) + i0 + item_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_size]);
            }
#وإلا

            constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;

#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[item_ct1.get_local_id(1)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D], ((ثابت عائم *) VKQ) + i0/warp_size);
            }
#endif // SYCL_FAST_FP16

            𐤀𐤌 (item_ct1.get_local_id(2) == 0) {
                KQ_sum_combine[item_ct1.get_local_id(1)] = KQ_sum[0];
            }
            أرجع;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        لكل (صحيح ip = 1; ip < np; ++ip) {
#ifdef SYCL_FAST_FP16
            constexpr صحيح cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   item_ct1.get_local_id(2) * cpy_ne_D]);
#pragma unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }
#وإلا
            constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                __dpct_align__(16) عائم tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
#pragma unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((عائم *)VKQ)[i0/warp_size + i1] += tmp[i1];
                }
            }
#endif // SYCL_FAST_FP16

            KQ_sum[0] += KQ_sum_combine[item_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    𐤀𐤌 (sinks && item_ct1.get_group(1) == 0) {
#pragma unroll
        لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
            ثابت صحيح   jc   = jc0 + (item_ct1.get_local_id(1) / np) * cpw;
            ثابت عائم sink = ((ثابت عائم *) sinks)[head0 + jc % ncols2];

            عائم       KQ_max_new_j = sycl::fmax((عائم) KQ_max[jc0], sink);
            ثابت عائم KQ_max_scale = sycl::native::exp((عائم) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            ثابت عائم val = sycl::native::exp((عائم) (sink - KQ_max[jc0]));
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef SYCL_FAST_FP16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#وإلا
#pragma unroll
            لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y() *= KQ_max_scale;
            }
#endif // SYCL_FAST_FP16
        }
    }

    // Write back results:
#pragma unroll
    لكل (صحيح jc0 = 0; jc0 < cpw; ++jc0) {
        ثابت صحيح jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        ثابت صحيح j = jc / ncols2;
        ثابت صحيح c = jc % ncols2;

        𐤀𐤌 (ncols1 > 1 && col_Q_0 + j >= صحيح(ne01.z())) {
            أرجع;
        }

        ثابت عائم scale = item_ct1.get_group_range(1) == 1 ? 1.0f / KQ_sum[jc0] : 1.0f;

        ثابت صحيح j_dst_unrolled =
            ((sequence * صحيح(ne01.z()) + col_Q_0 + j) * ne02 + head0 + c) * item_ct1.get_group_range(1) +
            item_ct1.get_group(1);

#ifdef SYCL_FAST_FP16
        constexpr صحيح cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        لكل (صحيح i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#pragma unroll
            لكل (صحيح i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size + i1]
                              .template convert<عائم, sycl::rounding_mode::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            𐤀𐤌 (i0 + warp_size * cpy_ne_D <= DV / 2 || i0 + item_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<حجم(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + item_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#وإلا
        constexpr صحيح cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        لكل (صحيح i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            𐤀𐤌 (i0 + warp_size*cpy_ne_D <= DV || i0 + item_ct1.get_local_id(2)*cpy_ne_D < DV) {
#pragma unroll
                لكل (صحيح i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // SYCL_FAST_FP16

        𐤀𐤌 (item_ct1.get_group_range(1) != 1 && item_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#وإلا
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#endif // SYCL_FLASH_ATTN
}

template <صحيح DKQ, صحيح DV, صحيح ncols2, منطقي use_logit_softcap>
ساكن فراغ launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * Q = dst->src[0];

    ثابت صحيح id        = ggml_sycl_get_device();
    ثابت صحيح cc        = ggml_sycl_info().devices[id].cc;
    ثابت صحيح warp_size = WARP_32_SIZE; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    𐤀𐤌 (DV < 512 && Q->ne[1] < 32) {
        𐤀𐤌 constexpr (ncols2 <= 32) {
            𐤀𐤌 (Q->ne[1] > 16/ncols2) {
                constexpr صحيح cols_per_block = 32;
                ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
                ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, خطأ);
                أرجع;
            }
        }
        𐤀𐤌 constexpr (ncols2 <= 16) {
            𐤀𐤌 (Q->ne[1] > 8/ncols2) {
                constexpr صحيح cols_per_block = 16;
                ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
                ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, خطأ);
                أرجع;
            }
        }
        𐤀𐤌 constexpr (ncols2 <= 8) {
            𐤀𐤌 (Q->ne[1] > 4/ncols2) {
                constexpr صحيح cols_per_block = 8;
                ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
                ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, خطأ);
                أرجع;
            }
        }
    }

    𐤀𐤌 constexpr (ncols2 <= 4) {
        𐤀𐤌 (Q->ne[1] > 2/ncols2) {
            constexpr صحيح cols_per_block = 4;
            ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            launch_fattn<DV, cols_per_block/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, خطأ);
            أرجع;
        }
    }

    𐤀𐤌 constexpr (ncols2 <= 2) {
        constexpr صحيح cols_per_block = 2;
        ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
        ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, خطأ);
        أرجع;
    }

    {
        constexpr صحيح cols_per_block = ncols2*2;
        ثابت صحيح nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
        ثابت صحيح nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, خطأ);
        أرجع;
    }

    GGML_ABORT("fatal error");
}

template <صحيح DKQ, صحيح DV, منطقي use_logit_softcap>
ساكن فراغ launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV  = dst;
    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * mask = dst->src[3];

    عائم max_bias = 0.0f;
    memcpy(&max_bias, (ثابت عائم *) KQV->op_params + 1, حجم(عائم));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    ثابت صحيح gqa_ratio = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    ثابت صحيح gqa_limit = gqa_ratio <= 4 && DV <= 256 ? 16 : INT_MAX;
    ثابت منطقي use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    𐤀𐤌 constexpr (DV == 512) {
        𐤀𐤌 (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            أرجع;
        }
        𐤀𐤌 (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            أرجع;
        }
        𐤀𐤌 (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            أرجع;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        𐤀𐤌 constexpr (DKQ == DV) {
            𐤀𐤌 (use_gqa_opt && gqa_ratio % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                أرجع;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            أرجع;
        }
    }

    𐤀𐤌 constexpr (DV <= 256) {
        𐤀𐤌 (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            أرجع;
        }

        𐤀𐤌 (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            أرجع;
        }

        𐤀𐤌 (use_gqa_opt && gqa_ratio % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            أرجع;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        أرجع;
    }
    GGML_ABORT("fatal error");
}

template <صحيح DKQ, صحيح DV>
فراغ ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV = dst;

    عائم logit_softcap;
    memcpy(&logit_softcap, (ثابت عائم *) KQV->op_params + 2, حجم(عائم));

    𐤀𐤌 (logit_softcap == 0.0f) {
        constexpr منطقي use_logit_softcap = خطأ;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } وإلا {
        constexpr منطقي use_logit_softcap = true;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

فراغ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    template فراغ ggml_sycl_flash_attn_ext_tile_case              \
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

