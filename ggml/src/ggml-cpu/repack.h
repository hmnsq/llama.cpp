#ਹਦਾਇਤ once

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

// GGML internal header

ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_repack_ਬਫਰ_type(ਖਾਲੀ);

ਸਾਂਚਾ <ਪੂਰਨ K> constexpr ਪੂਰਨ QK_0() {
    ਜੇ constexpr (K == 1) {
        ਵਾਪਸ QK1_0;
    }
    ਜੇ constexpr (K == 4) {
        ਵਾਪਸ QK4_0;
    }
    ਜੇ constexpr (K == 8) {
        ਵਾਪਸ QK8_0;
    }
    ਵਾਪਸ -1;
}

ਸਾਂਚਾ <ਪੂਰਨ K, ਪੂਰਨ N> ਬਣਤਰ ਖੰਡ {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(ਆਕਾਰ(ਖੰਡ<1, 4>) == 4 * ਆਕਾਰ(ggml_half) + QK1_0 / 2, "wrong block<1,4> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ<4, 4>) == 4 * ਆਕਾਰ(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ<4, 8>) == 8 * ਆਕਾਰ(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ<4, 16>) == 16 * ਆਕਾਰ(ggml_half) + QK8_0 * 8, "wrong block<4,16> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ<8, 4>) == 4 * ਆਕਾਰ(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ<8, 8>) == 8 * ਆਕਾਰ(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ<8, 16>) == 16 * ਆਕਾਰ(ggml_half) + QK8_0 * 16, "wrong block<8,16> size/padding");

using ਖੰਡ_q1_0x4 = ਖੰਡ<1, 4>;
using ਖੰਡ_q4_0x4 = ਖੰਡ<4, 4>;
using ਖੰਡ_q4_0x8 = ਖੰਡ<4, 8>;
using ਖੰਡ_q4_0x16 = ਖੰਡ<4, 16>;
using ਖੰਡ_q8_0x4 = ਖੰਡ<8, 4>;
using ਖੰਡ_q8_0x8 = ਖੰਡ<8, 8>;
using ਖੰਡ_q8_0x16 = ਖੰਡ<8, 16>;

ਬਣਤਰ ਖੰਡ_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static_assert(ਆਕਾਰ(ਖੰਡ_q4_Kx8) == ਆਕਾਰ(ggml_half) * 16 + K_SCALE_ਮਾਪ * 8 + QK_K * 4, "wrong q4_K block size/padding");
ਬਣਤਰ ਖੰਡ_q4_Kx16 {
    ggml_half d[16];      // super-block scale for quantized scales
    ggml_half dmin[16];   // super-block scale for quantized mins
    uint8_t scales[192];  // scales and mins, quantized with 6 bits
    uint8_t qs[2048];    // 4--bit quants
};

static_assert(ਆਕਾਰ(ਖੰਡ_q4_Kx16) == ਆਕਾਰ(ggml_half) * 32 + K_SCALE_ਮਾਪ * 16 + QK_K * 8, "wrong q4_K block size/padding");
ਬਣਤਰ ਖੰਡ_q2_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[128];  // scales and mins, quantized with 4 bits
    uint8_t qs[512];    // 2--bit quants
};

static_assert(ਆਕਾਰ(ਖੰਡ_q2_Kx8) == ਆਕਾਰ(ggml_half) * 16 + QK_K/2 + QK_K * 2, "wrong q2_K block size/padding");
ਬਣਤਰ ਖੰਡ_q2_Kx16 {
    ggml_half d[16];       // Super-block scale for quantized scales
    ggml_half dmin[16];    // Super-block scale for quantized mins
    uint8_t   scales[256]; // Sub-block scales (16 cols * 16 sub-blocks)
    uint8_t   qs[1024];    // Data (16 cols * 64 bytes per block)
};
static_assert(ਆਕਾਰ(ਖੰਡ_q2_Kx16) == ਆਕਾਰ(ggml_half) * 32 + QK_K + QK_K * 4, "wrong q2_K block size/padding");

ਬਣਤਰ ਖੰਡ_q5_Kx8 {
    ggml_half d[8];              // super-block scale for quantized scales
    ggml_half dmin[8];           // super-block scale for quantized mins
    uint8_t   scales[96];        // scales and mins, quantized with 6 bits
    uint8_t   qh[QK_K * 8 / 8];  // high bits of 5-bit quants
    uint8_t   qs[QK_K * 8 / 2];  // low bits of 5-bit quants (in groups of 4)
};

static_assert(ਆਕਾਰ(ਖੰਡ_q5_Kx8) == ਆਕਾਰ(ggml_half) * 16 + K_SCALE_ਮਾਪ * 8 + QK_K * 5,
              "wrong q5_K block size/padding");

ਬਣਤਰ ਖੰਡ_q6_Kx8 {
    ggml_half d[8];
    int8_t    scales[QK_K / 16 * 8];
    uint8_t   ql[QK_K / 2 * 8];  // low bits of 6-bit quants (groups of 2)
    uint8_t   qh[QK_K / 4 * 8];  // high bits of 6-bit quants (groups of 4)
};

static_assert(ਆਕਾਰ(ਖੰਡ_q6_Kx8) == ਆਕਾਰ(ggml_half) * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8,
              "wrong q6_K block size/padding");

ਬਣਤਰ ਖੰਡ_q8_Kx4 {
    ਦਸ਼ਮਲਵ d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(ਆਕਾਰ(ਖੰਡ_q8_Kx4) == ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4 + QK_K * 4 + (QK_K / 4) * ਆਕਾਰ(int16_t), "wrong q8_K block size/padding");

ਬਣਤਰ ਖੰਡ_iq4_nlx4 {
    ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(ਆਕਾਰ(ਖੰਡ_iq4_nlx4) == 4 * ਆਕਾਰ(ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

ਬਣਤਰ ਖੰਡ_iq4_nlx8 {
    ggml_half d[8];            // deltas for 8 iq4_nl blocks
    uint8_t   qs[QK4_NL * 4];  // nibbles / quants for 8 iq4_nl blocks
};

static_assert(ਆਕਾਰ(ਖੰਡ_iq4_nlx8) == 8 * ਆਕਾਰ(ggml_half) + QK4_NL * 4, "wrong iq4_nlx8 block size/padding");

ਬਣਤਰ ਖੰਡ_iq4_nlx16 {
    ggml_half d[16];            // deltas for 16 iq4_nl blocks
    uint8_t   qs[QK4_NL * 8];  // nibbles / quants for 16 iq4_nl blocks
};

static_assert(ਆਕਾਰ(ਖੰਡ_iq4_nlx16) == 16 * ਆਕਾਰ(ggml_half) + QK4_NL * 8, "wrong iq4_nlx16 block size/padding");
ਬਣਤਰ ਖੰਡ_mxfp4x4 {
    uint8_t e[4];
    uint8_t qs[QK_MXFP4 * 2];
};
static_assert(ਆਕਾਰ(ਖੰਡ_mxfp4x4) == 4 + QK_MXFP4 * 2, "wrong mxfp4x4 block size/padding");

ਬਣਤਰ ਖੰਡ_mxfp4x8 {
    uint8_t e[8];
    uint8_t qs[QK_MXFP4 * 4];
};
static_assert(ਆਕਾਰ(ਖੰਡ_mxfp4x8) == 8 + QK_MXFP4 * 4, "wrong mxfp4x8 block size/padding");

#ਜੇ defined(__cplusplus)
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_quantize_mat_q8_0_4x4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_0_4x8(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_K_4x4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_K_4x8(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_gemv_q1_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q1_0_4x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_0_4x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_0_8x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q2_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_K_8x4_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q5_K_8x4_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q5_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q6_K_8x4_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q6_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_iq4_nl_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_iq4_nl_8x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_mxfp4_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_mxfp4_8x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q8_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q8_0_4x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q1_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q1_0_4x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_4x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_8x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q2_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_K_8x4_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q5_K_8x4_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q5_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q6_K_8x4_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q6_K_8x8_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_iq4_nl_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_iq4_nl_8x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_mxfp4_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_mxfp4_8x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q8_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q8_0_4x8_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
#ਜੇ defined __riscv_zvfh
ਖਾਲੀ ggml_quantize_mat_q8_0_4x1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_K_4x1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_gemv_q4_0_16x1_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_K_16x1_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_iq4_nl_16x1_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q8_0_16x1_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q2_K_16x1_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_16x1_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_K_16x1_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_iq4_nl_16x1_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q8_0_16x1_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q2_K_16x1_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
#ਜੇ_ਅੰਤ

// Native implementations
ਖਾਲੀ ggml_quantize_mat_q8_0_4x4_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_0_4x8_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_K_4x4_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_K_4x8_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_gemv_q1_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q1_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_0_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q2_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q5_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q5_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q6_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q6_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_iq4_nl_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_iq4_nl_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_mxfp4_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_mxfp4_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q8_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q8_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q1_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q1_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q2_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q5_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q5_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q6_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q6_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_iq4_nl_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_iq4_nl_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_mxfp4_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_mxfp4_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q8_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q8_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
#ਜੇ defined __riscv_zvfh
ਖਾਲੀ ggml_quantize_mat_q8_0_4x1_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_quantize_mat_q8_K_4x1_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ ggml_gemv_q4_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q4_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q8_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_q2_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemv_iq4_nl_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q4_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q8_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_q2_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
ਖਾਲੀ ggml_gemm_iq4_nl_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc);
#ਜੇ_ਅੰਤ

#ਜੇ defined(__cplusplus)
} // extern "C"
#ਜੇ_ਅੰਤ
