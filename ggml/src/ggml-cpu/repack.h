#pragma once

#define GGML_COMMON_DECL_CPP
#অন্তর্ভাব "ggml-common.h"

#অন্তর্ভাব "traits.h"
#অন্তর্ভাব "ggml.h"

// GGML internal header

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(শূন্য);

template <পূর্ণাঙ্ক K> constexpr পূর্ণাঙ্ক QK_0() {
    যদি constexpr (K == 1) {
        প্রত্যাগম QK1_0;
    }
    যদি constexpr (K == 4) {
        প্রত্যাগম QK4_0;
    }
    যদি constexpr (K == 8) {
        প্রত্যাগম QK8_0;
    }
    প্রত্যাগম -1;
}

template <পূর্ণাঙ্ক K, পূর্ণাঙ্ক N> সংরচনা block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(পরিমাণ(block<1, 4>) == 4 * পরিমাণ(ggml_half) + QK1_0 / 2, "wrong block<1,4> size/padding");
static_assert(পরিমাণ(block<4, 4>) == 4 * পরিমাণ(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(পরিমাণ(block<4, 8>) == 8 * পরিমাণ(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(পরিমাণ(block<4, 16>) == 16 * পরিমাণ(ggml_half) + QK8_0 * 8, "wrong block<4,16> size/padding");
static_assert(পরিমাণ(block<8, 4>) == 4 * পরিমাণ(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(পরিমাণ(block<8, 8>) == 8 * পরিমাণ(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");
static_assert(পরিমাণ(block<8, 16>) == 16 * পরিমাণ(ggml_half) + QK8_0 * 16, "wrong block<8,16> size/padding");

using block_q1_0x4 = block<1, 4>;
using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q4_0x16 = block<4, 16>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x16 = block<8, 16>;

সংরচনা block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static_assert(পরিমাণ(block_q4_Kx8) == পরিমাণ(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4, "wrong q4_K block size/padding");
সংরচনা block_q4_Kx16 {
    ggml_half d[16];      // super-block scale for quantized scales
    ggml_half dmin[16];   // super-block scale for quantized mins
    uint8_t scales[192];  // scales and mins, quantized with 6 bits
    uint8_t qs[2048];    // 4--bit quants
};

static_assert(পরিমাণ(block_q4_Kx16) == পরিমাণ(ggml_half) * 32 + K_SCALE_SIZE * 16 + QK_K * 8, "wrong q4_K block size/padding");
সংরচনা block_q2_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[128];  // scales and mins, quantized with 4 bits
    uint8_t qs[512];    // 2--bit quants
};

static_assert(পরিমাণ(block_q2_Kx8) == পরিমাণ(ggml_half) * 16 + QK_K/2 + QK_K * 2, "wrong q2_K block size/padding");
সংরচনা block_q2_Kx16 {
    ggml_half d[16];       // Super-block scale for quantized scales
    ggml_half dmin[16];    // Super-block scale for quantized mins
    uint8_t   scales[256]; // Sub-block scales (16 cols * 16 sub-blocks)
    uint8_t   qs[1024];    // Data (16 cols * 64 bytes per block)
};
static_assert(পরিমাণ(block_q2_Kx16) == পরিমাণ(ggml_half) * 32 + QK_K + QK_K * 4, "wrong q2_K block size/padding");

সংরচনা block_q5_Kx8 {
    ggml_half d[8];              // super-block scale for quantized scales
    ggml_half dmin[8];           // super-block scale for quantized mins
    uint8_t   scales[96];        // scales and mins, quantized with 6 bits
    uint8_t   qh[QK_K * 8 / 8];  // high bits of 5-bit quants
    uint8_t   qs[QK_K * 8 / 2];  // low bits of 5-bit quants (in groups of 4)
};

static_assert(পরিমাণ(block_q5_Kx8) == পরিমাণ(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 5,
              "wrong q5_K block size/padding");

সংরচনা block_q6_Kx8 {
    ggml_half d[8];
    int8_t    scales[QK_K / 16 * 8];
    uint8_t   ql[QK_K / 2 * 8];  // low bits of 6-bit quants (groups of 2)
    uint8_t   qh[QK_K / 4 * 8];  // high bits of 6-bit quants (groups of 4)
};

static_assert(পরিমাণ(block_q6_Kx8) == পরিমাণ(ggml_half) * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8,
              "wrong q6_K block size/padding");

সংরচনা block_q8_Kx4 {
    ভিন্ন d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(পরিমাণ(block_q8_Kx4) == পরিমাণ(ভিন্ন) * 4 + QK_K * 4 + (QK_K / 4) * পরিমাণ(int16_t), "wrong q8_K block size/padding");

সংরচনা block_iq4_nlx4 {
    ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(পরিমাণ(block_iq4_nlx4) == 4 * পরিমাণ(ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

সংরচনা block_iq4_nlx8 {
    ggml_half d[8];            // deltas for 8 iq4_nl blocks
    uint8_t   qs[QK4_NL * 4];  // nibbles / quants for 8 iq4_nl blocks
};

static_assert(পরিমাণ(block_iq4_nlx8) == 8 * পরিমাণ(ggml_half) + QK4_NL * 4, "wrong iq4_nlx8 block size/padding");

সংরচনা block_iq4_nlx16 {
    ggml_half d[16];            // deltas for 16 iq4_nl blocks
    uint8_t   qs[QK4_NL * 8];  // nibbles / quants for 16 iq4_nl blocks
};

static_assert(পরিমাণ(block_iq4_nlx16) == 16 * পরিমাণ(ggml_half) + QK4_NL * 8, "wrong iq4_nlx16 block size/padding");
সংরচনা block_mxfp4x4 {
    uint8_t e[4];
    uint8_t qs[QK_MXFP4 * 2];
};
static_assert(পরিমাণ(block_mxfp4x4) == 4 + QK_MXFP4 * 2, "wrong mxfp4x4 block size/padding");

সংরচনা block_mxfp4x8 {
    uint8_t e[8];
    uint8_t qs[QK_MXFP4 * 4];
};
static_assert(পরিমাণ(block_mxfp4x8) == 8 + QK_MXFP4 * 4, "wrong mxfp4x8 block size/padding");

#যদি defined(__cplusplus)
বাহ্য "C" {
#endif

শূন্য ggml_quantize_mat_q8_0_4x4(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_0_4x8(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_K_4x4(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_K_4x8(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_gemv_q1_0_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q1_0_4x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_0_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_0_4x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_0_8x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q2_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_K_8x4_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q5_K_8x4_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q5_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q6_K_8x4_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q6_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_iq4_nl_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_iq4_nl_8x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_mxfp4_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_mxfp4_8x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q8_0_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q8_0_4x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q1_0_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q1_0_4x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_4x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_8x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q2_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_K_8x4_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q5_K_8x4_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q5_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q6_K_8x4_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q6_K_8x8_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_iq4_nl_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_iq4_nl_8x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_mxfp4_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_mxfp4_8x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q8_0_4x4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q8_0_4x8_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
#যদি defined __riscv_zvfh
শূন্য ggml_quantize_mat_q8_0_4x1(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_K_4x1(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_gemv_q4_0_16x1_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_K_16x1_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_iq4_nl_16x1_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q8_0_16x1_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q2_K_16x1_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_16x1_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_K_16x1_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_iq4_nl_16x1_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q8_0_16x1_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q2_K_16x1_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
#endif

// Native implementations
শূন্য ggml_quantize_mat_q8_0_4x4_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_0_4x8_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_K_4x4_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_K_4x8_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_gemv_q1_0_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q1_0_4x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_0_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_0_4x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_0_8x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q2_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_K_8x4_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q5_K_8x4_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q5_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q6_K_8x4_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q6_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_iq4_nl_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_iq4_nl_8x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_mxfp4_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_mxfp4_8x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q8_0_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q8_0_4x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q1_0_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q1_0_4x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_4x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_8x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q2_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_K_8x4_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q5_K_8x4_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q5_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q6_K_8x4_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q6_K_8x8_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_iq4_nl_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_iq4_nl_8x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_mxfp4_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_mxfp4_8x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q8_0_4x4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q8_0_4x8_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
#যদি defined __riscv_zvfh
শূন্য ggml_quantize_mat_q8_0_4x1_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_quantize_mat_q8_K_4x1_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য ggml_gemv_q4_0_16x1_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q4_K_16x1_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q8_0_16x1_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_q2_K_16x1_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemv_iq4_nl_16x1_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_0_16x1_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q4_K_16x1_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q8_0_16x1_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_q2_K_16x1_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
শূন্য ggml_gemm_iq4_nl_16x1_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, ধ্রুব শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক nr, পূর্ণাঙ্ক nc);
#endif

#যদি defined(__cplusplus)
} // extern "C"
#endif
