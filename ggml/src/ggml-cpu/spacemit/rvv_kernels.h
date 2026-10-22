#pragma once

#অন্তর্ভাব "ggml-cpu-impl.h"

#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <functional>

namespace spacemit_kernels {

constexpr auto div_round_up(auto up, auto down) {
    প্রত্যাগম (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    ধ্রুব size_t blk_size = পরিমাণ(ভিন্ন) + blk_len * পরিমাণ(int8_t) + (with_blk_sum ? পরিমাণ(int16_t) : 0);
    প্রত্যাগম blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    ধ্রুব size_t subblk_গুনতি = div_round_up(blk_len, size_t(32));
    ধ্রুব size_t blk_size     = blk_len * পরিমাণ(int8_t) + subblk_গুনতি * পরিমাণ(_Float16) +
                            (with_blk_sum ? subblk_গুনতি * পরিমাণ(_Float16) : 0) +
                            (with_blk_scale ? পরিমাণ(_Float16) : 0);
    প্রত্যাগম blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    ধ্রুব size_t blk_size = পরিমাণ(ভিন্ন) + blk_len * পরিমাণ(int8_t) + পরিমাণ(int16_t) * blk_len / 16;
    প্রত্যাগম blk_size;
}

using quantize_a_row_def = std::function<শূন্য(size_t, ধ্রুব ভিন্ন *, size_t, uint8_t *)>;

namespace rvv {
শূন্য memcpy1d(শূন্য * dst, ধ্রুব শূন্য * src, int64_t size);

শূন্য memcpy2d(শূন্য * dst, int64_t dst_stride, ধ্রুব শূন্য * src, int64_t src_stride, int64_t tile_rows, int64_t size);

শূন্য forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ধ্রুব ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        পূর্ণাঙ্ক                         ir0,
                                                        পূর্ণাঙ্ক                         ir1,
                                                        শূন্য *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size);

শূন্য forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ধ্রুব ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    পূর্ণাঙ্ক                         ir0,
                                                    পূর্ণাঙ্ক                         ir1,
                                                    শূন্য *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size);

শূন্য forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

শূন্য forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

শূন্য forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

শূন্য forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

template <typename T> শূন্য forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> শূন্য forward_concat(ggml_compute_params * params, ggml_tensor * op);

template <ggml_op op_type, typename T> শূন্য forward_দ্বিমিক(ggml_compute_params * params, ggml_tensor * op);

template <typename T> শূন্য forward_sum_rows(ধ্রুব ggml_compute_params * params, ggml_tensor * op);

template <typename T> শূন্য forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> শূন্য forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

শূন্য quantize_a_row_i8(size_t blk_len, ধ্রুব ভিন্ন * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr);

শূন্য quantize_a_4row_i8(size_t blk_len, ধ্রুব ভিন্ন * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr);

শূন্য quantize_a_row_i8_hp(size_t blk_len, ধ্রুব ভিন্ন * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr);

শূন্য quantize_a_4row_i8_hp(size_t blk_len, ধ্রুব ভিন্ন * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr);

শূন্য quantize_a_row_i8k(size_t blk_len, ধ্রুব ভিন্ন * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr);

শূন্য quantize_a_4row_i8k(size_t blk_len, ধ্রুব ভিন্ন * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
