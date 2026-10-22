#pragma once

#அந்தர்பாவ "ggml-cpu-impl.h"

#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <functional>

namespace spacemit_kernels {

constexpr auto div_round_up(auto up, auto down) {
    திரும்பு (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    மாறிலி size_t blk_size = அளவு(தசம) + blk_len * அளவு(int8_t) + (with_blk_sum ? அளவு(int16_t) : 0);
    திரும்பு blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    மாறிலி size_t subblk_மொத்தம் = div_round_up(blk_len, size_t(32));
    மாறிலி size_t blk_size     = blk_len * அளவு(int8_t) + subblk_மொத்தம் * அளவு(_Float16) +
                            (with_blk_sum ? subblk_மொத்தம் * அளவு(_Float16) : 0) +
                            (with_blk_scale ? அளவு(_Float16) : 0);
    திரும்பு blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    மாறிலி size_t blk_size = அளவு(தசம) + blk_len * அளவு(int8_t) + அளவு(int16_t) * blk_len / 16;
    திரும்பு blk_size;
}

using quantize_a_row_def = std::function<வெறுமை(size_t, மாறிலி தசம *, size_t, uint8_t *)>;

namespace rvv {
வெறுமை memcpy1d(வெறுமை * dst, மாறிலி வெறுமை * src, int64_t size);

வெறுமை memcpy2d(வெறுமை * dst, int64_t dst_stride, மாறிலி வெறுமை * src, int64_t src_stride, int64_t tile_rows, int64_t size);

வெறுமை forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(மாறிலி ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        முழு                         ir0,
                                                        முழு                         ir1,
                                                        வெறுமை *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size);

வெறுமை forward_flash_attn_ext_f16_tiled_vlen1024_vf16(மாறிலி ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    முழு                         ir0,
                                                    முழு                         ir1,
                                                    வெறுமை *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size);

வெறுமை forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

வெறுமை forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

வெறுமை forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

வெறுமை forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

template <typename T> வெறுமை forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> வெறுமை forward_concat(ggml_compute_params * params, ggml_tensor * op);

template <ggml_op op_type, typename T> வெறுமை forward_இருமம்(ggml_compute_params * params, ggml_tensor * op);

template <typename T> வெறுமை forward_sum_rows(மாறிலி ggml_compute_params * params, ggml_tensor * op);

template <typename T> வெறுமை forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> வெறுமை forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

வெறுமை quantize_a_row_i8(size_t blk_len, மாறிலி தசம * a_ptr, size_t மொத்தம்_k, uint8_t * quant_a_ptr);

வெறுமை quantize_a_4row_i8(size_t blk_len, மாறிலி தசம * a_ptr, size_t மொத்தம்_k, uint8_t * quant_a_ptr);

வெறுமை quantize_a_row_i8_hp(size_t blk_len, மாறிலி தசம * a_ptr, size_t மொத்தம்_k, uint8_t * quant_a_ptr);

வெறுமை quantize_a_4row_i8_hp(size_t blk_len, மாறிலி தசம * a_ptr, size_t மொத்தம்_k, uint8_t * quant_a_ptr);

வெறுமை quantize_a_row_i8k(size_t blk_len, மாறிலி தசம * a_ptr, size_t மொத்தம்_k, uint8_t * quant_a_ptr);

வெறுமை quantize_a_4row_i8k(size_t blk_len, மாறிலி தசம * a_ptr, size_t மொத்தம்_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
