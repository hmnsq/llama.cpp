#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <functional>

namespace spacemit_kernels {

constexpr auto div_round_up(auto up, auto down) {
    ಪ್ರತ್ಯಾಗಮ (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    ಧ್ರುವ size_t blk_size = ಪರಿಮಾಣ(ಭಿನ್ನ) + blk_len * ಪರಿಮಾಣ(int8_t) + (with_blk_sum ? ಪರಿಮಾಣ(int16_t) : 0);
    ಪ್ರತ್ಯಾಗಮ blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    ಧ್ರುವ size_t subblk_ಎಣಿಕೆ = div_round_up(blk_len, size_t(32));
    ಧ್ರುವ size_t blk_size     = blk_len * ಪರಿಮಾಣ(int8_t) + subblk_ಎಣಿಕೆ * ಪರಿಮಾಣ(_Float16) +
                            (with_blk_sum ? subblk_ಎಣಿಕೆ * ಪರಿಮಾಣ(_Float16) : 0) +
                            (with_blk_scale ? ಪರಿಮಾಣ(_Float16) : 0);
    ಪ್ರತ್ಯಾಗಮ blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    ಧ್ರುವ size_t blk_size = ಪರಿಮಾಣ(ಭಿನ್ನ) + blk_len * ಪರಿಮಾಣ(int8_t) + ಪರಿಮಾಣ(int16_t) * blk_len / 16;
    ಪ್ರತ್ಯಾಗಮ blk_size;
}

using quantize_a_row_def = std::function<ಶೂನ್ಯ(size_t, ಧ್ರುವ ಭಿನ್ನ *, size_t, uint8_t *)>;

namespace rvv {
ಶೂನ್ಯ memcpy1d(ಶೂನ್ಯ * dst, ಧ್ರುವ ಶೂನ್ಯ * src, int64_t size);

ಶೂನ್ಯ memcpy2d(ಶೂನ್ಯ * dst, int64_t dst_stride, ಧ್ರುವ ಶೂನ್ಯ * src, int64_t src_stride, int64_t tile_rows, int64_t size);

ಶೂನ್ಯ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ಧ್ರುವ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        ಪೂರ್ಣಾಙ್ಕ                         ir0,
                                                        ಪೂರ್ಣಾಙ್ಕ                         ir1,
                                                        ಶೂನ್ಯ *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size);

ಶೂನ್ಯ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ಧ್ರುವ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    ಪೂರ್ಣಾಙ್ಕ                         ir0,
                                                    ಪೂರ್ಣಾಙ್ಕ                         ir1,
                                                    ಶೂನ್ಯ *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size);

ಶೂನ್ಯ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ಶೂನ್ಯ forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ಶೂನ್ಯ forward_concat(ggml_compute_params * params, ggml_tensor * op);

template <ggml_op op_type, typename T> ಶೂನ್ಯ forward_ದ್ವಿಮಾನ(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ಶೂನ್ಯ forward_sum_rows(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * op);

template <typename T> ಶೂನ್ಯ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ಶೂನ್ಯ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ quantize_a_row_i8(size_t blk_len, ಧ್ರುವ ಭಿನ್ನ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_4row_i8(size_t blk_len, ಧ್ರುವ ಭಿನ್ನ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_row_i8_hp(size_t blk_len, ಧ್ರುವ ಭಿನ್ನ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_4row_i8_hp(size_t blk_len, ಧ್ರುವ ಭಿನ್ನ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_row_i8k(size_t blk_len, ಧ್ರುವ ಭಿನ್ನ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_4row_i8k(size_t blk_len, ಧ್ರುವ ಭಿನ್ನ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
