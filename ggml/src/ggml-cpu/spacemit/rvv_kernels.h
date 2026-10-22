#pragma once

#अन्तर्भाव "ggml-cpu-impl.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <functional>

namespace spacemit_kernels {

constexpr स्वतः div_round_up(स्वतः up, स्वतः down) {
    प्रत्यागम (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr माप_प्रकार q8_blk_size(माप_प्रकार blk_len, bool with_blk_sum = false) {
    ध्रुव माप_प्रकार blk_size = परिमाण(भिन्न) + blk_len * परिमाण(int8_t) + (with_blk_sum ? परिमाण(int16_t) : 0);
    प्रत्यागम blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr माप_प्रकार q8_hp_blk_size(माप_प्रकार blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    ध्रुव माप_प्रकार subblk_गिनती = div_round_up(blk_len, माप_प्रकार(32));
    ध्रुव माप_प्रकार blk_size     = blk_len * परिमाण(int8_t) + subblk_गिनती * परिमाण(_Float16) +
                            (with_blk_sum ? subblk_गिनती * परिमाण(_Float16) : 0) +
                            (with_blk_scale ? परिमाण(_Float16) : 0);
    प्रत्यागम blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr माप_प्रकार q8k_blk_size(माप_प्रकार blk_len) {
    ध्रुव माप_प्रकार blk_size = परिमाण(भिन्न) + blk_len * परिमाण(int8_t) + परिमाण(int16_t) * blk_len / 16;
    प्रत्यागम blk_size;
}

using quantize_a_row_def = std::function<शून्य(माप_प्रकार, ध्रुव भिन्न *, माप_प्रकार, uint8_t *)>;

namespace rvv {
शून्य memcpy1d(शून्य * dst, ध्रुव शून्य * src, int64_t size);

शून्य memcpy2d(शून्य * dst, int64_t dst_stride, ध्रुव शून्य * src, int64_t src_stride, int64_t tile_rows, int64_t size);

शून्य forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ध्रुव ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        पूर्णाङ्क                         ir0,
                                                        पूर्णाङ्क                         ir1,
                                                        शून्य *                      tcm_buffer,
                                                        माप_प्रकार                      tcm_buffer_size);

शून्य forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ध्रुव ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    पूर्णाङ्क                         ir0,
                                                    पूर्णाङ्क                         ir1,
                                                    शून्य *                      tcm_buffer,
                                                    माप_प्रकार                      tcm_buffer_size);

शून्य forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

शून्य forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

शून्य forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

शून्य forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> शून्य forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> शून्य forward_concat(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <ggml_op op_type, typename T> शून्य forward_द्विआधारी(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> शून्य forward_sum_rows(ध्रुव ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> शून्य forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> शून्य forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

शून्य quantize_a_row_i8(माप_प्रकार blk_len, ध्रुव भिन्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

शून्य quantize_a_4row_i8(माप_प्रकार blk_len, ध्रुव भिन्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

शून्य quantize_a_row_i8_hp(माप_प्रकार blk_len, ध्रुव भिन्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

शून्य quantize_a_4row_i8_hp(माप_प्रकार blk_len, ध्रुव भिन्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

शून्य quantize_a_row_i8k(माप_प्रकार blk_len, ध्रुव भिन्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

शून्य quantize_a_4row_i8k(माप_प्रकार blk_len, ध्रुव भिन्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
