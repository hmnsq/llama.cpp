#pragma once

#અન્તર્ભાવ "ggml-cpu-impl.h"

#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <functional>

namespace spacemit_kernels {

constexpr auto div_round_up(auto up, auto down) {
    પાછા (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    અચલ size_t blk_size = કદ(દશાંશ) + blk_len * કદ(int8_t) + (with_blk_sum ? કદ(int16_t) : 0);
    પાછા blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    અચલ size_t subblk_ગણતરી = div_round_up(blk_len, size_t(32));
    અચલ size_t blk_size     = blk_len * કદ(int8_t) + subblk_ગણતરી * કદ(_Float16) +
                            (with_blk_sum ? subblk_ગણતરી * કદ(_Float16) : 0) +
                            (with_blk_scale ? કદ(_Float16) : 0);
    પાછા blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    અચલ size_t blk_size = કદ(દશાંશ) + blk_len * કદ(int8_t) + કદ(int16_t) * blk_len / 16;
    પાછા blk_size;
}

using quantize_a_row_def = std::function<શૂન્ય(size_t, અચલ દશાંશ *, size_t, uint8_t *)>;

namespace rvv {
શૂન્ય memcpy1d(શૂન્ય * dst, અચલ શૂન્ય * src, int64_t size);

શૂન્ય memcpy2d(શૂન્ય * dst, int64_t dst_stride, અચલ શૂન્ય * src, int64_t src_stride, int64_t tile_rows, int64_t size);

શૂન્ય forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(અચલ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        પૂર્ણાંક                         ir0,
                                                        પૂર્ણાંક                         ir1,
                                                        શૂન્ય *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size);

શૂન્ય forward_flash_attn_ext_f16_tiled_vlen1024_vf16(અચલ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    પૂર્ણાંક                         ir0,
                                                    પૂર્ણાંક                         ir1,
                                                    શૂન્ય *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size);

શૂન્ય forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

શૂન્ય forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

શૂન્ય forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

શૂન્ય forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

template <typename T> શૂન્ય forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> શૂન્ય forward_concat(ggml_compute_params * params, ggml_tensor * op);

template <ggml_op op_type, typename T> શૂન્ય forward_દ્વિઅંકી(ggml_compute_params * params, ggml_tensor * op);

template <typename T> શૂન્ય forward_sum_rows(અચલ ggml_compute_params * params, ggml_tensor * op);

template <typename T> શૂન્ય forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> શૂન્ય forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

શૂન્ય quantize_a_row_i8(size_t blk_len, અચલ દશાંશ * a_ptr, size_t ગણતરી_k, uint8_t * quant_a_ptr);

શૂન્ય quantize_a_4row_i8(size_t blk_len, અચલ દશાંશ * a_ptr, size_t ગણતરી_k, uint8_t * quant_a_ptr);

શૂન્ય quantize_a_row_i8_hp(size_t blk_len, અચલ દશાંશ * a_ptr, size_t ગણતરી_k, uint8_t * quant_a_ptr);

શૂન્ય quantize_a_4row_i8_hp(size_t blk_len, અચલ દશાંશ * a_ptr, size_t ગણતરી_k, uint8_t * quant_a_ptr);

શૂન્ય quantize_a_row_i8k(size_t blk_len, અચલ દશાંશ * a_ptr, size_t ગણતરી_k, uint8_t * quant_a_ptr);

શૂન્ય quantize_a_4row_i8k(size_t blk_len, અચલ દશાંશ * a_ptr, size_t ગણતરી_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
