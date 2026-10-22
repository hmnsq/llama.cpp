#pragma once

#അന്തര്ഭാവ "ggml-cpu-impl.h"

#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <functional>

namespace spacemit_kernels {

constexpr auto div_round_up(auto up, auto down) {
    തിരികെ (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    സ്ഥിരം size_t blk_size = വലുപ്പം(ദശാംശം) + blk_len * വലുപ്പം(int8_t) + (with_blk_sum ? വലുപ്പം(int16_t) : 0);
    തിരികെ blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    സ്ഥിരം size_t subblk_എണ്ണം = div_round_up(blk_len, size_t(32));
    സ്ഥിരം size_t blk_size     = blk_len * വലുപ്പം(int8_t) + subblk_എണ്ണം * വലുപ്പം(_Float16) +
                            (with_blk_sum ? subblk_എണ്ണം * വലുപ്പം(_Float16) : 0) +
                            (with_blk_scale ? വലുപ്പം(_Float16) : 0);
    തിരികെ blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    സ്ഥിരം size_t blk_size = വലുപ്പം(ദശാംശം) + blk_len * വലുപ്പം(int8_t) + വലുപ്പം(int16_t) * blk_len / 16;
    തിരികെ blk_size;
}

using quantize_a_row_def = std::function<ശൂന്യം(size_t, സ്ഥിരം ദശാംശം *, size_t, uint8_t *)>;

namespace rvv {
ശൂന്യം memcpy1d(ശൂന്യം * dst, സ്ഥിരം ശൂന്യം * src, int64_t size);

ശൂന്യം memcpy2d(ശൂന്യം * dst, int64_t dst_stride, സ്ഥിരം ശൂന്യം * src, int64_t src_stride, int64_t tile_rows, int64_t size);

ശൂന്യം forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(സ്ഥിരം ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        പൂർണ്ണസംഖ്യ                         ir0,
                                                        പൂർണ്ണസംഖ്യ                         ir1,
                                                        ശൂന്യം *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size);

ശൂന്യം forward_flash_attn_ext_f16_tiled_vlen1024_vf16(സ്ഥിരം ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    പൂർണ്ണസംഖ്യ                         ir0,
                                                    പൂർണ്ണസംഖ്യ                         ir1,
                                                    ശൂന്യം *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size);

ശൂന്യം forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ശൂന്യം forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ശൂന്യം forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

ശൂന്യം forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ശൂന്യം forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ശൂന്യം forward_concat(ggml_compute_params * params, ggml_tensor * op);

template <ggml_op op_type, typename T> ശൂന്യം forward_ദ്വിമാന(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ശൂന്യം forward_sum_rows(സ്ഥിരം ggml_compute_params * params, ggml_tensor * op);

template <typename T> ശൂന്യം forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> ശൂന്യം forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

ശൂന്യം quantize_a_row_i8(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr);

ശൂന്യം quantize_a_4row_i8(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr);

ശൂന്യം quantize_a_row_i8_hp(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr);

ശൂന്യം quantize_a_4row_i8_hp(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr);

ശൂന്യം quantize_a_row_i8k(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr);

ശൂന്യം quantize_a_4row_i8k(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
