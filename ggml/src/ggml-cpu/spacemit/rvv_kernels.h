#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <functional>

namespace spacemit_kernels {

constexpr ਸਵੈਚਾਲਿਤ div_ਗੇੜਾ_ਉੱਪਰ(ਸਵੈਚਾਲਿਤ ਉੱਪਰ, ਸਵੈਚਾਲਿਤ ਥੱਲੇ) {
    ਵਾਪਸ (ਉੱਪਰ + ਥੱਲੇ - 1) / ਥੱਲੇ;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_ਮਾਪ(size_t blk_len, bool with_blk_ਜੋੜ = false) {
    ਅਚਲ size_t blk_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ) + blk_len * ਆਕਾਰ(int8_t) + (with_blk_ਜੋੜ ? ਆਕਾਰ(int16_t) : 0);
    ਵਾਪਸ blk_ਮਾਪ;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_ਮਾਪ(size_t blk_len, bool with_blk_ਜੋੜ = false, bool with_blk_scale = false) {
    ਅਚਲ size_t subblk_ਗਿਣਤੀ = div_ਗੇੜਾ_ਉੱਪਰ(blk_len, size_t(32));
    ਅਚਲ size_t blk_ਮਾਪ     = blk_len * ਆਕਾਰ(int8_t) + subblk_ਗਿਣਤੀ * ਆਕਾਰ(_Float16) +
                            (with_blk_ਜੋੜ ? subblk_ਗਿਣਤੀ * ਆਕਾਰ(_Float16) : 0) +
                            (with_blk_scale ? ਆਕਾਰ(_Float16) : 0);
    ਵਾਪਸ blk_ਮਾਪ;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_ਮਾਪ(size_t blk_len) {
    ਅਚਲ size_t blk_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ) + blk_len * ਆਕਾਰ(int8_t) + ਆਕਾਰ(int16_t) * blk_len / 16;
    ਵਾਪਸ blk_ਮਾਪ;
}

using quantize_a_ਪੰਕਤੀ_def = std::function<ਖਾਲੀ(size_t, ਅਚਲ ਦਸ਼ਮਲਵ *, size_t, uint8_t *)>;

namespace rvv {
ਖਾਲੀ memcpy1d(ਖਾਲੀ * dst, ਅਚਲ ਖਾਲੀ * src, int64_t ਮਾਪ);

ਖਾਲੀ memcpy2d(ਖਾਲੀ * dst, int64_t dst_stride, ਅਚਲ ਖਾਲੀ * src, int64_t src_stride, int64_t tile_ਪੰਕਤੀਆਂ, int64_t ਮਾਪ);

ਖਾਲੀ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ਅਚਲ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        ਪੂਰਨ                         ir0,
                                                        ਪੂਰਨ                         ir1,
                                                        ਖਾਲੀ *                      tcm_ਬਫਰ,
                                                        size_t                      tcm_ਬਫਰ_ਮਾਪ);

ਖਾਲੀ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ਅਚਲ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    ਪੂਰਨ                         ir0,
                                                    ਪੂਰਨ                         ir1,
                                                    ਖਾਲੀ *                      tcm_ਬਫਰ,
                                                    size_t                      tcm_ਬਫਰ_ਮਾਪ);

ਖਾਲੀ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ਖਾਲੀ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ਖਾਲੀ forward_cont_with_ਕ੍ਰਮ_ਬਦਲੋ(ggml_compute_params * params, ggml_tensor * op);

ਖਾਲੀ forward_cpy_with_ਕ੍ਰਮ_ਬਦਲੋ(ggml_compute_params * params, ggml_tensor * op);

ਸਾਂਚਾ <typename T> ਖਾਲੀ forward_get_ਪੰਕਤੀਆਂ(ggml_compute_params * params, ggml_tensor * op);

ਸਾਂਚਾ <typename T> ਖਾਲੀ forward_concat(ggml_compute_params * params, ggml_tensor * op);

ਸਾਂਚਾ <ggml_op op_type, typename T> ਖਾਲੀ forward_ਦੁਆਧਾਰੀ(ggml_compute_params * params, ggml_tensor * op);

ਸਾਂਚਾ <typename T> ਖਾਲੀ forward_ਜੋੜ_ਪੰਕਤੀਆਂ(ਅਚਲ ggml_compute_params * params, ggml_tensor * op);

ਸਾਂਚਾ <typename T> ਖਾਲੀ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

ਸਾਂਚਾ <typename T> ਖਾਲੀ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

ਖਾਲੀ quantize_a_ਪੰਕਤੀ_i8(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr);

ਖਾਲੀ quantize_a_4row_i8(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr);

ਖਾਲੀ quantize_a_ਪੰਕਤੀ_i8_hp(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr);

ਖਾਲੀ quantize_a_4row_i8_hp(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr);

ਖਾਲੀ quantize_a_ਪੰਕਤੀ_i8k(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr);

ਖਾਲੀ quantize_a_4row_i8k(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
