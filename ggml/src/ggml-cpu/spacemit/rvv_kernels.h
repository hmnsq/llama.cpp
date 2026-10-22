#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <functional>

namespace spacemit_kernels {

constexpr ಸ್ವಯಂಚಾಲಿತ div_ಸುತ್ತು_ಮೇಲೆ(ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ, ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ) {
    ಹಿಂತಿರುಗಿ (ಮೇಲೆ + ಕೆಳಕ್ಕೆ - 1) / ಕೆಳಕ್ಕೆ;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_ಗಾತ್ರ(size_t blk_len, bool with_blk_ಮೊತ್ತ = false) {
    ಸ್ಥಿರಾಂಕ size_t blk_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ) + blk_len * ಗಾತ್ರ(int8_t) + (with_blk_ಮೊತ್ತ ? ಗಾತ್ರ(int16_t) : 0);
    ಹಿಂತಿರುಗಿ blk_ಗಾತ್ರ;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_ಗಾತ್ರ(size_t blk_len, bool with_blk_ಮೊತ್ತ = false, bool with_blk_scale = false) {
    ಸ್ಥಿರಾಂಕ size_t subblk_ಎಣಿಕೆ = div_ಸುತ್ತು_ಮೇಲೆ(blk_len, size_t(32));
    ಸ್ಥಿರಾಂಕ size_t blk_ಗಾತ್ರ     = blk_len * ಗಾತ್ರ(int8_t) + subblk_ಎಣಿಕೆ * ಗಾತ್ರ(_Float16) +
                            (with_blk_ಮೊತ್ತ ? subblk_ಎಣಿಕೆ * ಗಾತ್ರ(_Float16) : 0) +
                            (with_blk_scale ? ಗಾತ್ರ(_Float16) : 0);
    ಹಿಂತಿರುಗಿ blk_ಗಾತ್ರ;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_ಗಾತ್ರ(size_t blk_len) {
    ಸ್ಥಿರಾಂಕ size_t blk_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ) + blk_len * ಗಾತ್ರ(int8_t) + ಗಾತ್ರ(int16_t) * blk_len / 16;
    ಹಿಂತಿರುಗಿ blk_ಗಾತ್ರ;
}

using quantize_a_ಅಡ್ಡಸಾಲು_def = std::function<ಶೂನ್ಯ(size_t, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *, size_t, uint8_t *)>;

namespace rvv {
ಶೂನ್ಯ memcpy1d(ಶೂನ್ಯ * dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, int64_t ಗಾತ್ರ_2);

ಶೂನ್ಯ memcpy2d(ಶೂನ್ಯ * dst, int64_t dst_stride, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, int64_t src_stride, int64_t tile_ಅಡ್ಡಸಾಲುಗಳು, int64_t ಗಾತ್ರ_2);

ಶೂನ್ಯ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        ಪೂರ್ಣಾಂಕ                         ir0,
                                                        ಪೂರ್ಣಾಂಕ                         ir1,
                                                        ಶೂನ್ಯ *                      tcm_ಬಫರ್,
                                                        size_t                      tcm_ಬಫರ್_ಗಾತ್ರ);

ಶೂನ್ಯ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    ಪೂರ್ಣಾಂಕ                         ir0,
                                                    ಪೂರ್ಣಾಂಕ                         ir1,
                                                    ಶೂನ್ಯ *                      tcm_ಬಫರ್,
                                                    size_t                      tcm_ಬಫರ್_ಗಾತ್ರ);

ಶೂನ್ಯ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ forward_cont_with_ಕ್ರಮಬದಲಾವಣೆ(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ forward_cpy_with_ಕ್ರಮಬದಲಾವಣೆ(ggml_compute_params * params, ggml_tensor * op);

ಮಾದರಿ <typename T> ಶೂನ್ಯ forward_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_compute_params * params, ggml_tensor * op);

ಮಾದರಿ <typename T> ಶೂನ್ಯ forward_concat(ggml_compute_params * params, ggml_tensor * op);

ಮಾದರಿ <ggml_op op_type, typename T> ಶೂನ್ಯ forward_ದ್ವಿಮಾನ(ggml_compute_params * params, ggml_tensor * op);

ಮಾದರಿ <typename T> ಶೂನ್ಯ forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * op);

ಮಾದರಿ <typename T> ಶೂನ್ಯ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

ಮಾದರಿ <typename T> ಶೂನ್ಯ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

ಶೂನ್ಯ quantize_a_ಅಡ್ಡಸಾಲು_i8(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_4row_i8(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_ಅಡ್ಡಸಾಲು_i8_hp(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_4row_i8_hp(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_ಅಡ್ಡಸಾಲು_i8k(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

ಶೂನ್ಯ quantize_a_4row_i8k(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
