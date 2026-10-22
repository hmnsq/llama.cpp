#ہدایت once

#ضمن "ggml-cpu-impl.h"

#ضمن <cassert>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <functional>

namespace spacemit_kernels {

constexpr خودکار div_گیڑا_اتے(خودکار اتے, خودکار تھلے) {
    أرجع (اتے + تھلے - 1) / تھلے;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_ناپ(size_t blk_len, bool with_blk_جوڑ = false) {
    ثابت size_t blk_ناپ = حجم(عائم) + blk_len * حجم(int8_t) + (with_blk_جوڑ ? حجم(int16_t) : 0);
    أرجع blk_ناپ;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_ناپ(size_t blk_len, bool with_blk_جوڑ = false, bool with_blk_scale = false) {
    ثابت size_t subblk_گنتی = div_گیڑا_اتے(blk_len, size_t(32));
    ثابت size_t blk_ناپ     = blk_len * حجم(int8_t) + subblk_گنتی * حجم(_Float16) +
                            (with_blk_جوڑ ? subblk_گنتی * حجم(_Float16) : 0) +
                            (with_blk_scale ? حجم(_Float16) : 0);
    أرجع blk_ناپ;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_ناپ(size_t blk_len) {
    ثابت size_t blk_ناپ = حجم(عائم) + blk_len * حجم(int8_t) + حجم(int16_t) * blk_len / 16;
    أرجع blk_ناپ;
}

using quantize_a_صف_def = std::function<فراغ(size_t, ثابت عائم *, size_t, uint8_t *)>;

namespace rvv {
فراغ memcpy1d(فراغ * dst, ثابت فراغ * src, int64_t ناپ);

فراغ memcpy2d(فراغ * dst, int64_t dst_stride, ثابت فراغ * src, int64_t src_stride, int64_t tile_صفاں, int64_t ناپ);

فراغ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        صحيح                         ir0,
                                                        صحيح                         ir1,
                                                        فراغ *                      tcm_بفر,
                                                        size_t                      tcm_بفر_ناپ);

فراغ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    صحيح                         ir0,
                                                    صحيح                         ir1,
                                                    فراغ *                      tcm_بفر,
                                                    size_t                      tcm_بفر_ناپ);

فراغ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

فراغ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

فراغ forward_cont_with_ترتیب_بدلو(ggml_compute_params * params, ggml_tensor * op);

فراغ forward_cpy_with_ترتیب_بدلو(ggml_compute_params * params, ggml_tensor * op);

سانچا <typename T> فراغ forward_get_صفاں(ggml_compute_params * params, ggml_tensor * op);

سانچا <typename T> فراغ forward_concat(ggml_compute_params * params, ggml_tensor * op);

سانچا <ggml_op op_type, typename T> فراغ forward_ثنائی(ggml_compute_params * params, ggml_tensor * op);

سانچا <typename T> فراغ forward_جوڑ_صفاں(ثابت ggml_compute_params * params, ggml_tensor * op);

سانچا <typename T> فراغ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

سانچا <typename T> فراغ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

فراغ quantize_a_صف_i8(size_t blk_len, ثابت عائم * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

فراغ quantize_a_4row_i8(size_t blk_len, ثابت عائم * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

فراغ quantize_a_صف_i8_hp(size_t blk_len, ثابت عائم * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

فراغ quantize_a_4row_i8_hp(size_t blk_len, ثابت عائم * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

فراغ quantize_a_صف_i8k(size_t blk_len, ثابت عائم * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

فراغ quantize_a_4row_i8k(size_t blk_len, ثابت عائم * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
