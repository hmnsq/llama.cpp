#निर्देशः once

#अन्तर्भाव "ggml-cpu-impl.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <functional>

namespace spacemit_kernels {

constexpr स्वचालित div_आवर्तनम्_ऊर्ध्वम्(स्वचालित ऊर्ध्वम्, स्वचालित अधस्तात्) {
    प्रत्यागम (ऊर्ध्वम् + अधस्तात् - 1) / अधस्तात्;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_आकार(size_t blk_len, bool with_blk_योग = false) {
    ध्रुव size_t blk_आकार = परिमाण(भिन्न) + blk_len * परिमाण(int8_t) + (with_blk_योग ? परिमाण(int16_t) : 0);
    प्रत्यागम blk_आकार;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_आकार(size_t blk_len, bool with_blk_योग = false, bool with_blk_scale = false) {
    ध्रुव size_t subblk_गणनम् = div_आवर्तनम्_ऊर्ध्वम्(blk_len, size_t(32));
    ध्रुव size_t blk_आकार     = blk_len * परिमाण(int8_t) + subblk_गणनम् * परिमाण(_Float16) +
                            (with_blk_योग ? subblk_गणनम् * परिमाण(_Float16) : 0) +
                            (with_blk_scale ? परिमाण(_Float16) : 0);
    प्रत्यागम blk_आकार;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_आकार(size_t blk_len) {
    ध्रुव size_t blk_आकार = परिमाण(भिन्न) + blk_len * परिमाण(int8_t) + परिमाण(int16_t) * blk_len / 16;
    प्रत्यागम blk_आकार;
}

using quantize_a_row_def = std::function<शून्य(size_t, ध्रुव भिन्न *, size_t, uint8_t *)>;

namespace rvv {
शून्य memcpy1d(शून्य * dst, ध्रुव शून्य * src, int64_t आकार);

शून्य memcpy2d(शून्य * dst, int64_t dst_stride, ध्रुव शून्य * src, int64_t src_stride, int64_t tile_पङ्क्तिकाः, int64_t आकार);

शून्य forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ध्रुव ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        पूर्णाङ्क                         ir0,
                                                        पूर्णाङ्क                         ir1,
                                                        शून्य *                      tcm_बफर,
                                                        size_t                      tcm_बफर_आकार);

शून्य forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ध्रुव ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    पूर्णाङ्क                         ir0,
                                                    पूर्णाङ्क                         ir1,
                                                    शून्य *                      tcm_बफर,
                                                    size_t                      tcm_बफर_आकार);

शून्य forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

शून्य forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

शून्य forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

शून्य forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

प्रतिरूपम् <typename T> शून्य forward_get_पङ्क्तिकाः(ggml_compute_params * params, ggml_tensor * op);

प्रतिरूपम् <typename T> शून्य forward_concat(ggml_compute_params * params, ggml_tensor * op);

प्रतिरूपम् <ggml_op op_type, typename T> शून्य forward_द्विमान(ggml_compute_params * params, ggml_tensor * op);

प्रतिरूपम् <typename T> शून्य forward_योग_पङ्क्तिकाः(ध्रुव ggml_compute_params * params, ggml_tensor * op);

प्रतिरूपम् <typename T> शून्य forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

प्रतिरूपम् <typename T> शून्य forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

शून्य quantize_a_row_i8(size_t blk_len, ध्रुव भिन्न * a_ptr, size_t गणनम्_k, uint8_t * quant_a_ptr);

शून्य quantize_a_4row_i8(size_t blk_len, ध्रुव भिन्न * a_ptr, size_t गणनम्_k, uint8_t * quant_a_ptr);

शून्य quantize_a_row_i8_hp(size_t blk_len, ध्रुव भिन्न * a_ptr, size_t गणनम्_k, uint8_t * quant_a_ptr);

शून्य quantize_a_4row_i8_hp(size_t blk_len, ध्रुव भिन्न * a_ptr, size_t गणनम्_k, uint8_t * quant_a_ptr);

शून्य quantize_a_row_i8k(size_t blk_len, ध्रुव भिन्न * a_ptr, size_t गणनम्_k, uint8_t * quant_a_ptr);

शून्य quantize_a_4row_i8k(size_t blk_len, ध्रुव भिन्न * a_ptr, size_t गणनम्_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
