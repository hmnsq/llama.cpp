#आशय once

#समावेश "ggml-cpu-impl.h"

#समावेश <cassert>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <functional>

namespace spacemit_kernels {

constexpr स्वतः div_round_up(स्वतः up, स्वतः down) {
    वापस (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr माप_प्रकार q8_blk_आकार(माप_प्रकार blk_len, bool with_blk_योग = false) {
    स्थिर माप_प्रकार blk_आकार = माप(भग्न) + blk_len * माप(int8_t) + (with_blk_योग ? माप(int16_t) : 0);
    वापस blk_आकार;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr माप_प्रकार q8_hp_blk_आकार(माप_प्रकार blk_len, bool with_blk_योग = false, bool with_blk_scale = false) {
    स्थिर माप_प्रकार subblk_गिनती = div_round_up(blk_len, माप_प्रकार(32));
    स्थिर माप_प्रकार blk_आकार     = blk_len * माप(int8_t) + subblk_गिनती * माप(_Float16) +
                            (with_blk_योग ? subblk_गिनती * माप(_Float16) : 0) +
                            (with_blk_scale ? माप(_Float16) : 0);
    वापस blk_आकार;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr माप_प्रकार q8k_blk_आकार(माप_प्रकार blk_len) {
    स्थिर माप_प्रकार blk_आकार = माप(भग्न) + blk_len * माप(int8_t) + माप(int16_t) * blk_len / 16;
    वापस blk_आकार;
}

using quantize_a_row_def = std::function<व्योम(माप_प्रकार, स्थिर भग्न *, माप_प्रकार, uint8_t *)>;

namespace rvv {
व्योम memcpy1d(व्योम * dst, स्थिर व्योम * src, int64_t आकार);

व्योम memcpy2d(व्योम * dst, int64_t dst_stride, स्थिर व्योम * src, int64_t src_stride, int64_t tile_rows, int64_t आकार);

व्योम forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(स्थिर ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        पूर्णांक                         ir0,
                                                        पूर्णांक                         ir1,
                                                        व्योम *                      tcm_बफर,
                                                        माप_प्रकार                      tcm_बफर_आकार);

व्योम forward_flash_attn_ext_f16_tiled_vlen1024_vf16(स्थिर ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    पूर्णांक                         ir0,
                                                    पूर्णांक                         ir1,
                                                    व्योम *                      tcm_बफर,
                                                    माप_प्रकार                      tcm_बफर_आकार);

व्योम forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

व्योम forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

व्योम forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

व्योम forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> व्योम forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> व्योम forward_concat(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <ggml_op op_type, typename T> व्योम forward_द्विआधारी(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> व्योम forward_योग_rows(स्थिर ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> व्योम forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

ढाँचा <typename T> व्योम forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

व्योम quantize_a_row_i8(माप_प्रकार blk_len, स्थिर भग्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

व्योम quantize_a_4row_i8(माप_प्रकार blk_len, स्थिर भग्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

व्योम quantize_a_row_i8_hp(माप_प्रकार blk_len, स्थिर भग्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

व्योम quantize_a_4row_i8_hp(माप_प्रकार blk_len, स्थिर भग्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

व्योम quantize_a_row_i8k(माप_प्रकार blk_len, स्थिर भग्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

व्योम quantize_a_4row_i8k(माप_प्रकार blk_len, स्थिर भग्न * a_ptr, माप_प्रकार गिनती_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
