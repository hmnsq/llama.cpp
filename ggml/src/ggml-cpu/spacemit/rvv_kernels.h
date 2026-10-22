#رهنمود once

#ضمن "ggml-cpu-impl.h"

#ضمن <cassert>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <functional>

namespace spacemit_kernels {

constexpr خودکار div_دور_به_بالا(خودکار به_بالا, خودکار پایین) {
    أرجع (به_بالا + پایین - 1) / پایین;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_اندازه(size_t blk_len, bool with_blk_جمع = false) {
    ثابت size_t blk_اندازه = حجم(عائم) + blk_len * حجم(int8_t) + (with_blk_جمع ? حجم(int16_t) : 0);
    أرجع blk_اندازه;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_اندازه(size_t blk_len, bool with_blk_جمع = false, bool with_blk_scale = false) {
    ثابت size_t subblk_شمار = div_دور_به_بالا(blk_len, size_t(32));
    ثابت size_t blk_اندازه     = blk_len * حجم(int8_t) + subblk_شمار * حجم(_Float16) +
                            (with_blk_جمع ? subblk_شمار * حجم(_Float16) : 0) +
                            (with_blk_scale ? حجم(_Float16) : 0);
    أرجع blk_اندازه;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_اندازه(size_t blk_len) {
    ثابت size_t blk_اندازه = حجم(عائم) + blk_len * حجم(int8_t) + حجم(int16_t) * blk_len / 16;
    أرجع blk_اندازه;
}

using quantize_a_ردیف_def = std::function<فراغ(size_t, ثابت عائم *, size_t, uint8_t *)>;

namespace rvv {
فراغ memcpy1d(فراغ * dst, ثابت فراغ * src, int64_t اندازه);

فراغ memcpy2d(فراغ * dst, int64_t dst_stride, ثابت فراغ * src, int64_t src_stride, int64_t tile_ردیفها, int64_t اندازه);

فراغ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        صحيح                         ir0,
                                                        صحيح                         ir1,
                                                        فراغ *                      tcm_میانگیر,
                                                        size_t                      tcm_میانگیر_اندازه);

فراغ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    صحيح                         ir0,
                                                    صحيح                         ir1,
                                                    فراغ *                      tcm_میانگیر,
                                                    size_t                      tcm_میانگیر_اندازه);

فراغ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

فراغ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

فراغ forward_cont_with_جایگشت(ggml_compute_params * params, ggml_tensor * op);

فراغ forward_cpy_with_جایگشت(ggml_compute_params * params, ggml_tensor * op);

قالب <typename T> فراغ forward_get_ردیفها(ggml_compute_params * params, ggml_tensor * op);

قالب <typename T> فراغ forward_concat(ggml_compute_params * params, ggml_tensor * op);

قالب <ggml_op op_type, typename T> فراغ forward_دودویی(ggml_compute_params * params, ggml_tensor * op);

قالب <typename T> فراغ forward_جمع_ردیفها(ثابت ggml_compute_params * params, ggml_tensor * op);

قالب <typename T> فراغ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

قالب <typename T> فراغ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

فراغ quantize_a_ردیف_i8(size_t blk_len, ثابت عائم * a_ptr, size_t شمار_k, uint8_t * quant_a_ptr);

فراغ quantize_a_4row_i8(size_t blk_len, ثابت عائم * a_ptr, size_t شمار_k, uint8_t * quant_a_ptr);

فراغ quantize_a_ردیف_i8_hp(size_t blk_len, ثابت عائم * a_ptr, size_t شمار_k, uint8_t * quant_a_ptr);

فراغ quantize_a_4row_i8_hp(size_t blk_len, ثابت عائم * a_ptr, size_t شمار_k, uint8_t * quant_a_ptr);

فراغ quantize_a_ردیف_i8k(size_t blk_len, ثابت عائم * a_ptr, size_t شمار_k, uint8_t * quant_a_ptr);

فراغ quantize_a_4row_i8k(size_t blk_len, ثابت عائم * a_ptr, size_t شمار_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
