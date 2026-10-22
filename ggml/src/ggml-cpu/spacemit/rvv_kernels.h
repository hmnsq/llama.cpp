#ہدایت once

#شامل "ggml-cpu-impl.h"

#شامل <cassert>
#شامل <cstddef>
#شامل <cstdint>
#شامل <functional>

namespace spacemit_kernels {

constexpr خودکار div_باری_اوپر(خودکار اوپر, خودکار نیچے_کو) {
    واپس (اوپر + نیچے_کو - 1) / نیچے_کو;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_سائز(size_t blk_len, bool with_blk_مجموع = false) {
    ثابت size_t blk_سائز = حجم(اعشاری) + blk_len * حجم(int8_t) + (with_blk_مجموع ? حجم(int16_t) : 0);
    واپس blk_سائز;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_سائز(size_t blk_len, bool with_blk_مجموع = false, bool with_blk_scale = false) {
    ثابت size_t subblk_گنتی = div_باری_اوپر(blk_len, size_t(32));
    ثابت size_t blk_سائز     = blk_len * حجم(int8_t) + subblk_گنتی * حجم(_Float16) +
                            (with_blk_مجموع ? subblk_گنتی * حجم(_Float16) : 0) +
                            (with_blk_scale ? حجم(_Float16) : 0);
    واپس blk_سائز;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_سائز(size_t blk_len) {
    ثابت size_t blk_سائز = حجم(اعشاری) + blk_len * حجم(int8_t) + حجم(int16_t) * blk_len / 16;
    واپس blk_سائز;
}

using quantize_a_صف_def = std::function<خالی(size_t, ثابت اعشاری *, size_t, uint8_t *)>;

namespace rvv {
خالی memcpy1d(خالی * dst, ثابت خالی * src, int64_t سائز);

خالی memcpy2d(خالی * dst, int64_t dst_stride, ثابت خالی * src, int64_t src_stride, int64_t tile_صفیں, int64_t سائز);

خالی forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        صحیح                         ir0,
                                                        صحیح                         ir1,
                                                        خالی *                      tcm_بفر,
                                                        size_t                      tcm_بفر_سائز);

خالی forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    صحیح                         ir0,
                                                    صحیح                         ir1,
                                                    خالی *                      tcm_بفر,
                                                    size_t                      tcm_بفر_سائز);

خالی forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

خالی forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

خالی forward_cont_with_ترتیب_بدلو(ggml_compute_params * params, ggml_tensor * op);

خالی forward_cpy_with_ترتیب_بدلو(ggml_compute_params * params, ggml_tensor * op);

سانچہ <typename T> خالی forward_get_صفیں(ggml_compute_params * params, ggml_tensor * op);

سانچہ <typename T> خالی forward_concat(ggml_compute_params * params, ggml_tensor * op);

سانچہ <ggml_op op_type, typename T> خالی forward_ثنائی(ggml_compute_params * params, ggml_tensor * op);

سانچہ <typename T> خالی forward_مجموع_صفیں(ثابت ggml_compute_params * params, ggml_tensor * op);

سانچہ <typename T> خالی forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

سانچہ <typename T> خالی forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

خالی quantize_a_صف_i8(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

خالی quantize_a_4row_i8(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

خالی quantize_a_صف_i8_hp(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

خالی quantize_a_4row_i8_hp(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

خالی quantize_a_صف_i8k(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

خالی quantize_a_4row_i8k(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
