#సూచన once

#అన్తర్భావ "ggml-cpu-impl.h"

#అన్తర్భావ <cassert>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <functional>

namespace spacemit_kernels {

constexpr స్వయంచాలక div_విడత_పైకి(స్వయంచాలక పైకి, స్వయంచాలక కిందికి) {
    తిరిగి (పైకి + కిందికి - 1) / కిందికి;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_కొలత(size_t blk_len, bool with_blk_కూడిక = false) {
    స్థిరరాశి size_t blk_కొలత = పరిమాణం(దశాంశం) + blk_len * పరిమాణం(int8_t) + (with_blk_కూడిక ? పరిమాణం(int16_t) : 0);
    తిరిగి blk_కొలత;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_కొలత(size_t blk_len, bool with_blk_కూడిక = false, bool with_blk_scale = false) {
    స్థిరరాశి size_t subblk_లెక్క = div_విడత_పైకి(blk_len, size_t(32));
    స్థిరరాశి size_t blk_కొలత     = blk_len * పరిమాణం(int8_t) + subblk_లెక్క * పరిమాణం(_Float16) +
                            (with_blk_కూడిక ? subblk_లెక్క * పరిమాణం(_Float16) : 0) +
                            (with_blk_scale ? పరిమాణం(_Float16) : 0);
    తిరిగి blk_కొలత;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_కొలత(size_t blk_len) {
    స్థిరరాశి size_t blk_కొలత = పరిమాణం(దశాంశం) + blk_len * పరిమాణం(int8_t) + పరిమాణం(int16_t) * blk_len / 16;
    తిరిగి blk_కొలత;
}

using quantize_a_అడ్డువరుస_def = std::function<శూన్య(size_t, స్థిరరాశి దశాంశం *, size_t, uint8_t *)>;

namespace rvv {
శూన్య memcpy1d(శూన్య * dst, స్థిరరాశి శూన్య * src, int64_t కొలత);

శూన్య memcpy2d(శూన్య * dst, int64_t dst_stride, స్థిరరాశి శూన్య * src, int64_t src_stride, int64_t tile_అడ్డువరుసలు, int64_t కొలత);

శూన్య forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(స్థిరరాశి ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        పూర్ణాంకం                         ir0,
                                                        పూర్ణాంకం                         ir1,
                                                        శూన్య *                      tcm_బఫరు,
                                                        size_t                      tcm_బఫరు_కొలత);

శూన్య forward_flash_attn_ext_f16_tiled_vlen1024_vf16(స్థిరరాశి ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    పూర్ణాంకం                         ir0,
                                                    పూర్ణాంకం                         ir1,
                                                    శూన్య *                      tcm_బఫరు,
                                                    size_t                      tcm_బఫరు_కొలత);

శూన్య forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

శూన్య forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

శూన్య forward_cont_with_క్రమమార్పు(ggml_compute_params * params, ggml_tensor * op);

శూన్య forward_cpy_with_క్రమమార్పు(ggml_compute_params * params, ggml_tensor * op);

మూస <typename T> శూన్య forward_get_అడ్డువరుసలు(ggml_compute_params * params, ggml_tensor * op);

మూస <typename T> శూన్య forward_concat(ggml_compute_params * params, ggml_tensor * op);

మూస <ggml_op op_type, typename T> శూన్య forward_ద్విమాన(ggml_compute_params * params, ggml_tensor * op);

మూస <typename T> శూన్య forward_కూడిక_అడ్డువరుసలు(స్థిరరాశి ggml_compute_params * params, ggml_tensor * op);

మూస <typename T> శూన్య forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

మూస <typename T> శూన్య forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

శూన్య quantize_a_అడ్డువరుస_i8(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr);

శూన్య quantize_a_4row_i8(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr);

శూన్య quantize_a_అడ్డువరుస_i8_hp(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr);

శూన్య quantize_a_4row_i8_hp(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr);

శూన్య quantize_a_అడ్డువరుస_i8k(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr);

శూన్య quantize_a_4row_i8k(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
