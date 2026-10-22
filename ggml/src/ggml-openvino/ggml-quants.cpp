#অন্তর্ভাব "ggml-quants.h"

#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-openvino-extra.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <limits>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/except.hpp>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/node_output.hpp>
#অন্তর্ভাব <openvino/core/parallel.hpp>
#অন্তর্ভাব <openvino/core/shape.hpp>
#অন্তর্ভাব <openvino/core/type/element_type.hpp>
#অন্তর্ভাব <openvino/core/type/element_type_traits.hpp>
#অন্তর্ভাব <openvino/core/type/float16.hpp>
#অন্তর্ভাব <openvino/core/type/float4_e2m1.hpp>
#অন্তর্ভাব <openvino/core/type/float8_e8m0.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/subtract.hpp>
#অন্তর্ভাব <openvino/op/util/attr_types.hpp>
#অন্তর্ভাব <openvino/pass/constant_folding.hpp>
#অন্তর্ভাব <openvino/runtime/tensor.hpp>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
নামাকাশ ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
যৌক্তিক get_single_মান(ধ্রুবক std::shared_ptr<ov::op::v0::Constant> & const_গ্রন্থি,
                      দশমিক & value,
                      যৌক্তিক check_মান_range = সত্য);
}  // namespace ov::op::util

নামাকাশ {
শূন্য unpack_32_4(ধ্রুবক uint8_t * তথ্য, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    জন্য (পূর্ণসংখ্যা j = 0; j < 16; ++j) {
        uint8_t x = (তথ্য[j] & 0x0F);
        uint8_t y = (তথ্য[j] >> 4);
        যদি (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_BLOCK_আকার = 32;
constexpr size_t MXFP4_BLOCK_QS_আকার = MXFP4_BLOCK_আকার / 2;
constexpr size_t MXFP4_BLOCK_BYTES = আকার(uint8_t) + MXFP4_BLOCK_QS_আকার;

শূন্য pack_32_mxfp4_for_openvino(ধ্রুবক uint8_t * তথ্য, uint8_t * dst) {
    জন্য (পূর্ণসংখ্যা j = 0; j < static_cast<পূর্ণসংখ্যা>(MXFP4_BLOCK_QS_আকার); j += 2) {
        ধ্রুবক uint8_t v0 = তথ্য[j] & 0x0F;
        ধ্রুবক uint8_t v1 = (তথ্য[j + 1] & 0x0F) << 4;
        ধ্রুবক uint8_t v16 = তথ্য[j] >> 4;
        ধ্রুবক uint8_t v17 = তথ্য[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_BLOCK_আকার / 4 + j / 2] = v16 | v17;
    }
}

শূন্য extract_mxfp4_তথ্য(ধ্রুবক ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ধ্রুবক auto * তথ্য = static_cast<ধ্রুবক uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ধ্রুবক size_t n_blocks = scales_arr.get_আকার();

    ov::parallel_for(n_blocks, [&](size_t i) {
        ধ্রুবক uint8_t * block = তথ্য + i * MXFP4_BLOCK_BYTES;
        pack_32_mxfp4_for_openvino(block + আকার(uint8_t), weights + i * MXFP4_BLOCK_QS_আকার);
        scales[i] = ov::float8_e8m0::from_bits(block[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
শূন্য extract_q4_0_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ধ্রুবক uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    যৌক্তিক is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    যদি (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (তথ্য + i * bytes_per_block)));
            // Pack two 4-bit zero points per byte
            যদি (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } অন্যথা {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(তথ্য + i * bytes_per_block + 2, weights + i * 16);
        });
    } অন্যথা {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (তথ্য + i * bytes_per_block)));
            unpack_32_4(তথ্য + i * bytes_per_block + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            জন্য (পূর্ণসংখ্যা j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
শূন্য extract_q4_1_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       যৌক্তিক use_bias) {
    ধ্রুবক uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    যদি (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        auto * bias = zp_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            দশমিক scale = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) (তথ্য + i * bytes_per_block))));
            দশমিক min = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) (তথ্য + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(তথ্য + i * bytes_per_block + 4, weights + i * 16);
        });
    } অন্যথা {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            দশমিক scale = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) (তথ্য + i * bytes_per_block))));
            দশমিক min = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) (তথ্য + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            যদি (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } অন্যথা {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(তথ্য + i * bytes_per_block + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
শূন্য extract_q5_1_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       যৌক্তিক use_bias) {
    ধ্রুবক uint64_t bytes_per_block = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ধ্রুবক পূর্ণসংখ্যা qk = 32;

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());  // u8 weights, one byte per weight
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    auto read_u16 = [](ধ্রুবক uint8_t * p) {
        uint16_t v;
        memcpy(&v, p, আকার(v));
        ফেরত v;
    };

    auto unpack_block = [&](ধ্রুবক uint8_t * block, uint8_t * dst) {
        uint32_t qh;
        memcpy(&qh, block + 4, আকার(uint32_t));
        ধ্রুবক uint8_t * qs = block + 8;
        জন্য (পূর্ণসংখ্যা j = 0; j < qk / 2; ++j) {
            ধ্রুবক uint8_t lo = qs[j] & 0x0F;
            ধ্রুবক uint8_t hi = qs[j] >> 4;
            ধ্রুবক uint8_t bit_lo = (qh >> j) & 1;
            ধ্রুবক uint8_t bit_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (bit_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (bit_hi << 4);  // last 16 weights
        }
    };

    যদি (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        auto * bias = zp_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            ধ্রুবক uint8_t * block = তথ্য + i * bytes_per_block;
            দশমিক scale = static_cast<দশমিক>(ov::float16::from_bits(read_u16(block)));
            দশমিক min = static_cast<দশমিক>(ov::float16::from_bits(read_u16(block + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_block(block, weights + i * qk);
        });
    } অন্যথা {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());  // u8 zero points
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            ধ্রুবক uint8_t * block = তথ্য + i * bytes_per_block;
            দশমিক scale = static_cast<দশমিক>(ov::float16::from_bits(read_u16(block)));
            দশমিক min = static_cast<দশমিক>(ov::float16::from_bits(read_u16(block + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_block(block, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
শূন্য extract_q8_0_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ধ্রুবক uint64_t weights_per_block = 32;
    ধ্রুবক uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    যৌক্তিক is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    যদি (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            uint8_t * block_তথ্য = তথ্য + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_তথ্য);
            zp[i] = 128;
            জন্য (size_t j = 0; j < weights_per_block; ++j) {
                uint8_t x = block_তথ্য[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_per_block + j] = x;
            }
        });
    } অন্যথা {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_আকার(), [&](size_t i) {
            uint8_t * block_তথ্য = তথ্য + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_তথ্য);
            // Copy int8 weights as-is (the tensor element type is i8)
            memcpy(weights + i * weights_per_block, block_তথ্য + 2, weights_per_block);
        });
    }
}

শূন্য unpack_256_4(ধ্রুবক uint8_t * তথ্য, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    জন্য (size_t i = 0; i < 4; ++i) {
        জন্য (পূর্ণসংখ্যা j = 0; j < 32; ++j) {
            uint8_t x = (তথ্য[i * 32 + j] & 0x0F);
            uint8_t y = (তথ্য[i * 32 + j] >> 4);
            যদি (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

শূন্য extract_q4_k_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       যৌক্তিক use_bias) {
    ধ্রুবক uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    ধ্রুবক uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    auto * zp_u4 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.তথ্য());
    auto * bias_f16 = use_bias ? zp_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_তথ্য = তথ্য + i * bytes_per_block;

        // Extract scale factors and offsets
        দশমিক scale_scales = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) block_তথ্য)));
        দশমিক scale_mins = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) block_তথ্য + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = block_তথ্য + 4;

        // Calculate scales
        দশমিক scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<দশমিক>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<দশমিক>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<দশমিক>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<দশমিক>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<দশমিক>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<দশমিক>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<দশমিক>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<দশমিক>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        দশমিক min_vals[8];
        min_vals[0] = scale_mins * static_cast<দশমিক>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<দশমিক>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<দশমিক>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<দশমিক>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<দশমিক>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<দশমিক>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<দশমিক>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<দশমিক>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            যদি (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } অন্যথা {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                যদি (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } অন্যথা {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(block_তথ্য + 16, weights + i * 128);
    });
}

শূন্য extract_q6_k_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ধ্রুবক uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    ধ্রুবক uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    যৌক্তিক is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    যদি (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_তথ্য = তথ্য + i * bytes_per_block;
            দশমিক scale_factor = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) block_তথ্য + 104)));
            জন্য (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<দশমিক>(*((int8_t *) (block_তথ্য + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = block_তথ্য;
            uint8_t * qh = block_তথ্য + 128;
            জন্য (int64_t j = 0; j < 32; ++j) {
                weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
                weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
                weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
                weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
                weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
                weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
                weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
                weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
            }
        });
    } অন্যথা {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_তথ্য = তথ্য + i * bytes_per_block;
            দশমিক scale_factor = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) block_তথ্য + 104)));
            জন্য (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<দশমিক>(*((int8_t *) (block_তথ্য + 128 + 64 + j))));
            }
            uint8_t * ql = block_তথ্য;
            uint8_t * qh = block_তথ্য + 128;
            auto * signed_weights = reinterpret_cast<int8_t *>(weights);
            জন্য (int64_t j = 0; j < 32; ++j) {
                signed_weights[i * 256 + j] = static_cast<int8_t>((ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 32] =
                    static_cast<int8_t>((ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 64] = static_cast<int8_t>((ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 96] =
                    static_cast<int8_t>((ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 128] =
                    static_cast<int8_t>((ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 160] =
                    static_cast<int8_t>((ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 192] =
                    static_cast<int8_t>((ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 224] =
                    static_cast<int8_t>((ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4)) - 32;
            }
        });
    }
}

অন্তঃরেখ শূন্য get_scale_min_k4(পূর্ণসংখ্যা j, ধ্রুবক uint8_t * q, uint8_t * d, uint8_t * m) {
    যদি (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } অন্যথা {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

শূন্য extract_q5_k_তথ্য(ধ্রুবক ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       যৌক্তিক use_bias) {
    ধ্রুবক uint64_t bytes_per_block = 4 + 12 + 32 + 128;
    ধ্রুবক uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * তথ্য = static_cast<uint8_t *>(tensor->তথ্য);
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    auto * zp_u8 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.তথ্য());
    auto * bias_f16 = use_bias ? zp_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_তথ্য = তথ্য + i * bytes_per_block;

        ধ্রুবক দশমিক d = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) block_তথ্য)));
        ধ্রুবক দশমিক min_factor = static_cast<দশমিক>(ov::float16::from_bits(*((uint16_t *) block_তথ্য + 1)));

        ধ্রুবক uint8_t * scales_তথ্য = block_তথ্য + 4;   // 12 bytes of scales
        ধ্রুবক uint8_t * qh = block_তথ্য + 4 + 12;       // 32 bytes of high bits
        ধ্রুবক uint8_t * ql = block_তথ্য + 4 + 12 + 32;  // 128 bytes of low bits

        পূর্ণসংখ্যা is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        জন্য (পূর্ণসংখ্যা j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_তথ্য, &sc, &m);
            ধ্রুবক দশমিক d1 = d * sc;
            ধ্রুবক দশমিক m1 = min_factor * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_তথ্য, &sc, &m);
            ধ্রুবক দশমিক d2 = d * sc;
            ধ্রুবক দশমিক m2 = min_factor * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            যদি (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } অন্যথা {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            জন্য (পূর্ণসংখ্যা l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            জন্য (পূর্ণসংখ্যা l = 0; l < 32; ++l) {
                weights[i * 256 + j + l + 32] = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}

// TODO Reorder for make_intX_weights

// If for_gather_matmul is true, weight may be N-D (e.g. 3D MoE expert weights [n_expert, rows, cols]).
// The dequantization chain below is built as usual but left in f16 (no final Convert to f32) --
// ov::pass::MarkDequantization (registered in translate_session.cpp) marks the chain so it survives
// model-build-time ConstantFolding. mul_mat_id.cpp constructs ov::op::internal::GatherMatmul directly
// on top of the resulting f16 chain.
ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_আকার = GGML_QUANTIZATION_GROUP_আকার,
                                       যৌক্তিক use_bias = মিথ্যা,
                                       যৌক্তিক for_gather_matmul = মিথ্যা) {
    ov::Shape orig_shape = weight.get_shape();
    যৌক্তিক is_signed = (weight.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    auto scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= group_আকার;
    packed_shape.push_back(group_আকার);
    ধ্রুবক size_t group_dim = packed_shape.size() - 2;

    যদি (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } অন্যথা {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        যদি (!is_signed && zp.get_আকার() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ফলাফল;
    যদি (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.তথ্য()), nullptr);
        weights_গ্রন্থি->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_গ্রন্থি, ov::element::f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        ফলাফল = mul;
    } অন্যথা {
        // Unsigned path
        auto weights_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.তথ্য()), nullptr);
        weights_গ্রন্থি->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_গ্রন্থি, ov::element::f16);

        যদি (use_bias && zp.get_আকার() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            auto * bias_zp_তথ্য = zp.তথ্য<ov::float16>();
            ধ্রুবক auto * scale_তথ্য = scales.তথ্য<ov::float16>();
            ধ্রুবক size_t n = zp.get_আকার();
            জন্য (size_t i = 0; i < n; i++) {
                দশমিক s = static_cast<দশমিক>(scale_তথ্য[i]);
                দশমিক b = static_cast<দশমিক>(bias_zp_তথ্য[i]);
                bias_zp_তথ্য[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            ফলাফল = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } অন্যথা {
            // Zero point path: (w - zp) * s
            auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
            দশমিক zp_মান;
            যদি (ov::op::util::get_single_মান(zero_point, zp_মান)) {
                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_মান});
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            auto mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            ফলাফল = mul;
        }
    }

    যদি (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(ফলাফল, final_shape, মিথ্যা);
        ফলাফল = reshaped;
    }

    যদি (for_gather_matmul) {
        ফেরত ফলাফল;
    }
    ফেরত std::make_shared<ov::op::v0::Convert>(ফলাফল, ov::element::f32);
}

// If for_gather_matmul is true, the weight tensor may be N-D (e.g. 3D MoE expert weights
// [n_expert, rows, cols]). The dequantization chain (Convert->[Subtract]->Multiply) is built as
// usual but left in f16 (no final Convert to f32) -- ov::pass::MarkDequantization (registered in
// translate_session.cpp) marks the chain so it survives model-build-time ConstantFolding -- see
// make_int8_weights.cpp/make_int4_weights.cpp. mul_mat_id.cpp constructs ov::op::internal::GatherMatmul
// directly from the resulting f16 dequant chain.
//
// When use_bias is true (explicitly, or implicitly because for_gather_matmul is true), the zp
// tensor is expected to hold an exact f16 bias value (rather than a rounded integer zero point);
// it is converted in place into an exact zero_point = -bias/scale and consumed via Subtract, not
// Add, so the chain still matches OpenVINO's Convert->Subtract->Multiply decompression pattern.
// See make_int8_weights for the meaning of for_gather_matmul.
ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_আকার = GGML_QUANTIZATION_GROUP_আকার,
                                       যৌক্তিক use_bias = মিথ্যা,
                                       যৌক্তিক for_gather_matmul = মিথ্যা) {
    ov::Shape orig_weight_shape = weight.get_shape();
    যৌক্তিক is_signed = (weight.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_weight_shape;
    packed_shape.back() /= group_আকার;
    packed_shape.push_back(group_আকার);
    ধ্রুবক size_t group_dim = packed_shape.size() - 2;

    যদি (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } অন্যথা {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        যদি (!is_signed && zp.get_আকার() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ফলাফল;
    যদি (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.তথ্য()), nullptr);
        weights_গ্রন্থি->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_গ্রন্থি, ov::element::f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        ফলাফল = mul;
    } অন্যথা {
        // Unsigned path
        auto weights_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.তথ্য()), nullptr);
        weights_গ্রন্থি->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_গ্রন্থি, ov::element::f16);

        যদি (use_bias && zp.get_আকার() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            auto * bias_zp_তথ্য = zp.তথ্য<ov::float16>();
            ধ্রুবক auto * scale_তথ্য = scales.তথ্য<ov::float16>();
            ধ্রুবক size_t n = zp.get_আকার();
            জন্য (size_t i = 0; i < n; i++) {
                দশমিক s = static_cast<দশমিক>(scale_তথ্য[i]);
                দশমিক b = static_cast<দশমিক>(bias_zp_তথ্য[i]);
                bias_zp_তথ্য[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            auto zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            ফলাফল = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } অন্যথা {
            // Zero point path: (w - zp) * s
            auto zero_points_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(zp);
            দশমিক zp_মান;
            যদি (ov::op::util::get_single_মান(zero_points_গ্রন্থি, zp_মান)) {
                zero_points_গ্রন্থি = ov::op::v0::Constant::create(zero_points_গ্রন্থি->get_element_type(), {}, {zp_মান});
            }
            auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_গ্রন্থি, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            auto mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            ফলাফল = mul;
        }
    }

    যদি (packed_shape.size() != orig_weight_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_weight_shape.size()},
                                                                  orig_weight_shape);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(ফলাফল, final_shape, মিথ্যা);
        ফলাফল = reshaped;
    }

    যদি (for_gather_matmul) {
        ফেরত ফলাফল;
    }
    ফেরত std::make_shared<ov::op::v0::Convert>(ফলাফল, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & weight, ov::Tensor & scales) {
    ধ্রুবক ov::Shape final_shape = weight.get_shape();
    GGML_ASSERT(!final_shape.empty());
    GGML_ASSERT(final_shape.back() % MXFP4_BLOCK_আকার == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_BLOCK_আকার;
    packed_shape.push_back(MXFP4_BLOCK_আকার);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.set_shape(scale_shape);

    auto weights_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(weight.তথ্য()), nullptr);
    weights_গ্রন্থি->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_গ্রন্থি, ov::element::f32);

    auto scales_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(scales);
    auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_গ্রন্থি, ov::element::f32);
    ov::Output<ov::Node> ফলাফল =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::AutoBroadcastType::NUMPY);

    auto final_shape_গ্রন্থি =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    ফেরত std::make_shared<ov::op::v1::Reshape>(ফলাফল, final_shape_গ্রন্থি, মিথ্যা);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & weight) {
    auto weights_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::u8, weight.get_shape(),
                                                               static_cast<uint8_t *>(weight.তথ্য()), nullptr);
    weights_গ্রন্থি->get_rt_info()["__gguf_tensor_holder"] = weight;
    weights_গ্রন্থি->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = সত্য;
    ফেরত weights_গ্রন্থি;
}

শূন্য quantize_q4_0(ধ্রুবক দশমিক * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    প্রতিজ্ঞা(k % qk == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();
    যৌক্তিক is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    যদি (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());
        জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
            দশমিক amax = 0.0f;
            দশমিক max = 0.0f;
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; j++) {
                ধ্রুবক দশমিক v = x[i * qk + j];
                যদি (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ধ্রুবক দশমিক d = max / -8;
            যদি (d == 0) {
                scales[i] = ov::float16(1.0f);
                যদি (i % 2 == 0) {
                    zp[i / 2] = 8;
                } অন্যথা {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                চালু;
            }
            ধ্রুবক দশমিক id = 1.0f / d;
            scales[i] = ov::float16(d);
            যদি (i % 2 == 0) {
                zp[i / 2] = 8;
            } অন্যথা {
                zp[i / 2] |= (8 << 4);
            }
            জন্য (পূর্ণসংখ্যা j = 0; j < qk / 2; ++j) {
                ধ্রুবক দশমিক x0 = x[i * qk + 2 * j] * id;
                ধ্রুবক দশমিক x1 = x[i * qk + 2 * j + 1] * id;
                ধ্রুবক uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                ধ্রুবক uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } অন্যথা {
        // Symmetric: produce signed i4 values in [-8, 7]
        জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
            দশমিক amax = 0.0f;
            দশমিক max = 0.0f;
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; j++) {
                ধ্রুবক দশমিক v = x[i * qk + j];
                যদি (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ধ্রুবক দশমিক d = max / -8;
            যদি (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                চালু;
            }
            ধ্রুবক দশমিক id = 1.0f / d;
            scales[i] = ov::float16(d);
            জন্য (পূর্ণসংখ্যা j = 0; j < qk / 2; ++j) {
                ধ্রুবক দশমিক x0 = x[i * qk + 2 * j] * id;
                ধ্রুবক দশমিক x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (পূর্ণসংখ্যা) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (পূর্ণসংখ্যা) roundf(x1)));
                weights[i * qk / 2 + j] = (si0 & 0x0F) | ((si1 & 0x0F) << 4);
            }
        }
    }
}

// Asymmetric u4 quantization with a per-group scale and zero point.
//
// Unlike quantize_q4_0's unsigned branch, which pins the zero point to 8 and is therefore
// symmetric, this keeps a real per-group zero point, so a group whose values are not centred on
// zero does not waste half its range.
শূন্য quantize_q4_1_asym(ধ্রুবক দশমিক * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    প্রতিজ্ঞা(k % qk == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য());
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    auto store_zp = [zp](পূর্ণসংখ্যা i, uint8_t v) {
        যদি (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } অন্যথা {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        দশমিক vmin = x[i * qk];
        দশমিক vmax = x[i * qk];
        জন্য (পূর্ণসংখ্যা j = 1; j < qk; j++) {
            ধ্রুবক দশমিক v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        ধ্রুবক দশমিক d = (vmax - vmin) / 15.0f;
        যদি (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            চালু;
        }
        ধ্রুবক দশমিক id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        ধ্রুবক পূর্ণসংখ্যা zq = std::max(0, std::min(15, (পূর্ণসংখ্যা) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        জন্য (পূর্ণসংখ্যা j = 0; j < qk / 2; ++j) {
            ধ্রুবক দশমিক x0 = x[i * qk + 2 * j] * id;
            ধ্রুবক দশমিক x1 = x[i * qk + 2 * j + 1] * id;
            ধ্রুবক uint8_t q0 = (uint8_t) std::max(0, std::min(15, (পূর্ণসংখ্যা) lroundf(x0) + zq));
            ধ্রুবক uint8_t q1 = (uint8_t) std::max(0, std::min(15, (পূর্ণসংখ্যা) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

শূন্য quantize_q8_0(ধ্রুবক দশমিক * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0) {
    প্রতিজ্ঞা(k % qk == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য()) + block_offset * qk;
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    যৌক্তিক is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    যদি (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য()) + block_offset;
        জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
            দশমিক amax = 0.0f;
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; j++) {
                ধ্রুবক দশমিক v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ধ্রুবক দশমিক d = amax / 127.0f;
            ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; ++j) {
                ধ্রুবক দশমিক x0 = x[i * qk + j] * id;
                ধ্রুবক int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } অন্যথা {
        // Symmetric: store signed int8 values directly
        auto * signed_weights = reinterpret_cast<int8_t *>(weights);
        জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
            দশমিক amax = 0.0f;
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; j++) {
                ধ্রুবক দশমিক v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ধ্রুবক দশমিক d = amax / 127.0f;
            ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; ++j) {
                ধ্রুবক দশমিক x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

শূন্য quantize_q8_1(ধ্রুবক দশমিক * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0) {
    প্রতিজ্ঞা(k % qk == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    auto * weights = static_cast<uint8_t *>(weights_arr.তথ্য()) + block_offset * qk;
    auto * scales = scales_arr.তথ্য<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    auto * zp = static_cast<uint8_t *>(zp_arr.তথ্য()) + block_offset;
    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        দশমিক min = std::numeric_limits<দশমিক>::max();
        দশমিক max = std::numeric_limits<দশমিক>::lowest();

        জন্য (পূর্ণসংখ্যা j = 0; j < qk; j++) {
            ধ্রুবক দশমিক v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        ধ্রুবক দশমিক d = (max - min) / ((1 << 8) - 1);
        ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        জন্য (পূর্ণসংখ্যা j = 0; j < qk; ++j) {
            ধ্রুবক দশমিক x0 = (x[i * qk + j] - min) * id;
            ধ্রুবক uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> extract_quantized_weights(ধ্রুবক ggml_tensor * tensor,
                                                    ধ্রুবক শূন্য * তথ্য,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    যৌক্তিক use_bias = মিথ্যা) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor অস্থায়ী_tensor = *tensor;
    অস্থায়ী_tensor.তথ্য = const_cast<শূন্য *>(তথ্য);

    যদি (tensor->type == GGML_TYPE_MXFP4) {
        extract_mxfp4_তথ্য(&অস্থায়ী_tensor, weights, scales);
        auto ফলাফল = make_mxfp4_weights(weights, scales).get_গ্রন্থি_shared_ptr();
        ফলাফল->set_friendly_name(tensor->name);
        ফেরত ফলাফল;
    }

    // Determine block size based on tensor type
    int64_t weights_per_block;
    যৌক্তিক is_u4;
    নির্বাচন (tensor->type) {
    ক্ষেত্র GGML_TYPE_Q4_0:
    ক্ষেত্র GGML_TYPE_Q4_1:
    ক্ষেত্র GGML_TYPE_Q4_K:
        is_u4 = সত্য;
        weights_per_block = 32;
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q8_0:
    ক্ষেত্র GGML_TYPE_Q5_1:
    ক্ষেত্র GGML_TYPE_Q5_K:
        is_u4 = মিথ্যা;
        weights_per_block = 32;
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q6_K:
        is_u4 = মিথ্যা;
        weights_per_block = 16;
        বিরতি;
    পূর্বনির্ধারিত:
        ছোড়ো std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    ধ্রুবক যৌক্তিক for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    নির্বাচন (tensor->type) {
    ক্ষেত্র GGML_TYPE_Q4_0:
        extract_q4_0_তথ্য(&অস্থায়ী_tensor, weights, scales, zp);
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q4_1:
        extract_q4_1_তথ্য(&অস্থায়ী_tensor, weights, scales, zp, use_bias);
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q4_K:
        extract_q4_k_তথ্য(&অস্থায়ী_tensor, weights, scales, zp, use_bias);
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q5_1:
        extract_q5_1_তথ্য(&অস্থায়ী_tensor, weights, scales, zp, use_bias);
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q8_0:
        extract_q8_0_তথ্য(&অস্থায়ী_tensor, weights, scales, zp);
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q6_K:
        extract_q6_k_তথ্য(&অস্থায়ী_tensor, weights, scales, zp);
        বিরতি;
    ক্ষেত্র GGML_TYPE_Q5_K:
        extract_q5_k_তথ্য(&অস্থায়ী_tensor, weights, scales, zp, use_bias);
        বিরতি;
    পূর্বনির্ধারিত:
        ছোড়ো std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> weight_গ্রন্থি;
    যদি (is_u4) {
        weight_গ্রন্থি = make_int4_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    } অন্যথা {
        weight_গ্রন্থি = make_int8_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    }

    auto ফলাফল = weight_গ্রন্থি.get_গ্রন্থি_shared_ptr();
    ফলাফল->set_friendly_name(tensor->name);
    ফেরত ফলাফল;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_to_buffers(ধ্রুবক ggml_tensor * tensor,
                                                ধ্রুবক শূন্য * তথ্য,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t block_আকার_2,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ধ্রুবক int64_t ne0 = tensor->ne[0];                 // elements per row
    ধ্রুবক int64_t n_rows = n_elements / ne0;
    ধ্রুবক auto * type_traits = ggml_get_type_traits(tensor->type);
    ধ্রুবক size_t src_row_bytes = ggml_row_আকার(tensor->type, ne0);

    যৌক্তিক is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
                  requant_type == ExtraQuantType::Q4_0_64 || requant_type == ExtraQuantType::Q4_1_64);

    // Streaming dequant (opt-in via GGML_OPENVINO_REDUCE_COMPILE_MEM or
    // GGML_OPENVINO_MEMORY_OPTIMIZE): instead of
    // materializing the full n_elements F32 array (e.g. ~1 GB for token_embd), dequantize
    // a chunk of complete rows into a small scratch and quantize/convert it straight into
    // the output buffers, capping the transient F32 footprint at CHUNK_ROWS*ne0 floats.
    //
    // Only valid (and only used) for the Q8_0_C / Q8_1_C / F16 targets whose block size
    // divides a row (channel-wise _C uses block_size == ne0) so no target block straddles
    // a row boundary, and Q8/F16 have no cross-block packing. The u4 (Q4_0) path packs two
    // weights per byte with running zp ORs that assume a single whole-array call, so it is
    // never streamed. When the flag is off, behavior is identical to the original
    // full-materialization path.
    ধ্রুবক যৌক্তিক stream_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(block_আকার_2 > 0 && ne0 % block_আকার_2 != 0);

    যদি (!stream_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<দশমিক> weights_f32(n_elements);
        type_traits->to_float(তথ্য, weights_f32.তথ্য(), n_elements);
        যদি (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.তথ্য(), weights.তথ্য(), n_elements);
            auto ফলাফল = std::make_shared<ov::op::v0::Constant>(weights);
            ফলাফল->set_friendly_name(tensor->name);
            ফেরত ফলাফল;
        }
        যদি (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.তথ্য(), weights, scales, zp, n_elements, block_আকার_2);
        } অন্যথা যদি (is_u4) {
            quantize_q4_0(weights_f32.তথ্য(), weights, scales, zp, n_elements, block_আকার_2);
        } অন্যথা যদি (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.তথ্য(), weights, scales, zp, n_elements, block_আকার_2);
        } অন্যথা {
            quantize_q8_0(weights_f32.তথ্য(), weights, scales, zp, n_elements, block_আকার_2);
        }
    } অন্যথা {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ধ্রুবক int64_t CHUNK_ROWS = std::min<int64_t>(n_rows, 256);
        std::vector<দশমিক> scratch(CHUNK_ROWS * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        auto * f16_base = static_cast<uint8_t *>(weights.তথ্য());
        জন্য (int64_t r0 = 0; r0 < n_rows; r0 += CHUNK_ROWS) {
            ধ্রুবক int64_t rows = std::min(CHUNK_ROWS, n_rows - r0);
            ধ্রুবক int64_t elems = rows * ne0;
            ধ্রুবক auto * src = static_cast<ধ্রুবক uint8_t *>(তথ্য) + r0 * src_row_bytes;
            type_traits->to_float(src, scratch.তথ্য(), elems);

            যদি (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.তথ্য(), f16_base + (r0 * ne0) * আকার(uint16_t), elems);
            } অন্যথা {
                ধ্রুবক int64_t block_offset = (r0 * ne0) / block_আকার_2;
                যদি (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.তথ্য(), weights, scales, zp, elems, block_আকার_2, block_offset);
                } অন্যথা {
                    quantize_q8_0(scratch.তথ্য(), weights, scales, zp, elems, block_আকার_2, block_offset);
                }
            }
        }
        যদি (requant_type == ExtraQuantType::F16) {
            auto ফলাফল = std::make_shared<ov::op::v0::Constant>(weights);
            ফলাফল->set_friendly_name(tensor->name);
            ফেরত ফলাফল;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_গ্রন্থি;
    যদি (is_u4) {
        weight_গ্রন্থি = make_int4_weights(weights, scales, zp, block_আকার_2);
    } অন্যথা {
        weight_গ্রন্থি = make_int8_weights(weights, scales, zp, block_আকার_2);
    }

    auto ফলাফল = weight_গ্রন্থি.get_গ্রন্থি_shared_ptr();
    ফলাফল->set_friendly_name(tensor->name);
    ফেরত ফলাফল;
}
}  // namespace

OvWeight process_weight_tensor(ধ্রুবক ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, শূন্য * output_base_ptr, যৌক্তিক use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(তথ্য != nullptr);

    OvWeight ফলাফল;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape গ্রন্থি_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    যদি (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        নির্বাচন (tensor->type) {
        ক্ষেত্র GGML_TYPE_F32:
            element_type = ov::element::f32;
            বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            element_type = ov::element::f16;
            বিরতি;
        ক্ষেত্র GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            বিরতি;
        পূর্বনির্ধারিত:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        যদি (output_base_ptr && output_base_ptr != তথ্য) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_bytes = ggml_nbytes(tensor);
            memcpy(output_base_ptr, তথ্য, tensor_bytes);
            ফলাফল.weights = ov::Tensor(element_type, গ্রন্থি_shape, output_base_ptr);
        } অন্যথা {
            ফলাফল.weights = ov::Tensor(element_type, গ্রন্থি_shape, তথ্য);
        }
        ফলাফল.weight_গ্রন্থি = std::make_shared<ov::op::v0::Constant>(ফলাফল.weights);
        ফেরত ফলাফল;
    }

    // Handle quantized weights
    যদি (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    ফলাফল.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ধ্রুবক auto & layout = ফলাফল.layout;
    যদি (layout.যোগ_আকার == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    ধ্রুবক যৌক্তিক for_gather_matmul = tensor->ne[2] > 1;
    ধ্রুবক যৌক্তিক zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    ধ্রুবক যৌক্তিক is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    যদি (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_BLOCK_আকার),
                                  MXFP4_BLOCK_BYTES};
        ধ্রুবক size_t tensor_bytes = ggml_nbytes(tensor);
        যদি (output_base_ptr) {
            auto * buf_base = static_cast<uint8_t *>(output_base_ptr);
            memcpy(buf_base + layout.weights_offset, তথ্য, tensor_bytes);
            ফলাফল.weights = ov::Tensor(ov::element::u8, packed_shape, buf_base + layout.weights_offset);
        } অন্যথা {
            ফলাফল.weights = ov::Tensor(ov::element::u8, packed_shape);
            memcpy(ফলাফল.weights.তথ্য(), তথ্য, tensor_bytes);
        }
        ফলাফল.weight_গ্রন্থি = make_mxfp4_moe_packed_weights(ফলাফল.weights).get_গ্রন্থি_shared_ptr();
        ফলাফল.weight_গ্রন্থি->set_friendly_name(tensor->name);
        ফেরত ফলাফল;
    }

    যদি (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    যদি (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        যদি (output_base_ptr) {
            ফলাফল.weights = ov::Tensor(ov::element::f16, গ্রন্থি_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } অন্যথা {
            ফলাফল.weights = ov::Tensor(ov::element::f16, গ্রন্থি_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        ফলাফল.weight_গ্রন্থি =
            requantize_to_buffers(tensor, তথ্য, ExtraQuantType::F16, 0, ফলাফল.weights, dummy_scales, dummy_zp);
        ফেরত ফলাফল;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type;
    যদি (tensor->type == GGML_TYPE_MXFP4) {
        weight_type = ov::element::f4e2m1;
    } অন্যথা যদি (layout.is_symmetric) {
        weight_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } অন্যথা {
        weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = গ্রন্থি_shape;
    scale_shape.back() /= layout.weights_per_block;

    যদি (tensor->type == GGML_TYPE_MXFP4) {
        যদি (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            গ্রন্থি_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } অন্যথা {
            গ্রন্থি_shape.clear();
            জন্য (পূর্ণসংখ্যা i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                গ্রন্থি_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = গ্রন্থি_shape;
        scale_shape.back() /= layout.weights_per_block;
    }

    ধ্রুবক ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    যদি (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    যদি (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        ফলাফল.weights = ov::Tensor(weight_type, গ্রন্থি_shape, buf_base + layout.weights_offset);
        ফলাফল.scales = ov::Tensor(scale_type, scale_shape, buf_base + layout.scales_offset);
        যদি (!layout.is_symmetric) {
            ফলাফল.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } অন্যথা {
        ফলাফল.weights = ov::Tensor(weight_type, গ্রন্থি_shape);
        ফলাফল.scales = ov::Tensor(scale_type, scale_shape);
        যদি (!layout.is_symmetric) {
            ফলাফল.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    যদি (layout.is_requant && layout.requant_type.has_value()) {
        ফলাফল.weight_গ্রন্থি = requantize_to_buffers(tensor, তথ্য, layout.requant_type.value(), layout.weights_per_block,
                                                   ফলাফল.weights, ফলাফল.scales, ফলাফল.zp);
    } অন্যথা {
        ফলাফল.weight_গ্রন্থি =
            extract_quantized_weights(tensor, তথ্য, ফলাফল.weights, ফলাফল.scales, ফলাফল.zp, use_bias);
    }

    ফেরত ফলাফল;
}
