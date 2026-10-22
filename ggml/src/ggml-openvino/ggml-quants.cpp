#அந்தர்பாவ "ggml-quants.h"

#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-openvino-extra.h"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <limits>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/except.hpp>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/core/parallel.hpp>
#அந்தர்பாவ <openvino/core/shape.hpp>
#அந்தர்பாவ <openvino/core/type/element_type.hpp>
#அந்தர்பாவ <openvino/core/type/element_type_traits.hpp>
#அந்தர்பாவ <openvino/core/type/float16.hpp>
#அந்தர்பாவ <openvino/core/type/float4_e2m1.hpp>
#அந்தர்பாவ <openvino/core/type/float8_e8m0.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>
#அந்தர்பாவ <openvino/op/util/attr_types.hpp>
#அந்தர்பாவ <openvino/pass/constant_folding.hpp>
#அந்தர்பாவ <openvino/runtime/tensor.hpp>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
நாமாகாஶ ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
தருக்க get_single_மதிப்பு(மாறிலி std::shared_ptr<ov::op::v0::Constant> & const_முனை,
                      தசம & value,
                      தருக்க check_மதிப்பு_range = உண்மை);
}  // namespace ov::op::util

நாமாகாஶ {
வெறுமை unpack_32_4(மாறிலி uint8_t * data, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    க்கு (முழு j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        எனில் (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_BLOCK_SIZE = 32;
constexpr size_t MXFP4_BLOCK_QS_SIZE = MXFP4_BLOCK_SIZE / 2;
constexpr size_t MXFP4_BLOCK_BYTES = அளவு(uint8_t) + MXFP4_BLOCK_QS_SIZE;

வெறுமை pack_32_mxfp4_for_openvino(மாறிலி uint8_t * data, uint8_t * dst) {
    க்கு (முழு j = 0; j < static_cast<முழு>(MXFP4_BLOCK_QS_SIZE); j += 2) {
        மாறிலி uint8_t v0 = data[j] & 0x0F;
        மாறிலி uint8_t v1 = (data[j + 1] & 0x0F) << 4;
        மாறிலி uint8_t v16 = data[j] >> 4;
        மாறிலி uint8_t v17 = data[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_BLOCK_SIZE / 4 + j / 2] = v16 | v17;
    }
}

வெறுமை extract_mxfp4_data(மாறிலி ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    மாறிலி auto * data = static_cast<மாறிலி uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    மாறிலி size_t n_blocks = scales_arr.get_size();

    ov::parallel_for(n_blocks, [&](size_t i) {
        மாறிலி uint8_t * block = data + i * MXFP4_BLOCK_BYTES;
        pack_32_mxfp4_for_openvino(block + அளவு(uint8_t), weights + i * MXFP4_BLOCK_QS_SIZE);
        scales[i] = ov::float8_e8m0::from_bits(block[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
வெறுமை extract_q4_0_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    மாறிலி uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    தருக்க is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    எனில் (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            // Pack two 4-bit zero points per byte
            எனில் (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } இல்லையெனில் {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
        });
    } இல்லையெனில் {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            க்கு (முழு j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
வெறுமை extract_q4_1_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       தருக்க use_bias) {
    மாறிலி uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    எனில் (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        auto * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            தசம scale = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            தசம min = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    } இல்லையெனில் {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            தசம scale = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            தசம min = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            எனில் (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } இல்லையெனில் {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
வெறுமை extract_q5_1_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       தருக்க use_bias) {
    மாறிலி uint64_t bytes_per_block = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    மாறிலி முழு qk = 32;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());  // u8 weights, one byte per weight
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    auto read_u16 = [](மாறிலி uint8_t * p) {
        uint16_t v;
        memcpy(&v, p, அளவு(v));
        திரும்பு v;
    };

    auto unpack_block = [&](மாறிலி uint8_t * block, uint8_t * dst) {
        uint32_t qh;
        memcpy(&qh, block + 4, அளவு(uint32_t));
        மாறிலி uint8_t * qs = block + 8;
        க்கு (முழு j = 0; j < qk / 2; ++j) {
            மாறிலி uint8_t lo = qs[j] & 0x0F;
            மாறிலி uint8_t hi = qs[j] >> 4;
            மாறிலி uint8_t bit_lo = (qh >> j) & 1;
            மாறிலி uint8_t bit_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (bit_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (bit_hi << 4);  // last 16 weights
        }
    };

    எனில் (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        auto * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            மாறிலி uint8_t * block = data + i * bytes_per_block;
            தசம scale = static_cast<தசம>(ov::float16::from_bits(read_u16(block)));
            தசம min = static_cast<தசம>(ov::float16::from_bits(read_u16(block + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_block(block, weights + i * qk);
        });
    } இல்லையெனில் {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());  // u8 zero points
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            மாறிலி uint8_t * block = data + i * bytes_per_block;
            தசம scale = static_cast<தசம>(ov::float16::from_bits(read_u16(block)));
            தசம min = static_cast<தசம>(ov::float16::from_bits(read_u16(block + 2)));
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
வெறுமை extract_q8_0_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    மாறிலி uint64_t weights_per_block = 32;
    மாறிலி uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    தருக்க is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    எனில் (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            zp[i] = 128;
            க்கு (size_t j = 0; j < weights_per_block; ++j) {
                uint8_t x = block_data[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_per_block + j] = x;
            }
        });
    } இல்லையெனில் {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            // Copy int8 weights as-is (the tensor element type is i8)
            memcpy(weights + i * weights_per_block, block_data + 2, weights_per_block);
        });
    }
}

வெறுமை unpack_256_4(மாறிலி uint8_t * data, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    க்கு (size_t i = 0; i < 4; ++i) {
        க்கு (முழு j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            எனில் (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

வெறுமை extract_q4_k_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       தருக்க use_bias) {
    மாறிலி uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    மாறிலி uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    auto * zp_u4 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets
        தசம scale_scales = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) block_data)));
        தசம scale_mins = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = block_data + 4;

        // Calculate scales
        தசம scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<தசம>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<தசம>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<தசம>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<தசம>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<தசம>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<தசம>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<தசம>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<தசம>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        தசம min_vals[8];
        min_vals[0] = scale_mins * static_cast<தசம>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<தசம>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<தசம>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<தசம>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<தசம>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<தசம>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<தசம>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<தசம>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        க்கு (முழு j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            எனில் (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } இல்லையெனில் {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                எனில் (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } இல்லையெனில் {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

வெறுமை extract_q6_k_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    மாறிலி uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    மாறிலி uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    தருக்க is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    எனில் (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            தசம scale_factor = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            க்கு (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<தசம>(*((int8_t *) (block_data + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            க்கு (int64_t j = 0; j < 32; ++j) {
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
    } இல்லையெனில் {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            தசம scale_factor = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            க்கு (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<தசம>(*((int8_t *) (block_data + 128 + 64 + j))));
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            auto * signed_weights = reinterpret_cast<int8_t *>(weights);
            க்கு (int64_t j = 0; j < 32; ++j) {
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

உள்வரி வெறுமை get_scale_min_k4(முழு j, மாறிலி uint8_t * q, uint8_t * d, uint8_t * m) {
    எனில் (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } இல்லையெனில் {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

வெறுமை extract_q5_k_data(மாறிலி ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       தருக்க use_bias) {
    மாறிலி uint64_t bytes_per_block = 4 + 12 + 32 + 128;
    மாறிலி uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    auto * zp_u8 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        மாறிலி தசம d = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) block_data)));
        மாறிலி தசம min_factor = static_cast<தசம>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        மாறிலி uint8_t * scales_data = block_data + 4;   // 12 bytes of scales
        மாறிலி uint8_t * qh = block_data + 4 + 12;       // 32 bytes of high bits
        மாறிலி uint8_t * ql = block_data + 4 + 12 + 32;  // 128 bytes of low bits

        முழு is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        க்கு (முழு j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_data, &sc, &m);
            மாறிலி தசம d1 = d * sc;
            மாறிலி தசம m1 = min_factor * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_data, &sc, &m);
            மாறிலி தசம d2 = d * sc;
            மாறிலி தசம m2 = min_factor * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            எனில் (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } இல்லையெனில் {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            க்கு (முழு l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            க்கு (முழு l = 0; l < 32; ++l) {
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
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       தருக்க use_bias = பொய்,
                                       தருக்க for_gather_matmul = பொய்) {
    ov::Shape orig_shape = weight.get_shape();
    தருக்க is_signed = (weight.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    auto scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= group_size;
    packed_shape.push_back(group_size);
    மாறிலி size_t group_dim = packed_shape.size() - 2;

    எனில் (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } இல்லையெனில் {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        எனில் (!is_signed && zp.get_size() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    எனில் (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_முனை->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_முனை, ov::element::f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = mul;
    } இல்லையெனில் {
        // Unsigned path
        auto weights_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_முனை->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_முனை, ov::element::f16);

        எனில் (use_bias && zp.get_size() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            auto * bias_zp_data = zp.data<ov::float16>();
            மாறிலி auto * scale_data = scales.data<ov::float16>();
            மாறிலி size_t n = zp.get_size();
            க்கு (size_t i = 0; i < n; i++) {
                தசம s = static_cast<தசம>(scale_data[i]);
                தசம b = static_cast<தசம>(bias_zp_data[i]);
                bias_zp_data[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } இல்லையெனில் {
            // Zero point path: (w - zp) * s
            auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
            தசம zp_மதிப்பு;
            எனில் (ov::op::util::get_single_மதிப்பு(zero_point, zp_மதிப்பு)) {
                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_மதிப்பு});
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            auto mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = mul;
        }
    }

    எனில் (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(result, final_shape, பொய்);
        result = reshaped;
    }

    எனில் (for_gather_matmul) {
        திரும்பு result;
    }
    திரும்பு std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
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
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       தருக்க use_bias = பொய்,
                                       தருக்க for_gather_matmul = பொய்) {
    ov::Shape orig_weight_shape = weight.get_shape();
    தருக்க is_signed = (weight.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_weight_shape;
    packed_shape.back() /= group_size;
    packed_shape.push_back(group_size);
    மாறிலி size_t group_dim = packed_shape.size() - 2;

    எனில் (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } இல்லையெனில் {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        எனில் (!is_signed && zp.get_size() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    எனில் (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_முனை->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_முனை, ov::element::f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = mul;
    } இல்லையெனில் {
        // Unsigned path
        auto weights_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_முனை->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_முனை, ov::element::f16);

        எனில் (use_bias && zp.get_size() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            auto * bias_zp_data = zp.data<ov::float16>();
            மாறிலி auto * scale_data = scales.data<ov::float16>();
            மாறிலி size_t n = zp.get_size();
            க்கு (size_t i = 0; i < n; i++) {
                தசம s = static_cast<தசம>(scale_data[i]);
                தசம b = static_cast<தசம>(bias_zp_data[i]);
                bias_zp_data[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            auto zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } இல்லையெனில் {
            // Zero point path: (w - zp) * s
            auto zero_points_முனை = std::make_shared<ov::op::v0::Constant>(zp);
            தசம zp_மதிப்பு;
            எனில் (ov::op::util::get_single_மதிப்பு(zero_points_முனை, zp_மதிப்பு)) {
                zero_points_முனை = ov::op::v0::Constant::create(zero_points_முனை->get_element_type(), {}, {zp_மதிப்பு});
            }
            auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_முனை, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            auto mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = mul;
        }
    }

    எனில் (packed_shape.size() != orig_weight_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_weight_shape.size()},
                                                                  orig_weight_shape);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(result, final_shape, பொய்);
        result = reshaped;
    }

    எனில் (for_gather_matmul) {
        திரும்பு result;
    }
    திரும்பு std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & weight, ov::Tensor & scales) {
    மாறிலி ov::Shape final_shape = weight.get_shape();
    GGML_ASSERT(!final_shape.empty());
    GGML_ASSERT(final_shape.back() % MXFP4_BLOCK_SIZE == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_BLOCK_SIZE;
    packed_shape.push_back(MXFP4_BLOCK_SIZE);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.set_shape(scale_shape);

    auto weights_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_முனை->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_முனை, ov::element::f32);

    auto scales_முனை = std::make_shared<ov::op::v0::Constant>(scales);
    auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_முனை, ov::element::f32);
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::AutoBroadcastType::NUMPY);

    auto final_shape_முனை =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    திரும்பு std::make_shared<ov::op::v1::Reshape>(result, final_shape_முனை, பொய்);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & weight) {
    auto weights_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::u8, weight.get_shape(),
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_முனை->get_rt_info()["__gguf_tensor_holder"] = weight;
    weights_முனை->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = உண்மை;
    திரும்பு weights_முனை;
}

வெறுமை quantize_q4_0(மாறிலி தசம * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    ப்ரதிஜ்ஞா(k % qk == 0);
    மாறிலி முழு nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    தருக்க is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    எனில் (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        க்கு (முழு i = 0; i < nb; i++) {
            தசம amax = 0.0f;
            தசம max = 0.0f;
            க்கு (முழு j = 0; j < qk; j++) {
                மாறிலி தசம v = x[i * qk + j];
                எனில் (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            மாறிலி தசம d = max / -8;
            எனில் (d == 0) {
                scales[i] = ov::float16(1.0f);
                எனில் (i % 2 == 0) {
                    zp[i / 2] = 8;
                } இல்லையெனில் {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                தொடர்;
            }
            மாறிலி தசம id = 1.0f / d;
            scales[i] = ov::float16(d);
            எனில் (i % 2 == 0) {
                zp[i / 2] = 8;
            } இல்லையெனில் {
                zp[i / 2] |= (8 << 4);
            }
            க்கு (முழு j = 0; j < qk / 2; ++j) {
                மாறிலி தசம x0 = x[i * qk + 2 * j] * id;
                மாறிலி தசம x1 = x[i * qk + 2 * j + 1] * id;
                மாறிலி uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                மாறிலி uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } இல்லையெனில் {
        // Symmetric: produce signed i4 values in [-8, 7]
        க்கு (முழு i = 0; i < nb; i++) {
            தசம amax = 0.0f;
            தசம max = 0.0f;
            க்கு (முழு j = 0; j < qk; j++) {
                மாறிலி தசம v = x[i * qk + j];
                எனில் (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            மாறிலி தசம d = max / -8;
            எனில் (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                தொடர்;
            }
            மாறிலி தசம id = 1.0f / d;
            scales[i] = ov::float16(d);
            க்கு (முழு j = 0; j < qk / 2; ++j) {
                மாறிலி தசம x0 = x[i * qk + 2 * j] * id;
                மாறிலி தசம x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (முழு) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (முழு) roundf(x1)));
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
வெறுமை quantize_q4_1_asym(மாறிலி தசம * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    ப்ரதிஜ்ஞா(k % qk == 0);
    மாறிலி முழு nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    auto store_zp = [zp](முழு i, uint8_t v) {
        எனில் (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } இல்லையெனில் {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    க்கு (முழு i = 0; i < nb; i++) {
        தசம vmin = x[i * qk];
        தசம vmax = x[i * qk];
        க்கு (முழு j = 1; j < qk; j++) {
            மாறிலி தசம v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        மாறிலி தசம d = (vmax - vmin) / 15.0f;
        எனில் (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            தொடர்;
        }
        மாறிலி தசம id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        மாறிலி முழு zq = std::max(0, std::min(15, (முழு) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        க்கு (முழு j = 0; j < qk / 2; ++j) {
            மாறிலி தசம x0 = x[i * qk + 2 * j] * id;
            மாறிலி தசம x1 = x[i * qk + 2 * j + 1] * id;
            மாறிலி uint8_t q0 = (uint8_t) std::max(0, std::min(15, (முழு) lroundf(x0) + zq));
            மாறிலி uint8_t q1 = (uint8_t) std::max(0, std::min(15, (முழு) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

வெறுமை quantize_q8_0(மாறிலி தசம * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0) {
    ப்ரதிஜ்ஞா(k % qk == 0);
    மாறிலி முழு nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    auto * weights = static_cast<uint8_t *>(weights_arr.data()) + block_offset * qk;
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    தருக்க is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    எனில் (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data()) + block_offset;
        க்கு (முழு i = 0; i < nb; i++) {
            தசம amax = 0.0f;
            க்கு (முழு j = 0; j < qk; j++) {
                மாறிலி தசம v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            மாறிலி தசம d = amax / 127.0f;
            மாறிலி தசம id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            க்கு (முழு j = 0; j < qk; ++j) {
                மாறிலி தசம x0 = x[i * qk + j] * id;
                மாறிலி int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } இல்லையெனில் {
        // Symmetric: store signed int8 values directly
        auto * signed_weights = reinterpret_cast<int8_t *>(weights);
        க்கு (முழு i = 0; i < nb; i++) {
            தசம amax = 0.0f;
            க்கு (முழு j = 0; j < qk; j++) {
                மாறிலி தசம v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            மாறிலி தசம d = amax / 127.0f;
            மாறிலி தசம id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            க்கு (முழு j = 0; j < qk; ++j) {
                மாறிலி தசம x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

வெறுமை quantize_q8_1(மாறிலி தசம * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0) {
    ப்ரதிஜ்ஞா(k % qk == 0);
    மாறிலி முழு nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    auto * weights = static_cast<uint8_t *>(weights_arr.data()) + block_offset * qk;
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    auto * zp = static_cast<uint8_t *>(zp_arr.data()) + block_offset;
    க்கு (முழு i = 0; i < nb; i++) {
        தசம min = std::numeric_limits<தசம>::max();
        தசம max = std::numeric_limits<தசம>::lowest();

        க்கு (முழு j = 0; j < qk; j++) {
            மாறிலி தசம v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        மாறிலி தசம d = (max - min) / ((1 << 8) - 1);
        மாறிலி தசம id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        க்கு (முழு j = 0; j < qk; ++j) {
            மாறிலி தசம x0 = (x[i * qk + j] - min) * id;
            மாறிலி uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> extract_quantized_weights(மாறிலி ggml_tensor * tensor,
                                                    மாறிலி வெறுமை * data,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    தருக்க use_bias = பொய்) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor temp_tensor = *tensor;
    temp_tensor.data = const_cast<வெறுமை *>(data);

    எனில் (tensor->type == GGML_TYPE_MXFP4) {
        extract_mxfp4_data(&temp_tensor, weights, scales);
        auto result = make_mxfp4_weights(weights, scales).get_முனை_shared_ptr();
        result->set_friendly_name(tensor->name);
        திரும்பு result;
    }

    // Determine block size based on tensor type
    int64_t weights_per_block;
    தருக்க is_u4;
    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_Q4_0:
    நிலை GGML_TYPE_Q4_1:
    நிலை GGML_TYPE_Q4_K:
        is_u4 = உண்மை;
        weights_per_block = 32;
        நிறுத்து;
    நிலை GGML_TYPE_Q8_0:
    நிலை GGML_TYPE_Q5_1:
    நிலை GGML_TYPE_Q5_K:
        is_u4 = பொய்;
        weights_per_block = 32;
        நிறுத்து;
    நிலை GGML_TYPE_Q6_K:
        is_u4 = பொய்;
        weights_per_block = 16;
        நிறுத்து;
    இயல்பு:
        எறி std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    மாறிலி தருக்க for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_Q4_0:
        extract_q4_0_data(&temp_tensor, weights, scales, zp);
        நிறுத்து;
    நிலை GGML_TYPE_Q4_1:
        extract_q4_1_data(&temp_tensor, weights, scales, zp, use_bias);
        நிறுத்து;
    நிலை GGML_TYPE_Q4_K:
        extract_q4_k_data(&temp_tensor, weights, scales, zp, use_bias);
        நிறுத்து;
    நிலை GGML_TYPE_Q5_1:
        extract_q5_1_data(&temp_tensor, weights, scales, zp, use_bias);
        நிறுத்து;
    நிலை GGML_TYPE_Q8_0:
        extract_q8_0_data(&temp_tensor, weights, scales, zp);
        நிறுத்து;
    நிலை GGML_TYPE_Q6_K:
        extract_q6_k_data(&temp_tensor, weights, scales, zp);
        நிறுத்து;
    நிலை GGML_TYPE_Q5_K:
        extract_q5_k_data(&temp_tensor, weights, scales, zp, use_bias);
        நிறுத்து;
    இயல்பு:
        எறி std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> weight_முனை;
    எனில் (is_u4) {
        weight_முனை = make_int4_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    } இல்லையெனில் {
        weight_முனை = make_int8_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    }

    auto result = weight_முனை.get_முனை_shared_ptr();
    result->set_friendly_name(tensor->name);
    திரும்பு result;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_to_buffers(மாறிலி ggml_tensor * tensor,
                                                மாறிலி வெறுமை * data,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    மாறிலி int64_t ne0 = tensor->ne[0];                 // elements per row
    மாறிலி int64_t n_rows = n_elements / ne0;
    மாறிலி auto * type_traits = ggml_get_type_traits(tensor->type);
    மாறிலி size_t src_row_bytes = ggml_row_size(tensor->type, ne0);

    தருக்க is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    மாறிலி தருக்க stream_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(block_size > 0 && ne0 % block_size != 0);

    எனில் (!stream_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<தசம> weights_f32(n_elements);
        type_traits->to_float(data, weights_f32.data(), n_elements);
        எனில் (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.data(), weights.data(), n_elements);
            auto result = std::make_shared<ov::op::v0::Constant>(weights);
            result->set_friendly_name(tensor->name);
            திரும்பு result;
        }
        எனில் (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } இல்லையெனில் எனில் (is_u4) {
            quantize_q4_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } இல்லையெனில் எனில் (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } இல்லையெனில் {
            quantize_q8_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        }
    } இல்லையெனில் {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        மாறிலி int64_t CHUNK_ROWS = std::min<int64_t>(n_rows, 256);
        std::vector<தசம> scratch(CHUNK_ROWS * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        auto * f16_base = static_cast<uint8_t *>(weights.data());
        க்கு (int64_t r0 = 0; r0 < n_rows; r0 += CHUNK_ROWS) {
            மாறிலி int64_t rows = std::min(CHUNK_ROWS, n_rows - r0);
            மாறிலி int64_t elems = rows * ne0;
            மாறிலி auto * src = static_cast<மாறிலி uint8_t *>(data) + r0 * src_row_bytes;
            type_traits->to_float(src, scratch.data(), elems);

            எனில் (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.data(), f16_base + (r0 * ne0) * அளவு(uint16_t), elems);
            } இல்லையெனில் {
                மாறிலி int64_t block_offset = (r0 * ne0) / block_size;
                எனில் (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.data(), weights, scales, zp, elems, block_size, block_offset);
                } இல்லையெனில் {
                    quantize_q8_0(scratch.data(), weights, scales, zp, elems, block_size, block_offset);
                }
            }
        }
        எனில் (requant_type == ExtraQuantType::F16) {
            auto result = std::make_shared<ov::op::v0::Constant>(weights);
            result->set_friendly_name(tensor->name);
            திரும்பு result;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_முனை;
    எனில் (is_u4) {
        weight_முனை = make_int4_weights(weights, scales, zp, block_size);
    } இல்லையெனில் {
        weight_முனை = make_int8_weights(weights, scales, zp, block_size);
    }

    auto result = weight_முனை.get_முனை_shared_ptr();
    result->set_friendly_name(tensor->name);
    திரும்பு result;
}
}  // namespace

OvWeight process_weight_tensor(மாறிலி ggml_tensor * tensor, மாறிலி வெறுமை * data, வெறுமை * output_base_ptr, தருக்க use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr);

    OvWeight result;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape முனை_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    எனில் (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        தேர்வு (tensor->type) {
        நிலை GGML_TYPE_F32:
            element_type = ov::element::f32;
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            element_type = ov::element::f16;
            நிறுத்து;
        நிலை GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            நிறுத்து;
        இயல்பு:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        எனில் (output_base_ptr && output_base_ptr != data) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_bytes = ggml_nbytes(tensor);
            memcpy(output_base_ptr, data, tensor_bytes);
            result.weights = ov::Tensor(element_type, முனை_shape, output_base_ptr);
        } இல்லையெனில் {
            result.weights = ov::Tensor(element_type, முனை_shape, data);
        }
        result.weight_முனை = std::make_shared<ov::op::v0::Constant>(result.weights);
        திரும்பு result;
    }

    // Handle quantized weights
    எனில் (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    result.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    மாறிலி auto & layout = result.layout;
    எனில் (layout.total_size == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    மாறிலி தருக்க for_gather_matmul = tensor->ne[2] > 1;
    மாறிலி தருக்க zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    மாறிலி தருக்க is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    எனில் (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_BLOCK_SIZE),
                                  MXFP4_BLOCK_BYTES};
        மாறிலி size_t tensor_bytes = ggml_nbytes(tensor);
        எனில் (output_base_ptr) {
            auto * buf_base = static_cast<uint8_t *>(output_base_ptr);
            memcpy(buf_base + layout.weights_offset, data, tensor_bytes);
            result.weights = ov::Tensor(ov::element::u8, packed_shape, buf_base + layout.weights_offset);
        } இல்லையெனில் {
            result.weights = ov::Tensor(ov::element::u8, packed_shape);
            memcpy(result.weights.data(), data, tensor_bytes);
        }
        result.weight_முனை = make_mxfp4_moe_packed_weights(result.weights).get_முனை_shared_ptr();
        result.weight_முனை->set_friendly_name(tensor->name);
        திரும்பு result;
    }

    எனில் (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    எனில் (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        எனில் (output_base_ptr) {
            result.weights = ov::Tensor(ov::element::f16, முனை_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } இல்லையெனில் {
            result.weights = ov::Tensor(ov::element::f16, முனை_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        result.weight_முனை =
            requantize_to_buffers(tensor, data, ExtraQuantType::F16, 0, result.weights, dummy_scales, dummy_zp);
        திரும்பு result;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type;
    எனில் (tensor->type == GGML_TYPE_MXFP4) {
        weight_type = ov::element::f4e2m1;
    } இல்லையெனில் எனில் (layout.is_symmetric) {
        weight_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } இல்லையெனில் {
        weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = முனை_shape;
    scale_shape.back() /= layout.weights_per_block;

    எனில் (tensor->type == GGML_TYPE_MXFP4) {
        எனில் (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            முனை_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } இல்லையெனில் {
            முனை_shape.clear();
            க்கு (முழு i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                முனை_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = முனை_shape;
        scale_shape.back() /= layout.weights_per_block;
    }

    மாறிலி ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    எனில் (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    எனில் (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        result.weights = ov::Tensor(weight_type, முனை_shape, buf_base + layout.weights_offset);
        result.scales = ov::Tensor(scale_type, scale_shape, buf_base + layout.scales_offset);
        எனில் (!layout.is_symmetric) {
            result.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } இல்லையெனில் {
        result.weights = ov::Tensor(weight_type, முனை_shape);
        result.scales = ov::Tensor(scale_type, scale_shape);
        எனில் (!layout.is_symmetric) {
            result.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    எனில் (layout.is_requant && layout.requant_type.has_value()) {
        result.weight_முனை = requantize_to_buffers(tensor, data, layout.requant_type.value(), layout.weights_per_block,
                                                   result.weights, result.scales, result.zp);
    } இல்லையெனில் {
        result.weight_முனை =
            extract_quantized_weights(tensor, data, result.weights, result.scales, result.zp, use_bias);
    }

    திரும்பு result;
}
