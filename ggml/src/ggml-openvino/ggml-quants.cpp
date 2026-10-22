#अन्तर्भाव "ggml-quants.h"

#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino-extra.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <limits>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/except.hpp>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/core/parallel.hpp>
#अन्तर्भाव <openvino/core/shape.hpp>
#अन्तर्भाव <openvino/core/type/element_type.hpp>
#अन्तर्भाव <openvino/core/type/element_type_traits.hpp>
#अन्तर्भाव <openvino/core/type/float16.hpp>
#अन्तर्भाव <openvino/core/type/float4_e2m1.hpp>
#अन्तर्भाव <openvino/core/type/float8_e8m0.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/util/attr_types.hpp>
#अन्तर्भाव <openvino/pass/constant_folding.hpp>
#अन्तर्भाव <openvino/runtime/tensor.hpp>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
नामाकाश ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
तार्किक get_single_मान(अचर std::shared_ptr<ov::op::v0::Constant> & const_गाँठो,
                      भिन्न & value,
                      तार्किक check_मान_range = सत्य);
}  // namespace ov::op::util

नामाकाश {
शून्य unpack_32_4(अचर uint8_t * data, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    लागि (पूर्णाङ्क j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        यदि (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr माप_प्रकार MXFP4_BLOCK_SIZE = 32;
constexpr माप_प्रकार MXFP4_BLOCK_QS_SIZE = MXFP4_BLOCK_SIZE / 2;
constexpr माप_प्रकार MXFP4_BLOCK_BYTES = परिमाण(uint8_t) + MXFP4_BLOCK_QS_SIZE;

शून्य pack_32_mxfp4_for_openvino(अचर uint8_t * data, uint8_t * dst) {
    लागि (पूर्णाङ्क j = 0; j < static_cast<पूर्णाङ्क>(MXFP4_BLOCK_QS_SIZE); j += 2) {
        अचर uint8_t v0 = data[j] & 0x0F;
        अचर uint8_t v1 = (data[j + 1] & 0x0F) << 4;
        अचर uint8_t v16 = data[j] >> 4;
        अचर uint8_t v17 = data[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_BLOCK_SIZE / 4 + j / 2] = v16 | v17;
    }
}

शून्य extract_mxfp4_data(अचर ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    अचर स्वतः * data = static_cast<अचर uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    अचर माप_प्रकार n_blocks = scales_arr.get_size();

    ov::parallel_for(n_blocks, [&](माप_प्रकार i) {
        अचर uint8_t * block = data + i * MXFP4_BLOCK_BYTES;
        pack_32_mxfp4_for_openvino(block + परिमाण(uint8_t), weights + i * MXFP4_BLOCK_QS_SIZE);
        scales[i] = ov::float8_e8m0::from_bits(block[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
शून्य extract_q4_0_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    अचर uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    तार्किक is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    यदि (!is_symmetric) {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            // Pack two 4-bit zero points per byte
            यदि (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } अन्यथा {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
        });
    } अन्यथा {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            लागि (पूर्णाङ्क j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
शून्य extract_q4_1_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    अचर uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    यदि (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        स्वतः * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    } अन्यथा {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            यदि (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } अन्यथा {
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
शून्य extract_q5_1_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    अचर uint64_t bytes_per_block = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    अचर पूर्णाङ्क qk = 32;

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());  // u8 weights, one byte per weight
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    स्वतः read_u16 = [](अचर uint8_t * p) {
        uint16_t v;
        स_नकल(&v, p, परिमाण(v));
        फर्काऊ v;
    };

    स्वतः unpack_block = [&](अचर uint8_t * block, uint8_t * dst) {
        uint32_t qh;
        स_नकल(&qh, block + 4, परिमाण(uint32_t));
        अचर uint8_t * qs = block + 8;
        लागि (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
            अचर uint8_t lo = qs[j] & 0x0F;
            अचर uint8_t hi = qs[j] >> 4;
            अचर uint8_t bit_lo = (qh >> j) & 1;
            अचर uint8_t bit_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (bit_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (bit_hi << 4);  // last 16 weights
        }
    };

    यदि (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        स्वतः * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            अचर uint8_t * block = data + i * bytes_per_block;
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_bits(read_u16(block)));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_bits(read_u16(block + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_block(block, weights + i * qk);
        });
    } अन्यथा {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());  // u8 zero points
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            अचर uint8_t * block = data + i * bytes_per_block;
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_bits(read_u16(block)));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_bits(read_u16(block + 2)));
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
शून्य extract_q8_0_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    अचर uint64_t weights_per_block = 32;
    अचर uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    तार्किक is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    यदि (!is_symmetric) {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            zp[i] = 128;
            लागि (माप_प्रकार j = 0; j < weights_per_block; ++j) {
                uint8_t x = block_data[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_per_block + j] = x;
            }
        });
    } अन्यथा {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_size(), [&](माप_प्रकार i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            // Copy int8 weights as-is (the tensor element type is i8)
            स_नकल(weights + i * weights_per_block, block_data + 2, weights_per_block);
        });
    }
}

शून्य unpack_256_4(अचर uint8_t * data, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    लागि (माप_प्रकार i = 0; i < 4; ++i) {
        लागि (पूर्णाङ्क j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            यदि (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

शून्य extract_q4_k_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    अचर uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    अचर uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    स्वतः * zp_u4 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    स्वतः * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](माप_प्रकार i) {
        uint8_t * block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets
        भिन्न scale_scales = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) block_data)));
        भिन्न scale_mins = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = block_data + 4;

        // Calculate scales
        भिन्न scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<भिन्न>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<भिन्न>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<भिन्न>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<भिन्न>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<भिन्न>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<भिन्न>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<भिन्न>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<भिन्न>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        भिन्न min_vals[8];
        min_vals[0] = scale_mins * static_cast<भिन्न>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<भिन्न>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<भिन्न>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<भिन्न>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<भिन्न>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<भिन्न>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<भिन्न>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<भिन्न>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        लागि (पूर्णाङ्क j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            यदि (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } अन्यथा {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                माप_प्रकार idx = i * 8 + j;
                यदि (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } अन्यथा {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

शून्य extract_q6_k_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    अचर uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    अचर uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    तार्किक is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    यदि (!is_symmetric) {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(n_super_block, [&](माप_प्रकार i) {
            uint8_t * block_data = data + i * bytes_per_block;
            भिन्न scale_factor = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            लागि (माप_प्रकार j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<भिन्न>(*((int8_t *) (block_data + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            लागि (int64_t j = 0; j < 32; ++j) {
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
    } अन्यथा {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_block, [&](माप_प्रकार i) {
            uint8_t * block_data = data + i * bytes_per_block;
            भिन्न scale_factor = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            लागि (माप_प्रकार j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<भिन्न>(*((int8_t *) (block_data + 128 + 64 + j))));
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            स्वतः * signed_weights = reinterpret_cast<int8_t *>(weights);
            लागि (int64_t j = 0; j < 32; ++j) {
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

अन्तःस्थ शून्य get_scale_min_k4(पूर्णाङ्क j, अचर uint8_t * q, uint8_t * d, uint8_t * m) {
    यदि (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } अन्यथा {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

शून्य extract_q5_k_data(अचर ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    अचर uint64_t bytes_per_block = 4 + 12 + 32 + 128;
    अचर uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    स्वतः * data = static_cast<uint8_t *>(tensor->data);
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    स्वतः * zp_u8 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    स्वतः * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](माप_प्रकार i) {
        uint8_t * block_data = data + i * bytes_per_block;

        अचर भिन्न d = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) block_data)));
        अचर भिन्न min_factor = static_cast<भिन्न>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        अचर uint8_t * scales_data = block_data + 4;   // 12 bytes of scales
        अचर uint8_t * qh = block_data + 4 + 12;       // 32 bytes of high bits
        अचर uint8_t * ql = block_data + 4 + 12 + 32;  // 128 bytes of low bits

        पूर्णाङ्क is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        लागि (पूर्णाङ्क j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_data, &sc, &m);
            अचर भिन्न d1 = d * sc;
            अचर भिन्न m1 = min_factor * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_data, &sc, &m);
            अचर भिन्न d2 = d * sc;
            अचर भिन्न m2 = min_factor * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            यदि (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } अन्यथा {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            लागि (पूर्णाङ्क l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            लागि (पूर्णाङ्क l = 0; l < 32; ++l) {
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
                                       माप_प्रकार group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       तार्किक use_bias = असत्य,
                                       तार्किक for_gather_matmul = असत्य) {
    ov::Shape orig_shape = weight.get_shape();
    तार्किक is_signed = (weight.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    स्वतः scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= group_size;
    packed_shape.push_back(group_size);
    अचर माप_प्रकार group_dim = packed_shape.size() - 2;

    यदि (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } अन्यथा {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        यदि (!is_signed && zp.get_size() > 0) {
            स्वतः zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    स्वतः scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    यदि (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        स्वतः weights_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_गाँठो->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वतः weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_गाँठो, ov::element::f16);
        स्वतः mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = mul;
    } अन्यथा {
        // Unsigned path
        स्वतः weights_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_गाँठो->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वतः weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_गाँठो, ov::element::f16);

        यदि (use_bias && zp.get_size() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            स्वतः * bias_zp_data = zp.data<ov::float16>();
            अचर स्वतः * scale_data = scales.data<ov::float16>();
            अचर माप_प्रकार n = zp.get_size();
            लागि (माप_प्रकार i = 0; i < n; i++) {
                भिन्न s = static_cast<भिन्न>(scale_data[i]);
                भिन्न b = static_cast<भिन्न>(bias_zp_data[i]);
                bias_zp_data[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            स्वतः zero_point_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            स्वतः w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } अन्यथा {
            // Zero point path: (w - zp) * s
            स्वतः zero_point = std::make_shared<ov::op::v0::Constant>(zp);
            भिन्न zp_मान;
            यदि (ov::op::util::get_single_मान(zero_point, zp_मान)) {
                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_मान});
            }
            स्वतः zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            स्वतः w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            स्वतः mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = mul;
        }
    }

    यदि (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        स्वतः final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(result, final_shape, असत्य);
        result = reshaped;
    }

    यदि (for_gather_matmul) {
        फर्काऊ result;
    }
    फर्काऊ std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
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
                                       माप_प्रकार group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       तार्किक use_bias = असत्य,
                                       तार्किक for_gather_matmul = असत्य) {
    ov::Shape orig_weight_shape = weight.get_shape();
    तार्किक is_signed = (weight.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_weight_shape;
    packed_shape.back() /= group_size;
    packed_shape.push_back(group_size);
    अचर माप_प्रकार group_dim = packed_shape.size() - 2;

    यदि (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } अन्यथा {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        यदि (!is_signed && zp.get_size() > 0) {
            स्वतः zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    स्वतः scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    यदि (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        स्वतः weights_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_गाँठो->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वतः weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_गाँठो, ov::element::f16);
        स्वतः mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = mul;
    } अन्यथा {
        // Unsigned path
        स्वतः weights_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_गाँठो->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वतः weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_गाँठो, ov::element::f16);

        यदि (use_bias && zp.get_size() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            स्वतः * bias_zp_data = zp.data<ov::float16>();
            अचर स्वतः * scale_data = scales.data<ov::float16>();
            अचर माप_प्रकार n = zp.get_size();
            लागि (माप_प्रकार i = 0; i < n; i++) {
                भिन्न s = static_cast<भिन्न>(scale_data[i]);
                भिन्न b = static_cast<भिन्न>(bias_zp_data[i]);
                bias_zp_data[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            स्वतः zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            स्वतः w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } अन्यथा {
            // Zero point path: (w - zp) * s
            स्वतः zero_points_गाँठो = std::make_shared<ov::op::v0::Constant>(zp);
            भिन्न zp_मान;
            यदि (ov::op::util::get_single_मान(zero_points_गाँठो, zp_मान)) {
                zero_points_गाँठो = ov::op::v0::Constant::create(zero_points_गाँठो->get_element_type(), {}, {zp_मान});
            }
            स्वतः zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_गाँठो, ov::element::f16);
            स्वतः w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            स्वतः mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = mul;
        }
    }

    यदि (packed_shape.size() != orig_weight_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        स्वतः final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_weight_shape.size()},
                                                                  orig_weight_shape);
        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(result, final_shape, असत्य);
        result = reshaped;
    }

    यदि (for_gather_matmul) {
        फर्काऊ result;
    }
    फर्काऊ std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & weight, ov::Tensor & scales) {
    अचर ov::Shape final_shape = weight.get_shape();
    GGML_ASSERT(!final_shape.empty());
    GGML_ASSERT(final_shape.back() % MXFP4_BLOCK_SIZE == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_BLOCK_SIZE;
    packed_shape.push_back(MXFP4_BLOCK_SIZE);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.set_shape(scale_shape);

    स्वतः weights_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_गाँठो->get_rt_info()["__gguf_tensor_holder"] = weight;
    स्वतः weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_गाँठो, ov::element::f32);

    स्वतः scales_गाँठो = std::make_shared<ov::op::v0::Constant>(scales);
    स्वतः scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_गाँठो, ov::element::f32);
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::AutoBroadcastType::NUMPY);

    स्वतः final_shape_गाँठो =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    फर्काऊ std::make_shared<ov::op::v1::Reshape>(result, final_shape_गाँठो, असत्य);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & weight) {
    स्वतः weights_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::u8, weight.get_shape(),
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_गाँठो->get_rt_info()["__gguf_tensor_holder"] = weight;
    weights_गाँठो->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = सत्य;
    फर्काऊ weights_गाँठो;
}

शून्य quantize_q4_0(अचर भिन्न * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    प्रतिज्ञा(k % qk == 0);
    अचर पूर्णाङ्क nb = k / qk;

    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    तार्किक is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    यदि (!is_symmetric) {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());
        लागि (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            भिन्न max = 0.0f;
            लागि (पूर्णाङ्क j = 0; j < qk; j++) {
                अचर भिन्न v = x[i * qk + j];
                यदि (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            अचर भिन्न d = max / -8;
            यदि (d == 0) {
                scales[i] = ov::float16(1.0f);
                यदि (i % 2 == 0) {
                    zp[i / 2] = 8;
                } अन्यथा {
                    zp[i / 2] |= (8 << 4);
                }
                स_रखो(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                जारी;
            }
            अचर भिन्न id = 1.0f / d;
            scales[i] = ov::float16(d);
            यदि (i % 2 == 0) {
                zp[i / 2] = 8;
            } अन्यथा {
                zp[i / 2] |= (8 << 4);
            }
            लागि (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
                अचर भिन्न x0 = x[i * qk + 2 * j] * id;
                अचर भिन्न x1 = x[i * qk + 2 * j + 1] * id;
                अचर uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                अचर uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } अन्यथा {
        // Symmetric: produce signed i4 values in [-8, 7]
        लागि (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            भिन्न max = 0.0f;
            लागि (पूर्णाङ्क j = 0; j < qk; j++) {
                अचर भिन्न v = x[i * qk + j];
                यदि (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            अचर भिन्न d = max / -8;
            यदि (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                स_रखो(weights + i * qk / 2, 0, qk / 2);
                जारी;
            }
            अचर भिन्न id = 1.0f / d;
            scales[i] = ov::float16(d);
            लागि (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
                अचर भिन्न x0 = x[i * qk + 2 * j] * id;
                अचर भिन्न x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (पूर्णाङ्क) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (पूर्णाङ्क) roundf(x1)));
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
शून्य quantize_q4_1_asym(अचर भिन्न * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    प्रतिज्ञा(k % qk == 0);
    अचर पूर्णाङ्क nb = k / qk;

    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data());
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    स्वतः * zp = static_cast<uint8_t *>(zp_arr.data());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    स्वतः store_zp = [zp](पूर्णाङ्क i, uint8_t v) {
        यदि (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } अन्यथा {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    लागि (पूर्णाङ्क i = 0; i < nb; i++) {
        भिन्न vmin = x[i * qk];
        भिन्न vmax = x[i * qk];
        लागि (पूर्णाङ्क j = 1; j < qk; j++) {
            अचर भिन्न v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        अचर भिन्न d = (vmax - vmin) / 15.0f;
        यदि (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            स_रखो(weights + i * qk / 2, 0, qk / 2);
            जारी;
        }
        अचर भिन्न id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        अचर पूर्णाङ्क zq = std::max(0, std::min(15, (पूर्णाङ्क) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        लागि (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
            अचर भिन्न x0 = x[i * qk + 2 * j] * id;
            अचर भिन्न x1 = x[i * qk + 2 * j + 1] * id;
            अचर uint8_t q0 = (uint8_t) std::max(0, std::min(15, (पूर्णाङ्क) lroundf(x0) + zq));
            अचर uint8_t q1 = (uint8_t) std::max(0, std::min(15, (पूर्णाङ्क) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

शून्य quantize_q8_0(अचर भिन्न * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0) {
    प्रतिज्ञा(k % qk == 0);
    अचर पूर्णाङ्क nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data()) + block_offset * qk;
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    तार्किक is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    यदि (!is_symmetric) {
        स्वतः * zp = static_cast<uint8_t *>(zp_arr.data()) + block_offset;
        लागि (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            लागि (पूर्णाङ्क j = 0; j < qk; j++) {
                अचर भिन्न v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            अचर भिन्न d = amax / 127.0f;
            अचर भिन्न id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            लागि (पूर्णाङ्क j = 0; j < qk; ++j) {
                अचर भिन्न x0 = x[i * qk + j] * id;
                अचर int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } अन्यथा {
        // Symmetric: store signed int8 values directly
        स्वतः * signed_weights = reinterpret_cast<int8_t *>(weights);
        लागि (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            लागि (पूर्णाङ्क j = 0; j < qk; j++) {
                अचर भिन्न v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            अचर भिन्न d = amax / 127.0f;
            अचर भिन्न id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            लागि (पूर्णाङ्क j = 0; j < qk; ++j) {
                अचर भिन्न x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

शून्य quantize_q8_1(अचर भिन्न * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0) {
    प्रतिज्ञा(k % qk == 0);
    अचर पूर्णाङ्क nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    स्वतः * weights = static_cast<uint8_t *>(weights_arr.data()) + block_offset * qk;
    स्वतः * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    स्वतः * zp = static_cast<uint8_t *>(zp_arr.data()) + block_offset;
    लागि (पूर्णाङ्क i = 0; i < nb; i++) {
        भिन्न min = std::numeric_limits<भिन्न>::max();
        भिन्न max = std::numeric_limits<भिन्न>::lowest();

        लागि (पूर्णाङ्क j = 0; j < qk; j++) {
            अचर भिन्न v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        अचर भिन्न d = (max - min) / ((1 << 8) - 1);
        अचर भिन्न id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        लागि (पूर्णाङ्क j = 0; j < qk; ++j) {
            अचर भिन्न x0 = (x[i * qk + j] - min) * id;
            अचर uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> extract_quantized_weights(अचर ggml_tensor * tensor,
                                                    अचर शून्य * data,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    तार्किक use_bias = असत्य) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor temp_tensor = *tensor;
    temp_tensor.data = const_cast<शून्य *>(data);

    यदि (tensor->type == GGML_TYPE_MXFP4) {
        extract_mxfp4_data(&temp_tensor, weights, scales);
        स्वतः result = make_mxfp4_weights(weights, scales).get_गाँठो_shared_ptr();
        result->set_friendly_name(tensor->name);
        फर्काऊ result;
    }

    // Determine block size based on tensor type
    int64_t weights_per_block;
    तार्किक is_u4;
    विकल्प (tensor->type) {
    पक्ष GGML_TYPE_Q4_0:
    पक्ष GGML_TYPE_Q4_1:
    पक्ष GGML_TYPE_Q4_K:
        is_u4 = सत्य;
        weights_per_block = 32;
        रोक;
    पक्ष GGML_TYPE_Q8_0:
    पक्ष GGML_TYPE_Q5_1:
    पक्ष GGML_TYPE_Q5_K:
        is_u4 = असत्य;
        weights_per_block = 32;
        रोक;
    पक्ष GGML_TYPE_Q6_K:
        is_u4 = असत्य;
        weights_per_block = 16;
        रोक;
    स्वभाव:
        क्षिप std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    अचर तार्किक for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    विकल्प (tensor->type) {
    पक्ष GGML_TYPE_Q4_0:
        extract_q4_0_data(&temp_tensor, weights, scales, zp);
        रोक;
    पक्ष GGML_TYPE_Q4_1:
        extract_q4_1_data(&temp_tensor, weights, scales, zp, use_bias);
        रोक;
    पक्ष GGML_TYPE_Q4_K:
        extract_q4_k_data(&temp_tensor, weights, scales, zp, use_bias);
        रोक;
    पक्ष GGML_TYPE_Q5_1:
        extract_q5_1_data(&temp_tensor, weights, scales, zp, use_bias);
        रोक;
    पक्ष GGML_TYPE_Q8_0:
        extract_q8_0_data(&temp_tensor, weights, scales, zp);
        रोक;
    पक्ष GGML_TYPE_Q6_K:
        extract_q6_k_data(&temp_tensor, weights, scales, zp);
        रोक;
    पक्ष GGML_TYPE_Q5_K:
        extract_q5_k_data(&temp_tensor, weights, scales, zp, use_bias);
        रोक;
    स्वभाव:
        क्षिप std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> weight_गाँठो;
    यदि (is_u4) {
        weight_गाँठो = make_int4_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    } अन्यथा {
        weight_गाँठो = make_int8_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    }

    स्वतः result = weight_गाँठो.get_गाँठो_shared_ptr();
    result->set_friendly_name(tensor->name);
    फर्काऊ result;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_to_buffers(अचर ggml_tensor * tensor,
                                                अचर शून्य * data,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    अचर int64_t ne0 = tensor->ne[0];                 // elements per row
    अचर int64_t n_rows = n_elements / ne0;
    अचर स्वतः * type_traits = ggml_get_type_traits(tensor->type);
    अचर माप_प्रकार src_row_bytes = ggml_row_size(tensor->type, ne0);

    तार्किक is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    अचर तार्किक stream_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(block_size > 0 && ne0 % block_size != 0);

    यदि (!stream_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<भिन्न> weights_f32(n_elements);
        type_traits->to_float(data, weights_f32.data(), n_elements);
        यदि (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.data(), weights.data(), n_elements);
            स्वतः result = std::make_shared<ov::op::v0::Constant>(weights);
            result->set_friendly_name(tensor->name);
            फर्काऊ result;
        }
        यदि (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } अन्यथा यदि (is_u4) {
            quantize_q4_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } अन्यथा यदि (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } अन्यथा {
            quantize_q8_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        }
    } अन्यथा {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        अचर int64_t CHUNK_ROWS = std::min<int64_t>(n_rows, 256);
        std::vector<भिन्न> scratch(CHUNK_ROWS * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        स्वतः * f16_base = static_cast<uint8_t *>(weights.data());
        लागि (int64_t r0 = 0; r0 < n_rows; r0 += CHUNK_ROWS) {
            अचर int64_t rows = std::min(CHUNK_ROWS, n_rows - r0);
            अचर int64_t elems = rows * ne0;
            अचर स्वतः * src = static_cast<अचर uint8_t *>(data) + r0 * src_row_bytes;
            type_traits->to_float(src, scratch.data(), elems);

            यदि (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.data(), f16_base + (r0 * ne0) * परिमाण(uint16_t), elems);
            } अन्यथा {
                अचर int64_t block_offset = (r0 * ne0) / block_size;
                यदि (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.data(), weights, scales, zp, elems, block_size, block_offset);
                } अन्यथा {
                    quantize_q8_0(scratch.data(), weights, scales, zp, elems, block_size, block_offset);
                }
            }
        }
        यदि (requant_type == ExtraQuantType::F16) {
            स्वतः result = std::make_shared<ov::op::v0::Constant>(weights);
            result->set_friendly_name(tensor->name);
            फर्काऊ result;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_गाँठो;
    यदि (is_u4) {
        weight_गाँठो = make_int4_weights(weights, scales, zp, block_size);
    } अन्यथा {
        weight_गाँठो = make_int8_weights(weights, scales, zp, block_size);
    }

    स्वतः result = weight_गाँठो.get_गाँठो_shared_ptr();
    result->set_friendly_name(tensor->name);
    फर्काऊ result;
}
}  // namespace

OvWeight process_weight_tensor(अचर ggml_tensor * tensor, अचर शून्य * data, शून्य * output_base_ptr, तार्किक use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr);

    OvWeight result;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape गाँठो_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<माप_प्रकार>(tensor->ne[2]), static_cast<माप_प्रकार>(tensor->ne[1]),
                                         static_cast<माप_प्रकार>(tensor->ne[0])} :
                               ov::Shape{static_cast<माप_प्रकार>(tensor->ne[1]), static_cast<माप_प्रकार>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    यदि (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        विकल्प (tensor->type) {
        पक्ष GGML_TYPE_F32:
            element_type = ov::element::f32;
            रोक;
        पक्ष GGML_TYPE_F16:
            element_type = ov::element::f16;
            रोक;
        पक्ष GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            रोक;
        स्वभाव:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        यदि (output_base_ptr && output_base_ptr != data) {
            // Using external buffer - copy data and create shared-memory constant
            माप_प्रकार tensor_bytes = ggml_nbytes(tensor);
            स_नकल(output_base_ptr, data, tensor_bytes);
            result.weights = ov::Tensor(element_type, गाँठो_shape, output_base_ptr);
        } अन्यथा {
            result.weights = ov::Tensor(element_type, गाँठो_shape, data);
        }
        result.weight_गाँठो = std::make_shared<ov::op::v0::Constant>(result.weights);
        फर्काऊ result;
    }

    // Handle quantized weights
    यदि (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    result.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    अचर स्वतः & layout = result.layout;
    यदि (layout.total_size == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    अचर तार्किक for_gather_matmul = tensor->ne[2] > 1;
    अचर तार्किक zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    अचर तार्किक is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    यदि (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<माप_प्रकार>(tensor->ne[3]),
                                  static_cast<माप_प्रकार>(tensor->ne[2]),
                                  static_cast<माप_प्रकार>(tensor->ne[1]),
                                  static_cast<माप_प्रकार>(tensor->ne[0] / MXFP4_BLOCK_SIZE),
                                  MXFP4_BLOCK_BYTES};
        अचर माप_प्रकार tensor_bytes = ggml_nbytes(tensor);
        यदि (output_base_ptr) {
            स्वतः * buf_base = static_cast<uint8_t *>(output_base_ptr);
            स_नकल(buf_base + layout.weights_offset, data, tensor_bytes);
            result.weights = ov::Tensor(ov::element::u8, packed_shape, buf_base + layout.weights_offset);
        } अन्यथा {
            result.weights = ov::Tensor(ov::element::u8, packed_shape);
            स_नकल(result.weights.data(), data, tensor_bytes);
        }
        result.weight_गाँठो = make_mxfp4_moe_packed_weights(result.weights).get_गाँठो_shared_ptr();
        result.weight_गाँठो->set_friendly_name(tensor->name);
        फर्काऊ result;
    }

    यदि (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    यदि (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        यदि (output_base_ptr) {
            result.weights = ov::Tensor(ov::element::f16, गाँठो_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } अन्यथा {
            result.weights = ov::Tensor(ov::element::f16, गाँठो_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        result.weight_गाँठो =
            requantize_to_buffers(tensor, data, ExtraQuantType::F16, 0, result.weights, dummy_scales, dummy_zp);
        फर्काऊ result;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type;
    यदि (tensor->type == GGML_TYPE_MXFP4) {
        weight_type = ov::element::f4e2m1;
    } अन्यथा यदि (layout.is_symmetric) {
        weight_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } अन्यथा {
        weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = गाँठो_shape;
    scale_shape.back() /= layout.weights_per_block;

    यदि (tensor->type == GGML_TYPE_MXFP4) {
        यदि (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            गाँठो_shape = {static_cast<माप_प्रकार>(tensor->ne[1]), static_cast<माप_प्रकार>(tensor->ne[0])};
        } अन्यथा {
            गाँठो_shape.clear();
            लागि (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                गाँठो_shape.push_back(static_cast<माप_प्रकार>(tensor->ne[i]));
            }
        }

        scale_shape = गाँठो_shape;
        scale_shape.back() /= layout.weights_per_block;
    }

    अचर ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    यदि (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    यदि (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        result.weights = ov::Tensor(weight_type, गाँठो_shape, buf_base + layout.weights_offset);
        result.scales = ov::Tensor(scale_type, scale_shape, buf_base + layout.scales_offset);
        यदि (!layout.is_symmetric) {
            result.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } अन्यथा {
        result.weights = ov::Tensor(weight_type, गाँठो_shape);
        result.scales = ov::Tensor(scale_type, scale_shape);
        यदि (!layout.is_symmetric) {
            result.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    यदि (layout.is_requant && layout.requant_type.has_value()) {
        result.weight_गाँठो = requantize_to_buffers(tensor, data, layout.requant_type.value(), layout.weights_per_block,
                                                   result.weights, result.scales, result.zp);
    } अन्यथा {
        result.weight_गाँठो =
            extract_quantized_weights(tensor, data, result.weights, result.scales, result.zp, use_bias);
    }

    फर्काऊ result;
}
