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
तार्किक get_single_मानम्(ध्रुव std::shared_ptr<ov::op::v0::Constant> & const_ग्रन्थिः,
                      भिन्न & value,
                      तार्किक check_मानम्_range = सत्य);
}  // namespace ov::op::util

नामाकाश {
शून्य unpack_32_4(ध्रुव uint8_t * दत्त, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    प्रति (पूर्णाङ्क j = 0; j < 16; ++j) {
        uint8_t x = (दत्त[j] & 0x0F);
        uint8_t y = (दत्त[j] >> 4);
        यदि (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_खण्डम्_आकार = 32;
constexpr size_t MXFP4_खण्डम्_QS_आकार = MXFP4_खण्डम्_आकार / 2;
constexpr size_t MXFP4_खण्डम्_अष्टकानि = परिमाण(uint8_t) + MXFP4_खण्डम्_QS_आकार;

शून्य pack_32_mxfp4_for_openvino(ध्रुव uint8_t * दत्त, uint8_t * dst) {
    प्रति (पूर्णाङ्क j = 0; j < static_cast<पूर्णाङ्क>(MXFP4_खण्डम्_QS_आकार); j += 2) {
        ध्रुव uint8_t v0 = दत्त[j] & 0x0F;
        ध्रुव uint8_t v1 = (दत्त[j + 1] & 0x0F) << 4;
        ध्रुव uint8_t v16 = दत्त[j] >> 4;
        ध्रुव uint8_t v17 = दत्त[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_खण्डम्_आकार / 4 + j / 2] = v16 | v17;
    }
}

शून्य extract_mxfp4_दत्त(ध्रुव ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ध्रुव स्वचालित * दत्त = static_cast<ध्रुव uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ध्रुव size_t n_खण्डानि = scales_arr.get_आकार();

    ov::parallel_for(n_खण्डानि, [&](size_t i) {
        ध्रुव uint8_t * खण्डम् = दत्त + i * MXFP4_खण्डम्_अष्टकानि;
        pack_32_mxfp4_for_openvino(खण्डम् + परिमाण(uint8_t), weights + i * MXFP4_खण्डम्_QS_आकार);
        scales[i] = ov::float8_e8m0::from_द्वयङ्काः(खण्डम्[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
शून्य extract_q4_0_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 18;  // 2 bytes scale, 32x0.5 byte weights

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    तार्किक is_सममितम् = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    यदि (!is_सममितम्) {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            scales[i] = ov::float16::from_द्वयङ्काः(*((uint16_t *) (दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्)));
            // Pack two 4-bit zero points per byte
            यदि (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } अन्यथा {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम् + 2, weights + i * 16);
        });
    } अन्यथा {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            scales[i] = ov::float16::from_द्वयङ्काः(*((uint16_t *) (दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्)));
            unpack_32_4(दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम् + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            प्रति (पूर्णाङ्क j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
शून्य extract_q4_1_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    यदि (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        स्वचालित * bias = zp_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) (दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्))));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) (दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम् + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम् + 4, weights + i * 16);
        });
    } अन्यथा {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) (दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्))));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) (दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम् + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            यदि (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } अन्यथा {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम् + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
शून्य extract_q5_1_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ध्रुव पूर्णाङ्क qk = 32;

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());  // u8 weights, one byte per weight
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    स्वचालित read_u16 = [](ध्रुव uint8_t * p) {
        uint16_t v;
        स्मृतिं_प्रतिलिख(&v, p, परिमाण(v));
        प्रत्यागम v;
    };

    स्वचालित unpack_खण्डम् = [&](ध्रुव uint8_t * खण्डम्, uint8_t * dst) {
        uint32_t qh;
        स्मृतिं_प्रतिलिख(&qh, खण्डम् + 4, परिमाण(uint32_t));
        ध्रुव uint8_t * qs = खण्डम् + 8;
        प्रति (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
            ध्रुव uint8_t lo = qs[j] & 0x0F;
            ध्रुव uint8_t hi = qs[j] >> 4;
            ध्रुव uint8_t bit_lo = (qh >> j) & 1;
            ध्रुव uint8_t bit_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (bit_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (bit_hi << 4);  // last 16 weights
        }
    };

    यदि (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        स्वचालित * bias = zp_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            ध्रुव uint8_t * खण्डम् = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(read_u16(खण्डम्)));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(read_u16(खण्डम् + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_खण्डम्(खण्डम्, weights + i * qk);
        });
    } अन्यथा {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());  // u8 zero points
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            ध्रुव uint8_t * खण्डम् = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;
            भिन्न scale = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(read_u16(खण्डम्)));
            भिन्न min = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(read_u16(खण्डम् + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_खण्डम्(खण्डम्, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
शून्य extract_q8_0_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ध्रुव uint64_t weights_प्रत्येकम्_खण्डम् = 32;
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 34;  // 2 bytes scale, 32x1 byte weights

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    तार्किक is_सममितम् = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    यदि (!is_सममितम्) {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            uint8_t * खण्डम्_दत्त = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;
            scales[i] = ov::float16::from_द्वयङ्काः(*(uint16_t *) खण्डम्_दत्त);
            zp[i] = 128;
            प्रति (size_t j = 0; j < weights_प्रत्येकम्_खण्डम्; ++j) {
                uint8_t x = खण्डम्_दत्त[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_प्रत्येकम्_खण्डम् + j] = x;
            }
        });
    } अन्यथा {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_आकार(), [&](size_t i) {
            uint8_t * खण्डम्_दत्त = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;
            scales[i] = ov::float16::from_द्वयङ्काः(*(uint16_t *) खण्डम्_दत्त);
            // Copy int8 weights as-is (the tensor element type is i8)
            स्मृतिं_प्रतिलिख(weights + i * weights_प्रत्येकम्_खण्डम्, खण्डम्_दत्त + 2, weights_प्रत्येकम्_खण्डम्);
        });
    }
}

शून्य unpack_256_4(ध्रुव uint8_t * दत्त, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    प्रति (size_t i = 0; i < 4; ++i) {
        प्रति (पूर्णाङ्क j = 0; j < 32; ++j) {
            uint8_t x = (दत्त[i * 32 + j] & 0x0F);
            uint8_t y = (दत्त[i * 32 + j] >> 4);
            यदि (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

शून्य extract_q4_k_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 2 + 2 + 12 + 128;
    ध्रुव uint64_t n_super_खण्डम् = tensor->nb[3] / अष्टकानि_प्रत्येकम्_खण्डम्;

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    स्वचालित * zp_u4 = use_bias ? शून्यसूचकः : static_cast<uint8_t *>(zp_arr.दत्त());
    स्वचालित * bias_f16 = use_bias ? zp_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>() : शून्यसूचकः;

    ov::parallel_for(n_super_खण्डम्, [&](size_t i) {
        uint8_t * खण्डम्_दत्त = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;

        // Extract scale factors and offsets
        भिन्न scale_scales = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) खण्डम्_दत्त)));
        भिन्न scale_mins = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) खण्डम्_दत्त + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = खण्डम्_दत्त + 4;

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
        प्रति (पूर्णाङ्क j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            यदि (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } अन्यथा {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                यदि (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } अन्यथा {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(खण्डम्_दत्त + 16, weights + i * 128);
    });
}

शून्य extract_q6_k_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 128 + 64 + 16 + 2;
    ध्रुव uint64_t n_super_खण्डम् = tensor->nb[3] / अष्टकानि_प्रत्येकम्_खण्डम्;

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    तार्किक is_सममितम् = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    यदि (!is_सममितम्) {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());
        ov::parallel_for(n_super_खण्डम्, [&](size_t i) {
            uint8_t * खण्डम्_दत्त = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;
            भिन्न scale_गुणकः = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) खण्डम्_दत्त + 104)));
            प्रति (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_गुणकः * static_cast<भिन्न>(*((int8_t *) (खण्डम्_दत्त + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = खण्डम्_दत्त;
            uint8_t * qh = खण्डम्_दत्त + 128;
            प्रति (int64_t j = 0; j < 32; ++j) {
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
        ov::parallel_for(n_super_खण्डम्, [&](size_t i) {
            uint8_t * खण्डम्_दत्त = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;
            भिन्न scale_गुणकः = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) खण्डम्_दत्त + 104)));
            प्रति (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_गुणकः * static_cast<भिन्न>(*((int8_t *) (खण्डम्_दत्त + 128 + 64 + j))));
            }
            uint8_t * ql = खण्डम्_दत्त;
            uint8_t * qh = खण्डम्_दत्त + 128;
            स्वचालित * signed_weights = reinterpret_cast<int8_t *>(weights);
            प्रति (int64_t j = 0; j < 32; ++j) {
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

अन्तःस्थ शून्य get_scale_min_k4(पूर्णाङ्क j, ध्रुव uint8_t * q, uint8_t * d, uint8_t * m) {
    यदि (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } अन्यथा {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

शून्य extract_q5_k_दत्त(ध्रुव ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       तार्किक use_bias) {
    ध्रुव uint64_t अष्टकानि_प्रत्येकम्_खण्डम् = 4 + 12 + 32 + 128;
    ध्रुव uint64_t n_super_खण्डम् = tensor->nb[3] / अष्टकानि_प्रत्येकम्_खण्डम्;

    स्वचालित * दत्त = static_cast<uint8_t *>(tensor->दत्त);
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    स्वचालित * zp_u8 = use_bias ? शून्यसूचकः : static_cast<uint8_t *>(zp_arr.दत्त());
    स्वचालित * bias_f16 = use_bias ? zp_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>() : शून्यसूचकः;

    ov::parallel_for(n_super_खण्डम्, [&](size_t i) {
        uint8_t * खण्डम्_दत्त = दत्त + i * अष्टकानि_प्रत्येकम्_खण्डम्;

        ध्रुव भिन्न d = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) खण्डम्_दत्त)));
        ध्रुव भिन्न min_गुणकः = static_cast<भिन्न>(ov::float16::from_द्वयङ्काः(*((uint16_t *) खण्डम्_दत्त + 1)));

        ध्रुव uint8_t * scales_दत्त = खण्डम्_दत्त + 4;   // 12 bytes of scales
        ध्रुव uint8_t * qh = खण्डम्_दत्त + 4 + 12;       // 32 bytes of high bits
        ध्रुव uint8_t * ql = खण्डम्_दत्त + 4 + 12 + 32;  // 128 bytes of low bits

        पूर्णाङ्क is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        प्रति (पूर्णाङ्क j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_दत्त, &sc, &m);
            ध्रुव भिन्न d1 = d * sc;
            ध्रुव भिन्न m1 = min_गुणकः * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_दत्त, &sc, &m);
            ध्रुव भिन्न d2 = d * sc;
            ध्रुव भिन्न m2 = min_गुणकः * m;

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
            प्रति (पूर्णाङ्क l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            प्रति (पूर्णाङ्क l = 0; l < 32; ++l) {
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
                                       size_t गणः_आकार = GGML_QUANTIZATION_गणः_आकार,
                                       तार्किक use_bias = असत्य,
                                       तार्किक for_gather_matmul = असत्य) {
    ov::Shape orig_shape = weight.get_shape();
    तार्किक is_signed = (weight.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    स्वचालित scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= गणः_आकार;
    packed_shape.push_back(गणः_आकार);
    ध्रुव size_t गणः_dim = packed_shape.size() - 2;

    यदि (packed_shape[गणः_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + गणः_dim);
    } अन्यथा {
        scale_shape.push_back(1);
        scales.समुच्चयः_shape(scale_shape);
        यदि (!is_signed && zp.get_आकार() > 0) {
            स्वचालित zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.समुच्चयः_shape(zp_shape);
        }
    }

    स्वचालित scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> परिणाम;
    यदि (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        स्वचालित weights_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.दत्त()), शून्यसूचकः);
        weights_ग्रन्थिः->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वचालित weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ग्रन्थिः, ov::element::f16);
        स्वचालित mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
        परिणाम = mul;
    } अन्यथा {
        // Unsigned path
        स्वचालित weights_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.दत्त()), शून्यसूचकः);
        weights_ग्रन्थिः->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वचालित weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ग्रन्थिः, ov::element::f16);

        यदि (use_bias && zp.get_आकार() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            स्वचालित * bias_zp_दत्त = zp.दत्त<ov::float16>();
            ध्रुव स्वचालित * scale_दत्त = scales.दत्त<ov::float16>();
            ध्रुव size_t n = zp.get_आकार();
            प्रति (size_t i = 0; i < n; i++) {
                भिन्न s = static_cast<भिन्न>(scale_दत्त[i]);
                भिन्न b = static_cast<भिन्न>(bias_zp_दत्त[i]);
                bias_zp_दत्त[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            स्वचालित zero_point_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            स्वचालित w_zp =
                std::make_shared<ov::op::v1::व्यवकलय>(weights_f16, zero_point_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
            परिणाम = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
        } अन्यथा {
            // Zero point path: (w - zp) * s
            स्वचालित zero_point = std::make_shared<ov::op::v0::Constant>(zp);
            भिन्न zp_मानम्;
            यदि (ov::op::util::get_single_मानम्(zero_point, zp_मानम्)) {
                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_मानम्});
            }
            स्वचालित zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            स्वचालित w_zp =
                std::make_shared<ov::op::v1::व्यवकलय>(weights_f16, zero_point_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
            स्वचालित mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
            परिणाम = mul;
        }
    }

    यदि (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        स्वचालित final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(परिणाम, final_shape, असत्य);
        परिणाम = reshaped;
    }

    यदि (for_gather_matmul) {
        प्रत्यागम परिणाम;
    }
    प्रत्यागम std::make_shared<ov::op::v0::Convert>(परिणाम, ov::element::f32);
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
                                       size_t गणः_आकार = GGML_QUANTIZATION_गणः_आकार,
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
    packed_shape.back() /= गणः_आकार;
    packed_shape.push_back(गणः_आकार);
    ध्रुव size_t गणः_dim = packed_shape.size() - 2;

    यदि (packed_shape[गणः_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + गणः_dim);
    } अन्यथा {
        scale_shape.push_back(1);
        scales.समुच्चयः_shape(scale_shape);
        यदि (!is_signed && zp.get_आकार() > 0) {
            स्वचालित zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.समुच्चयः_shape(zp_shape);
        }
    }

    स्वचालित scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> परिणाम;
    यदि (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        स्वचालित weights_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.दत्त()), शून्यसूचकः);
        weights_ग्रन्थिः->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वचालित weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ग्रन्थिः, ov::element::f16);
        स्वचालित mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
        परिणाम = mul;
    } अन्यथा {
        // Unsigned path
        स्वचालित weights_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.दत्त()), शून्यसूचकः);
        weights_ग्रन्थिः->get_rt_info()["__gguf_tensor_holder"] = weight;
        स्वचालित weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ग्रन्थिः, ov::element::f16);

        यदि (use_bias && zp.get_आकार() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            स्वचालित * bias_zp_दत्त = zp.दत्त<ov::float16>();
            ध्रुव स्वचालित * scale_दत्त = scales.दत्त<ov::float16>();
            ध्रुव size_t n = zp.get_आकार();
            प्रति (size_t i = 0; i < n; i++) {
                भिन्न s = static_cast<भिन्न>(scale_दत्त[i]);
                भिन्न b = static_cast<भिन्न>(bias_zp_दत्त[i]);
                bias_zp_दत्त[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            स्वचालित zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            स्वचालित w_zp =
                std::make_shared<ov::op::v1::व्यवकलय>(weights_f16, zero_points_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
            परिणाम = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
        } अन्यथा {
            // Zero point path: (w - zp) * s
            स्वचालित zero_points_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(zp);
            भिन्न zp_मानम्;
            यदि (ov::op::util::get_single_मानम्(zero_points_ग्रन्थिः, zp_मानम्)) {
                zero_points_ग्रन्थिः = ov::op::v0::Constant::create(zero_points_ग्रन्थिः->get_element_type(), {}, {zp_मानम्});
            }
            स्वचालित zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_ग्रन्थिः, ov::element::f16);
            स्वचालित w_zp =
                std::make_shared<ov::op::v1::व्यवकलय>(weights_f16, zero_points_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
            स्वचालित mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_प्रसारणम्_Type::NUMPY);
            परिणाम = mul;
        }
    }

    यदि (packed_shape.size() != orig_weight_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        स्वचालित final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_weight_shape.size()},
                                                                  orig_weight_shape);
        स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(परिणाम, final_shape, असत्य);
        परिणाम = reshaped;
    }

    यदि (for_gather_matmul) {
        प्रत्यागम परिणाम;
    }
    प्रत्यागम std::make_shared<ov::op::v0::Convert>(परिणाम, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & weight, ov::Tensor & scales) {
    ध्रुव ov::Shape final_shape = weight.get_shape();
    GGML_ASSERT(!final_shape.रिक्तानि());
    GGML_ASSERT(final_shape.back() % MXFP4_खण्डम्_आकार == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_खण्डम्_आकार;
    packed_shape.push_back(MXFP4_खण्डम्_आकार);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.समुच्चयः_shape(scale_shape);

    स्वचालित weights_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(weight.दत्त()), शून्यसूचकः);
    weights_ग्रन्थिः->get_rt_info()["__gguf_tensor_holder"] = weight;
    स्वचालित weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_ग्रन्थिः, ov::element::f32);

    स्वचालित scales_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(scales);
    स्वचालित scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_ग्रन्थिः, ov::element::f32);
    ov::Output<ov::Node> परिणाम =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_प्रसारणम्_Type::NUMPY);

    स्वचालित final_shape_ग्रन्थिः =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    प्रत्यागम std::make_shared<ov::op::v1::Reshape>(परिणाम, final_shape_ग्रन्थिः, असत्य);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & weight) {
    स्वचालित weights_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::u8, weight.get_shape(),
                                                               static_cast<uint8_t *>(weight.दत्त()), शून्यसूचकः);
    weights_ग्रन्थिः->get_rt_info()["__gguf_tensor_holder"] = weight;
    weights_ग्रन्थिः->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = सत्य;
    प्रत्यागम weights_ग्रन्थिः;
}

शून्य quantize_q4_0(ध्रुव भिन्न * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    प्रतिज्ञा(k % qk == 0);
    ध्रुव पूर्णाङ्क nb = k / qk;

    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();
    तार्किक is_सममितम् = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    यदि (!is_सममितम्) {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());
        प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            भिन्न max = 0.0f;
            प्रति (पूर्णाङ्क j = 0; j < qk; j++) {
                ध्रुव भिन्न v = x[i * qk + j];
                यदि (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ध्रुव भिन्न d = max / -8;
            यदि (d == 0) {
                scales[i] = ov::float16(1.0f);
                यदि (i % 2 == 0) {
                    zp[i / 2] = 8;
                } अन्यथा {
                    zp[i / 2] |= (8 << 4);
                }
                स्मृतिं_पूरय(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                अनुवर्त;
            }
            ध्रुव भिन्न id = 1.0f / d;
            scales[i] = ov::float16(d);
            यदि (i % 2 == 0) {
                zp[i / 2] = 8;
            } अन्यथा {
                zp[i / 2] |= (8 << 4);
            }
            प्रति (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
                ध्रुव भिन्न x0 = x[i * qk + 2 * j] * id;
                ध्रुव भिन्न x1 = x[i * qk + 2 * j + 1] * id;
                ध्रुव uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                ध्रुव uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } अन्यथा {
        // Symmetric: produce signed i4 values in [-8, 7]
        प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            भिन्न max = 0.0f;
            प्रति (पूर्णाङ्क j = 0; j < qk; j++) {
                ध्रुव भिन्न v = x[i * qk + j];
                यदि (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ध्रुव भिन्न d = max / -8;
            यदि (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                स्मृतिं_पूरय(weights + i * qk / 2, 0, qk / 2);
                अनुवर्त;
            }
            ध्रुव भिन्न id = 1.0f / d;
            scales[i] = ov::float16(d);
            प्रति (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
                ध्रुव भिन्न x0 = x[i * qk + 2 * j] * id;
                ध्रुव भिन्न x1 = x[i * qk + 2 * j + 1] * id;
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
शून्य quantize_q4_1_asym(ध्रुव भिन्न * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    प्रतिज्ञा(k % qk == 0);
    ध्रुव पूर्णाङ्क nb = k / qk;

    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त());
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>();
    स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    स्वचालित store_zp = [zp](पूर्णाङ्क i, uint8_t v) {
        यदि (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } अन्यथा {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
        भिन्न vmin = x[i * qk];
        भिन्न vmax = x[i * qk];
        प्रति (पूर्णाङ्क j = 1; j < qk; j++) {
            ध्रुव भिन्न v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        ध्रुव भिन्न d = (vmax - vmin) / 15.0f;
        यदि (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            स्मृतिं_पूरय(weights + i * qk / 2, 0, qk / 2);
            अनुवर्त;
        }
        ध्रुव भिन्न id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        ध्रुव पूर्णाङ्क zq = std::max(0, std::min(15, (पूर्णाङ्क) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        प्रति (पूर्णाङ्क j = 0; j < qk / 2; ++j) {
            ध्रुव भिन्न x0 = x[i * qk + 2 * j] * id;
            ध्रुव भिन्न x1 = x[i * qk + 2 * j + 1] * id;
            ध्रुव uint8_t q0 = (uint8_t) std::max(0, std::min(15, (पूर्णाङ्क) lroundf(x0) + zq));
            ध्रुव uint8_t q1 = (uint8_t) std::max(0, std::min(15, (पूर्णाङ्क) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

शून्य quantize_q8_0(ध्रुव भिन्न * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t खण्डम्_विस्थापनम् = 0) {
    प्रतिज्ञा(k % qk == 0);
    ध्रुव पूर्णाङ्क nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त()) + खण्डम्_विस्थापनम् * qk;
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>() + खण्डम्_विस्थापनम्;
    तार्किक is_सममितम् = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    यदि (!is_सममितम्) {
        स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त()) + खण्डम्_विस्थापनम्;
        प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            प्रति (पूर्णाङ्क j = 0; j < qk; j++) {
                ध्रुव भिन्न v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ध्रुव भिन्न d = amax / 127.0f;
            ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            प्रति (पूर्णाङ्क j = 0; j < qk; ++j) {
                ध्रुव भिन्न x0 = x[i * qk + j] * id;
                ध्रुव int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } अन्यथा {
        // Symmetric: store signed int8 values directly
        स्वचालित * signed_weights = reinterpret_cast<int8_t *>(weights);
        प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
            भिन्न amax = 0.0f;
            प्रति (पूर्णाङ्क j = 0; j < qk; j++) {
                ध्रुव भिन्न v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ध्रुव भिन्न d = amax / 127.0f;
            ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            प्रति (पूर्णाङ्क j = 0; j < qk; ++j) {
                ध्रुव भिन्न x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

शून्य quantize_q8_1(ध्रुव भिन्न * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t खण्डम्_विस्थापनम् = 0) {
    प्रतिज्ञा(k % qk == 0);
    ध्रुव पूर्णाङ्क nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    स्वचालित * weights = static_cast<uint8_t *>(weights_arr.दत्त()) + खण्डम्_विस्थापनम् * qk;
    स्वचालित * scales = scales_arr.दत्त<ov::element_type_traits<ov::element::f16>::value_type>() + खण्डम्_विस्थापनम्;
    स्वचालित * zp = static_cast<uint8_t *>(zp_arr.दत्त()) + खण्डम्_विस्थापनम्;
    प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
        भिन्न min = std::numeric_limits<भिन्न>::max();
        भिन्न max = std::numeric_limits<भिन्न>::lowest();

        प्रति (पूर्णाङ्क j = 0; j < qk; j++) {
            ध्रुव भिन्न v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        ध्रुव भिन्न d = (max - min) / ((1 << 8) - 1);
        ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        प्रति (पूर्णाङ्क j = 0; j < qk; ++j) {
            ध्रुव भिन्न x0 = (x[i * qk + j] - min) * id;
            ध्रुव uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> extract_quantized_weights(ध्रुव ggml_tensor * tensor,
                                                    ध्रुव शून्य * दत्त,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    तार्किक use_bias = असत्य) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor अस्थायी_tensor = *tensor;
    अस्थायी_tensor.दत्त = const_cast<शून्य *>(दत्त);

    यदि (tensor->type == GGML_TYPE_MXFP4) {
        extract_mxfp4_दत्त(&अस्थायी_tensor, weights, scales);
        स्वचालित परिणाम = make_mxfp4_weights(weights, scales).get_ग्रन्थिः_shared_ptr();
        परिणाम->समुच्चयः_friendly_name(tensor->name);
        प्रत्यागम परिणाम;
    }

    // Determine block size based on tensor type
    int64_t weights_प्रत्येकम्_खण्डम्;
    तार्किक is_u4;
    विकल्प (tensor->type) {
    पक्ष GGML_TYPE_Q4_0:
    पक्ष GGML_TYPE_Q4_1:
    पक्ष GGML_TYPE_Q4_K:
        is_u4 = सत्य;
        weights_प्रत्येकम्_खण्डम् = 32;
        भङ्ग;
    पक्ष GGML_TYPE_Q8_0:
    पक्ष GGML_TYPE_Q5_1:
    पक्ष GGML_TYPE_Q5_K:
        is_u4 = असत्य;
        weights_प्रत्येकम्_खण्डम् = 32;
        भङ्ग;
    पक्ष GGML_TYPE_Q6_K:
        is_u4 = असत्य;
        weights_प्रत्येकम्_खण्डम् = 16;
        भङ्ग;
    स्वभाव:
        क्षिप std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    ध्रुव तार्किक for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    विकल्प (tensor->type) {
    पक्ष GGML_TYPE_Q4_0:
        extract_q4_0_दत्त(&अस्थायी_tensor, weights, scales, zp);
        भङ्ग;
    पक्ष GGML_TYPE_Q4_1:
        extract_q4_1_दत्त(&अस्थायी_tensor, weights, scales, zp, use_bias);
        भङ्ग;
    पक्ष GGML_TYPE_Q4_K:
        extract_q4_k_दत्त(&अस्थायी_tensor, weights, scales, zp, use_bias);
        भङ्ग;
    पक्ष GGML_TYPE_Q5_1:
        extract_q5_1_दत्त(&अस्थायी_tensor, weights, scales, zp, use_bias);
        भङ्ग;
    पक्ष GGML_TYPE_Q8_0:
        extract_q8_0_दत्त(&अस्थायी_tensor, weights, scales, zp);
        भङ्ग;
    पक्ष GGML_TYPE_Q6_K:
        extract_q6_k_दत्त(&अस्थायी_tensor, weights, scales, zp);
        भङ्ग;
    पक्ष GGML_TYPE_Q5_K:
        extract_q5_k_दत्त(&अस्थायी_tensor, weights, scales, zp, use_bias);
        भङ्ग;
    स्वभाव:
        क्षिप std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> weight_ग्रन्थिः;
    यदि (is_u4) {
        weight_ग्रन्थिः = make_int4_weights(weights, scales, zp, weights_प्रत्येकम्_खण्डम्, use_bias, for_gather_matmul);
    } अन्यथा {
        weight_ग्रन्थिः = make_int8_weights(weights, scales, zp, weights_प्रत्येकम्_खण्डम्, use_bias, for_gather_matmul);
    }

    स्वचालित परिणाम = weight_ग्रन्थिः.get_ग्रन्थिः_shared_ptr();
    परिणाम->समुच्चयः_friendly_name(tensor->name);
    प्रत्यागम परिणाम;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_गन्तव्यम्_buffers(ध्रुव ggml_tensor * tensor,
                                                ध्रुव शून्य * दत्त,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t खण्डम्_आकार_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ध्रुव int64_t ne0 = tensor->ne[0];                 // elements per row
    ध्रुव int64_t n_पङ्क्तिकाः = n_elements / ne0;
    ध्रुव स्वचालित * type_traits = ggml_get_type_traits(tensor->type);
    ध्रुव size_t src_row_अष्टकानि = ggml_row_आकार(tensor->type, ne0);

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
    ध्रुव तार्किक प्रवाहः_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(खण्डम्_आकार_3 > 0 && ne0 % खण्डम्_आकार_3 != 0);

    यदि (!प्रवाहः_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<भिन्न> weights_f32(n_elements);
        type_traits->गन्तव्यम्_float(दत्त, weights_f32.दत्त(), n_elements);
        यदि (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.दत्त(), weights.दत्त(), n_elements);
            स्वचालित परिणाम = std::make_shared<ov::op::v0::Constant>(weights);
            परिणाम->समुच्चयः_friendly_name(tensor->name);
            प्रत्यागम परिणाम;
        }
        यदि (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.दत्त(), weights, scales, zp, n_elements, खण्डम्_आकार_3);
        } अन्यथा यदि (is_u4) {
            quantize_q4_0(weights_f32.दत्त(), weights, scales, zp, n_elements, खण्डम्_आकार_3);
        } अन्यथा यदि (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.दत्त(), weights, scales, zp, n_elements, खण्डम्_आकार_3);
        } अन्यथा {
            quantize_q8_0(weights_f32.दत्त(), weights, scales, zp, n_elements, खण्डम्_आकार_3);
        }
    } अन्यथा {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ध्रुव int64_t CHUNK_पङ्क्तिकाः = std::min<int64_t>(n_पङ्क्तिकाः, 256);
        std::vector<भिन्न> scratch(CHUNK_पङ्क्तिकाः * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        स्वचालित * f16_आधारः = static_cast<uint8_t *>(weights.दत्त());
        प्रति (int64_t r0 = 0; r0 < n_पङ्क्तिकाः; r0 += CHUNK_पङ्क्तिकाः) {
            ध्रुव int64_t पङ्क्तिकाः = std::min(CHUNK_पङ्क्तिकाः, n_पङ्क्तिकाः - r0);
            ध्रुव int64_t elems = पङ्क्तिकाः * ne0;
            ध्रुव स्वचालित * src = static_cast<ध्रुव uint8_t *>(दत्त) + r0 * src_row_अष्टकानि;
            type_traits->गन्तव्यम्_float(src, scratch.दत्त(), elems);

            यदि (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.दत्त(), f16_आधारः + (r0 * ne0) * परिमाण(uint16_t), elems);
            } अन्यथा {
                ध्रुव int64_t खण्डम्_विस्थापनम् = (r0 * ne0) / खण्डम्_आकार_3;
                यदि (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.दत्त(), weights, scales, zp, elems, खण्डम्_आकार_3, खण्डम्_विस्थापनम्);
                } अन्यथा {
                    quantize_q8_0(scratch.दत्त(), weights, scales, zp, elems, खण्डम्_आकार_3, खण्डम्_विस्थापनम्);
                }
            }
        }
        यदि (requant_type == ExtraQuantType::F16) {
            स्वचालित परिणाम = std::make_shared<ov::op::v0::Constant>(weights);
            परिणाम->समुच्चयः_friendly_name(tensor->name);
            प्रत्यागम परिणाम;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_ग्रन्थिः;
    यदि (is_u4) {
        weight_ग्रन्थिः = make_int4_weights(weights, scales, zp, खण्डम्_आकार_3);
    } अन्यथा {
        weight_ग्रन्थिः = make_int8_weights(weights, scales, zp, खण्डम्_आकार_3);
    }

    स्वचालित परिणाम = weight_ग्रन्थिः.get_ग्रन्थिः_shared_ptr();
    परिणाम->समुच्चयः_friendly_name(tensor->name);
    प्रत्यागम परिणाम;
}
}  // namespace

OvWeight process_weight_tensor(ध्रुव ggml_tensor * tensor, ध्रुव शून्य * दत्त, शून्य * output_आधारः_ptr, तार्किक use_bias) {
    GGML_ASSERT(tensor != शून्यसूचकः);
    GGML_ASSERT(दत्त != शून्यसूचकः);

    OvWeight परिणाम;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape ग्रन्थिः_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    यदि (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        विकल्प (tensor->type) {
        पक्ष GGML_TYPE_F32:
            element_type = ov::element::f32;
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            element_type = ov::element::f16;
            भङ्ग;
        पक्ष GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            भङ्ग;
        स्वभाव:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        यदि (output_आधारः_ptr && output_आधारः_ptr != दत्त) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_अष्टकानि = ggml_nbytes(tensor);
            स्मृतिं_प्रतिलिख(output_आधारः_ptr, दत्त, tensor_अष्टकानि);
            परिणाम.weights = ov::Tensor(element_type, ग्रन्थिः_shape, output_आधारः_ptr);
        } अन्यथा {
            परिणाम.weights = ov::Tensor(element_type, ग्रन्थिः_shape, दत्त);
        }
        परिणाम.weight_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(परिणाम.weights);
        प्रत्यागम परिणाम;
    }

    // Handle quantized weights
    यदि (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    परिणाम.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ध्रुव स्वचालित & layout = परिणाम.layout;
    यदि (layout.योग_आकार == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    ध्रुव तार्किक for_gather_matmul = tensor->ne[2] > 1;
    ध्रुव तार्किक zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    ध्रुव तार्किक is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    यदि (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_खण्डम्_आकार),
                                  MXFP4_खण्डम्_अष्टकानि};
        ध्रुव size_t tensor_अष्टकानि = ggml_nbytes(tensor);
        यदि (output_आधारः_ptr) {
            स्वचालित * buf_आधारः = static_cast<uint8_t *>(output_आधारः_ptr);
            स्मृतिं_प्रतिलिख(buf_आधारः + layout.weights_विस्थापनम्, दत्त, tensor_अष्टकानि);
            परिणाम.weights = ov::Tensor(ov::element::u8, packed_shape, buf_आधारः + layout.weights_विस्थापनम्);
        } अन्यथा {
            परिणाम.weights = ov::Tensor(ov::element::u8, packed_shape);
            स्मृतिं_प्रतिलिख(परिणाम.weights.दत्त(), दत्त, tensor_अष्टकानि);
        }
        परिणाम.weight_ग्रन्थिः = make_mxfp4_moe_packed_weights(परिणाम.weights).get_ग्रन्थिः_shared_ptr();
        परिणाम.weight_ग्रन्थिः->समुच्चयः_friendly_name(tensor->name);
        प्रत्यागम परिणाम;
    }

    यदि (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_आधारः_ptr = शून्यसूचकः;
    }

    // F16 requant path - no separate scales/zp needed in result
    यदि (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        यदि (output_आधारः_ptr) {
            परिणाम.weights = ov::Tensor(ov::element::f16, ग्रन्थिः_shape,
                                        static_cast<uint8_t *>(output_आधारः_ptr) + layout.weights_विस्थापनम्);
        } अन्यथा {
            परिणाम.weights = ov::Tensor(ov::element::f16, ग्रन्थिः_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        परिणाम.weight_ग्रन्थिः =
            requantize_गन्तव्यम्_buffers(tensor, दत्त, ExtraQuantType::F16, 0, परिणाम.weights, dummy_scales, dummy_zp);
        प्रत्यागम परिणाम;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type;
    यदि (tensor->type == GGML_TYPE_MXFP4) {
        weight_type = ov::element::f4e2m1;
    } अन्यथा यदि (layout.is_सममितम्) {
        weight_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } अन्यथा {
        weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = ग्रन्थिः_shape;
    scale_shape.back() /= layout.weights_प्रत्येकम्_खण्डम्;

    यदि (tensor->type == GGML_TYPE_MXFP4) {
        यदि (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            ग्रन्थिः_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } अन्यथा {
            ग्रन्थिः_shape.clear();
            प्रति (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                ग्रन्थिः_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = ग्रन्थिः_shape;
        scale_shape.back() /= layout.weights_प्रत्येकम्_खण्डम्;
    }

    ध्रुव ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    यदि (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    यदि (output_आधारः_ptr) {
        uint8_t * buf_आधारः = static_cast<uint8_t *>(output_आधारः_ptr);
        परिणाम.weights = ov::Tensor(weight_type, ग्रन्थिः_shape, buf_आधारः + layout.weights_विस्थापनम्);
        परिणाम.scales = ov::Tensor(scale_type, scale_shape, buf_आधारः + layout.scales_विस्थापनम्);
        यदि (!layout.is_सममितम्) {
            परिणाम.zp = ov::Tensor(zp_type, scale_shape, buf_आधारः + layout.zp_विस्थापनम्);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } अन्यथा {
        परिणाम.weights = ov::Tensor(weight_type, ग्रन्थिः_shape);
        परिणाम.scales = ov::Tensor(scale_type, scale_shape);
        यदि (!layout.is_सममितम्) {
            परिणाम.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    यदि (layout.is_requant && layout.requant_type.has_value()) {
        परिणाम.weight_ग्रन्थिः = requantize_गन्तव्यम्_buffers(tensor, दत्त, layout.requant_type.value(), layout.weights_प्रत्येकम्_खण्डम्,
                                                   परिणाम.weights, परिणाम.scales, परिणाम.zp);
    } अन्यथा {
        परिणाम.weight_ग्रन्थिः =
            extract_quantized_weights(tensor, दत्त, परिणाम.weights, परिणाम.scales, परिणाम.zp, use_bias);
    }

    प्रत्यागम परिणाम;
}
