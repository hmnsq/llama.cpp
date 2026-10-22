#ಅನ್ತರ್ಭಾವ "ggml-quants.h"

#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/except.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/parallel.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/element_type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/element_type_traits.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/float16.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/float4_e2m1.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/float8_e8m0.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/util/attr_types.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/constant_folding.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
ನಾಮಾಕಾಶ ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
ತಾರ್ಕಿಕ get_single_ಮೌಲ್ಯ(ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::op::v0::Constant> & const_ಗಂಟು,
                      ದಶಮಾಂಶ & value,
                      ತಾರ್ಕಿಕ check_ಮೌಲ್ಯ_range = ನಿಜ);
}  // namespace ov::op::util

ನಾಮಾಕಾಶ {
ಶೂನ್ಯ unpack_32_4(ಸ್ಥಿರಾಂಕ uint8_t * ದತ್ತಾಂಶ, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) {
        uint8_t x = (ದತ್ತಾಂಶ[j] & 0x0F);
        uint8_t y = (ದತ್ತಾಂಶ[j] >> 4);
        ಆದರೆ (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_ಖಂಡ_ಗಾತ್ರ = 32;
constexpr size_t MXFP4_ಖಂಡ_QS_ಗಾತ್ರ = MXFP4_ಖಂಡ_ಗಾತ್ರ / 2;
constexpr size_t MXFP4_ಖಂಡ_ಬೈಟುಗಳು = ಗಾತ್ರ(uint8_t) + MXFP4_ಖಂಡ_QS_ಗಾತ್ರ;

ಶೂನ್ಯ pack_32_mxfp4_for_openvino(ಸ್ಥಿರಾಂಕ uint8_t * ದತ್ತಾಂಶ, uint8_t * dst) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < static_cast<ಪೂರ್ಣಾಂಕ>(MXFP4_ಖಂಡ_QS_ಗಾತ್ರ); j += 2) {
        ಸ್ಥಿರಾಂಕ uint8_t v0 = ದತ್ತಾಂಶ[j] & 0x0F;
        ಸ್ಥಿರಾಂಕ uint8_t v1 = (ದತ್ತಾಂಶ[j + 1] & 0x0F) << 4;
        ಸ್ಥಿರಾಂಕ uint8_t v16 = ದತ್ತಾಂಶ[j] >> 4;
        ಸ್ಥಿರಾಂಕ uint8_t v17 = ದತ್ತಾಂಶ[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_ಖಂಡ_ಗಾತ್ರ / 4 + j / 2] = v16 | v17;
    }
}

ಶೂನ್ಯ ಹೊರತೆಗೆ_mxfp4_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ಸ್ಥಿರಾಂಕ size_t n_ಖಂಡಗಳು = scales_arr.get_ಗಾತ್ರ();

    ov::parallel_for(n_ಖಂಡಗಳು, [&](size_t i) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಖಂಡ = ದತ್ತಾಂಶ + i * MXFP4_ಖಂಡ_ಬೈಟುಗಳು;
        pack_32_mxfp4_for_openvino(ಖಂಡ + ಗಾತ್ರ(uint8_t), weights + i * MXFP4_ಖಂಡ_QS_ಗಾತ್ರ);
        scales[i] = ov::float8_e8m0::from_ಬಿಟ್ಟುಗಳು(ಖಂಡ[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
ಶೂನ್ಯ ಹೊರತೆಗೆ_q4_0_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 18;  // 2 bytes scale, 32x0.5 byte weights

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    ತಾರ್ಕಿಕ is_ಸಮಮಿತ = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    ಆದರೆ (!is_ಸಮಮಿತ) {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            scales[i] = ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) (ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ)));
            // Pack two 4-bit zero points per byte
            ಆದರೆ (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } ಇಲ್ಲದಿದ್ದರೆ {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ + 2, weights + i * 16);
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            scales[i] = ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) (ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ)));
            unpack_32_4(ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
ಶೂನ್ಯ ಹೊರತೆಗೆ_q4_1_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ತಾರ್ಕಿಕ use_bias) {
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    ಆದರೆ (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        ಸ್ವಯಂಚಾಲಿತ * bias = zp_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            ದಶಮಾಂಶ scale = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) (ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ))));
            ದಶಮಾಂಶ min = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) (ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ + 4, weights + i * 16);
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            ದಶಮಾಂಶ scale = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) (ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ))));
            ದಶಮಾಂಶ min = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) (ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            ಆದರೆ (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } ಇಲ್ಲದಿದ್ದರೆ {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
ಶೂನ್ಯ ಹೊರತೆಗೆ_q5_1_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ತಾರ್ಕಿಕ use_bias) {
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = 32;

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());  // u8 weights, one byte per weight
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    ಸ್ವಯಂಚಾಲಿತ read_u16 = [](ಸ್ಥಿರಾಂಕ uint8_t * p) {
        uint16_t v;
        ಸ್ಮರಣೆ_ನಕಲು(&v, p, ಗಾತ್ರ(v));
        ಹಿಂತಿರುಗಿ v;
    };

    ಸ್ವಯಂಚಾಲಿತ unpack_ಖಂಡ = [&](ಸ್ಥಿರಾಂಕ uint8_t * ಖಂಡ, uint8_t * dst) {
        uint32_t qh;
        ಸ್ಮರಣೆ_ನಕಲು(&qh, ಖಂಡ + 4, ಗಾತ್ರ(uint32_t));
        ಸ್ಥಿರಾಂಕ uint8_t * qs = ಖಂಡ + 8;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk / 2; ++j) {
            ಸ್ಥಿರಾಂಕ uint8_t lo = qs[j] & 0x0F;
            ಸ್ಥಿರಾಂಕ uint8_t hi = qs[j] >> 4;
            ಸ್ಥಿರಾಂಕ uint8_t ಬಿಟ್_lo = (qh >> j) & 1;
            ಸ್ಥಿರಾಂಕ uint8_t ಬಿಟ್_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (ಬಿಟ್_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (ಬಿಟ್_hi << 4);  // last 16 weights
        }
    };

    ಆದರೆ (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        ಸ್ವಯಂಚಾಲಿತ * bias = zp_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಖಂಡ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;
            ದಶಮಾಂಶ scale = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(read_u16(ಖಂಡ)));
            ದಶಮಾಂಶ min = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(read_u16(ಖಂಡ + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_ಖಂಡ(ಖಂಡ, weights + i * qk);
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());  // u8 zero points
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಖಂಡ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;
            ದಶಮಾಂಶ scale = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(read_u16(ಖಂಡ)));
            ದಶಮಾಂಶ min = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(read_u16(ಖಂಡ + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_ಖಂಡ(ಖಂಡ, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
ಶೂನ್ಯ ಹೊರತೆಗೆ_q8_0_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ಸ್ಥಿರಾಂಕ uint64_t weights_ಪ್ರತಿ_ಖಂಡ = 32;
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 34;  // 2 bytes scale, 32x1 byte weights

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    ತಾರ್ಕಿಕ is_ಸಮಮಿತ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ಆದರೆ (!is_ಸಮಮಿತ) {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            uint8_t * ಖಂಡ_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;
            scales[i] = ov::float16::from_ಬಿಟ್ಟುಗಳು(*(uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ);
            zp[i] = 128;
            ಗಾಗಿ (size_t j = 0; j < weights_ಪ್ರತಿ_ಖಂಡ; ++j) {
                uint8_t x = ಖಂಡ_ದತ್ತಾಂಶ[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_ಪ್ರತಿ_ಖಂಡ + j] = x;
            }
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_ಗಾತ್ರ(), [&](size_t i) {
            uint8_t * ಖಂಡ_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;
            scales[i] = ov::float16::from_ಬಿಟ್ಟುಗಳು(*(uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ);
            // Copy int8 weights as-is (the tensor element type is i8)
            ಸ್ಮರಣೆ_ನಕಲು(weights + i * weights_ಪ್ರತಿ_ಖಂಡ, ಖಂಡ_ದತ್ತಾಂಶ + 2, weights_ಪ್ರತಿ_ಖಂಡ);
        });
    }
}

ಶೂನ್ಯ unpack_256_4(ಸ್ಥಿರಾಂಕ uint8_t * ದತ್ತಾಂಶ, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    ಗಾಗಿ (size_t i = 0; i < 4; ++i) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; ++j) {
            uint8_t x = (ದತ್ತಾಂಶ[i * 32 + j] & 0x0F);
            uint8_t y = (ದತ್ತಾಂಶ[i * 32 + j] >> 4);
            ಆದರೆ (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

ಶೂನ್ಯ ಹೊರತೆಗೆ_q4_k_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ತಾರ್ಕಿಕ use_bias) {
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 2 + 2 + 12 + 128;
    ಸ್ಥಿರಾಂಕ uint64_t n_super_ಖಂಡ = tensor->nb[3] / ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    ಸ್ವಯಂಚಾಲಿತ * zp_u4 = use_bias ? ಶೂನ್ಯಸೂಚಕ : static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * bias_f16 = use_bias ? zp_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>() : ಶೂನ್ಯಸೂಚಕ;

    ov::parallel_for(n_super_ಖಂಡ, [&](size_t i) {
        uint8_t * ಖಂಡ_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;

        // Extract scale factors and offsets
        ದಶಮಾಂಶ scale_scales = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ)));
        ದಶಮಾಂಶ scale_mins = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = ಖಂಡ_ದತ್ತಾಂಶ + 4;

        // Calculate scales
        ದಶಮಾಂಶ scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<ದಶಮಾಂಶ>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        ದಶಮಾಂಶ min_vals[8];
        min_vals[0] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<ದಶಮಾಂಶ>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            ಆದರೆ (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                ಆದರೆ (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(ಖಂಡ_ದತ್ತಾಂಶ + 16, weights + i * 128);
    });
}

ಶೂನ್ಯ ಹೊರತೆಗೆ_q6_k_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 128 + 64 + 16 + 2;
    ಸ್ಥಿರಾಂಕ uint64_t n_super_ಖಂಡ = tensor->nb[3] / ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    ತಾರ್ಕಿಕ is_ಸಮಮಿತ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ಆದರೆ (!is_ಸಮಮಿತ) {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
        ov::parallel_for(n_super_ಖಂಡ, [&](size_t i) {
            uint8_t * ಖಂಡ_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;
            ದಶಮಾಂಶ scale_ಗುಣಕ = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ + 104)));
            ಗಾಗಿ (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ಗುಣಕ * static_cast<ದಶಮಾಂಶ>(*((int8_t *) (ಖಂಡ_ದತ್ತಾಂಶ + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = ಖಂಡ_ದತ್ತಾಂಶ;
            uint8_t * qh = ಖಂಡ_ದತ್ತಾಂಶ + 128;
            ಗಾಗಿ (int64_t j = 0; j < 32; ++j) {
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
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_ಖಂಡ, [&](size_t i) {
            uint8_t * ಖಂಡ_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;
            ದಶಮಾಂಶ scale_ಗುಣಕ = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ + 104)));
            ಗಾಗಿ (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ಗುಣಕ * static_cast<ದಶಮಾಂಶ>(*((int8_t *) (ಖಂಡ_ದತ್ತಾಂಶ + 128 + 64 + j))));
            }
            uint8_t * ql = ಖಂಡ_ದತ್ತಾಂಶ;
            uint8_t * qh = ಖಂಡ_ದತ್ತಾಂಶ + 128;
            ಸ್ವಯಂಚಾಲಿತ * signed_weights = reinterpret_cast<int8_t *>(weights);
            ಗಾಗಿ (int64_t j = 0; j < 32; ++j) {
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

ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * q, uint8_t * d, uint8_t * m) {
    ಆದರೆ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ಶೂನ್ಯ ಹೊರತೆಗೆ_q5_k_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ತಾರ್ಕಿಕ use_bias) {
    ಸ್ಥಿರಾಂಕ uint64_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ = 4 + 12 + 32 + 128;
    ಸ್ಥಿರಾಂಕ uint64_t n_super_ಖಂಡ = tensor->nb[3] / ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;

    ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    ಸ್ವಯಂಚಾಲಿತ * zp_u8 = use_bias ? ಶೂನ್ಯಸೂಚಕ : static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * bias_f16 = use_bias ? zp_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>() : ಶೂನ್ಯಸೂಚಕ;

    ov::parallel_for(n_super_ಖಂಡ, [&](size_t i) {
        uint8_t * ಖಂಡ_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + i * ಬೈಟುಗಳು_ಪ್ರತಿ_ಖಂಡ;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ)));
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min_ಗುಣಕ = static_cast<ದಶಮಾಂಶ>(ov::float16::from_ಬಿಟ್ಟುಗಳು(*((uint16_t *) ಖಂಡ_ದತ್ತಾಂಶ + 1)));

        ಸ್ಥಿರಾಂಕ uint8_t * scales_ದತ್ತಾಂಶ = ಖಂಡ_ದತ್ತಾಂಶ + 4;   // 12 bytes of scales
        ಸ್ಥಿರಾಂಕ uint8_t * qh = ಖಂಡ_ದತ್ತಾಂಶ + 4 + 12;       // 32 bytes of high bits
        ಸ್ಥಿರಾಂಕ uint8_t * ql = ಖಂಡ_ದತ್ತಾಂಶ + 4 + 12 + 32;  // 128 bytes of low bits

        ಪೂರ್ಣಾಂಕ is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_ದತ್ತಾಂಶ, &sc, &m);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = d * sc;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = min_ಗುಣಕ * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_ದತ್ತಾಂಶ, &sc, &m);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2 = d * sc;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m2 = min_ಗುಣಕ * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            ಆದರೆ (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
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
ov::Output<ov::Node> make_int8_weights(ov::Tensor & ತೂಕ,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t ಗುಂಪು_ಗಾತ್ರ = GGML_QUANTIZATION_ಗುಂಪು_ಗಾತ್ರ,
                                       ತಾರ್ಕಿಕ use_bias = ಸುಳ್ಳು,
                                       ತಾರ್ಕಿಕ for_gather_matmul = ಸುಳ್ಳು) {
    ov::Shape orig_shape = ತೂಕ.get_shape();
    ತಾರ್ಕಿಕ is_signed = (ತೂಕ.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ಸ್ವಯಂಚಾಲಿತ scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= ಗುಂಪು_ಗಾತ್ರ;
    packed_shape.push_back(ಗುಂಪು_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_dim = packed_shape.size() - 2;

    ಆದರೆ (packed_shape[ಗುಂಪು_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + ಗುಂಪು_dim);
    } ಇಲ್ಲದಿದ್ದರೆ {
        scale_shape.push_back(1);
        scales.ಗಣ_shape(scale_shape);
        ಆದರೆ (!is_signed && zp.get_ಗಾತ್ರ() > 0) {
            ಸ್ವಯಂಚಾಲಿತ zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.ಗಣ_shape(zp_shape);
        }
    }

    ಸ್ವಯಂಚಾಲಿತ scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ಫಲಿತಾಂಶ;
    ಆದರೆ (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        ಸ್ವಯಂಚಾಲಿತ weights_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(ತೂಕ.ದತ್ತಾಂಶ()), ಶೂನ್ಯಸೂಚಕ);
        weights_ಗಂಟು->get_rt_info()["__gguf_tensor_holder"] = ತೂಕ;
        ಸ್ವಯಂಚಾಲಿತ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ಗಂಟು, ov::element::f16);
        ಸ್ವಯಂಚಾಲಿತ mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
        ಫಲಿತಾಂಶ = mul;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Unsigned path
        ಸ್ವಯಂಚಾಲಿತ weights_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(ತೂಕ.ದತ್ತಾಂಶ()), ಶೂನ್ಯಸೂಚಕ);
        weights_ಗಂಟು->get_rt_info()["__gguf_tensor_holder"] = ತೂಕ;
        ಸ್ವಯಂಚಾಲಿತ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ಗಂಟು, ov::element::f16);

        ಆದರೆ (use_bias && zp.get_ಗಾತ್ರ() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            ಸ್ವಯಂಚಾಲಿತ * bias_zp_ದತ್ತಾಂಶ = zp.ದತ್ತಾಂಶ<ov::float16>();
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * scale_ದತ್ತಾಂಶ = scales.ದತ್ತಾಂಶ<ov::float16>();
            ಸ್ಥಿರಾಂಕ size_t n = zp.get_ಗಾತ್ರ();
            ಗಾಗಿ (size_t i = 0; i < n; i++) {
                ದಶಮಾಂಶ s = static_cast<ದಶಮಾಂಶ>(scale_ದತ್ತಾಂಶ[i]);
                ದಶಮಾಂಶ b = static_cast<ದಶಮಾಂಶ>(bias_zp_ದತ್ತಾಂಶ[i]);
                bias_zp_ದತ್ತಾಂಶ[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            ಸ್ವಯಂಚಾಲಿತ zero_ಬಿಂದು_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            ಸ್ವಯಂಚಾಲಿತ w_zp =
                std::make_shared<ov::op::v1::ಕಳೆ>(weights_f16, zero_ಬಿಂದು_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
            ಫಲಿತಾಂಶ = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Zero point path: (w - zp) * s
            ಸ್ವಯಂಚಾಲಿತ zero_ಬಿಂದು = std::make_shared<ov::op::v0::Constant>(zp);
            ದಶಮಾಂಶ zp_ಮೌಲ್ಯ;
            ಆದರೆ (ov::op::util::get_single_ಮೌಲ್ಯ(zero_ಬಿಂದು, zp_ಮೌಲ್ಯ)) {
                zero_ಬಿಂದು = ov::op::v0::Constant::create(zero_ಬಿಂದು->get_element_type(), {}, {zp_ಮೌಲ್ಯ});
            }
            ಸ್ವಯಂಚಾಲಿತ zero_ಬಿಂದು_f16 = std::make_shared<ov::op::v0::Convert>(zero_ಬಿಂದು, ov::element::f16);
            ಸ್ವಯಂಚಾಲಿತ w_zp =
                std::make_shared<ov::op::v1::ಕಳೆ>(weights_f16, zero_ಬಿಂದು_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
            ಸ್ವಯಂಚಾಲಿತ mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
            ಫಲಿತಾಂಶ = mul;
        }
    }

    ಆದರೆ (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        ಸ್ವಯಂಚಾಲಿತ final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(ಫಲಿತಾಂಶ, final_shape, ಸುಳ್ಳು);
        ಫಲಿತಾಂಶ = reshaped;
    }

    ಆದರೆ (for_gather_matmul) {
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }
    ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v0::Convert>(ಫಲಿತಾಂಶ, ov::element::f32);
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
ov::Output<ov::Node> make_int4_weights(ov::Tensor & ತೂಕ,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t ಗುಂಪು_ಗಾತ್ರ = GGML_QUANTIZATION_ಗುಂಪು_ಗಾತ್ರ,
                                       ತಾರ್ಕಿಕ use_bias = ಸುಳ್ಳು,
                                       ತಾರ್ಕಿಕ for_gather_matmul = ಸುಳ್ಳು) {
    ov::Shape orig_ತೂಕ_shape = ತೂಕ.get_shape();
    ತಾರ್ಕಿಕ is_signed = (ತೂಕ.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_ತೂಕ_shape;
    packed_shape.back() /= ಗುಂಪು_ಗಾತ್ರ;
    packed_shape.push_back(ಗುಂಪು_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_dim = packed_shape.size() - 2;

    ಆದರೆ (packed_shape[ಗುಂಪು_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + ಗುಂಪು_dim);
    } ಇಲ್ಲದಿದ್ದರೆ {
        scale_shape.push_back(1);
        scales.ಗಣ_shape(scale_shape);
        ಆದರೆ (!is_signed && zp.get_ಗಾತ್ರ() > 0) {
            ಸ್ವಯಂಚಾಲಿತ zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.ಗಣ_shape(zp_shape);
        }
    }

    ಸ್ವಯಂಚಾಲಿತ scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ಫಲಿತಾಂಶ;
    ಆದರೆ (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        ಸ್ವಯಂಚಾಲಿತ weights_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(ತೂಕ.ದತ್ತಾಂಶ()), ಶೂನ್ಯಸೂಚಕ);
        weights_ಗಂಟು->get_rt_info()["__gguf_tensor_holder"] = ತೂಕ;
        ಸ್ವಯಂಚಾಲಿತ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ಗಂಟು, ov::element::f16);
        ಸ್ವಯಂಚಾಲಿತ mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
        ಫಲಿತಾಂಶ = mul;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Unsigned path
        ಸ್ವಯಂಚಾಲಿತ weights_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(ತೂಕ.ದತ್ತಾಂಶ()), ಶೂನ್ಯಸೂಚಕ);
        weights_ಗಂಟು->get_rt_info()["__gguf_tensor_holder"] = ತೂಕ;
        ಸ್ವಯಂಚಾಲಿತ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ಗಂಟು, ov::element::f16);

        ಆದರೆ (use_bias && zp.get_ಗಾತ್ರ() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            ಸ್ವಯಂಚಾಲಿತ * bias_zp_ದತ್ತಾಂಶ = zp.ದತ್ತಾಂಶ<ov::float16>();
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * scale_ದತ್ತಾಂಶ = scales.ದತ್ತಾಂಶ<ov::float16>();
            ಸ್ಥಿರಾಂಕ size_t n = zp.get_ಗಾತ್ರ();
            ಗಾಗಿ (size_t i = 0; i < n; i++) {
                ದಶಮಾಂಶ s = static_cast<ದಶಮಾಂಶ>(scale_ದತ್ತಾಂಶ[i]);
                ದಶಮಾಂಶ b = static_cast<ದಶಮಾಂಶ>(bias_zp_ದತ್ತಾಂಶ[i]);
                bias_zp_ದತ್ತಾಂಶ[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            ಸ್ವಯಂಚಾಲಿತ zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            ಸ್ವಯಂಚಾಲಿತ w_zp =
                std::make_shared<ov::op::v1::ಕಳೆ>(weights_f16, zero_points_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
            ಫಲಿತಾಂಶ = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Zero point path: (w - zp) * s
            ಸ್ವಯಂಚಾಲಿತ zero_points_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(zp);
            ದಶಮಾಂಶ zp_ಮೌಲ್ಯ;
            ಆದರೆ (ov::op::util::get_single_ಮೌಲ್ಯ(zero_points_ಗಂಟು, zp_ಮೌಲ್ಯ)) {
                zero_points_ಗಂಟು = ov::op::v0::Constant::create(zero_points_ಗಂಟು->get_element_type(), {}, {zp_ಮೌಲ್ಯ});
            }
            ಸ್ವಯಂಚಾಲಿತ zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_ಗಂಟು, ov::element::f16);
            ಸ್ವಯಂಚಾಲಿತ w_zp =
                std::make_shared<ov::op::v1::ಕಳೆ>(weights_f16, zero_points_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
            ಸ್ವಯಂಚಾಲಿತ mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
            ಫಲಿತಾಂಶ = mul;
        }
    }

    ಆದರೆ (packed_shape.size() != orig_ತೂಕ_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        ಸ್ವಯಂಚಾಲಿತ final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_ತೂಕ_shape.size()},
                                                                  orig_ತೂಕ_shape);
        ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(ಫಲಿತಾಂಶ, final_shape, ಸುಳ್ಳು);
        ಫಲಿತಾಂಶ = reshaped;
    }

    ಆದರೆ (for_gather_matmul) {
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }
    ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v0::Convert>(ಫಲಿತಾಂಶ, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & ತೂಕ, ov::Tensor & scales) {
    ಸ್ಥಿರಾಂಕ ov::Shape final_shape = ತೂಕ.get_shape();
    GGML_ASSERT(!final_shape.ಖಾಲಿಗಳು());
    GGML_ASSERT(final_shape.back() % MXFP4_ಖಂಡ_ಗಾತ್ರ == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_ಖಂಡ_ಗಾತ್ರ;
    packed_shape.push_back(MXFP4_ಖಂಡ_ಗಾತ್ರ);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.ಗಣ_shape(scale_shape);

    ಸ್ವಯಂಚಾಲಿತ weights_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(ತೂಕ.ದತ್ತಾಂಶ()), ಶೂನ್ಯಸೂಚಕ);
    weights_ಗಂಟು->get_rt_info()["__gguf_tensor_holder"] = ತೂಕ;
    ಸ್ವಯಂಚಾಲಿತ weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_ಗಂಟು, ov::element::f32);

    ಸ್ವಯಂಚಾಲಿತ scales_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(scales);
    ಸ್ವಯಂಚಾಲಿತ scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_ಗಂಟು, ov::element::f32);
    ov::Output<ov::Node> ಫಲಿತಾಂಶ =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);

    ಸ್ವಯಂಚಾಲಿತ final_shape_ಗಂಟು =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v1::Reshape>(ಫಲಿತಾಂಶ, final_shape_ಗಂಟು, ಸುಳ್ಳು);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & ತೂಕ) {
    ಸ್ವಯಂಚಾಲಿತ weights_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ತೂಕ.get_shape(),
                                                               static_cast<uint8_t *>(ತೂಕ.ದತ್ತಾಂಶ()), ಶೂನ್ಯಸೂಚಕ);
    weights_ಗಂಟು->get_rt_info()["__gguf_tensor_holder"] = ತೂಕ;
    weights_ಗಂಟು->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = ನಿಜ;
    ಹಿಂತಿರುಗಿ weights_ಗಂಟು;
}

ಶೂನ್ಯ quantize_q4_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    ಪ್ರತಿಜ್ಞಾ(k % qk == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / qk;

    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();
    ತಾರ್ಕಿಕ is_ಸಮಮಿತ = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    ಆದರೆ (!is_ಸಮಮಿತ) {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
            ದಶಮಾಂಶ amax = 0.0f;
            ದಶಮಾಂಶ max = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; j++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = x[i * qk + j];
                ಆದರೆ (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = max / -8;
            ಆದರೆ (d == 0) {
                scales[i] = ov::float16(1.0f);
                ಆದರೆ (i % 2 == 0) {
                    zp[i / 2] = 8;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = 1.0f / d;
            scales[i] = ov::float16(d);
            ಆದರೆ (i % 2 == 0) {
                zp[i / 2] = 8;
            } ಇಲ್ಲದಿದ್ದರೆ {
                zp[i / 2] |= (8 << 4);
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk / 2; ++j) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[i * qk + 2 * j] * id;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[i * qk + 2 * j + 1] * id;
                ಸ್ಥಿರಾಂಕ uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                ಸ್ಥಿರಾಂಕ uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Symmetric: produce signed i4 values in [-8, 7]
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
            ದಶಮಾಂಶ amax = 0.0f;
            ದಶಮಾಂಶ max = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; j++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = x[i * qk + j];
                ಆದರೆ (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = max / -8;
            ಆದರೆ (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = 1.0f / d;
            scales[i] = ov::float16(d);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk / 2; ++j) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[i * qk + 2 * j] * id;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (ಪೂರ್ಣಾಂಕ) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (ಪೂರ್ಣಾಂಕ) roundf(x1)));
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
ಶೂನ್ಯ quantize_q4_1_asym(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    ಪ್ರತಿಜ್ಞಾ(k % qk == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / qk;

    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>();
    ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    ಸ್ವಯಂಚಾಲಿತ store_zp = [zp](ಪೂರ್ಣಾಂಕ i, uint8_t v) {
        ಆದರೆ (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } ಇಲ್ಲದಿದ್ದರೆ {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        ದಶಮಾಂಶ vmin = x[i * qk];
        ದಶಮಾಂಶ vmax = x[i * qk];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < qk; j++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (vmax - vmin) / 15.0f;
        ಆದರೆ (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ zq = std::max(0, std::min(15, (ಪೂರ್ಣಾಂಕ) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk / 2; ++j) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[i * qk + 2 * j] * id;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[i * qk + 2 * j + 1] * id;
            ಸ್ಥಿರಾಂಕ uint8_t q0 = (uint8_t) std::max(0, std::min(15, (ಪೂರ್ಣಾಂಕ) lroundf(x0) + zq));
            ಸ್ಥಿರಾಂಕ uint8_t q1 = (uint8_t) std::max(0, std::min(15, (ಪೂರ್ಣಾಂಕ) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

ಶೂನ್ಯ quantize_q8_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t ಖಂಡ_ಸ್ಥಾನಾಂತರ = 0) {
    ಪ್ರತಿಜ್ಞಾ(k % qk == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ()) + ಖಂಡ_ಸ್ಥಾನಾಂತರ * qk;
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>() + ಖಂಡ_ಸ್ಥಾನಾಂತರ;
    ತಾರ್ಕಿಕ is_ಸಮಮಿತ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ಆದರೆ (!is_ಸಮಮಿತ) {
        ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ()) + ಖಂಡ_ಸ್ಥಾನಾಂತರ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
            ದಶಮಾಂಶ amax = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; j++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / 127.0f;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; ++j) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[i * qk + j] * id;
                ಸ್ಥಿರಾಂಕ int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Symmetric: store signed int8 values directly
        ಸ್ವಯಂಚಾಲಿತ * signed_weights = reinterpret_cast<int8_t *>(weights);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
            ದಶಮಾಂಶ amax = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; j++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / 127.0f;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; ++j) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

ಶೂನ್ಯ quantize_q8_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t ಖಂಡ_ಸ್ಥಾನಾಂತರ = 0) {
    ಪ್ರತಿಜ್ಞಾ(k % qk == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    ಸ್ವಯಂಚಾಲಿತ * weights = static_cast<uint8_t *>(weights_arr.ದತ್ತಾಂಶ()) + ಖಂಡ_ಸ್ಥಾನಾಂತರ * qk;
    ಸ್ವಯಂಚಾಲಿತ * scales = scales_arr.ದತ್ತಾಂಶ<ov::element_type_traits<ov::element::f16>::value_type>() + ಖಂಡ_ಸ್ಥಾನಾಂತರ;
    ಸ್ವಯಂಚಾಲಿತ * zp = static_cast<uint8_t *>(zp_arr.ದತ್ತಾಂಶ()) + ಖಂಡ_ಸ್ಥಾನಾಂತರ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        ದಶಮಾಂಶ min = std::numeric_limits<ದಶಮಾಂಶ>::max();
        ದಶಮಾಂಶ max = std::numeric_limits<ದಶಮಾಂಶ>::lowest();

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; j++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (max - min) / ((1 << 8) - 1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; ++j) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = (x[i * qk + j] - min) * id;
            ಸ್ಥಿರಾಂಕ uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> ಹೊರತೆಗೆ_quantized_weights(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                                                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    ತಾರ್ಕಿಕ use_bias = ಸುಳ್ಳು) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor ತಾತ್ಕಾಲಿಕ_tensor = *tensor;
    ತಾತ್ಕಾಲಿಕ_tensor.ದತ್ತಾಂಶ = const_cast<ಶೂನ್ಯ *>(ದತ್ತಾಂಶ);

    ಆದರೆ (tensor->type == GGML_TYPE_MXFP4) {
        ಹೊರತೆಗೆ_mxfp4_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales);
        ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = make_mxfp4_weights(weights, scales).get_ಗಂಟು_shared_ptr();
        ಫಲಿತಾಂಶ->ಗಣ_friendly_name(tensor->name);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Determine block size based on tensor type
    int64_t weights_ಪ್ರತಿ_ಖಂಡ;
    ತಾರ್ಕಿಕ is_u4;
    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        is_u4 = ನಿಜ;
        weights_ಪ್ರತಿ_ಖಂಡ = 32;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        is_u4 = ಸುಳ್ಳು;
        weights_ಪ್ರತಿ_ಖಂಡ = 32;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        is_u4 = ಸುಳ್ಳು;
        weights_ಪ್ರತಿ_ಖಂಡ = 16;
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಎಸೆ std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಹೊರತೆಗೆ_q4_0_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಹೊರತೆಗೆ_q4_1_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp, use_bias);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಹೊರತೆಗೆ_q4_k_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp, use_bias);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಹೊರತೆಗೆ_q5_1_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp, use_bias);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಹೊರತೆಗೆ_q8_0_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಹೊರತೆಗೆ_q6_k_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಹೊರತೆಗೆ_q5_k_ದತ್ತಾಂಶ(&ತಾತ್ಕಾಲಿಕ_tensor, weights, scales, zp, use_bias);
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಎಸೆ std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> ತೂಕ_ಗಂಟು;
    ಆದರೆ (is_u4) {
        ತೂಕ_ಗಂಟು = make_int4_weights(weights, scales, zp, weights_ಪ್ರತಿ_ಖಂಡ, use_bias, for_gather_matmul);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ತೂಕ_ಗಂಟು = make_int8_weights(weights, scales, zp, weights_ಪ್ರತಿ_ಖಂಡ, use_bias, for_gather_matmul);
    }

    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = ತೂಕ_ಗಂಟು.get_ಗಂಟು_shared_ptr();
    ಫಲಿತಾಂಶ->ಗಣ_friendly_name(tensor->name);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_ಗಮ್ಯ_buffers(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t ಖಂಡ_ಗಾತ್ರ_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ಸ್ಥಿರಾಂಕ int64_t ne0 = tensor->ne[0];                 // elements per row
    ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು = n_elements / ne0;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * type_traits = ggml_get_type_traits(tensor->type);
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ne0);

    ತಾರ್ಕಿಕ is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಹರಿವು_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(ಖಂಡ_ಗಾತ್ರ_3 > 0 && ne0 % ಖಂಡ_ಗಾತ್ರ_3 != 0);

    ಆದರೆ (!ಹರಿವು_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<ದಶಮಾಂಶ> weights_f32(n_elements);
        type_traits->ಗಮ್ಯ_float(ದತ್ತಾಂಶ, weights_f32.ದತ್ತಾಂಶ(), n_elements);
        ಆದರೆ (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.ದತ್ತಾಂಶ(), weights.ದತ್ತಾಂಶ(), n_elements);
            ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Constant>(weights);
            ಫಲಿತಾಂಶ->ಗಣ_friendly_name(tensor->name);
            ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
        }
        ಆದರೆ (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.ದತ್ತಾಂಶ(), weights, scales, zp, n_elements, ಖಂಡ_ಗಾತ್ರ_3);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_u4) {
            quantize_q4_0(weights_f32.ದತ್ತಾಂಶ(), weights, scales, zp, n_elements, ಖಂಡ_ಗಾತ್ರ_3);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.ದತ್ತಾಂಶ(), weights, scales, zp, n_elements, ಖಂಡ_ಗಾತ್ರ_3);
        } ಇಲ್ಲದಿದ್ದರೆ {
            quantize_q8_0(weights_f32.ದತ್ತಾಂಶ(), weights, scales, zp, n_elements, ಖಂಡ_ಗಾತ್ರ_3);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ಸ್ಥಿರಾಂಕ int64_t CHUNK_ಅಡ್ಡಸಾಲುಗಳು = std::min<int64_t>(n_ಅಡ್ಡಸಾಲುಗಳು, 256);
        std::vector<ದಶಮಾಂಶ> scratch(CHUNK_ಅಡ್ಡಸಾಲುಗಳು * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        ಸ್ವಯಂಚಾಲಿತ * f16_ತಳ = static_cast<uint8_t *>(weights.ದತ್ತಾಂಶ());
        ಗಾಗಿ (int64_t r0 = 0; r0 < n_ಅಡ್ಡಸಾಲುಗಳು; r0 += CHUNK_ಅಡ್ಡಸಾಲುಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು = std::min(CHUNK_ಅಡ್ಡಸಾಲುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು - r0);
            ಸ್ಥಿರಾಂಕ int64_t elems = ಅಡ್ಡಸಾಲುಗಳು * ne0;
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ) + r0 * src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
            type_traits->ಗಮ್ಯ_float(src, scratch.ದತ್ತಾಂಶ(), elems);

            ಆದರೆ (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.ದತ್ತಾಂಶ(), f16_ತಳ + (r0 * ne0) * ಗಾತ್ರ(uint16_t), elems);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಸ್ಥಾನಾಂತರ = (r0 * ne0) / ಖಂಡ_ಗಾತ್ರ_3;
                ಆದರೆ (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.ದತ್ತಾಂಶ(), weights, scales, zp, elems, ಖಂಡ_ಗಾತ್ರ_3, ಖಂಡ_ಸ್ಥಾನಾಂತರ);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    quantize_q8_0(scratch.ದತ್ತಾಂಶ(), weights, scales, zp, elems, ಖಂಡ_ಗಾತ್ರ_3, ಖಂಡ_ಸ್ಥಾನಾಂತರ);
                }
            }
        }
        ಆದರೆ (requant_type == ExtraQuantType::F16) {
            ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Constant>(weights);
            ಫಲಿತಾಂಶ->ಗಣ_friendly_name(tensor->name);
            ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> ತೂಕ_ಗಂಟು;
    ಆದರೆ (is_u4) {
        ತೂಕ_ಗಂಟು = make_int4_weights(weights, scales, zp, ಖಂಡ_ಗಾತ್ರ_3);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ತೂಕ_ಗಂಟು = make_int8_weights(weights, scales, zp, ಖಂಡ_ಗಾತ್ರ_3);
    }

    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = ತೂಕ_ಗಂಟು.get_ಗಂಟು_shared_ptr();
    ಫಲಿತಾಂಶ->ಗಣ_friendly_name(tensor->name);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}
}  // namespace

Ov_ತೂಕ process_ತೂಕ_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಶೂನ್ಯ * output_ತಳ_ptr, ತಾರ್ಕಿಕ use_bias) {
    GGML_ASSERT(tensor != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ);

    Ov_ತೂಕ ಫಲಿತಾಂಶ;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape ಗಂಟು_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    ಆದರೆ (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            element_type = ov::element::f32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            element_type = ov::element::f16;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        ಆದರೆ (output_ತಳ_ptr && output_ತಳ_ptr != ದತ್ತಾಂಶ) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_ಬೈಟುಗಳು = ggml_nbytes(tensor);
            ಸ್ಮರಣೆ_ನಕಲು(output_ತಳ_ptr, ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು);
            ಫಲಿತಾಂಶ.weights = ov::Tensor(element_type, ಗಂಟು_shape, output_ತಳ_ptr);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ.weights = ov::Tensor(element_type, ಗಂಟು_shape, ದತ್ತಾಂಶ);
        }
        ಫಲಿತಾಂಶ.ತೂಕ_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ಫಲಿತಾಂಶ.weights);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Handle quantized weights
    ಆದರೆ (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    ಫಲಿತಾಂಶ.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & layout = ಫಲಿತಾಂಶ.layout;
    ಆದರೆ (layout.ಒಟ್ಟು_ಗಾತ್ರ == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ for_gather_matmul = tensor->ne[2] > 1;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    ಆದರೆ (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_ಖಂಡ_ಗಾತ್ರ),
                                  MXFP4_ಖಂಡ_ಬೈಟುಗಳು};
        ಸ್ಥಿರಾಂಕ size_t tensor_ಬೈಟುಗಳು = ggml_nbytes(tensor);
        ಆದರೆ (output_ತಳ_ptr) {
            ಸ್ವಯಂಚಾಲಿತ * buf_ತಳ = static_cast<uint8_t *>(output_ತಳ_ptr);
            ಸ್ಮರಣೆ_ನಕಲು(buf_ತಳ + layout.weights_ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು);
            ಫಲಿತಾಂಶ.weights = ov::Tensor(ov::element::u8, packed_shape, buf_ತಳ + layout.weights_ಸ್ಥಾನಾಂತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ.weights = ov::Tensor(ov::element::u8, packed_shape);
            ಸ್ಮರಣೆ_ನಕಲು(ಫಲಿತಾಂಶ.weights.ದತ್ತಾಂಶ(), ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು);
        }
        ಫಲಿತಾಂಶ.ತೂಕ_ಗಂಟು = make_mxfp4_moe_packed_weights(ಫಲಿತಾಂಶ.weights).get_ಗಂಟು_shared_ptr();
        ಫಲಿತಾಂಶ.ತೂಕ_ಗಂಟು->ಗಣ_friendly_name(tensor->name);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    ಆದರೆ (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_ತಳ_ptr = ಶೂನ್ಯಸೂಚಕ;
    }

    // F16 requant path - no separate scales/zp needed in result
    ಆದರೆ (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        ಆದರೆ (output_ತಳ_ptr) {
            ಫಲಿತಾಂಶ.weights = ov::Tensor(ov::element::f16, ಗಂಟು_shape,
                                        static_cast<uint8_t *>(output_ತಳ_ptr) + layout.weights_ಸ್ಥಾನಾಂತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ.weights = ov::Tensor(ov::element::f16, ಗಂಟು_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        ಫಲಿತಾಂಶ.ತೂಕ_ಗಂಟು =
            requantize_ಗಮ್ಯ_buffers(tensor, ದತ್ತಾಂಶ, ExtraQuantType::F16, 0, ಫಲಿತಾಂಶ.weights, dummy_scales, dummy_zp);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type ತೂಕ_type;
    ಆದರೆ (tensor->type == GGML_TYPE_MXFP4) {
        ತೂಕ_type = ov::element::f4e2m1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (layout.is_ಸಮಮಿತ) {
        ತೂಕ_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ತೂಕ_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = ಗಂಟು_shape;
    scale_shape.back() /= layout.weights_ಪ್ರತಿ_ಖಂಡ;

    ಆದರೆ (tensor->type == GGML_TYPE_MXFP4) {
        ಆದರೆ (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            ಗಂಟು_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಂಟು_shape.clear();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                ಗಂಟು_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = ಗಂಟು_shape;
        scale_shape.back() /= layout.weights_ಪ್ರತಿ_ಖಂಡ;
    }

    ಸ್ಥಿರಾಂಕ ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    ಆದರೆ (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    ಆದರೆ (output_ತಳ_ptr) {
        uint8_t * buf_ತಳ = static_cast<uint8_t *>(output_ತಳ_ptr);
        ಫಲಿತಾಂಶ.weights = ov::Tensor(ತೂಕ_type, ಗಂಟು_shape, buf_ತಳ + layout.weights_ಸ್ಥಾನಾಂತರ);
        ಫಲಿತಾಂಶ.scales = ov::Tensor(scale_type, scale_shape, buf_ತಳ + layout.scales_ಸ್ಥಾನಾಂತರ);
        ಆದರೆ (!layout.is_ಸಮಮಿತ) {
            ಫಲಿತಾಂಶ.zp = ov::Tensor(zp_type, scale_shape, buf_ತಳ + layout.zp_ಸ್ಥಾನಾಂತರ);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.weights = ov::Tensor(ತೂಕ_type, ಗಂಟು_shape);
        ಫಲಿತಾಂಶ.scales = ov::Tensor(scale_type, scale_shape);
        ಆದರೆ (!layout.is_ಸಮಮಿತ) {
            ಫಲಿತಾಂಶ.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    ಆದರೆ (layout.is_requant && layout.requant_type.has_value()) {
        ಫಲಿತಾಂಶ.ತೂಕ_ಗಂಟು = requantize_ಗಮ್ಯ_buffers(tensor, ದತ್ತಾಂಶ, layout.requant_type.value(), layout.weights_ಪ್ರತಿ_ಖಂಡ,
                                                   ಫಲಿತಾಂಶ.weights, ಫಲಿತಾಂಶ.scales, ಫಲಿತಾಂಶ.zp);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.ತೂಕ_ಗಂಟು =
            ಹೊರತೆಗೆ_quantized_weights(tensor, ದತ್ತಾಂಶ, ಫಲಿತಾಂಶ.weights, ಫಲಿತಾಂಶ.scales, ಫಲಿತಾಂಶ.zp, use_bias);
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}
