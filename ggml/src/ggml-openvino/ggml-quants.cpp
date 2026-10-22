#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/except.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/parallel.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/element_type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/element_type_traits.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/float16.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/float4_e2m1.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/float8_e8m0.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/util/attr_types.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/constant_folding.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/tensor.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
ਨਾਮਾਕਾਸ਼ ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
ਤਾਰਕਿਕ get_single_ਮੁੱਲ(ਅਚਲ std::shared_ptr<ov::op::v0::Constant> & const_ਗੰਢ,
                      ਦਸ਼ਮਲਵ & value,
                      ਤਾਰਕਿਕ check_ਮੁੱਲ_range = ਸੱਚ);
}  // namespace ov::op::util

ਨਾਮਾਕਾਸ਼ {
ਖਾਲੀ unpack_32_4(ਅਚਲ uint8_t * ਅੰਕੜੇ, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
        uint8_t x = (ਅੰਕੜੇ[j] & 0x0F);
        uint8_t y = (ਅੰਕੜੇ[j] >> 4);
        ਜੇ (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_ਖੰਡ_ਮਾਪ = 32;
constexpr size_t MXFP4_ਖੰਡ_QS_ਮਾਪ = MXFP4_ਖੰਡ_ਮਾਪ / 2;
constexpr size_t MXFP4_ਖੰਡ_ਬਾਈਟਾਂ = ਆਕਾਰ(uint8_t) + MXFP4_ਖੰਡ_QS_ਮਾਪ;

ਖਾਲੀ pack_32_mxfp4_for_openvino(ਅਚਲ uint8_t * ਅੰਕੜੇ, uint8_t * dst) {
    ਲਈ (ਪੂਰਨ j = 0; j < static_cast<ਪੂਰਨ>(MXFP4_ਖੰਡ_QS_ਮਾਪ); j += 2) {
        ਅਚਲ uint8_t v0 = ਅੰਕੜੇ[j] & 0x0F;
        ਅਚਲ uint8_t v1 = (ਅੰਕੜੇ[j + 1] & 0x0F) << 4;
        ਅਚਲ uint8_t v16 = ਅੰਕੜੇ[j] >> 4;
        ਅਚਲ uint8_t v17 = ਅੰਕੜੇ[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_ਖੰਡ_ਮਾਪ / 4 + j / 2] = v16 | v17;
    }
}

ਖਾਲੀ ਕੱਢੋ_mxfp4_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<ਅਚਲ uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ਅਚਲ size_t n_ਖੰਡਾਂ = scales_arr.get_ਮਾਪ();

    ov::parallel_for(n_ਖੰਡਾਂ, [&](size_t i) {
        ਅਚਲ uint8_t * ਖੰਡ = ਅੰਕੜੇ + i * MXFP4_ਖੰਡ_ਬਾਈਟਾਂ;
        pack_32_mxfp4_for_openvino(ਖੰਡ + ਆਕਾਰ(uint8_t), weights + i * MXFP4_ਖੰਡ_QS_ਮਾਪ);
        scales[i] = ov::float8_e8m0::from_ਬਿੱਟ(ਖੰਡ[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
ਖਾਲੀ ਕੱਢੋ_q4_0_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 18;  // 2 bytes scale, 32x0.5 byte weights

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    ਤਾਰਕਿਕ is_ਸਮਮਿਤ = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    ਜੇ (!is_ਸਮਮਿਤ) {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            scales[i] = ov::float16::from_ਬਿੱਟ(*((uint16_t *) (ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ)));
            // Pack two 4-bit zero points per byte
            ਜੇ (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } ਨਹੀਂਤਾਂ {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ + 2, weights + i * 16);
        });
    } ਨਹੀਂਤਾਂ {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            scales[i] = ov::float16::from_ਬਿੱਟ(*((uint16_t *) (ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ)));
            unpack_32_4(ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
ਖਾਲੀ ਕੱਢੋ_q4_1_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ਤਾਰਕਿਕ use_bias) {
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    ਜੇ (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        ਸਵੈਚਾਲਿਤ * bias = zp_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            ਦਸ਼ਮਲਵ scale = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) (ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ))));
            ਦਸ਼ਮਲਵ min = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) (ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ + 4, weights + i * 16);
        });
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            ਦਸ਼ਮਲਵ scale = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) (ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ))));
            ਦਸ਼ਮਲਵ min = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) (ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            ਜੇ (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } ਨਹੀਂਤਾਂ {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
ਖਾਲੀ ਕੱਢੋ_q5_1_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ਤਾਰਕਿਕ use_bias) {
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ਅਚਲ ਪੂਰਨ qk = 32;

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());  // u8 weights, one byte per weight
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    ਸਵੈਚਾਲਿਤ read_u16 = [](ਅਚਲ uint8_t * p) {
        uint16_t v;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&v, p, ਆਕਾਰ(v));
        ਵਾਪਸ v;
    };

    ਸਵੈਚਾਲਿਤ unpack_ਖੰਡ = [&](ਅਚਲ uint8_t * ਖੰਡ, uint8_t * dst) {
        uint32_t qh;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&qh, ਖੰਡ + 4, ਆਕਾਰ(uint32_t));
        ਅਚਲ uint8_t * qs = ਖੰਡ + 8;
        ਲਈ (ਪੂਰਨ j = 0; j < qk / 2; ++j) {
            ਅਚਲ uint8_t lo = qs[j] & 0x0F;
            ਅਚਲ uint8_t hi = qs[j] >> 4;
            ਅਚਲ uint8_t ਦੋਆਧਾਰੀ_ਅੰਕ_lo = (qh >> j) & 1;
            ਅਚਲ uint8_t ਦੋਆਧਾਰੀ_ਅੰਕ_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (ਦੋਆਧਾਰੀ_ਅੰਕ_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (ਦੋਆਧਾਰੀ_ਅੰਕ_hi << 4);  // last 16 weights
        }
    };

    ਜੇ (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        ਸਵੈਚਾਲਿਤ * bias = zp_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            ਅਚਲ uint8_t * ਖੰਡ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;
            ਦਸ਼ਮਲਵ scale = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(read_u16(ਖੰਡ)));
            ਦਸ਼ਮਲਵ min = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(read_u16(ਖੰਡ + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_ਖੰਡ(ਖੰਡ, weights + i * qk);
        });
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());  // u8 zero points
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            ਅਚਲ uint8_t * ਖੰਡ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;
            ਦਸ਼ਮਲਵ scale = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(read_u16(ਖੰਡ)));
            ਦਸ਼ਮਲਵ min = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(read_u16(ਖੰਡ + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_ਖੰਡ(ਖੰਡ, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
ਖਾਲੀ ਕੱਢੋ_q8_0_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ਅਚਲ uint64_t weights_ਪ੍ਰਤੀ_ਖੰਡ = 32;
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 34;  // 2 bytes scale, 32x1 byte weights

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    ਤਾਰਕਿਕ is_ਸਮਮਿਤ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ਜੇ (!is_ਸਮਮਿਤ) {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            uint8_t * ਖੰਡ_ਅੰਕੜੇ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;
            scales[i] = ov::float16::from_ਬਿੱਟ(*(uint16_t *) ਖੰਡ_ਅੰਕੜੇ);
            zp[i] = 128;
            ਲਈ (size_t j = 0; j < weights_ਪ੍ਰਤੀ_ਖੰਡ; ++j) {
                uint8_t x = ਖੰਡ_ਅੰਕੜੇ[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_ਪ੍ਰਤੀ_ਖੰਡ + j] = x;
            }
        });
    } ਨਹੀਂਤਾਂ {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_ਮਾਪ(), [&](size_t i) {
            uint8_t * ਖੰਡ_ਅੰਕੜੇ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;
            scales[i] = ov::float16::from_ਬਿੱਟ(*(uint16_t *) ਖੰਡ_ਅੰਕੜੇ);
            // Copy int8 weights as-is (the tensor element type is i8)
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(weights + i * weights_ਪ੍ਰਤੀ_ਖੰਡ, ਖੰਡ_ਅੰਕੜੇ + 2, weights_ਪ੍ਰਤੀ_ਖੰਡ);
        });
    }
}

ਖਾਲੀ unpack_256_4(ਅਚਲ uint8_t * ਅੰਕੜੇ, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    ਲਈ (size_t i = 0; i < 4; ++i) {
        ਲਈ (ਪੂਰਨ j = 0; j < 32; ++j) {
            uint8_t x = (ਅੰਕੜੇ[i * 32 + j] & 0x0F);
            uint8_t y = (ਅੰਕੜੇ[i * 32 + j] >> 4);
            ਜੇ (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

ਖਾਲੀ ਕੱਢੋ_q4_k_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ਤਾਰਕਿਕ use_bias) {
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 2 + 2 + 12 + 128;
    ਅਚਲ uint64_t n_super_ਖੰਡ = tensor->nb[3] / ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    ਸਵੈਚਾਲਿਤ * zp_u4 = use_bias ? ਖਾਲੀ_ਸੂਚਕ : static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * bias_f16 = use_bias ? zp_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>() : ਖਾਲੀ_ਸੂਚਕ;

    ov::parallel_for(n_super_ਖੰਡ, [&](size_t i) {
        uint8_t * ਖੰਡ_ਅੰਕੜੇ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;

        // Extract scale factors and offsets
        ਦਸ਼ਮਲਵ scale_scales = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) ਖੰਡ_ਅੰਕੜੇ)));
        ਦਸ਼ਮਲਵ scale_mins = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) ਖੰਡ_ਅੰਕੜੇ + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = ਖੰਡ_ਅੰਕੜੇ + 4;

        // Calculate scales
        ਦਸ਼ਮਲਵ scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        ਦਸ਼ਮਲਵ min_vals[8];
        min_vals[0] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<ਦਸ਼ਮਲਵ>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            ਜੇ (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } ਨਹੀਂਤਾਂ {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                ਜੇ (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } ਨਹੀਂਤਾਂ {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(ਖੰਡ_ਅੰਕੜੇ + 16, weights + i * 128);
    });
}

ਖਾਲੀ ਕੱਢੋ_q6_k_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 128 + 64 + 16 + 2;
    ਅਚਲ uint64_t n_super_ਖੰਡ = tensor->nb[3] / ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    ਤਾਰਕਿਕ is_ਸਮਮਿਤ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ਜੇ (!is_ਸਮਮਿਤ) {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
        ov::parallel_for(n_super_ਖੰਡ, [&](size_t i) {
            uint8_t * ਖੰਡ_ਅੰਕੜੇ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;
            ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) ਖੰਡ_ਅੰਕੜੇ + 104)));
            ਲਈ (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ਗੁਣਕ * static_cast<ਦਸ਼ਮਲਵ>(*((int8_t *) (ਖੰਡ_ਅੰਕੜੇ + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = ਖੰਡ_ਅੰਕੜੇ;
            uint8_t * qh = ਖੰਡ_ਅੰਕੜੇ + 128;
            ਲਈ (int64_t j = 0; j < 32; ++j) {
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
    } ਨਹੀਂਤਾਂ {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_ਖੰਡ, [&](size_t i) {
            uint8_t * ਖੰਡ_ਅੰਕੜੇ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;
            ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) ਖੰਡ_ਅੰਕੜੇ + 104)));
            ਲਈ (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ਗੁਣਕ * static_cast<ਦਸ਼ਮਲਵ>(*((int8_t *) (ਖੰਡ_ਅੰਕੜੇ + 128 + 64 + j))));
            }
            uint8_t * ql = ਖੰਡ_ਅੰਕੜੇ;
            uint8_t * qh = ਖੰਡ_ਅੰਕੜੇ + 128;
            ਸਵੈਚਾਲਿਤ * signed_weights = reinterpret_cast<int8_t *>(weights);
            ਲਈ (int64_t j = 0; j < 32; ++j) {
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

ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4(ਪੂਰਨ j, ਅਚਲ uint8_t * q, uint8_t * d, uint8_t * m) {
    ਜੇ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ਖਾਲੀ ਕੱਢੋ_q5_k_ਅੰਕੜੇ(ਅਚਲ ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       ਤਾਰਕਿਕ use_bias) {
    ਅਚਲ uint64_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ = 4 + 12 + 32 + 128;
    ਅਚਲ uint64_t n_super_ਖੰਡ = tensor->nb[3] / ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;

    ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    ਸਵੈਚਾਲਿਤ * zp_u8 = use_bias ? ਖਾਲੀ_ਸੂਚਕ : static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * bias_f16 = use_bias ? zp_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>() : ਖਾਲੀ_ਸੂਚਕ;

    ov::parallel_for(n_super_ਖੰਡ, [&](size_t i) {
        uint8_t * ਖੰਡ_ਅੰਕੜੇ = ਅੰਕੜੇ + i * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖੰਡ;

        ਅਚਲ ਦਸ਼ਮਲਵ d = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) ਖੰਡ_ਅੰਕੜੇ)));
        ਅਚਲ ਦਸ਼ਮਲਵ min_ਗੁਣਕ = static_cast<ਦਸ਼ਮਲਵ>(ov::float16::from_ਬਿੱਟ(*((uint16_t *) ਖੰਡ_ਅੰਕੜੇ + 1)));

        ਅਚਲ uint8_t * scales_ਅੰਕੜੇ = ਖੰਡ_ਅੰਕੜੇ + 4;   // 12 bytes of scales
        ਅਚਲ uint8_t * qh = ਖੰਡ_ਅੰਕੜੇ + 4 + 12;       // 32 bytes of high bits
        ਅਚਲ uint8_t * ql = ਖੰਡ_ਅੰਕੜੇ + 4 + 12 + 32;  // 128 bytes of low bits

        ਪੂਰਨ is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        ਲਈ (ਪੂਰਨ j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_ਅੰਕੜੇ, &sc, &m);
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = d * sc;
            ਅਚਲ ਦਸ਼ਮਲਵ m1 = min_ਗੁਣਕ * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_ਅੰਕੜੇ, &sc, &m);
            ਅਚਲ ਦਸ਼ਮਲਵ d2 = d * sc;
            ਅਚਲ ਦਸ਼ਮਲਵ m2 = min_ਗੁਣਕ * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            ਜੇ (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } ਨਹੀਂਤਾਂ {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
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
ov::Output<ov::Node> make_int8_weights(ov::Tensor & ਭਾਰ,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t ਸਮੂਹ_ਮਾਪ = GGML_QUANTIZATION_ਸਮੂਹ_ਮਾਪ,
                                       ਤਾਰਕਿਕ use_bias = ਝੂਠ,
                                       ਤਾਰਕਿਕ for_gather_matmul = ਝੂਠ) {
    ov::Shape orig_shape = ਭਾਰ.get_shape();
    ਤਾਰਕਿਕ is_signed = (ਭਾਰ.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ਸਵੈਚਾਲਿਤ scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= ਸਮੂਹ_ਮਾਪ;
    packed_shape.push_back(ਸਮੂਹ_ਮਾਪ);
    ਅਚਲ size_t ਸਮੂਹ_dim = packed_shape.size() - 2;

    ਜੇ (packed_shape[ਸਮੂਹ_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + ਸਮੂਹ_dim);
    } ਨਹੀਂਤਾਂ {
        scale_shape.push_back(1);
        scales.ਸੈੱਟ_shape(scale_shape);
        ਜੇ (!is_signed && zp.get_ਮਾਪ() > 0) {
            ਸਵੈਚਾਲਿਤ zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.ਸੈੱਟ_shape(zp_shape);
        }
    }

    ਸਵੈਚਾਲਿਤ scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ਨਤੀਜਾ;
    ਜੇ (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        ਸਵੈਚਾਲਿਤ weights_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(ਭਾਰ.ਅੰਕੜੇ()), ਖਾਲੀ_ਸੂਚਕ);
        weights_ਗੰਢ->get_rt_info()["__gguf_tensor_holder"] = ਭਾਰ;
        ਸਵੈਚਾਲਿਤ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ਗੰਢ, ov::element::f16);
        ਸਵੈਚਾਲਿਤ mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
        ਨਤੀਜਾ = mul;
    } ਨਹੀਂਤਾਂ {
        // Unsigned path
        ਸਵੈਚਾਲਿਤ weights_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(ਭਾਰ.ਅੰਕੜੇ()), ਖਾਲੀ_ਸੂਚਕ);
        weights_ਗੰਢ->get_rt_info()["__gguf_tensor_holder"] = ਭਾਰ;
        ਸਵੈਚਾਲਿਤ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ਗੰਢ, ov::element::f16);

        ਜੇ (use_bias && zp.get_ਮਾਪ() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            ਸਵੈਚਾਲਿਤ * bias_zp_ਅੰਕੜੇ = zp.ਅੰਕੜੇ<ov::float16>();
            ਅਚਲ ਸਵੈਚਾਲਿਤ * scale_ਅੰਕੜੇ = scales.ਅੰਕੜੇ<ov::float16>();
            ਅਚਲ size_t n = zp.get_ਮਾਪ();
            ਲਈ (size_t i = 0; i < n; i++) {
                ਦਸ਼ਮਲਵ s = static_cast<ਦਸ਼ਮਲਵ>(scale_ਅੰਕੜੇ[i]);
                ਦਸ਼ਮਲਵ b = static_cast<ਦਸ਼ਮਲਵ>(bias_zp_ਅੰਕੜੇ[i]);
                bias_zp_ਅੰਕੜੇ[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            ਸਵੈਚਾਲਿਤ zero_ਬਿੰਦੂ_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            ਸਵੈਚਾਲਿਤ w_zp =
                std::make_shared<ov::op::v1::ਘਟਾਓ>(weights_f16, zero_ਬਿੰਦੂ_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
            ਨਤੀਜਾ = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
        } ਨਹੀਂਤਾਂ {
            // Zero point path: (w - zp) * s
            ਸਵੈਚਾਲਿਤ zero_ਬਿੰਦੂ = std::make_shared<ov::op::v0::Constant>(zp);
            ਦਸ਼ਮਲਵ zp_ਮੁੱਲ;
            ਜੇ (ov::op::util::get_single_ਮੁੱਲ(zero_ਬਿੰਦੂ, zp_ਮੁੱਲ)) {
                zero_ਬਿੰਦੂ = ov::op::v0::Constant::create(zero_ਬਿੰਦੂ->get_element_type(), {}, {zp_ਮੁੱਲ});
            }
            ਸਵੈਚਾਲਿਤ zero_ਬਿੰਦੂ_f16 = std::make_shared<ov::op::v0::Convert>(zero_ਬਿੰਦੂ, ov::element::f16);
            ਸਵੈਚਾਲਿਤ w_zp =
                std::make_shared<ov::op::v1::ਘਟਾਓ>(weights_f16, zero_ਬਿੰਦੂ_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
            ਸਵੈਚਾਲਿਤ mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
            ਨਤੀਜਾ = mul;
        }
    }

    ਜੇ (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        ਸਵੈਚਾਲਿਤ final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(ਨਤੀਜਾ, final_shape, ਝੂਠ);
        ਨਤੀਜਾ = reshaped;
    }

    ਜੇ (for_gather_matmul) {
        ਵਾਪਸ ਨਤੀਜਾ;
    }
    ਵਾਪਸ std::make_shared<ov::op::v0::Convert>(ਨਤੀਜਾ, ov::element::f32);
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
ov::Output<ov::Node> make_int4_weights(ov::Tensor & ਭਾਰ,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t ਸਮੂਹ_ਮਾਪ = GGML_QUANTIZATION_ਸਮੂਹ_ਮਾਪ,
                                       ਤਾਰਕਿਕ use_bias = ਝੂਠ,
                                       ਤਾਰਕਿਕ for_gather_matmul = ਝੂਠ) {
    ov::Shape orig_ਭਾਰ_shape = ਭਾਰ.get_shape();
    ਤਾਰਕਿਕ is_signed = (ਭਾਰ.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_ਭਾਰ_shape;
    packed_shape.back() /= ਸਮੂਹ_ਮਾਪ;
    packed_shape.push_back(ਸਮੂਹ_ਮਾਪ);
    ਅਚਲ size_t ਸਮੂਹ_dim = packed_shape.size() - 2;

    ਜੇ (packed_shape[ਸਮੂਹ_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + ਸਮੂਹ_dim);
    } ਨਹੀਂਤਾਂ {
        scale_shape.push_back(1);
        scales.ਸੈੱਟ_shape(scale_shape);
        ਜੇ (!is_signed && zp.get_ਮਾਪ() > 0) {
            ਸਵੈਚਾਲਿਤ zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.ਸੈੱਟ_shape(zp_shape);
        }
    }

    ਸਵੈਚਾਲਿਤ scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ਨਤੀਜਾ;
    ਜੇ (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        ਸਵੈਚਾਲਿਤ weights_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(ਭਾਰ.ਅੰਕੜੇ()), ਖਾਲੀ_ਸੂਚਕ);
        weights_ਗੰਢ->get_rt_info()["__gguf_tensor_holder"] = ਭਾਰ;
        ਸਵੈਚਾਲਿਤ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ਗੰਢ, ov::element::f16);
        ਸਵੈਚਾਲਿਤ mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
        ਨਤੀਜਾ = mul;
    } ਨਹੀਂਤਾਂ {
        // Unsigned path
        ਸਵੈਚਾਲਿਤ weights_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(ਭਾਰ.ਅੰਕੜੇ()), ਖਾਲੀ_ਸੂਚਕ);
        weights_ਗੰਢ->get_rt_info()["__gguf_tensor_holder"] = ਭਾਰ;
        ਸਵੈਚਾਲਿਤ weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_ਗੰਢ, ov::element::f16);

        ਜੇ (use_bias && zp.get_ਮਾਪ() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            ਸਵੈਚਾਲਿਤ * bias_zp_ਅੰਕੜੇ = zp.ਅੰਕੜੇ<ov::float16>();
            ਅਚਲ ਸਵੈਚਾਲਿਤ * scale_ਅੰਕੜੇ = scales.ਅੰਕੜੇ<ov::float16>();
            ਅਚਲ size_t n = zp.get_ਮਾਪ();
            ਲਈ (size_t i = 0; i < n; i++) {
                ਦਸ਼ਮਲਵ s = static_cast<ਦਸ਼ਮਲਵ>(scale_ਅੰਕੜੇ[i]);
                ਦਸ਼ਮਲਵ b = static_cast<ਦਸ਼ਮਲਵ>(bias_zp_ਅੰਕੜੇ[i]);
                bias_zp_ਅੰਕੜੇ[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            ਸਵੈਚਾਲਿਤ zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            ਸਵੈਚਾਲਿਤ w_zp =
                std::make_shared<ov::op::v1::ਘਟਾਓ>(weights_f16, zero_points_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
            ਨਤੀਜਾ = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
        } ਨਹੀਂਤਾਂ {
            // Zero point path: (w - zp) * s
            ਸਵੈਚਾਲਿਤ zero_points_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(zp);
            ਦਸ਼ਮਲਵ zp_ਮੁੱਲ;
            ਜੇ (ov::op::util::get_single_ਮੁੱਲ(zero_points_ਗੰਢ, zp_ਮੁੱਲ)) {
                zero_points_ਗੰਢ = ov::op::v0::Constant::create(zero_points_ਗੰਢ->get_element_type(), {}, {zp_ਮੁੱਲ});
            }
            ਸਵੈਚਾਲਿਤ zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_ਗੰਢ, ov::element::f16);
            ਸਵੈਚਾਲਿਤ w_zp =
                std::make_shared<ov::op::v1::ਘਟਾਓ>(weights_f16, zero_points_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
            ਸਵੈਚਾਲਿਤ mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
            ਨਤੀਜਾ = mul;
        }
    }

    ਜੇ (packed_shape.size() != orig_ਭਾਰ_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        ਸਵੈਚਾਲਿਤ final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_ਭਾਰ_shape.size()},
                                                                  orig_ਭਾਰ_shape);
        ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(ਨਤੀਜਾ, final_shape, ਝੂਠ);
        ਨਤੀਜਾ = reshaped;
    }

    ਜੇ (for_gather_matmul) {
        ਵਾਪਸ ਨਤੀਜਾ;
    }
    ਵਾਪਸ std::make_shared<ov::op::v0::Convert>(ਨਤੀਜਾ, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & ਭਾਰ, ov::Tensor & scales) {
    ਅਚਲ ov::Shape final_shape = ਭਾਰ.get_shape();
    GGML_ASSERT(!final_shape.ਖਾਲੀ_ਥਾਵਾਂ());
    GGML_ASSERT(final_shape.back() % MXFP4_ਖੰਡ_ਮਾਪ == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_ਖੰਡ_ਮਾਪ;
    packed_shape.push_back(MXFP4_ਖੰਡ_ਮਾਪ);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.ਸੈੱਟ_shape(scale_shape);

    ਸਵੈਚਾਲਿਤ weights_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(ਭਾਰ.ਅੰਕੜੇ()), ਖਾਲੀ_ਸੂਚਕ);
    weights_ਗੰਢ->get_rt_info()["__gguf_tensor_holder"] = ਭਾਰ;
    ਸਵੈਚਾਲਿਤ weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_ਗੰਢ, ov::element::f32);

    ਸਵੈਚਾਲਿਤ scales_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(scales);
    ਸਵੈਚਾਲਿਤ scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_ਗੰਢ, ov::element::f32);
    ov::Output<ov::Node> ਨਤੀਜਾ =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);

    ਸਵੈਚਾਲਿਤ final_shape_ਗੰਢ =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    ਵਾਪਸ std::make_shared<ov::op::v1::Reshape>(ਨਤੀਜਾ, final_shape_ਗੰਢ, ਝੂਠ);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & ਭਾਰ) {
    ਸਵੈਚਾਲਿਤ weights_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ਭਾਰ.get_shape(),
                                                               static_cast<uint8_t *>(ਭਾਰ.ਅੰਕੜੇ()), ਖਾਲੀ_ਸੂਚਕ);
    weights_ਗੰਢ->get_rt_info()["__gguf_tensor_holder"] = ਭਾਰ;
    weights_ਗੰਢ->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = ਸੱਚ;
    ਵਾਪਸ weights_ਗੰਢ;
}

ਖਾਲੀ quantize_q4_0(ਅਚਲ ਦਸ਼ਮਲਵ * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % qk == 0);
    ਅਚਲ ਪੂਰਨ nb = k / qk;

    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();
    ਤਾਰਕਿਕ is_ਸਮਮਿਤ = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    ਜੇ (!is_ਸਮਮਿਤ) {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());
        ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
            ਦਸ਼ਮਲਵ amax = 0.0f;
            ਦਸ਼ਮਲਵ max = 0.0f;
            ਲਈ (ਪੂਰਨ j = 0; j < qk; j++) {
                ਅਚਲ ਦਸ਼ਮਲਵ v = x[i * qk + j];
                ਜੇ (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ਅਚਲ ਦਸ਼ਮਲਵ d = max / -8;
            ਜੇ (d == 0) {
                scales[i] = ov::float16(1.0f);
                ਜੇ (i % 2 == 0) {
                    zp[i / 2] = 8;
                } ਨਹੀਂਤਾਂ {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                ਜਾਰੀ;
            }
            ਅਚਲ ਦਸ਼ਮਲਵ id = 1.0f / d;
            scales[i] = ov::float16(d);
            ਜੇ (i % 2 == 0) {
                zp[i / 2] = 8;
            } ਨਹੀਂਤਾਂ {
                zp[i / 2] |= (8 << 4);
            }
            ਲਈ (ਪੂਰਨ j = 0; j < qk / 2; ++j) {
                ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[i * qk + 2 * j] * id;
                ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[i * qk + 2 * j + 1] * id;
                ਅਚਲ uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                ਅਚਲ uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } ਨਹੀਂਤਾਂ {
        // Symmetric: produce signed i4 values in [-8, 7]
        ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
            ਦਸ਼ਮਲਵ amax = 0.0f;
            ਦਸ਼ਮਲਵ max = 0.0f;
            ਲਈ (ਪੂਰਨ j = 0; j < qk; j++) {
                ਅਚਲ ਦਸ਼ਮਲਵ v = x[i * qk + j];
                ਜੇ (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ਅਚਲ ਦਸ਼ਮਲਵ d = max / -8;
            ਜੇ (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                ਜਾਰੀ;
            }
            ਅਚਲ ਦਸ਼ਮਲਵ id = 1.0f / d;
            scales[i] = ov::float16(d);
            ਲਈ (ਪੂਰਨ j = 0; j < qk / 2; ++j) {
                ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[i * qk + 2 * j] * id;
                ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (ਪੂਰਨ) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (ਪੂਰਨ) roundf(x1)));
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
ਖਾਲੀ quantize_q4_1_asym(ਅਚਲ ਦਸ਼ਮਲਵ * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % qk == 0);
    ਅਚਲ ਪੂਰਨ nb = k / qk;

    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>();
    ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    ਸਵੈਚਾਲਿਤ store_zp = [zp](ਪੂਰਨ i, uint8_t v) {
        ਜੇ (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } ਨਹੀਂਤਾਂ {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਦਸ਼ਮਲਵ vmin = x[i * qk];
        ਦਸ਼ਮਲਵ vmax = x[i * qk];
        ਲਈ (ਪੂਰਨ j = 1; j < qk; j++) {
            ਅਚਲ ਦਸ਼ਮਲਵ v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        ਅਚਲ ਦਸ਼ਮਲਵ d = (vmax - vmin) / 15.0f;
        ਜੇ (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            ਜਾਰੀ;
        }
        ਅਚਲ ਦਸ਼ਮਲਵ id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        ਅਚਲ ਪੂਰਨ zq = std::max(0, std::min(15, (ਪੂਰਨ) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        ਲਈ (ਪੂਰਨ j = 0; j < qk / 2; ++j) {
            ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[i * qk + 2 * j] * id;
            ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[i * qk + 2 * j + 1] * id;
            ਅਚਲ uint8_t q0 = (uint8_t) std::max(0, std::min(15, (ਪੂਰਨ) lroundf(x0) + zq));
            ਅਚਲ uint8_t q1 = (uint8_t) std::max(0, std::min(15, (ਪੂਰਨ) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

ਖਾਲੀ quantize_q8_0(ਅਚਲ ਦਸ਼ਮਲਵ * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t ਖੰਡ_ਵਿੱਥ = 0) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % qk == 0);
    ਅਚਲ ਪੂਰਨ nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ()) + ਖੰਡ_ਵਿੱਥ * qk;
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>() + ਖੰਡ_ਵਿੱਥ;
    ਤਾਰਕਿਕ is_ਸਮਮਿਤ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ਜੇ (!is_ਸਮਮਿਤ) {
        ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ()) + ਖੰਡ_ਵਿੱਥ;
        ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
            ਦਸ਼ਮਲਵ amax = 0.0f;
            ਲਈ (ਪੂਰਨ j = 0; j < qk; j++) {
                ਅਚਲ ਦਸ਼ਮਲਵ v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ਅਚਲ ਦਸ਼ਮਲਵ d = amax / 127.0f;
            ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            ਲਈ (ਪੂਰਨ j = 0; j < qk; ++j) {
                ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[i * qk + j] * id;
                ਅਚਲ int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } ਨਹੀਂਤਾਂ {
        // Symmetric: store signed int8 values directly
        ਸਵੈਚਾਲਿਤ * signed_weights = reinterpret_cast<int8_t *>(weights);
        ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
            ਦਸ਼ਮਲਵ amax = 0.0f;
            ਲਈ (ਪੂਰਨ j = 0; j < qk; j++) {
                ਅਚਲ ਦਸ਼ਮਲਵ v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ਅਚਲ ਦਸ਼ਮਲਵ d = amax / 127.0f;
            ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            ਲਈ (ਪੂਰਨ j = 0; j < qk; ++j) {
                ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

ਖਾਲੀ quantize_q8_1(ਅਚਲ ਦਸ਼ਮਲਵ * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t ਖੰਡ_ਵਿੱਥ = 0) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % qk == 0);
    ਅਚਲ ਪੂਰਨ nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    ਸਵੈਚਾਲਿਤ * weights = static_cast<uint8_t *>(weights_arr.ਅੰਕੜੇ()) + ਖੰਡ_ਵਿੱਥ * qk;
    ਸਵੈਚਾਲਿਤ * scales = scales_arr.ਅੰਕੜੇ<ov::element_type_traits<ov::element::f16>::value_type>() + ਖੰਡ_ਵਿੱਥ;
    ਸਵੈਚਾਲਿਤ * zp = static_cast<uint8_t *>(zp_arr.ਅੰਕੜੇ()) + ਖੰਡ_ਵਿੱਥ;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਦਸ਼ਮਲਵ min = std::numeric_limits<ਦਸ਼ਮਲਵ>::max();
        ਦਸ਼ਮਲਵ max = std::numeric_limits<ਦਸ਼ਮਲਵ>::lowest();

        ਲਈ (ਪੂਰਨ j = 0; j < qk; j++) {
            ਅਚਲ ਦਸ਼ਮਲਵ v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        ਅਚਲ ਦਸ਼ਮਲਵ d = (max - min) / ((1 << 8) - 1);
        ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        ਲਈ (ਪੂਰਨ j = 0; j < qk; ++j) {
            ਅਚਲ ਦਸ਼ਮਲਵ x0 = (x[i * qk + j] - min) * id;
            ਅਚਲ uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> ਕੱਢੋ_quantized_weights(ਅਚਲ ggml_tensor * tensor,
                                                    ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    ਤਾਰਕਿਕ use_bias = ਝੂਠ) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor ਆਰਜ਼ੀ_tensor = *tensor;
    ਆਰਜ਼ੀ_tensor.ਅੰਕੜੇ = const_cast<ਖਾਲੀ *>(ਅੰਕੜੇ);

    ਜੇ (tensor->type == GGML_TYPE_MXFP4) {
        ਕੱਢੋ_mxfp4_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales);
        ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = make_mxfp4_weights(weights, scales).get_ਗੰਢ_shared_ptr();
        ਨਤੀਜਾ->ਸੈੱਟ_friendly_name(tensor->name);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Determine block size based on tensor type
    int64_t weights_ਪ੍ਰਤੀ_ਖੰਡ;
    ਤਾਰਕਿਕ is_u4;
    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_Q4_0:
    ਹਾਲਤ GGML_TYPE_Q4_1:
    ਹਾਲਤ GGML_TYPE_Q4_K:
        is_u4 = ਸੱਚ;
        weights_ਪ੍ਰਤੀ_ਖੰਡ = 32;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q8_0:
    ਹਾਲਤ GGML_TYPE_Q5_1:
    ਹਾਲਤ GGML_TYPE_Q5_K:
        is_u4 = ਝੂਠ;
        weights_ਪ੍ਰਤੀ_ਖੰਡ = 32;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q6_K:
        is_u4 = ਝੂਠ;
        weights_ਪ੍ਰਤੀ_ਖੰਡ = 16;
        ਰੋਕੋ;
    ਮੂਲ:
        ਸੁੱਟੋ std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    ਅਚਲ ਤਾਰਕਿਕ for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_Q4_0:
        ਕੱਢੋ_q4_0_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q4_1:
        ਕੱਢੋ_q4_1_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp, use_bias);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q4_K:
        ਕੱਢੋ_q4_k_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp, use_bias);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q5_1:
        ਕੱਢੋ_q5_1_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp, use_bias);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q8_0:
        ਕੱਢੋ_q8_0_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q6_K:
        ਕੱਢੋ_q6_k_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q5_K:
        ਕੱਢੋ_q5_k_ਅੰਕੜੇ(&ਆਰਜ਼ੀ_tensor, weights, scales, zp, use_bias);
        ਰੋਕੋ;
    ਮੂਲ:
        ਸੁੱਟੋ std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> ਭਾਰ_ਗੰਢ;
    ਜੇ (is_u4) {
        ਭਾਰ_ਗੰਢ = make_int4_weights(weights, scales, zp, weights_ਪ੍ਰਤੀ_ਖੰਡ, use_bias, for_gather_matmul);
    } ਨਹੀਂਤਾਂ {
        ਭਾਰ_ਗੰਢ = make_int8_weights(weights, scales, zp, weights_ਪ੍ਰਤੀ_ਖੰਡ, use_bias, for_gather_matmul);
    }

    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = ਭਾਰ_ਗੰਢ.get_ਗੰਢ_shared_ptr();
    ਨਤੀਜਾ->ਸੈੱਟ_friendly_name(tensor->name);
    ਵਾਪਸ ਨਤੀਜਾ;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_ਮੰਜ਼ਿਲ_buffers(ਅਚਲ ggml_tensor * tensor,
                                                ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t ਖੰਡ_ਮਾਪ_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ਅਚਲ int64_t ne0 = tensor->ne[0];                 // elements per row
    ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ = n_elements / ne0;
    ਅਚਲ ਸਵੈਚਾਲਿਤ * type_traits = ggml_get_type_traits(tensor->type);
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ne0);

    ਤਾਰਕਿਕ is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    ਅਚਲ ਤਾਰਕਿਕ ਧਾਰਾ_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(ਖੰਡ_ਮਾਪ_3 > 0 && ne0 % ਖੰਡ_ਮਾਪ_3 != 0);

    ਜੇ (!ਧਾਰਾ_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<ਦਸ਼ਮਲਵ> weights_f32(n_elements);
        type_traits->ਮੰਜ਼ਿਲ_float(ਅੰਕੜੇ, weights_f32.ਅੰਕੜੇ(), n_elements);
        ਜੇ (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.ਅੰਕੜੇ(), weights.ਅੰਕੜੇ(), n_elements);
            ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = std::make_shared<ov::op::v0::Constant>(weights);
            ਨਤੀਜਾ->ਸੈੱਟ_friendly_name(tensor->name);
            ਵਾਪਸ ਨਤੀਜਾ;
        }
        ਜੇ (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.ਅੰਕੜੇ(), weights, scales, zp, n_elements, ਖੰਡ_ਮਾਪ_3);
        } ਨਹੀਂਤਾਂ ਜੇ (is_u4) {
            quantize_q4_0(weights_f32.ਅੰਕੜੇ(), weights, scales, zp, n_elements, ਖੰਡ_ਮਾਪ_3);
        } ਨਹੀਂਤਾਂ ਜੇ (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.ਅੰਕੜੇ(), weights, scales, zp, n_elements, ਖੰਡ_ਮਾਪ_3);
        } ਨਹੀਂਤਾਂ {
            quantize_q8_0(weights_f32.ਅੰਕੜੇ(), weights, scales, zp, n_elements, ਖੰਡ_ਮਾਪ_3);
        }
    } ਨਹੀਂਤਾਂ {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ਅਚਲ int64_t CHUNK_ਪੰਕਤੀਆਂ = std::min<int64_t>(n_ਪੰਕਤੀਆਂ, 256);
        std::vector<ਦਸ਼ਮਲਵ> scratch(CHUNK_ਪੰਕਤੀਆਂ * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        ਸਵੈਚਾਲਿਤ * f16_ਅਧਾਰ = static_cast<uint8_t *>(weights.ਅੰਕੜੇ());
        ਲਈ (int64_t r0 = 0; r0 < n_ਪੰਕਤੀਆਂ; r0 += CHUNK_ਪੰਕਤੀਆਂ) {
            ਅਚਲ int64_t ਪੰਕਤੀਆਂ = std::min(CHUNK_ਪੰਕਤੀਆਂ, n_ਪੰਕਤੀਆਂ - r0);
            ਅਚਲ int64_t elems = ਪੰਕਤੀਆਂ * ne0;
            ਅਚਲ ਸਵੈਚਾਲਿਤ * src = static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ) + r0 * src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
            type_traits->ਮੰਜ਼ਿਲ_float(src, scratch.ਅੰਕੜੇ(), elems);

            ਜੇ (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.ਅੰਕੜੇ(), f16_ਅਧਾਰ + (r0 * ne0) * ਆਕਾਰ(uint16_t), elems);
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ int64_t ਖੰਡ_ਵਿੱਥ = (r0 * ne0) / ਖੰਡ_ਮਾਪ_3;
                ਜੇ (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.ਅੰਕੜੇ(), weights, scales, zp, elems, ਖੰਡ_ਮਾਪ_3, ਖੰਡ_ਵਿੱਥ);
                } ਨਹੀਂਤਾਂ {
                    quantize_q8_0(scratch.ਅੰਕੜੇ(), weights, scales, zp, elems, ਖੰਡ_ਮਾਪ_3, ਖੰਡ_ਵਿੱਥ);
                }
            }
        }
        ਜੇ (requant_type == ExtraQuantType::F16) {
            ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = std::make_shared<ov::op::v0::Constant>(weights);
            ਨਤੀਜਾ->ਸੈੱਟ_friendly_name(tensor->name);
            ਵਾਪਸ ਨਤੀਜਾ;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> ਭਾਰ_ਗੰਢ;
    ਜੇ (is_u4) {
        ਭਾਰ_ਗੰਢ = make_int4_weights(weights, scales, zp, ਖੰਡ_ਮਾਪ_3);
    } ਨਹੀਂਤਾਂ {
        ਭਾਰ_ਗੰਢ = make_int8_weights(weights, scales, zp, ਖੰਡ_ਮਾਪ_3);
    }

    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = ਭਾਰ_ਗੰਢ.get_ਗੰਢ_shared_ptr();
    ਨਤੀਜਾ->ਸੈੱਟ_friendly_name(tensor->name);
    ਵਾਪਸ ਨਤੀਜਾ;
}
}  // namespace

Ov_ਭਾਰ process_ਭਾਰ_tensor(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, ਖਾਲੀ * output_ਅਧਾਰ_ptr, ਤਾਰਕਿਕ use_bias) {
    GGML_ASSERT(tensor != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ);

    Ov_ਭਾਰ ਨਤੀਜਾ;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape ਗੰਢ_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    ਜੇ (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            element_type = ov::element::f32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            element_type = ov::element::f16;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            ਰੋਕੋ;
        ਮੂਲ:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        ਜੇ (output_ਅਧਾਰ_ptr && output_ਅਧਾਰ_ptr != ਅੰਕੜੇ) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_ਬਾਈਟਾਂ = ggml_nbytes(tensor);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(output_ਅਧਾਰ_ptr, ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ);
            ਨਤੀਜਾ.weights = ov::Tensor(element_type, ਗੰਢ_shape, output_ਅਧਾਰ_ptr);
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ.weights = ov::Tensor(element_type, ਗੰਢ_shape, ਅੰਕੜੇ);
        }
        ਨਤੀਜਾ.ਭਾਰ_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ਨਤੀਜਾ.weights);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Handle quantized weights
    ਜੇ (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    ਨਤੀਜਾ.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ਅਚਲ ਸਵੈਚਾਲਿਤ & layout = ਨਤੀਜਾ.layout;
    ਜੇ (layout.ਕੁੱਲ_ਮਾਪ == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    ਅਚਲ ਤਾਰਕਿਕ for_gather_matmul = tensor->ne[2] > 1;
    ਅਚਲ ਤਾਰਕਿਕ zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    ਅਚਲ ਤਾਰਕਿਕ is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    ਜੇ (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_ਖੰਡ_ਮਾਪ),
                                  MXFP4_ਖੰਡ_ਬਾਈਟਾਂ};
        ਅਚਲ size_t tensor_ਬਾਈਟਾਂ = ggml_nbytes(tensor);
        ਜੇ (output_ਅਧਾਰ_ptr) {
            ਸਵੈਚਾਲਿਤ * buf_ਅਧਾਰ = static_cast<uint8_t *>(output_ਅਧਾਰ_ptr);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(buf_ਅਧਾਰ + layout.weights_ਵਿੱਥ, ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ);
            ਨਤੀਜਾ.weights = ov::Tensor(ov::element::u8, packed_shape, buf_ਅਧਾਰ + layout.weights_ਵਿੱਥ);
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ.weights = ov::Tensor(ov::element::u8, packed_shape);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਨਤੀਜਾ.weights.ਅੰਕੜੇ(), ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ);
        }
        ਨਤੀਜਾ.ਭਾਰ_ਗੰਢ = make_mxfp4_moe_packed_weights(ਨਤੀਜਾ.weights).get_ਗੰਢ_shared_ptr();
        ਨਤੀਜਾ.ਭਾਰ_ਗੰਢ->ਸੈੱਟ_friendly_name(tensor->name);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    ਜੇ (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_ਅਧਾਰ_ptr = ਖਾਲੀ_ਸੂਚਕ;
    }

    // F16 requant path - no separate scales/zp needed in result
    ਜੇ (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        ਜੇ (output_ਅਧਾਰ_ptr) {
            ਨਤੀਜਾ.weights = ov::Tensor(ov::element::f16, ਗੰਢ_shape,
                                        static_cast<uint8_t *>(output_ਅਧਾਰ_ptr) + layout.weights_ਵਿੱਥ);
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ.weights = ov::Tensor(ov::element::f16, ਗੰਢ_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        ਨਤੀਜਾ.ਭਾਰ_ਗੰਢ =
            requantize_ਮੰਜ਼ਿਲ_buffers(tensor, ਅੰਕੜੇ, ExtraQuantType::F16, 0, ਨਤੀਜਾ.weights, dummy_scales, dummy_zp);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type ਭਾਰ_type;
    ਜੇ (tensor->type == GGML_TYPE_MXFP4) {
        ਭਾਰ_type = ov::element::f4e2m1;
    } ਨਹੀਂਤਾਂ ਜੇ (layout.is_ਸਮਮਿਤ) {
        ਭਾਰ_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } ਨਹੀਂਤਾਂ {
        ਭਾਰ_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = ਗੰਢ_shape;
    scale_shape.back() /= layout.weights_ਪ੍ਰਤੀ_ਖੰਡ;

    ਜੇ (tensor->type == GGML_TYPE_MXFP4) {
        ਜੇ (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            ਗੰਢ_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } ਨਹੀਂਤਾਂ {
            ਗੰਢ_shape.clear();
            ਲਈ (ਪੂਰਨ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                ਗੰਢ_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = ਗੰਢ_shape;
        scale_shape.back() /= layout.weights_ਪ੍ਰਤੀ_ਖੰਡ;
    }

    ਅਚਲ ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    ਜੇ (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    ਜੇ (output_ਅਧਾਰ_ptr) {
        uint8_t * buf_ਅਧਾਰ = static_cast<uint8_t *>(output_ਅਧਾਰ_ptr);
        ਨਤੀਜਾ.weights = ov::Tensor(ਭਾਰ_type, ਗੰਢ_shape, buf_ਅਧਾਰ + layout.weights_ਵਿੱਥ);
        ਨਤੀਜਾ.scales = ov::Tensor(scale_type, scale_shape, buf_ਅਧਾਰ + layout.scales_ਵਿੱਥ);
        ਜੇ (!layout.is_ਸਮਮਿਤ) {
            ਨਤੀਜਾ.zp = ov::Tensor(zp_type, scale_shape, buf_ਅਧਾਰ + layout.zp_ਵਿੱਥ);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.weights = ov::Tensor(ਭਾਰ_type, ਗੰਢ_shape);
        ਨਤੀਜਾ.scales = ov::Tensor(scale_type, scale_shape);
        ਜੇ (!layout.is_ਸਮਮਿਤ) {
            ਨਤੀਜਾ.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    ਜੇ (layout.is_requant && layout.requant_type.has_value()) {
        ਨਤੀਜਾ.ਭਾਰ_ਗੰਢ = requantize_ਮੰਜ਼ਿਲ_buffers(tensor, ਅੰਕੜੇ, layout.requant_type.value(), layout.weights_ਪ੍ਰਤੀ_ਖੰਡ,
                                                   ਨਤੀਜਾ.weights, ਨਤੀਜਾ.scales, ਨਤੀਜਾ.zp);
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.ਭਾਰ_ਗੰਢ =
            ਕੱਢੋ_quantized_weights(tensor, ਅੰਕੜੇ, ਨਤੀਜਾ.weights, ਨਤੀਜਾ.scales, ਨਤੀਜਾ.zp, use_bias);
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}
