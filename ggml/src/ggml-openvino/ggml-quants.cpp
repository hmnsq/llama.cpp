#شامل "ggml-quants.h"

#شامل "ggml-common.h"
#شامل "ggml-impl.h"
#شامل "ggml-openvino-extra.h"
#شامل "ggml.h"

#شامل <algorithm>
#شامل <cassert>
#شامل <cmath>
#شامل <cstddef>
#شامل <cstdint>
#شامل <limits>
#شامل <memory>
#شامل <openvino/core/except.hpp>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/core/parallel.hpp>
#شامل <openvino/core/shape.hpp>
#شامل <openvino/core/type/element_type.hpp>
#شامل <openvino/core/type/element_type_traits.hpp>
#شامل <openvino/core/type/float16.hpp>
#شامل <openvino/core/type/float4_e2m1.hpp>
#شامل <openvino/core/type/float8_e8m0.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <openvino/op/util/attr_types.hpp>
#شامل <openvino/pass/constant_folding.hpp>
#شامل <openvino/runtime/tensor.hpp>
#شامل <string>
#شامل <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
نامخلا ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
منطقی get_single_قدر(ثابت std::shared_ptr<ov::op::v0::Constant> & const_گرہ,
                      اعشاری & value,
                      منطقی check_قدر_range = سچ);
}  // namespace ov::op::util

نامخلا {
خالی unpack_32_4(ثابت uint8_t * ڈیٹا, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    برائے (صحیح j = 0; j < 16; ++j) {
        uint8_t x = (ڈیٹا[j] & 0x0F);
        uint8_t y = (ڈیٹا[j] >> 4);
        اگر (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_بلاک_سائز = 32;
constexpr size_t MXFP4_بلاک_QS_سائز = MXFP4_بلاک_سائز / 2;
constexpr size_t MXFP4_بلاک_بائٹیں = حجم(uint8_t) + MXFP4_بلاک_QS_سائز;

خالی pack_32_mxfp4_for_openvino(ثابت uint8_t * ڈیٹا, uint8_t * dst) {
    برائے (صحیح j = 0; j < static_cast<صحیح>(MXFP4_بلاک_QS_سائز); j += 2) {
        ثابت uint8_t v0 = ڈیٹا[j] & 0x0F;
        ثابت uint8_t v1 = (ڈیٹا[j + 1] & 0x0F) << 4;
        ثابت uint8_t v16 = ڈیٹا[j] >> 4;
        ثابت uint8_t v17 = ڈیٹا[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_بلاک_سائز / 4 + j / 2] = v16 | v17;
    }
}

خالی نکالو_mxfp4_ڈیٹا(ثابت ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ثابت خودکار * ڈیٹا = static_cast<ثابت uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ثابت size_t n_بلاکیں = scales_arr.get_سائز();

    ov::parallel_for(n_بلاکیں, [&](size_t i) {
        ثابت uint8_t * بلاک = ڈیٹا + i * MXFP4_بلاک_بائٹیں;
        pack_32_mxfp4_for_openvino(بلاک + حجم(uint8_t), weights + i * MXFP4_بلاک_QS_سائز);
        scales[i] = ov::float8_e8m0::from_بٹ(بلاک[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
خالی نکالو_q4_0_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t بائٹیں_فی_بلاک = 18;  // 2 bytes scale, 32x0.5 byte weights

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقی is_متشاکل = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    اگر (!is_متشاکل) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            scales[i] = ov::float16::from_بٹ(*((uint16_t *) (ڈیٹا + i * بائٹیں_فی_بلاک)));
            // Pack two 4-bit zero points per byte
            اگر (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } ورنہ {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(ڈیٹا + i * بائٹیں_فی_بلاک + 2, weights + i * 16);
        });
    } ورنہ {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            scales[i] = ov::float16::from_بٹ(*((uint16_t *) (ڈیٹا + i * بائٹیں_فی_بلاک)));
            unpack_32_4(ڈیٹا + i * بائٹیں_فی_بلاک + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            برائے (صحیح j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
خالی نکالو_q4_1_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقی use_bias) {
    ثابت uint64_t بائٹیں_فی_بلاک = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    اگر (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        خودکار * bias = zp_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            اعشاری scale = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) (ڈیٹا + i * بائٹیں_فی_بلاک))));
            اعشاری min = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) (ڈیٹا + i * بائٹیں_فی_بلاک + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(ڈیٹا + i * بائٹیں_فی_بلاک + 4, weights + i * 16);
        });
    } ورنہ {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            اعشاری scale = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) (ڈیٹا + i * بائٹیں_فی_بلاک))));
            اعشاری min = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) (ڈیٹا + i * بائٹیں_فی_بلاک + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            اگر (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } ورنہ {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(ڈیٹا + i * بائٹیں_فی_بلاک + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
خالی نکالو_q5_1_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقی use_bias) {
    ثابت uint64_t بائٹیں_فی_بلاک = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ثابت صحیح qk = 32;

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());  // u8 weights, one byte per weight
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    خودکار read_u16 = [](ثابت uint8_t * p) {
        uint16_t v;
        حافظہ_نقل(&v, p, حجم(v));
        واپس v;
    };

    خودکار unpack_بلاک = [&](ثابت uint8_t * بلاک, uint8_t * dst) {
        uint32_t qh;
        حافظہ_نقل(&qh, بلاک + 4, حجم(uint32_t));
        ثابت uint8_t * qs = بلاک + 8;
        برائے (صحیح j = 0; j < qk / 2; ++j) {
            ثابت uint8_t lo = qs[j] & 0x0F;
            ثابت uint8_t hi = qs[j] >> 4;
            ثابت uint8_t ثنائی_ہندسہ_lo = (qh >> j) & 1;
            ثابت uint8_t ثنائی_ہندسہ_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (ثنائی_ہندسہ_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (ثنائی_ہندسہ_hi << 4);  // last 16 weights
        }
    };

    اگر (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        خودکار * bias = zp_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            ثابت uint8_t * بلاک = ڈیٹا + i * بائٹیں_فی_بلاک;
            اعشاری scale = static_cast<اعشاری>(ov::float16::from_بٹ(read_u16(بلاک)));
            اعشاری min = static_cast<اعشاری>(ov::float16::from_بٹ(read_u16(بلاک + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_بلاک(بلاک, weights + i * qk);
        });
    } ورنہ {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());  // u8 zero points
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            ثابت uint8_t * بلاک = ڈیٹا + i * بائٹیں_فی_بلاک;
            اعشاری scale = static_cast<اعشاری>(ov::float16::from_بٹ(read_u16(بلاک)));
            اعشاری min = static_cast<اعشاری>(ov::float16::from_بٹ(read_u16(بلاک + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_بلاک(بلاک, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
خالی نکالو_q8_0_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t weights_فی_بلاک = 32;
    ثابت uint64_t بائٹیں_فی_بلاک = 34;  // 2 bytes scale, 32x1 byte weights

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقی is_متشاکل = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    اگر (!is_متشاکل) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            uint8_t * بلاک_ڈیٹا = ڈیٹا + i * بائٹیں_فی_بلاک;
            scales[i] = ov::float16::from_بٹ(*(uint16_t *) بلاک_ڈیٹا);
            zp[i] = 128;
            برائے (size_t j = 0; j < weights_فی_بلاک; ++j) {
                uint8_t x = بلاک_ڈیٹا[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_فی_بلاک + j] = x;
            }
        });
    } ورنہ {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_سائز(), [&](size_t i) {
            uint8_t * بلاک_ڈیٹا = ڈیٹا + i * بائٹیں_فی_بلاک;
            scales[i] = ov::float16::from_بٹ(*(uint16_t *) بلاک_ڈیٹا);
            // Copy int8 weights as-is (the tensor element type is i8)
            حافظہ_نقل(weights + i * weights_فی_بلاک, بلاک_ڈیٹا + 2, weights_فی_بلاک);
        });
    }
}

خالی unpack_256_4(ثابت uint8_t * ڈیٹا, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    برائے (size_t i = 0; i < 4; ++i) {
        برائے (صحیح j = 0; j < 32; ++j) {
            uint8_t x = (ڈیٹا[i * 32 + j] & 0x0F);
            uint8_t y = (ڈیٹا[i * 32 + j] >> 4);
            اگر (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

خالی نکالو_q4_k_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقی use_bias) {
    ثابت uint64_t بائٹیں_فی_بلاک = 2 + 2 + 12 + 128;
    ثابت uint64_t n_super_بلاک = tensor->nb[3] / بائٹیں_فی_بلاک;

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    خودکار * zp_u4 = use_bias ? خالی_اشاریہ : static_cast<uint8_t *>(zp_arr.ڈیٹا());
    خودکار * bias_f16 = use_bias ? zp_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>() : خالی_اشاریہ;

    ov::parallel_for(n_super_بلاک, [&](size_t i) {
        uint8_t * بلاک_ڈیٹا = ڈیٹا + i * بائٹیں_فی_بلاک;

        // Extract scale factors and offsets
        اعشاری scale_scales = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) بلاک_ڈیٹا)));
        اعشاری scale_mins = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) بلاک_ڈیٹا + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = بلاک_ڈیٹا + 4;

        // Calculate scales
        اعشاری scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<اعشاری>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<اعشاری>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<اعشاری>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<اعشاری>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<اعشاری>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<اعشاری>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<اعشاری>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<اعشاری>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        اعشاری min_vals[8];
        min_vals[0] = scale_mins * static_cast<اعشاری>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<اعشاری>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<اعشاری>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<اعشاری>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<اعشاری>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<اعشاری>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<اعشاری>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<اعشاری>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        برائے (صحیح j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            اگر (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } ورنہ {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                اگر (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } ورنہ {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(بلاک_ڈیٹا + 16, weights + i * 128);
    });
}

خالی نکالو_q6_k_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t بائٹیں_فی_بلاک = 128 + 64 + 16 + 2;
    ثابت uint64_t n_super_بلاک = tensor->nb[3] / بائٹیں_فی_بلاک;

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقی is_متشاکل = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    اگر (!is_متشاکل) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());
        ov::parallel_for(n_super_بلاک, [&](size_t i) {
            uint8_t * بلاک_ڈیٹا = ڈیٹا + i * بائٹیں_فی_بلاک;
            اعشاری scale_ضربی_عدد = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) بلاک_ڈیٹا + 104)));
            برائے (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ضربی_عدد * static_cast<اعشاری>(*((int8_t *) (بلاک_ڈیٹا + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = بلاک_ڈیٹا;
            uint8_t * qh = بلاک_ڈیٹا + 128;
            برائے (int64_t j = 0; j < 32; ++j) {
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
    } ورنہ {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_بلاک, [&](size_t i) {
            uint8_t * بلاک_ڈیٹا = ڈیٹا + i * بائٹیں_فی_بلاک;
            اعشاری scale_ضربی_عدد = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) بلاک_ڈیٹا + 104)));
            برائے (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ضربی_عدد * static_cast<اعشاری>(*((int8_t *) (بلاک_ڈیٹا + 128 + 64 + j))));
            }
            uint8_t * ql = بلاک_ڈیٹا;
            uint8_t * qh = بلاک_ڈیٹا + 128;
            خودکار * signed_weights = reinterpret_cast<int8_t *>(weights);
            برائے (int64_t j = 0; j < 32; ++j) {
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

اندرونی خالی get_scale_min_k4(صحیح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    اگر (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ورنہ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

خالی نکالو_q5_k_ڈیٹا(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقی use_bias) {
    ثابت uint64_t بائٹیں_فی_بلاک = 4 + 12 + 32 + 128;
    ثابت uint64_t n_super_بلاک = tensor->nb[3] / بائٹیں_فی_بلاک;

    خودکار * ڈیٹا = static_cast<uint8_t *>(tensor->ڈیٹا);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    خودکار * zp_u8 = use_bias ? خالی_اشاریہ : static_cast<uint8_t *>(zp_arr.ڈیٹا());
    خودکار * bias_f16 = use_bias ? zp_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>() : خالی_اشاریہ;

    ov::parallel_for(n_super_بلاک, [&](size_t i) {
        uint8_t * بلاک_ڈیٹا = ڈیٹا + i * بائٹیں_فی_بلاک;

        ثابت اعشاری d = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) بلاک_ڈیٹا)));
        ثابت اعشاری min_ضربی_عدد = static_cast<اعشاری>(ov::float16::from_بٹ(*((uint16_t *) بلاک_ڈیٹا + 1)));

        ثابت uint8_t * scales_ڈیٹا = بلاک_ڈیٹا + 4;   // 12 bytes of scales
        ثابت uint8_t * qh = بلاک_ڈیٹا + 4 + 12;       // 32 bytes of high bits
        ثابت uint8_t * ql = بلاک_ڈیٹا + 4 + 12 + 32;  // 128 bytes of low bits

        صحیح is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        برائے (صحیح j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_ڈیٹا, &sc, &m);
            ثابت اعشاری d1 = d * sc;
            ثابت اعشاری m1 = min_ضربی_عدد * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_ڈیٹا, &sc, &m);
            ثابت اعشاری d2 = d * sc;
            ثابت اعشاری m2 = min_ضربی_عدد * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            اگر (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } ورنہ {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            برائے (صحیح l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            برائے (صحیح l = 0; l < 32; ++l) {
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
ov::Output<ov::Node> make_int8_weights(ov::Tensor & وزن,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t گروہ_سائز = GGML_QUANTIZATION_گروہ_سائز,
                                       منطقی use_bias = جھوٹ,
                                       منطقی for_gather_matmul = جھوٹ) {
    ov::Shape orig_shape = وزن.get_shape();
    منطقی is_signed = (وزن.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    خودکار scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= گروہ_سائز;
    packed_shape.push_back(گروہ_سائز);
    ثابت size_t گروہ_dim = packed_shape.size() - 2;

    اگر (packed_shape[گروہ_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + گروہ_dim);
    } ورنہ {
        scale_shape.push_back(1);
        scales.سیٹ_shape(scale_shape);
        اگر (!is_signed && zp.get_سائز() > 0) {
            خودکار zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.سیٹ_shape(zp_shape);
        }
    }

    خودکار scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> نتیجہ;
    اگر (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        خودکار weights_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.ڈیٹا()), خالی_اشاریہ);
        weights_گرہ->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گرہ, ov::element::f16);
        خودکار mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_نشر_Type::NUMPY);
        نتیجہ = mul;
    } ورنہ {
        // Unsigned path
        خودکار weights_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.ڈیٹا()), خالی_اشاریہ);
        weights_گرہ->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گرہ, ov::element::f16);

        اگر (use_bias && zp.get_سائز() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            خودکار * bias_zp_ڈیٹا = zp.ڈیٹا<ov::float16>();
            ثابت خودکار * scale_ڈیٹا = scales.ڈیٹا<ov::float16>();
            ثابت size_t n = zp.get_سائز();
            برائے (size_t i = 0; i < n; i++) {
                اعشاری s = static_cast<اعشاری>(scale_ڈیٹا[i]);
                اعشاری b = static_cast<اعشاری>(bias_zp_ڈیٹا[i]);
                bias_zp_ڈیٹا[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            خودکار zero_نقطہ_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_نقطہ_f16, ov::op::Auto_نشر_Type::NUMPY);
            نتیجہ = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_نشر_Type::NUMPY);
        } ورنہ {
            // Zero point path: (w - zp) * s
            خودکار zero_نقطہ = std::make_shared<ov::op::v0::Constant>(zp);
            اعشاری zp_قدر;
            اگر (ov::op::util::get_single_قدر(zero_نقطہ, zp_قدر)) {
                zero_نقطہ = ov::op::v0::Constant::create(zero_نقطہ->get_element_type(), {}, {zp_قدر});
            }
            خودکار zero_نقطہ_f16 = std::make_shared<ov::op::v0::Convert>(zero_نقطہ, ov::element::f16);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_نقطہ_f16, ov::op::Auto_نشر_Type::NUMPY);
            خودکار mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_نشر_Type::NUMPY);
            نتیجہ = mul;
        }
    }

    اگر (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        خودکار final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(نتیجہ, final_shape, جھوٹ);
        نتیجہ = reshaped;
    }

    اگر (for_gather_matmul) {
        واپس نتیجہ;
    }
    واپس std::make_shared<ov::op::v0::Convert>(نتیجہ, ov::element::f32);
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
ov::Output<ov::Node> make_int4_weights(ov::Tensor & وزن,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t گروہ_سائز = GGML_QUANTIZATION_گروہ_سائز,
                                       منطقی use_bias = جھوٹ,
                                       منطقی for_gather_matmul = جھوٹ) {
    ov::Shape orig_وزن_shape = وزن.get_shape();
    منطقی is_signed = (وزن.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_وزن_shape;
    packed_shape.back() /= گروہ_سائز;
    packed_shape.push_back(گروہ_سائز);
    ثابت size_t گروہ_dim = packed_shape.size() - 2;

    اگر (packed_shape[گروہ_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + گروہ_dim);
    } ورنہ {
        scale_shape.push_back(1);
        scales.سیٹ_shape(scale_shape);
        اگر (!is_signed && zp.get_سائز() > 0) {
            خودکار zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.سیٹ_shape(zp_shape);
        }
    }

    خودکار scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> نتیجہ;
    اگر (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        خودکار weights_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.ڈیٹا()), خالی_اشاریہ);
        weights_گرہ->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گرہ, ov::element::f16);
        خودکار mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_نشر_Type::NUMPY);
        نتیجہ = mul;
    } ورنہ {
        // Unsigned path
        خودکار weights_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.ڈیٹا()), خالی_اشاریہ);
        weights_گرہ->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گرہ, ov::element::f16);

        اگر (use_bias && zp.get_سائز() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            خودکار * bias_zp_ڈیٹا = zp.ڈیٹا<ov::float16>();
            ثابت خودکار * scale_ڈیٹا = scales.ڈیٹا<ov::float16>();
            ثابت size_t n = zp.get_سائز();
            برائے (size_t i = 0; i < n; i++) {
                اعشاری s = static_cast<اعشاری>(scale_ڈیٹا[i]);
                اعشاری b = static_cast<اعشاری>(bias_zp_ڈیٹا[i]);
                bias_zp_ڈیٹا[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            خودکار zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_points_f16, ov::op::Auto_نشر_Type::NUMPY);
            نتیجہ = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_نشر_Type::NUMPY);
        } ورنہ {
            // Zero point path: (w - zp) * s
            خودکار zero_points_گرہ = std::make_shared<ov::op::v0::Constant>(zp);
            اعشاری zp_قدر;
            اگر (ov::op::util::get_single_قدر(zero_points_گرہ, zp_قدر)) {
                zero_points_گرہ = ov::op::v0::Constant::create(zero_points_گرہ->get_element_type(), {}, {zp_قدر});
            }
            خودکار zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_گرہ, ov::element::f16);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_points_f16, ov::op::Auto_نشر_Type::NUMPY);
            خودکار mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_نشر_Type::NUMPY);
            نتیجہ = mul;
        }
    }

    اگر (packed_shape.size() != orig_وزن_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        خودکار final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_وزن_shape.size()},
                                                                  orig_وزن_shape);
        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(نتیجہ, final_shape, جھوٹ);
        نتیجہ = reshaped;
    }

    اگر (for_gather_matmul) {
        واپس نتیجہ;
    }
    واپس std::make_shared<ov::op::v0::Convert>(نتیجہ, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & وزن, ov::Tensor & scales) {
    ثابت ov::Shape final_shape = وزن.get_shape();
    GGML_ASSERT(!final_shape.خالی_خانے());
    GGML_ASSERT(final_shape.back() % MXFP4_بلاک_سائز == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_بلاک_سائز;
    packed_shape.push_back(MXFP4_بلاک_سائز);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.سیٹ_shape(scale_shape);

    خودکار weights_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(وزن.ڈیٹا()), خالی_اشاریہ);
    weights_گرہ->get_rt_info()["__gguf_tensor_holder"] = وزن;
    خودکار weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_گرہ, ov::element::f32);

    خودکار scales_گرہ = std::make_shared<ov::op::v0::Constant>(scales);
    خودکار scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_گرہ, ov::element::f32);
    ov::Output<ov::Node> نتیجہ =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_نشر_Type::NUMPY);

    خودکار final_shape_گرہ =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    واپس std::make_shared<ov::op::v1::Reshape>(نتیجہ, final_shape_گرہ, جھوٹ);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & وزن) {
    خودکار weights_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::u8, وزن.get_shape(),
                                                               static_cast<uint8_t *>(وزن.ڈیٹا()), خالی_اشاریہ);
    weights_گرہ->get_rt_info()["__gguf_tensor_holder"] = وزن;
    weights_گرہ->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = سچ;
    واپس weights_گرہ;
}

خالی quantize_q4_0(ثابت اعشاری * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    أكد(k % qk == 0);
    ثابت صحیح nb = k / qk;

    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();
    منطقی is_متشاکل = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    اگر (!is_متشاکل) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());
        برائے (صحیح i = 0; i < nb; i++) {
            اعشاری amax = 0.0f;
            اعشاری max = 0.0f;
            برائے (صحیح j = 0; j < qk; j++) {
                ثابت اعشاری v = x[i * qk + j];
                اگر (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ثابت اعشاری d = max / -8;
            اگر (d == 0) {
                scales[i] = ov::float16(1.0f);
                اگر (i % 2 == 0) {
                    zp[i / 2] = 8;
                } ورنہ {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                جاری;
            }
            ثابت اعشاری id = 1.0f / d;
            scales[i] = ov::float16(d);
            اگر (i % 2 == 0) {
                zp[i / 2] = 8;
            } ورنہ {
                zp[i / 2] |= (8 << 4);
            }
            برائے (صحیح j = 0; j < qk / 2; ++j) {
                ثابت اعشاری x0 = x[i * qk + 2 * j] * id;
                ثابت اعشاری x1 = x[i * qk + 2 * j + 1] * id;
                ثابت uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                ثابت uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } ورنہ {
        // Symmetric: produce signed i4 values in [-8, 7]
        برائے (صحیح i = 0; i < nb; i++) {
            اعشاری amax = 0.0f;
            اعشاری max = 0.0f;
            برائے (صحیح j = 0; j < qk; j++) {
                ثابت اعشاری v = x[i * qk + j];
                اگر (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ثابت اعشاری d = max / -8;
            اگر (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                جاری;
            }
            ثابت اعشاری id = 1.0f / d;
            scales[i] = ov::float16(d);
            برائے (صحیح j = 0; j < qk / 2; ++j) {
                ثابت اعشاری x0 = x[i * qk + 2 * j] * id;
                ثابت اعشاری x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (صحیح) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (صحیح) roundf(x1)));
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
خالی quantize_q4_1_asym(ثابت اعشاری * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    أكد(k % qk == 0);
    ثابت صحیح nb = k / qk;

    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا());
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>();
    خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    خودکار store_zp = [zp](صحیح i, uint8_t v) {
        اگر (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } ورنہ {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    برائے (صحیح i = 0; i < nb; i++) {
        اعشاری vmin = x[i * qk];
        اعشاری vmax = x[i * qk];
        برائے (صحیح j = 1; j < qk; j++) {
            ثابت اعشاری v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        ثابت اعشاری d = (vmax - vmin) / 15.0f;
        اگر (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            جاری;
        }
        ثابت اعشاری id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        ثابت صحیح zq = std::max(0, std::min(15, (صحیح) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        برائے (صحیح j = 0; j < qk / 2; ++j) {
            ثابت اعشاری x0 = x[i * qk + 2 * j] * id;
            ثابت اعشاری x1 = x[i * qk + 2 * j + 1] * id;
            ثابت uint8_t q0 = (uint8_t) std::max(0, std::min(15, (صحیح) lroundf(x0) + zq));
            ثابت uint8_t q1 = (uint8_t) std::max(0, std::min(15, (صحیح) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

خالی quantize_q8_0(ثابت اعشاری * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t بلاک_انحراف = 0) {
    أكد(k % qk == 0);
    ثابت صحیح nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا()) + بلاک_انحراف * qk;
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>() + بلاک_انحراف;
    منطقی is_متشاکل = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    اگر (!is_متشاکل) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا()) + بلاک_انحراف;
        برائے (صحیح i = 0; i < nb; i++) {
            اعشاری amax = 0.0f;
            برائے (صحیح j = 0; j < qk; j++) {
                ثابت اعشاری v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ثابت اعشاری d = amax / 127.0f;
            ثابت اعشاری id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            برائے (صحیح j = 0; j < qk; ++j) {
                ثابت اعشاری x0 = x[i * qk + j] * id;
                ثابت int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } ورنہ {
        // Symmetric: store signed int8 values directly
        خودکار * signed_weights = reinterpret_cast<int8_t *>(weights);
        برائے (صحیح i = 0; i < nb; i++) {
            اعشاری amax = 0.0f;
            برائے (صحیح j = 0; j < qk; j++) {
                ثابت اعشاری v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ثابت اعشاری d = amax / 127.0f;
            ثابت اعشاری id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            برائے (صحیح j = 0; j < qk; ++j) {
                ثابت اعشاری x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

خالی quantize_q8_1(ثابت اعشاری * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t بلاک_انحراف = 0) {
    أكد(k % qk == 0);
    ثابت صحیح nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    خودکار * weights = static_cast<uint8_t *>(weights_arr.ڈیٹا()) + بلاک_انحراف * qk;
    خودکار * scales = scales_arr.ڈیٹا<ov::element_type_traits<ov::element::f16>::value_type>() + بلاک_انحراف;
    خودکار * zp = static_cast<uint8_t *>(zp_arr.ڈیٹا()) + بلاک_انحراف;
    برائے (صحیح i = 0; i < nb; i++) {
        اعشاری min = std::numeric_limits<اعشاری>::max();
        اعشاری max = std::numeric_limits<اعشاری>::lowest();

        برائے (صحیح j = 0; j < qk; j++) {
            ثابت اعشاری v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        ثابت اعشاری d = (max - min) / ((1 << 8) - 1);
        ثابت اعشاری id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        برائے (صحیح j = 0; j < qk; ++j) {
            ثابت اعشاری x0 = (x[i * qk + j] - min) * id;
            ثابت uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> نکالو_quantized_weights(ثابت ggml_tensor * tensor,
                                                    ثابت خالی * ڈیٹا,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    منطقی use_bias = جھوٹ) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor عارضی_tensor = *tensor;
    عارضی_tensor.ڈیٹا = const_cast<خالی *>(ڈیٹا);

    اگر (tensor->type == GGML_TYPE_MXFP4) {
        نکالو_mxfp4_ڈیٹا(&عارضی_tensor, weights, scales);
        خودکار نتیجہ = make_mxfp4_weights(weights, scales).get_گرہ_shared_ptr();
        نتیجہ->سیٹ_friendly_name(tensor->name);
        واپس نتیجہ;
    }

    // Determine block size based on tensor type
    int64_t weights_فی_بلاک;
    منطقی is_u4;
    انتخاب (tensor->type) {
    حالت GGML_TYPE_Q4_0:
    حالت GGML_TYPE_Q4_1:
    حالت GGML_TYPE_Q4_K:
        is_u4 = سچ;
        weights_فی_بلاک = 32;
        توڑو;
    حالت GGML_TYPE_Q8_0:
    حالت GGML_TYPE_Q5_1:
    حالت GGML_TYPE_Q5_K:
        is_u4 = جھوٹ;
        weights_فی_بلاک = 32;
        توڑو;
    حالت GGML_TYPE_Q6_K:
        is_u4 = جھوٹ;
        weights_فی_بلاک = 16;
        توڑو;
    طےشدہ:
        پھینکو std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    ثابت منطقی for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    انتخاب (tensor->type) {
    حالت GGML_TYPE_Q4_0:
        نکالو_q4_0_ڈیٹا(&عارضی_tensor, weights, scales, zp);
        توڑو;
    حالت GGML_TYPE_Q4_1:
        نکالو_q4_1_ڈیٹا(&عارضی_tensor, weights, scales, zp, use_bias);
        توڑو;
    حالت GGML_TYPE_Q4_K:
        نکالو_q4_k_ڈیٹا(&عارضی_tensor, weights, scales, zp, use_bias);
        توڑو;
    حالت GGML_TYPE_Q5_1:
        نکالو_q5_1_ڈیٹا(&عارضی_tensor, weights, scales, zp, use_bias);
        توڑو;
    حالت GGML_TYPE_Q8_0:
        نکالو_q8_0_ڈیٹا(&عارضی_tensor, weights, scales, zp);
        توڑو;
    حالت GGML_TYPE_Q6_K:
        نکالو_q6_k_ڈیٹا(&عارضی_tensor, weights, scales, zp);
        توڑو;
    حالت GGML_TYPE_Q5_K:
        نکالو_q5_k_ڈیٹا(&عارضی_tensor, weights, scales, zp, use_bias);
        توڑو;
    طےشدہ:
        پھینکو std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> وزن_گرہ;
    اگر (is_u4) {
        وزن_گرہ = make_int4_weights(weights, scales, zp, weights_فی_بلاک, use_bias, for_gather_matmul);
    } ورنہ {
        وزن_گرہ = make_int8_weights(weights, scales, zp, weights_فی_بلاک, use_bias, for_gather_matmul);
    }

    خودکار نتیجہ = وزن_گرہ.get_گرہ_shared_ptr();
    نتیجہ->سیٹ_friendly_name(tensor->name);
    واپس نتیجہ;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_مقصد_buffers(ثابت ggml_tensor * tensor,
                                                ثابت خالی * ڈیٹا,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t بلاک_سائز_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ثابت int64_t ne0 = tensor->ne[0];                 // elements per row
    ثابت int64_t n_صفیں = n_elements / ne0;
    ثابت خودکار * type_traits = ggml_get_type_traits(tensor->type);
    ثابت size_t src_صف_بائٹیں = ggml_صف_سائز(tensor->type, ne0);

    منطقی is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    ثابت منطقی دھارا_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(بلاک_سائز_3 > 0 && ne0 % بلاک_سائز_3 != 0);

    اگر (!دھارا_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<اعشاری> weights_f32(n_elements);
        type_traits->مقصد_float(ڈیٹا, weights_f32.ڈیٹا(), n_elements);
        اگر (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.ڈیٹا(), weights.ڈیٹا(), n_elements);
            خودکار نتیجہ = std::make_shared<ov::op::v0::Constant>(weights);
            نتیجہ->سیٹ_friendly_name(tensor->name);
            واپس نتیجہ;
        }
        اگر (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.ڈیٹا(), weights, scales, zp, n_elements, بلاک_سائز_3);
        } ورنہ اگر (is_u4) {
            quantize_q4_0(weights_f32.ڈیٹا(), weights, scales, zp, n_elements, بلاک_سائز_3);
        } ورنہ اگر (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.ڈیٹا(), weights, scales, zp, n_elements, بلاک_سائز_3);
        } ورنہ {
            quantize_q8_0(weights_f32.ڈیٹا(), weights, scales, zp, n_elements, بلاک_سائز_3);
        }
    } ورنہ {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ثابت int64_t CHUNK_صفیں = std::min<int64_t>(n_صفیں, 256);
        std::vector<اعشاری> scratch(CHUNK_صفیں * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        خودکار * f16_بنیاد = static_cast<uint8_t *>(weights.ڈیٹا());
        برائے (int64_t r0 = 0; r0 < n_صفیں; r0 += CHUNK_صفیں) {
            ثابت int64_t صفیں = std::min(CHUNK_صفیں, n_صفیں - r0);
            ثابت int64_t elems = صفیں * ne0;
            ثابت خودکار * src = static_cast<ثابت uint8_t *>(ڈیٹا) + r0 * src_صف_بائٹیں;
            type_traits->مقصد_float(src, scratch.ڈیٹا(), elems);

            اگر (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.ڈیٹا(), f16_بنیاد + (r0 * ne0) * حجم(uint16_t), elems);
            } ورنہ {
                ثابت int64_t بلاک_انحراف = (r0 * ne0) / بلاک_سائز_3;
                اگر (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.ڈیٹا(), weights, scales, zp, elems, بلاک_سائز_3, بلاک_انحراف);
                } ورنہ {
                    quantize_q8_0(scratch.ڈیٹا(), weights, scales, zp, elems, بلاک_سائز_3, بلاک_انحراف);
                }
            }
        }
        اگر (requant_type == ExtraQuantType::F16) {
            خودکار نتیجہ = std::make_shared<ov::op::v0::Constant>(weights);
            نتیجہ->سیٹ_friendly_name(tensor->name);
            واپس نتیجہ;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> وزن_گرہ;
    اگر (is_u4) {
        وزن_گرہ = make_int4_weights(weights, scales, zp, بلاک_سائز_3);
    } ورنہ {
        وزن_گرہ = make_int8_weights(weights, scales, zp, بلاک_سائز_3);
    }

    خودکار نتیجہ = وزن_گرہ.get_گرہ_shared_ptr();
    نتیجہ->سیٹ_friendly_name(tensor->name);
    واپس نتیجہ;
}
}  // namespace

Ov_وزن process_وزن_tensor(ثابت ggml_tensor * tensor, ثابت خالی * ڈیٹا, خالی * output_بنیاد_ptr, منطقی use_bias) {
    GGML_ASSERT(tensor != خالی_اشاریہ);
    GGML_ASSERT(ڈیٹا != خالی_اشاریہ);

    Ov_وزن نتیجہ;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape گرہ_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    اگر (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        انتخاب (tensor->type) {
        حالت GGML_TYPE_F32:
            element_type = ov::element::f32;
            توڑو;
        حالت GGML_TYPE_F16:
            element_type = ov::element::f16;
            توڑو;
        حالت GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            توڑو;
        طےشدہ:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        اگر (output_بنیاد_ptr && output_بنیاد_ptr != ڈیٹا) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_بائٹیں = ggml_nbytes(tensor);
            حافظہ_نقل(output_بنیاد_ptr, ڈیٹا, tensor_بائٹیں);
            نتیجہ.weights = ov::Tensor(element_type, گرہ_shape, output_بنیاد_ptr);
        } ورنہ {
            نتیجہ.weights = ov::Tensor(element_type, گرہ_shape, ڈیٹا);
        }
        نتیجہ.وزن_گرہ = std::make_shared<ov::op::v0::Constant>(نتیجہ.weights);
        واپس نتیجہ;
    }

    // Handle quantized weights
    اگر (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    نتیجہ.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ثابت خودکار & layout = نتیجہ.layout;
    اگر (layout.کل_سائز == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    ثابت منطقی for_gather_matmul = tensor->ne[2] > 1;
    ثابت منطقی zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    ثابت منطقی is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    اگر (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_بلاک_سائز),
                                  MXFP4_بلاک_بائٹیں};
        ثابت size_t tensor_بائٹیں = ggml_nbytes(tensor);
        اگر (output_بنیاد_ptr) {
            خودکار * buf_بنیاد = static_cast<uint8_t *>(output_بنیاد_ptr);
            حافظہ_نقل(buf_بنیاد + layout.weights_انحراف, ڈیٹا, tensor_بائٹیں);
            نتیجہ.weights = ov::Tensor(ov::element::u8, packed_shape, buf_بنیاد + layout.weights_انحراف);
        } ورنہ {
            نتیجہ.weights = ov::Tensor(ov::element::u8, packed_shape);
            حافظہ_نقل(نتیجہ.weights.ڈیٹا(), ڈیٹا, tensor_بائٹیں);
        }
        نتیجہ.وزن_گرہ = make_mxfp4_moe_packed_weights(نتیجہ.weights).get_گرہ_shared_ptr();
        نتیجہ.وزن_گرہ->سیٹ_friendly_name(tensor->name);
        واپس نتیجہ;
    }

    اگر (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_بنیاد_ptr = خالی_اشاریہ;
    }

    // F16 requant path - no separate scales/zp needed in result
    اگر (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        اگر (output_بنیاد_ptr) {
            نتیجہ.weights = ov::Tensor(ov::element::f16, گرہ_shape,
                                        static_cast<uint8_t *>(output_بنیاد_ptr) + layout.weights_انحراف);
        } ورنہ {
            نتیجہ.weights = ov::Tensor(ov::element::f16, گرہ_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        نتیجہ.وزن_گرہ =
            requantize_مقصد_buffers(tensor, ڈیٹا, ExtraQuantType::F16, 0, نتیجہ.weights, dummy_scales, dummy_zp);
        واپس نتیجہ;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type وزن_type;
    اگر (tensor->type == GGML_TYPE_MXFP4) {
        وزن_type = ov::element::f4e2m1;
    } ورنہ اگر (layout.is_متشاکل) {
        وزن_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } ورنہ {
        وزن_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = گرہ_shape;
    scale_shape.back() /= layout.weights_فی_بلاک;

    اگر (tensor->type == GGML_TYPE_MXFP4) {
        اگر (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            گرہ_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } ورنہ {
            گرہ_shape.clear();
            برائے (صحیح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                گرہ_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = گرہ_shape;
        scale_shape.back() /= layout.weights_فی_بلاک;
    }

    ثابت ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    اگر (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    اگر (output_بنیاد_ptr) {
        uint8_t * buf_بنیاد = static_cast<uint8_t *>(output_بنیاد_ptr);
        نتیجہ.weights = ov::Tensor(وزن_type, گرہ_shape, buf_بنیاد + layout.weights_انحراف);
        نتیجہ.scales = ov::Tensor(scale_type, scale_shape, buf_بنیاد + layout.scales_انحراف);
        اگر (!layout.is_متشاکل) {
            نتیجہ.zp = ov::Tensor(zp_type, scale_shape, buf_بنیاد + layout.zp_انحراف);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } ورنہ {
        نتیجہ.weights = ov::Tensor(وزن_type, گرہ_shape);
        نتیجہ.scales = ov::Tensor(scale_type, scale_shape);
        اگر (!layout.is_متشاکل) {
            نتیجہ.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    اگر (layout.is_requant && layout.requant_type.has_value()) {
        نتیجہ.وزن_گرہ = requantize_مقصد_buffers(tensor, ڈیٹا, layout.requant_type.value(), layout.weights_فی_بلاک,
                                                   نتیجہ.weights, نتیجہ.scales, نتیجہ.zp);
    } ورنہ {
        نتیجہ.وزن_گرہ =
            نکالو_quantized_weights(tensor, ڈیٹا, نتیجہ.weights, نتیجہ.scales, نتیجہ.zp, use_bias);
    }

    واپس نتیجہ;
}
