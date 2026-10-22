#ضمن "ggml-quants.h"

#ضمن "ggml-common.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"
#ضمن "ggml.h"

#ضمن <algorithm>
#ضمن <cassert>
#ضمن <cmath>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <limits>
#ضمن <memory>
#ضمن <openvino/core/except.hpp>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/core/parallel.hpp>
#ضمن <openvino/core/shape.hpp>
#ضمن <openvino/core/type/element_type.hpp>
#ضمن <openvino/core/type/element_type_traits.hpp>
#ضمن <openvino/core/type/float16.hpp>
#ضمن <openvino/core/type/float4_e2m1.hpp>
#ضمن <openvino/core/type/float8_e8m0.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/util/attr_types.hpp>
#ضمن <openvino/pass/constant_folding.hpp>
#ضمن <openvino/runtime/tensor.hpp>
#ضمن <string>
#ضمن <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
نطاق ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
منطقي get_single_مقدار(ثابت std::shared_ptr<ov::op::v0::Constant> & const_گره,
                      عائم & value,
                      منطقي check_مقدار_range = درست);
}  // namespace ov::op::util

نطاق {
فراغ unpack_32_4(ثابت uint8_t * داده, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    لكل (صحيح j = 0; j < 16; ++j) {
        uint8_t x = (داده[j] & 0x0F);
        uint8_t y = (داده[j] >> 4);
        إذا (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_بلوک_اندازه = 32;
constexpr size_t MXFP4_بلوک_QS_اندازه = MXFP4_بلوک_اندازه / 2;
constexpr size_t MXFP4_بلوک_بایتها = حجم(uint8_t) + MXFP4_بلوک_QS_اندازه;

فراغ pack_32_mxfp4_for_openvino(ثابت uint8_t * داده, uint8_t * dst) {
    لكل (صحيح j = 0; j < static_cast<صحيح>(MXFP4_بلوک_QS_اندازه); j += 2) {
        ثابت uint8_t v0 = داده[j] & 0x0F;
        ثابت uint8_t v1 = (داده[j + 1] & 0x0F) << 4;
        ثابت uint8_t v16 = داده[j] >> 4;
        ثابت uint8_t v17 = داده[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_بلوک_اندازه / 4 + j / 2] = v16 | v17;
    }
}

فراغ استخراج_mxfp4_داده(ثابت ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ثابت خودکار * داده = static_cast<ثابت uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ثابت size_t n_بلوکها = scales_arr.get_اندازه();

    ov::parallel_for(n_بلوکها, [&](size_t i) {
        ثابت uint8_t * بلوک = داده + i * MXFP4_بلوک_بایتها;
        pack_32_mxfp4_for_openvino(بلوک + حجم(uint8_t), weights + i * MXFP4_بلوک_QS_اندازه);
        scales[i] = ov::float8_e8m0::from_بیتها(بلوک[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
فراغ استخراج_q4_0_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t بایتها_در_هر_بلوک = 18;  // 2 bytes scale, 32x0.5 byte weights

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقي is_متقارن = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    إذا (!is_متقارن) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            scales[i] = ov::float16::from_بیتها(*((uint16_t *) (داده + i * بایتها_در_هر_بلوک)));
            // Pack two 4-bit zero points per byte
            إذا (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } وإلا {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(داده + i * بایتها_در_هر_بلوک + 2, weights + i * 16);
        });
    } وإلا {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            scales[i] = ov::float16::from_بیتها(*((uint16_t *) (داده + i * بایتها_در_هر_بلوک)));
            unpack_32_4(داده + i * بایتها_در_هر_بلوک + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            لكل (صحيح j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
فراغ استخراج_q4_1_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایتها_در_هر_بلوک = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    إذا (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        خودکار * bias = zp_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            عائم scale = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) (داده + i * بایتها_در_هر_بلوک))));
            عائم min = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) (داده + i * بایتها_در_هر_بلوک + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(داده + i * بایتها_در_هر_بلوک + 4, weights + i * 16);
        });
    } وإلا {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            عائم scale = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) (داده + i * بایتها_در_هر_بلوک))));
            عائم min = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) (داده + i * بایتها_در_هر_بلوک + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            إذا (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } وإلا {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(داده + i * بایتها_در_هر_بلوک + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
فراغ استخراج_q5_1_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایتها_در_هر_بلوک = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ثابت صحيح qk = 32;

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());  // u8 weights, one byte per weight
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    خودکار read_u16 = [](ثابت uint8_t * p) {
        uint16_t v;
        کپی_حافظه(&v, p, حجم(v));
        أرجع v;
    };

    خودکار unpack_بلوک = [&](ثابت uint8_t * بلوک, uint8_t * dst) {
        uint32_t qh;
        کپی_حافظه(&qh, بلوک + 4, حجم(uint32_t));
        ثابت uint8_t * qs = بلوک + 8;
        لكل (صحيح j = 0; j < qk / 2; ++j) {
            ثابت uint8_t lo = qs[j] & 0x0F;
            ثابت uint8_t hi = qs[j] >> 4;
            ثابت uint8_t بیت_lo = (qh >> j) & 1;
            ثابت uint8_t بیت_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (بیت_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (بیت_hi << 4);  // last 16 weights
        }
    };

    إذا (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        خودکار * bias = zp_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            ثابت uint8_t * بلوک = داده + i * بایتها_در_هر_بلوک;
            عائم scale = static_cast<عائم>(ov::float16::from_بیتها(read_u16(بلوک)));
            عائم min = static_cast<عائم>(ov::float16::from_بیتها(read_u16(بلوک + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_بلوک(بلوک, weights + i * qk);
        });
    } وإلا {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());  // u8 zero points
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            ثابت uint8_t * بلوک = داده + i * بایتها_در_هر_بلوک;
            عائم scale = static_cast<عائم>(ov::float16::from_بیتها(read_u16(بلوک)));
            عائم min = static_cast<عائم>(ov::float16::from_بیتها(read_u16(بلوک + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_بلوک(بلوک, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
فراغ استخراج_q8_0_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t weights_در_هر_بلوک = 32;
    ثابت uint64_t بایتها_در_هر_بلوک = 34;  // 2 bytes scale, 32x1 byte weights

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقي is_متقارن = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    إذا (!is_متقارن) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            uint8_t * بلوک_داده = داده + i * بایتها_در_هر_بلوک;
            scales[i] = ov::float16::from_بیتها(*(uint16_t *) بلوک_داده);
            zp[i] = 128;
            لكل (size_t j = 0; j < weights_در_هر_بلوک; ++j) {
                uint8_t x = بلوک_داده[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_در_هر_بلوک + j] = x;
            }
        });
    } وإلا {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            uint8_t * بلوک_داده = داده + i * بایتها_در_هر_بلوک;
            scales[i] = ov::float16::from_بیتها(*(uint16_t *) بلوک_داده);
            // Copy int8 weights as-is (the tensor element type is i8)
            کپی_حافظه(weights + i * weights_در_هر_بلوک, بلوک_داده + 2, weights_در_هر_بلوک);
        });
    }
}

فراغ unpack_256_4(ثابت uint8_t * داده, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    لكل (size_t i = 0; i < 4; ++i) {
        لكل (صحيح j = 0; j < 32; ++j) {
            uint8_t x = (داده[i * 32 + j] & 0x0F);
            uint8_t y = (داده[i * 32 + j] >> 4);
            إذا (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

فراغ استخراج_q4_k_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایتها_در_هر_بلوک = 2 + 2 + 12 + 128;
    ثابت uint64_t n_super_بلوک = tensor->nb[3] / بایتها_در_هر_بلوک;

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    خودکار * zp_u4 = use_bias ? اشاره_تهی : static_cast<uint8_t *>(zp_arr.داده());
    خودکار * bias_f16 = use_bias ? zp_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>() : اشاره_تهی;

    ov::parallel_for(n_super_بلوک, [&](size_t i) {
        uint8_t * بلوک_داده = داده + i * بایتها_در_هر_بلوک;

        // Extract scale factors and offsets
        عائم scale_scales = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) بلوک_داده)));
        عائم scale_mins = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) بلوک_داده + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = بلوک_داده + 4;

        // Calculate scales
        عائم scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<عائم>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<عائم>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<عائم>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<عائم>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<عائم>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<عائم>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<عائم>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<عائم>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        عائم min_vals[8];
        min_vals[0] = scale_mins * static_cast<عائم>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<عائم>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<عائم>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<عائم>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<عائم>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<عائم>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<عائم>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<عائم>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        لكل (صحيح j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            إذا (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } وإلا {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                إذا (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } وإلا {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(بلوک_داده + 16, weights + i * 128);
    });
}

فراغ استخراج_q6_k_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t بایتها_در_هر_بلوک = 128 + 64 + 16 + 2;
    ثابت uint64_t n_super_بلوک = tensor->nb[3] / بایتها_در_هر_بلوک;

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقي is_متقارن = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    إذا (!is_متقارن) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());
        ov::parallel_for(n_super_بلوک, [&](size_t i) {
            uint8_t * بلوک_داده = داده + i * بایتها_در_هر_بلوک;
            عائم scale_ضریب = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) بلوک_داده + 104)));
            لكل (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ضریب * static_cast<عائم>(*((int8_t *) (بلوک_داده + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = بلوک_داده;
            uint8_t * qh = بلوک_داده + 128;
            لكل (int64_t j = 0; j < 32; ++j) {
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
    } وإلا {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_بلوک, [&](size_t i) {
            uint8_t * بلوک_داده = داده + i * بایتها_در_هر_بلوک;
            عائم scale_ضریب = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) بلوک_داده + 104)));
            لكل (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ضریب * static_cast<عائم>(*((int8_t *) (بلوک_داده + 128 + 64 + j))));
            }
            uint8_t * ql = بلوک_داده;
            uint8_t * qh = بلوک_داده + 128;
            خودکار * signed_weights = reinterpret_cast<int8_t *>(weights);
            لكل (int64_t j = 0; j < 32; ++j) {
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

مضمن فراغ get_scale_min_k4(صحيح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    إذا (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } وإلا {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

فراغ استخراج_q5_k_داده(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایتها_در_هر_بلوک = 4 + 12 + 32 + 128;
    ثابت uint64_t n_super_بلوک = tensor->nb[3] / بایتها_در_هر_بلوک;

    خودکار * داده = static_cast<uint8_t *>(tensor->داده);
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    خودکار * zp_u8 = use_bias ? اشاره_تهی : static_cast<uint8_t *>(zp_arr.داده());
    خودکار * bias_f16 = use_bias ? zp_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>() : اشاره_تهی;

    ov::parallel_for(n_super_بلوک, [&](size_t i) {
        uint8_t * بلوک_داده = داده + i * بایتها_در_هر_بلوک;

        ثابت عائم d = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) بلوک_داده)));
        ثابت عائم min_ضریب = static_cast<عائم>(ov::float16::from_بیتها(*((uint16_t *) بلوک_داده + 1)));

        ثابت uint8_t * scales_داده = بلوک_داده + 4;   // 12 bytes of scales
        ثابت uint8_t * qh = بلوک_داده + 4 + 12;       // 32 bytes of high bits
        ثابت uint8_t * ql = بلوک_داده + 4 + 12 + 32;  // 128 bytes of low bits

        صحيح is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        لكل (صحيح j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_داده, &sc, &m);
            ثابت عائم d1 = d * sc;
            ثابت عائم m1 = min_ضریب * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_داده, &sc, &m);
            ثابت عائم d2 = d * sc;
            ثابت عائم m2 = min_ضریب * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            إذا (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } وإلا {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            لكل (صحيح l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            لكل (صحيح l = 0; l < 32; ++l) {
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
                                       size_t گروه_اندازه = GGML_QUANTIZATION_گروه_اندازه,
                                       منطقي use_bias = خطأ,
                                       منطقي for_gather_matmul = خطأ) {
    ov::Shape orig_shape = وزن.get_shape();
    منطقي is_signed = (وزن.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    خودکار scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= گروه_اندازه;
    packed_shape.push_back(گروه_اندازه);
    ثابت size_t گروه_dim = packed_shape.size() - 2;

    إذا (packed_shape[گروه_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + گروه_dim);
    } وإلا {
        scale_shape.push_back(1);
        scales.مجموعه_shape(scale_shape);
        إذا (!is_signed && zp.get_اندازه() > 0) {
            خودکار zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.مجموعه_shape(zp_shape);
        }
    }

    خودکار scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> نتیجه;
    إذا (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        خودکار weights_گره = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.داده()), اشاره_تهی);
        weights_گره->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گره, ov::element::f16);
        خودکار mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_پخش_Type::NUMPY);
        نتیجه = mul;
    } وإلا {
        // Unsigned path
        خودکار weights_گره = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.داده()), اشاره_تهی);
        weights_گره->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گره, ov::element::f16);

        إذا (use_bias && zp.get_اندازه() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            خودکار * bias_zp_داده = zp.داده<ov::float16>();
            ثابت خودکار * scale_داده = scales.داده<ov::float16>();
            ثابت size_t n = zp.get_اندازه();
            لكل (size_t i = 0; i < n; i++) {
                عائم s = static_cast<عائم>(scale_داده[i]);
                عائم b = static_cast<عائم>(bias_zp_داده[i]);
                bias_zp_داده[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            خودکار zero_نقطه_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_نقطه_f16, ov::op::Auto_پخش_Type::NUMPY);
            نتیجه = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_پخش_Type::NUMPY);
        } وإلا {
            // Zero point path: (w - zp) * s
            خودکار zero_نقطه = std::make_shared<ov::op::v0::Constant>(zp);
            عائم zp_مقدار;
            إذا (ov::op::util::get_single_مقدار(zero_نقطه, zp_مقدار)) {
                zero_نقطه = ov::op::v0::Constant::create(zero_نقطه->get_element_type(), {}, {zp_مقدار});
            }
            خودکار zero_نقطه_f16 = std::make_shared<ov::op::v0::Convert>(zero_نقطه, ov::element::f16);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_نقطه_f16, ov::op::Auto_پخش_Type::NUMPY);
            خودکار mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_پخش_Type::NUMPY);
            نتیجه = mul;
        }
    }

    إذا (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        خودکار final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(نتیجه, final_shape, خطأ);
        نتیجه = reshaped;
    }

    إذا (for_gather_matmul) {
        أرجع نتیجه;
    }
    أرجع std::make_shared<ov::op::v0::Convert>(نتیجه, ov::element::f32);
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
                                       size_t گروه_اندازه = GGML_QUANTIZATION_گروه_اندازه,
                                       منطقي use_bias = خطأ,
                                       منطقي for_gather_matmul = خطأ) {
    ov::Shape orig_وزن_shape = وزن.get_shape();
    منطقي is_signed = (وزن.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_وزن_shape;
    packed_shape.back() /= گروه_اندازه;
    packed_shape.push_back(گروه_اندازه);
    ثابت size_t گروه_dim = packed_shape.size() - 2;

    إذا (packed_shape[گروه_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + گروه_dim);
    } وإلا {
        scale_shape.push_back(1);
        scales.مجموعه_shape(scale_shape);
        إذا (!is_signed && zp.get_اندازه() > 0) {
            خودکار zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.مجموعه_shape(zp_shape);
        }
    }

    خودکار scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> نتیجه;
    إذا (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        خودکار weights_گره = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.داده()), اشاره_تهی);
        weights_گره->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گره, ov::element::f16);
        خودکار mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_پخش_Type::NUMPY);
        نتیجه = mul;
    } وإلا {
        // Unsigned path
        خودکار weights_گره = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.داده()), اشاره_تهی);
        weights_گره->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خودکار weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_گره, ov::element::f16);

        إذا (use_bias && zp.get_اندازه() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            خودکار * bias_zp_داده = zp.داده<ov::float16>();
            ثابت خودکار * scale_داده = scales.داده<ov::float16>();
            ثابت size_t n = zp.get_اندازه();
            لكل (size_t i = 0; i < n; i++) {
                عائم s = static_cast<عائم>(scale_داده[i]);
                عائم b = static_cast<عائم>(bias_zp_داده[i]);
                bias_zp_داده[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            خودکار zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_points_f16, ov::op::Auto_پخش_Type::NUMPY);
            نتیجه = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_پخش_Type::NUMPY);
        } وإلا {
            // Zero point path: (w - zp) * s
            خودکار zero_points_گره = std::make_shared<ov::op::v0::Constant>(zp);
            عائم zp_مقدار;
            إذا (ov::op::util::get_single_مقدار(zero_points_گره, zp_مقدار)) {
                zero_points_گره = ov::op::v0::Constant::create(zero_points_گره->get_element_type(), {}, {zp_مقدار});
            }
            خودکار zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_گره, ov::element::f16);
            خودکار w_zp =
                std::make_shared<ov::op::v1::تفریق>(weights_f16, zero_points_f16, ov::op::Auto_پخش_Type::NUMPY);
            خودکار mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_پخش_Type::NUMPY);
            نتیجه = mul;
        }
    }

    إذا (packed_shape.size() != orig_وزن_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        خودکار final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_وزن_shape.size()},
                                                                  orig_وزن_shape);
        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(نتیجه, final_shape, خطأ);
        نتیجه = reshaped;
    }

    إذا (for_gather_matmul) {
        أرجع نتیجه;
    }
    أرجع std::make_shared<ov::op::v0::Convert>(نتیجه, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & وزن, ov::Tensor & scales) {
    ثابت ov::Shape final_shape = وزن.get_shape();
    GGML_ASSERT(!final_shape.خالی());
    GGML_ASSERT(final_shape.back() % MXFP4_بلوک_اندازه == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_بلوک_اندازه;
    packed_shape.push_back(MXFP4_بلوک_اندازه);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.مجموعه_shape(scale_shape);

    خودکار weights_گره = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(وزن.داده()), اشاره_تهی);
    weights_گره->get_rt_info()["__gguf_tensor_holder"] = وزن;
    خودکار weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_گره, ov::element::f32);

    خودکار scales_گره = std::make_shared<ov::op::v0::Constant>(scales);
    خودکار scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_گره, ov::element::f32);
    ov::Output<ov::Node> نتیجه =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_پخش_Type::NUMPY);

    خودکار final_shape_گره =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    أرجع std::make_shared<ov::op::v1::Reshape>(نتیجه, final_shape_گره, خطأ);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & وزن) {
    خودکار weights_گره = std::make_shared<ov::op::v0::Constant>(ov::element::u8, وزن.get_shape(),
                                                               static_cast<uint8_t *>(وزن.داده()), اشاره_تهی);
    weights_گره->get_rt_info()["__gguf_tensor_holder"] = وزن;
    weights_گره->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = درست;
    أرجع weights_گره;
}

فراغ quantize_q4_0(ثابت عائم * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();
    منطقي is_متقارن = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    إذا (!is_متقارن) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());
        لكل (صحيح i = 0; i < nb; i++) {
            عائم amax = 0.0f;
            عائم max = 0.0f;
            لكل (صحيح j = 0; j < qk; j++) {
                ثابت عائم v = x[i * qk + j];
                إذا (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ثابت عائم d = max / -8;
            إذا (d == 0) {
                scales[i] = ov::float16(1.0f);
                إذا (i % 2 == 0) {
                    zp[i / 2] = 8;
                } وإلا {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                استمر;
            }
            ثابت عائم id = 1.0f / d;
            scales[i] = ov::float16(d);
            إذا (i % 2 == 0) {
                zp[i / 2] = 8;
            } وإلا {
                zp[i / 2] |= (8 << 4);
            }
            لكل (صحيح j = 0; j < qk / 2; ++j) {
                ثابت عائم x0 = x[i * qk + 2 * j] * id;
                ثابت عائم x1 = x[i * qk + 2 * j + 1] * id;
                ثابت uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                ثابت uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } وإلا {
        // Symmetric: produce signed i4 values in [-8, 7]
        لكل (صحيح i = 0; i < nb; i++) {
            عائم amax = 0.0f;
            عائم max = 0.0f;
            لكل (صحيح j = 0; j < qk; j++) {
                ثابت عائم v = x[i * qk + j];
                إذا (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            ثابت عائم d = max / -8;
            إذا (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                استمر;
            }
            ثابت عائم id = 1.0f / d;
            scales[i] = ov::float16(d);
            لكل (صحيح j = 0; j < qk / 2; ++j) {
                ثابت عائم x0 = x[i * qk + 2 * j] * id;
                ثابت عائم x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (صحيح) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (صحيح) roundf(x1)));
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
فراغ quantize_q4_1_asym(ثابت عائم * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده());
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>();
    خودکار * zp = static_cast<uint8_t *>(zp_arr.داده());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    خودکار store_zp = [zp](صحيح i, uint8_t v) {
        إذا (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } وإلا {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    لكل (صحيح i = 0; i < nb; i++) {
        عائم vmin = x[i * qk];
        عائم vmax = x[i * qk];
        لكل (صحيح j = 1; j < qk; j++) {
            ثابت عائم v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        ثابت عائم d = (vmax - vmin) / 15.0f;
        إذا (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            استمر;
        }
        ثابت عائم id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        ثابت صحيح zq = std::max(0, std::min(15, (صحيح) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        لكل (صحيح j = 0; j < qk / 2; ++j) {
            ثابت عائم x0 = x[i * qk + 2 * j] * id;
            ثابت عائم x1 = x[i * qk + 2 * j + 1] * id;
            ثابت uint8_t q0 = (uint8_t) std::max(0, std::min(15, (صحيح) lroundf(x0) + zq));
            ثابت uint8_t q1 = (uint8_t) std::max(0, std::min(15, (صحيح) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

فراغ quantize_q8_0(ثابت عائم * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t بلوک_جابه_جایی = 0) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده()) + بلوک_جابه_جایی * qk;
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>() + بلوک_جابه_جایی;
    منطقي is_متقارن = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    إذا (!is_متقارن) {
        خودکار * zp = static_cast<uint8_t *>(zp_arr.داده()) + بلوک_جابه_جایی;
        لكل (صحيح i = 0; i < nb; i++) {
            عائم amax = 0.0f;
            لكل (صحيح j = 0; j < qk; j++) {
                ثابت عائم v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ثابت عائم d = amax / 127.0f;
            ثابت عائم id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            لكل (صحيح j = 0; j < qk; ++j) {
                ثابت عائم x0 = x[i * qk + j] * id;
                ثابت int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } وإلا {
        // Symmetric: store signed int8 values directly
        خودکار * signed_weights = reinterpret_cast<int8_t *>(weights);
        لكل (صحيح i = 0; i < nb; i++) {
            عائم amax = 0.0f;
            لكل (صحيح j = 0; j < qk; j++) {
                ثابت عائم v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            ثابت عائم d = amax / 127.0f;
            ثابت عائم id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            لكل (صحيح j = 0; j < qk; ++j) {
                ثابت عائم x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

فراغ quantize_q8_1(ثابت عائم * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t بلوک_جابه_جایی = 0) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    خودکار * weights = static_cast<uint8_t *>(weights_arr.داده()) + بلوک_جابه_جایی * qk;
    خودکار * scales = scales_arr.داده<ov::element_type_traits<ov::element::f16>::value_type>() + بلوک_جابه_جایی;
    خودکار * zp = static_cast<uint8_t *>(zp_arr.داده()) + بلوک_جابه_جایی;
    لكل (صحيح i = 0; i < nb; i++) {
        عائم min = std::numeric_limits<عائم>::max();
        عائم max = std::numeric_limits<عائم>::lowest();

        لكل (صحيح j = 0; j < qk; j++) {
            ثابت عائم v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        ثابت عائم d = (max - min) / ((1 << 8) - 1);
        ثابت عائم id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        لكل (صحيح j = 0; j < qk; ++j) {
            ثابت عائم x0 = (x[i * qk + j] - min) * id;
            ثابت uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> استخراج_quantized_weights(ثابت ggml_tensor * tensor,
                                                    ثابت فراغ * داده,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    منطقي use_bias = خطأ) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor موقت_tensor = *tensor;
    موقت_tensor.داده = const_cast<فراغ *>(داده);

    إذا (tensor->type == GGML_TYPE_MXFP4) {
        استخراج_mxfp4_داده(&موقت_tensor, weights, scales);
        خودکار نتیجه = make_mxfp4_weights(weights, scales).get_گره_shared_ptr();
        نتیجه->مجموعه_friendly_name(tensor->name);
        أرجع نتیجه;
    }

    // Determine block size based on tensor type
    int64_t weights_در_هر_بلوک;
    منطقي is_u4;
    بدل (tensor->type) {
    حالة GGML_TYPE_Q4_0:
    حالة GGML_TYPE_Q4_1:
    حالة GGML_TYPE_Q4_K:
        is_u4 = درست;
        weights_در_هر_بلوک = 32;
        اكسر;
    حالة GGML_TYPE_Q8_0:
    حالة GGML_TYPE_Q5_1:
    حالة GGML_TYPE_Q5_K:
        is_u4 = خطأ;
        weights_در_هر_بلوک = 32;
        اكسر;
    حالة GGML_TYPE_Q6_K:
        is_u4 = خطأ;
        weights_در_هر_بلوک = 16;
        اكسر;
    افتراضي:
        ارم std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    ثابت منطقي for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    بدل (tensor->type) {
    حالة GGML_TYPE_Q4_0:
        استخراج_q4_0_داده(&موقت_tensor, weights, scales, zp);
        اكسر;
    حالة GGML_TYPE_Q4_1:
        استخراج_q4_1_داده(&موقت_tensor, weights, scales, zp, use_bias);
        اكسر;
    حالة GGML_TYPE_Q4_K:
        استخراج_q4_k_داده(&موقت_tensor, weights, scales, zp, use_bias);
        اكسر;
    حالة GGML_TYPE_Q5_1:
        استخراج_q5_1_داده(&موقت_tensor, weights, scales, zp, use_bias);
        اكسر;
    حالة GGML_TYPE_Q8_0:
        استخراج_q8_0_داده(&موقت_tensor, weights, scales, zp);
        اكسر;
    حالة GGML_TYPE_Q6_K:
        استخراج_q6_k_داده(&موقت_tensor, weights, scales, zp);
        اكسر;
    حالة GGML_TYPE_Q5_K:
        استخراج_q5_k_داده(&موقت_tensor, weights, scales, zp, use_bias);
        اكسر;
    افتراضي:
        ارم std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> وزن_گره;
    إذا (is_u4) {
        وزن_گره = make_int4_weights(weights, scales, zp, weights_در_هر_بلوک, use_bias, for_gather_matmul);
    } وإلا {
        وزن_گره = make_int8_weights(weights, scales, zp, weights_در_هر_بلوک, use_bias, for_gather_matmul);
    }

    خودکار نتیجه = وزن_گره.get_گره_shared_ptr();
    نتیجه->مجموعه_friendly_name(tensor->name);
    أرجع نتیجه;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_مقصد_buffers(ثابت ggml_tensor * tensor,
                                                ثابت فراغ * داده,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t بلوک_اندازه_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ثابت int64_t ne0 = tensor->ne[0];                 // elements per row
    ثابت int64_t n_ردیفها = n_elements / ne0;
    ثابت خودکار * type_traits = ggml_get_type_traits(tensor->type);
    ثابت size_t src_ردیف_بایتها = ggml_ردیف_اندازه(tensor->type, ne0);

    منطقي is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    ثابت منطقي جریان_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(بلوک_اندازه_3 > 0 && ne0 % بلوک_اندازه_3 != 0);

    إذا (!جریان_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<عائم> weights_f32(n_elements);
        type_traits->مقصد_float(داده, weights_f32.داده(), n_elements);
        إذا (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.داده(), weights.داده(), n_elements);
            خودکار نتیجه = std::make_shared<ov::op::v0::Constant>(weights);
            نتیجه->مجموعه_friendly_name(tensor->name);
            أرجع نتیجه;
        }
        إذا (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.داده(), weights, scales, zp, n_elements, بلوک_اندازه_3);
        } وإلا إذا (is_u4) {
            quantize_q4_0(weights_f32.داده(), weights, scales, zp, n_elements, بلوک_اندازه_3);
        } وإلا إذا (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.داده(), weights, scales, zp, n_elements, بلوک_اندازه_3);
        } وإلا {
            quantize_q8_0(weights_f32.داده(), weights, scales, zp, n_elements, بلوک_اندازه_3);
        }
    } وإلا {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ثابت int64_t CHUNK_ردیفها = std::min<int64_t>(n_ردیفها, 256);
        std::vector<عائم> scratch(CHUNK_ردیفها * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        خودکار * f16_پایه = static_cast<uint8_t *>(weights.داده());
        لكل (int64_t r0 = 0; r0 < n_ردیفها; r0 += CHUNK_ردیفها) {
            ثابت int64_t ردیفها = std::min(CHUNK_ردیفها, n_ردیفها - r0);
            ثابت int64_t elems = ردیفها * ne0;
            ثابت خودکار * src = static_cast<ثابت uint8_t *>(داده) + r0 * src_ردیف_بایتها;
            type_traits->مقصد_float(src, scratch.داده(), elems);

            إذا (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.داده(), f16_پایه + (r0 * ne0) * حجم(uint16_t), elems);
            } وإلا {
                ثابت int64_t بلوک_جابه_جایی = (r0 * ne0) / بلوک_اندازه_3;
                إذا (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.داده(), weights, scales, zp, elems, بلوک_اندازه_3, بلوک_جابه_جایی);
                } وإلا {
                    quantize_q8_0(scratch.داده(), weights, scales, zp, elems, بلوک_اندازه_3, بلوک_جابه_جایی);
                }
            }
        }
        إذا (requant_type == ExtraQuantType::F16) {
            خودکار نتیجه = std::make_shared<ov::op::v0::Constant>(weights);
            نتیجه->مجموعه_friendly_name(tensor->name);
            أرجع نتیجه;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> وزن_گره;
    إذا (is_u4) {
        وزن_گره = make_int4_weights(weights, scales, zp, بلوک_اندازه_3);
    } وإلا {
        وزن_گره = make_int8_weights(weights, scales, zp, بلوک_اندازه_3);
    }

    خودکار نتیجه = وزن_گره.get_گره_shared_ptr();
    نتیجه->مجموعه_friendly_name(tensor->name);
    أرجع نتیجه;
}
}  // namespace

Ov_وزن process_وزن_tensor(ثابت ggml_tensor * tensor, ثابت فراغ * داده, فراغ * output_پایه_ptr, منطقي use_bias) {
    GGML_ASSERT(tensor != اشاره_تهی);
    GGML_ASSERT(داده != اشاره_تهی);

    Ov_وزن نتیجه;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape گره_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    إذا (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        بدل (tensor->type) {
        حالة GGML_TYPE_F32:
            element_type = ov::element::f32;
            اكسر;
        حالة GGML_TYPE_F16:
            element_type = ov::element::f16;
            اكسر;
        حالة GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            اكسر;
        افتراضي:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        إذا (output_پایه_ptr && output_پایه_ptr != داده) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_بایتها = ggml_nbytes(tensor);
            کپی_حافظه(output_پایه_ptr, داده, tensor_بایتها);
            نتیجه.weights = ov::Tensor(element_type, گره_shape, output_پایه_ptr);
        } وإلا {
            نتیجه.weights = ov::Tensor(element_type, گره_shape, داده);
        }
        نتیجه.وزن_گره = std::make_shared<ov::op::v0::Constant>(نتیجه.weights);
        أرجع نتیجه;
    }

    // Handle quantized weights
    إذا (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    نتیجه.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ثابت خودکار & layout = نتیجه.layout;
    إذا (layout.مجموع_اندازه == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    ثابت منطقي for_gather_matmul = tensor->ne[2] > 1;
    ثابت منطقي zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    ثابت منطقي is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    إذا (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_بلوک_اندازه),
                                  MXFP4_بلوک_بایتها};
        ثابت size_t tensor_بایتها = ggml_nbytes(tensor);
        إذا (output_پایه_ptr) {
            خودکار * buf_پایه = static_cast<uint8_t *>(output_پایه_ptr);
            کپی_حافظه(buf_پایه + layout.weights_جابه_جایی, داده, tensor_بایتها);
            نتیجه.weights = ov::Tensor(ov::element::u8, packed_shape, buf_پایه + layout.weights_جابه_جایی);
        } وإلا {
            نتیجه.weights = ov::Tensor(ov::element::u8, packed_shape);
            کپی_حافظه(نتیجه.weights.داده(), داده, tensor_بایتها);
        }
        نتیجه.وزن_گره = make_mxfp4_moe_packed_weights(نتیجه.weights).get_گره_shared_ptr();
        نتیجه.وزن_گره->مجموعه_friendly_name(tensor->name);
        أرجع نتیجه;
    }

    إذا (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_پایه_ptr = اشاره_تهی;
    }

    // F16 requant path - no separate scales/zp needed in result
    إذا (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        إذا (output_پایه_ptr) {
            نتیجه.weights = ov::Tensor(ov::element::f16, گره_shape,
                                        static_cast<uint8_t *>(output_پایه_ptr) + layout.weights_جابه_جایی);
        } وإلا {
            نتیجه.weights = ov::Tensor(ov::element::f16, گره_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        نتیجه.وزن_گره =
            requantize_مقصد_buffers(tensor, داده, ExtraQuantType::F16, 0, نتیجه.weights, dummy_scales, dummy_zp);
        أرجع نتیجه;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type وزن_type;
    إذا (tensor->type == GGML_TYPE_MXFP4) {
        وزن_type = ov::element::f4e2m1;
    } وإلا إذا (layout.is_متقارن) {
        وزن_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } وإلا {
        وزن_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = گره_shape;
    scale_shape.back() /= layout.weights_در_هر_بلوک;

    إذا (tensor->type == GGML_TYPE_MXFP4) {
        إذا (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            گره_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } وإلا {
            گره_shape.clear();
            لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                گره_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = گره_shape;
        scale_shape.back() /= layout.weights_در_هر_بلوک;
    }

    ثابت ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    إذا (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    إذا (output_پایه_ptr) {
        uint8_t * buf_پایه = static_cast<uint8_t *>(output_پایه_ptr);
        نتیجه.weights = ov::Tensor(وزن_type, گره_shape, buf_پایه + layout.weights_جابه_جایی);
        نتیجه.scales = ov::Tensor(scale_type, scale_shape, buf_پایه + layout.scales_جابه_جایی);
        إذا (!layout.is_متقارن) {
            نتیجه.zp = ov::Tensor(zp_type, scale_shape, buf_پایه + layout.zp_جابه_جایی);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } وإلا {
        نتیجه.weights = ov::Tensor(وزن_type, گره_shape);
        نتیجه.scales = ov::Tensor(scale_type, scale_shape);
        إذا (!layout.is_متقارن) {
            نتیجه.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    إذا (layout.is_requant && layout.requant_type.has_value()) {
        نتیجه.وزن_گره = requantize_مقصد_buffers(tensor, داده, layout.requant_type.value(), layout.weights_در_هر_بلوک,
                                                   نتیجه.weights, نتیجه.scales, نتیجه.zp);
    } وإلا {
        نتیجه.وزن_گره =
            استخراج_quantized_weights(tensor, داده, نتیجه.weights, نتیجه.scales, نتیجه.zp, use_bias);
    }

    أرجع نتیجه;
}
