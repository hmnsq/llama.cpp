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
منطقي get_single_ارزښت(ثابت std::shared_ptr<ov::op::v0::Constant> & const_غوټه,
                      عائم & value,
                      منطقي check_ارزښت_range = ریښتیا);
}  // namespace ov::op::util

نطاق {
فراغ unpack_32_4(ثابت uint8_t * معلومات, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    لكل (صحيح j = 0; j < 16; ++j) {
        uint8_t x = (معلومات[j] & 0x0F);
        uint8_t y = (معلومات[j] >> 4);
        إذا (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_بلاک_اندازه = 32;
constexpr size_t MXFP4_بلاک_QS_اندازه = MXFP4_بلاک_اندازه / 2;
constexpr size_t MXFP4_بلاک_بایټونه = حجم(uint8_t) + MXFP4_بلاک_QS_اندازه;

فراغ pack_32_mxfp4_for_openvino(ثابت uint8_t * معلومات, uint8_t * dst) {
    لكل (صحيح j = 0; j < static_cast<صحيح>(MXFP4_بلاک_QS_اندازه); j += 2) {
        ثابت uint8_t v0 = معلومات[j] & 0x0F;
        ثابت uint8_t v1 = (معلومات[j + 1] & 0x0F) << 4;
        ثابت uint8_t v16 = معلومات[j] >> 4;
        ثابت uint8_t v17 = معلومات[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_بلاک_اندازه / 4 + j / 2] = v16 | v17;
    }
}

فراغ ایستل_mxfp4_معلومات(ثابت ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    ثابت خپلکاری * معلومات = static_cast<ثابت uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    ثابت size_t n_بلاکونه = scales_arr.get_اندازه();

    ov::parallel_for(n_بلاکونه, [&](size_t i) {
        ثابت uint8_t * بلاک = معلومات + i * MXFP4_بلاک_بایټونه;
        pack_32_mxfp4_for_openvino(بلاک + حجم(uint8_t), weights + i * MXFP4_بلاک_QS_اندازه);
        scales[i] = ov::float8_e8m0::from_بټونه(بلاک[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
فراغ ایستل_q4_0_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t بایټونه_فی_بلاک = 18;  // 2 bytes scale, 32x0.5 byte weights

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقي is_متناظر = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    إذا (!is_متناظر) {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            scales[i] = ov::float16::from_بټونه(*((uint16_t *) (معلومات + i * بایټونه_فی_بلاک)));
            // Pack two 4-bit zero points per byte
            إذا (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } وإلا {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(معلومات + i * بایټونه_فی_بلاک + 2, weights + i * 16);
        });
    } وإلا {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            scales[i] = ov::float16::from_بټونه(*((uint16_t *) (معلومات + i * بایټونه_فی_بلاک)));
            unpack_32_4(معلومات + i * بایټونه_فی_بلاک + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            لكل (صحيح j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
فراغ ایستل_q4_1_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایټونه_فی_بلاک = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    إذا (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        خپلکاری * bias = zp_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            عائم scale = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) (معلومات + i * بایټونه_فی_بلاک))));
            عائم min = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) (معلومات + i * بایټونه_فی_بلاک + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(معلومات + i * بایټونه_فی_بلاک + 4, weights + i * 16);
        });
    } وإلا {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            عائم scale = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) (معلومات + i * بایټونه_فی_بلاک))));
            عائم min = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) (معلومات + i * بایټونه_فی_بلاک + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            إذا (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } وإلا {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(معلومات + i * بایټونه_فی_بلاک + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
فراغ ایستل_q5_1_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایټونه_فی_بلاک = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    ثابت صحيح qk = 32;

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());  // u8 weights, one byte per weight
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    خپلکاری read_u16 = [](ثابت uint8_t * p) {
        uint16_t v;
        حافظه_نقل(&v, p, حجم(v));
        أرجع v;
    };

    خپلکاری unpack_بلاک = [&](ثابت uint8_t * بلاک, uint8_t * dst) {
        uint32_t qh;
        حافظه_نقل(&qh, بلاک + 4, حجم(uint32_t));
        ثابت uint8_t * qs = بلاک + 8;
        لكل (صحيح j = 0; j < qk / 2; ++j) {
            ثابت uint8_t lo = qs[j] & 0x0F;
            ثابت uint8_t hi = qs[j] >> 4;
            ثابت uint8_t بټ_lo = (qh >> j) & 1;
            ثابت uint8_t بټ_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (بټ_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (بټ_hi << 4);  // last 16 weights
        }
    };

    إذا (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        خپلکاری * bias = zp_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            ثابت uint8_t * بلاک = معلومات + i * بایټونه_فی_بلاک;
            عائم scale = static_cast<عائم>(ov::float16::from_بټونه(read_u16(بلاک)));
            عائم min = static_cast<عائم>(ov::float16::from_بټونه(read_u16(بلاک + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_بلاک(بلاک, weights + i * qk);
        });
    } وإلا {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());  // u8 zero points
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            ثابت uint8_t * بلاک = معلومات + i * بایټونه_فی_بلاک;
            عائم scale = static_cast<عائم>(ov::float16::from_بټونه(read_u16(بلاک)));
            عائم min = static_cast<عائم>(ov::float16::from_بټونه(read_u16(بلاک + 2)));
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
فراغ ایستل_q8_0_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t weights_فی_بلاک = 32;
    ثابت uint64_t بایټونه_فی_بلاک = 34;  // 2 bytes scale, 32x1 byte weights

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقي is_متناظر = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    إذا (!is_متناظر) {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            uint8_t * بلاک_معلومات = معلومات + i * بایټونه_فی_بلاک;
            scales[i] = ov::float16::from_بټونه(*(uint16_t *) بلاک_معلومات);
            zp[i] = 128;
            لكل (size_t j = 0; j < weights_فی_بلاک; ++j) {
                uint8_t x = بلاک_معلومات[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_فی_بلاک + j] = x;
            }
        });
    } وإلا {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_اندازه(), [&](size_t i) {
            uint8_t * بلاک_معلومات = معلومات + i * بایټونه_فی_بلاک;
            scales[i] = ov::float16::from_بټونه(*(uint16_t *) بلاک_معلومات);
            // Copy int8 weights as-is (the tensor element type is i8)
            حافظه_نقل(weights + i * weights_فی_بلاک, بلاک_معلومات + 2, weights_فی_بلاک);
        });
    }
}

فراغ unpack_256_4(ثابت uint8_t * معلومات, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    لكل (size_t i = 0; i < 4; ++i) {
        لكل (صحيح j = 0; j < 32; ++j) {
            uint8_t x = (معلومات[i * 32 + j] & 0x0F);
            uint8_t y = (معلومات[i * 32 + j] >> 4);
            إذا (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

فراغ ایستل_q4_k_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایټونه_فی_بلاک = 2 + 2 + 12 + 128;
    ثابت uint64_t n_super_بلاک = tensor->nb[3] / بایټونه_فی_بلاک;

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    خپلکاری * zp_u4 = use_bias ? تش_اشاره : static_cast<uint8_t *>(zp_arr.معلومات());
    خپلکاری * bias_f16 = use_bias ? zp_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>() : تش_اشاره;

    ov::parallel_for(n_super_بلاک, [&](size_t i) {
        uint8_t * بلاک_معلومات = معلومات + i * بایټونه_فی_بلاک;

        // Extract scale factors and offsets
        عائم scale_scales = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) بلاک_معلومات)));
        عائم scale_mins = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) بلاک_معلومات + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = بلاک_معلومات + 4;

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
        unpack_256_4(بلاک_معلومات + 16, weights + i * 128);
    });
}

فراغ ایستل_q6_k_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    ثابت uint64_t بایټونه_فی_بلاک = 128 + 64 + 16 + 2;
    ثابت uint64_t n_super_بلاک = tensor->nb[3] / بایټونه_فی_بلاک;

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    منطقي is_متناظر = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    إذا (!is_متناظر) {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());
        ov::parallel_for(n_super_بلاک, [&](size_t i) {
            uint8_t * بلاک_معلومات = معلومات + i * بایټونه_فی_بلاک;
            عائم scale_ضریب = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) بلاک_معلومات + 104)));
            لكل (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ضریب * static_cast<عائم>(*((int8_t *) (بلاک_معلومات + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = بلاک_معلومات;
            uint8_t * qh = بلاک_معلومات + 128;
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
        ov::parallel_for(n_super_بلاک, [&](size_t i) {
            uint8_t * بلاک_معلومات = معلومات + i * بایټونه_فی_بلاک;
            عائم scale_ضریب = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) بلاک_معلومات + 104)));
            لكل (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_ضریب * static_cast<عائم>(*((int8_t *) (بلاک_معلومات + 128 + 64 + j))));
            }
            uint8_t * ql = بلاک_معلومات;
            uint8_t * qh = بلاک_معلومات + 128;
            خپلکاری * signed_weights = reinterpret_cast<int8_t *>(weights);
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

فراغ ایستل_q5_k_معلومات(ثابت ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       منطقي use_bias) {
    ثابت uint64_t بایټونه_فی_بلاک = 4 + 12 + 32 + 128;
    ثابت uint64_t n_super_بلاک = tensor->nb[3] / بایټونه_فی_بلاک;

    خپلکاری * معلومات = static_cast<uint8_t *>(tensor->معلومات);
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    خپلکاری * zp_u8 = use_bias ? تش_اشاره : static_cast<uint8_t *>(zp_arr.معلومات());
    خپلکاری * bias_f16 = use_bias ? zp_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>() : تش_اشاره;

    ov::parallel_for(n_super_بلاک, [&](size_t i) {
        uint8_t * بلاک_معلومات = معلومات + i * بایټونه_فی_بلاک;

        ثابت عائم d = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) بلاک_معلومات)));
        ثابت عائم min_ضریب = static_cast<عائم>(ov::float16::from_بټونه(*((uint16_t *) بلاک_معلومات + 1)));

        ثابت uint8_t * scales_معلومات = بلاک_معلومات + 4;   // 12 bytes of scales
        ثابت uint8_t * qh = بلاک_معلومات + 4 + 12;       // 32 bytes of high bits
        ثابت uint8_t * ql = بلاک_معلومات + 4 + 12 + 32;  // 128 bytes of low bits

        صحيح is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        لكل (صحيح j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_معلومات, &sc, &m);
            ثابت عائم d1 = d * sc;
            ثابت عائم m1 = min_ضریب * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_معلومات, &sc, &m);
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
                                       size_t ډله_اندازه = GGML_QUANTIZATION_ډله_اندازه,
                                       منطقي use_bias = خطأ,
                                       منطقي for_gather_matmul = خطأ) {
    ov::Shape orig_shape = وزن.get_shape();
    منطقي is_signed = (وزن.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    خپلکاری scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= ډله_اندازه;
    packed_shape.push_back(ډله_اندازه);
    ثابت size_t ډله_dim = packed_shape.size() - 2;

    إذا (packed_shape[ډله_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + ډله_dim);
    } وإلا {
        scale_shape.push_back(1);
        scales.سېټ_shape(scale_shape);
        إذا (!is_signed && zp.get_اندازه() > 0) {
            خپلکاری zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.سېټ_shape(zp_shape);
        }
    }

    خپلکاری scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> پایله;
    إذا (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        خپلکاری weights_غوټه = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.معلومات()), تش_اشاره);
        weights_غوټه->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خپلکاری weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_غوټه, ov::element::f16);
        خپلکاری mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_خپرونه_Type::NUMPY);
        پایله = mul;
    } وإلا {
        // Unsigned path
        خپلکاری weights_غوټه = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.معلومات()), تش_اشاره);
        weights_غوټه->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خپلکاری weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_غوټه, ov::element::f16);

        إذا (use_bias && zp.get_اندازه() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            خپلکاری * bias_zp_معلومات = zp.معلومات<ov::float16>();
            ثابت خپلکاری * scale_معلومات = scales.معلومات<ov::float16>();
            ثابت size_t n = zp.get_اندازه();
            لكل (size_t i = 0; i < n; i++) {
                عائم s = static_cast<عائم>(scale_معلومات[i]);
                عائم b = static_cast<عائم>(bias_zp_معلومات[i]);
                bias_zp_معلومات[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            خپلکاری zero_ټکی_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            خپلکاری w_zp =
                std::make_shared<ov::op::v1::کمول>(weights_f16, zero_ټکی_f16, ov::op::Auto_خپرونه_Type::NUMPY);
            پایله = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_خپرونه_Type::NUMPY);
        } وإلا {
            // Zero point path: (w - zp) * s
            خپلکاری zero_ټکی = std::make_shared<ov::op::v0::Constant>(zp);
            عائم zp_ارزښت;
            إذا (ov::op::util::get_single_ارزښت(zero_ټکی, zp_ارزښت)) {
                zero_ټکی = ov::op::v0::Constant::create(zero_ټکی->get_element_type(), {}, {zp_ارزښت});
            }
            خپلکاری zero_ټکی_f16 = std::make_shared<ov::op::v0::Convert>(zero_ټکی, ov::element::f16);
            خپلکاری w_zp =
                std::make_shared<ov::op::v1::کمول>(weights_f16, zero_ټکی_f16, ov::op::Auto_خپرونه_Type::NUMPY);
            خپلکاری mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_خپرونه_Type::NUMPY);
            پایله = mul;
        }
    }

    إذا (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        خپلکاری final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(پایله, final_shape, خطأ);
        پایله = reshaped;
    }

    إذا (for_gather_matmul) {
        أرجع پایله;
    }
    أرجع std::make_shared<ov::op::v0::Convert>(پایله, ov::element::f32);
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
                                       size_t ډله_اندازه = GGML_QUANTIZATION_ډله_اندازه,
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
    packed_shape.back() /= ډله_اندازه;
    packed_shape.push_back(ډله_اندازه);
    ثابت size_t ډله_dim = packed_shape.size() - 2;

    إذا (packed_shape[ډله_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + ډله_dim);
    } وإلا {
        scale_shape.push_back(1);
        scales.سېټ_shape(scale_shape);
        إذا (!is_signed && zp.get_اندازه() > 0) {
            خپلکاری zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.سېټ_shape(zp_shape);
        }
    }

    خپلکاری scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> پایله;
    إذا (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        خپلکاری weights_غوټه = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.معلومات()), تش_اشاره);
        weights_غوټه->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خپلکاری weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_غوټه, ov::element::f16);
        خپلکاری mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_خپرونه_Type::NUMPY);
        پایله = mul;
    } وإلا {
        // Unsigned path
        خپلکاری weights_غوټه = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(وزن.معلومات()), تش_اشاره);
        weights_غوټه->get_rt_info()["__gguf_tensor_holder"] = وزن;
        خپلکاری weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_غوټه, ov::element::f16);

        إذا (use_bias && zp.get_اندازه() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            خپلکاری * bias_zp_معلومات = zp.معلومات<ov::float16>();
            ثابت خپلکاری * scale_معلومات = scales.معلومات<ov::float16>();
            ثابت size_t n = zp.get_اندازه();
            لكل (size_t i = 0; i < n; i++) {
                عائم s = static_cast<عائم>(scale_معلومات[i]);
                عائم b = static_cast<عائم>(bias_zp_معلومات[i]);
                bias_zp_معلومات[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            خپلکاری zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            خپلکاری w_zp =
                std::make_shared<ov::op::v1::کمول>(weights_f16, zero_points_f16, ov::op::Auto_خپرونه_Type::NUMPY);
            پایله = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_خپرونه_Type::NUMPY);
        } وإلا {
            // Zero point path: (w - zp) * s
            خپلکاری zero_points_غوټه = std::make_shared<ov::op::v0::Constant>(zp);
            عائم zp_ارزښت;
            إذا (ov::op::util::get_single_ارزښت(zero_points_غوټه, zp_ارزښت)) {
                zero_points_غوټه = ov::op::v0::Constant::create(zero_points_غوټه->get_element_type(), {}, {zp_ارزښت});
            }
            خپلکاری zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_غوټه, ov::element::f16);
            خپلکاری w_zp =
                std::make_shared<ov::op::v1::کمول>(weights_f16, zero_points_f16, ov::op::Auto_خپرونه_Type::NUMPY);
            خپلکاری mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_خپرونه_Type::NUMPY);
            پایله = mul;
        }
    }

    إذا (packed_shape.size() != orig_وزن_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        خپلکاری final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_وزن_shape.size()},
                                                                  orig_وزن_shape);
        خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(پایله, final_shape, خطأ);
        پایله = reshaped;
    }

    إذا (for_gather_matmul) {
        أرجع پایله;
    }
    أرجع std::make_shared<ov::op::v0::Convert>(پایله, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & وزن, ov::Tensor & scales) {
    ثابت ov::Shape final_shape = وزن.get_shape();
    GGML_ASSERT(!final_shape.تشه());
    GGML_ASSERT(final_shape.back() % MXFP4_بلاک_اندازه == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_بلاک_اندازه;
    packed_shape.push_back(MXFP4_بلاک_اندازه);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.سېټ_shape(scale_shape);

    خپلکاری weights_غوټه = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(وزن.معلومات()), تش_اشاره);
    weights_غوټه->get_rt_info()["__gguf_tensor_holder"] = وزن;
    خپلکاری weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_غوټه, ov::element::f32);

    خپلکاری scales_غوټه = std::make_shared<ov::op::v0::Constant>(scales);
    خپلکاری scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_غوټه, ov::element::f32);
    ov::Output<ov::Node> پایله =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_خپرونه_Type::NUMPY);

    خپلکاری final_shape_غوټه =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    أرجع std::make_shared<ov::op::v1::Reshape>(پایله, final_shape_غوټه, خطأ);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & وزن) {
    خپلکاری weights_غوټه = std::make_shared<ov::op::v0::Constant>(ov::element::u8, وزن.get_shape(),
                                                               static_cast<uint8_t *>(وزن.معلومات()), تش_اشاره);
    weights_غوټه->get_rt_info()["__gguf_tensor_holder"] = وزن;
    weights_غوټه->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = ریښتیا;
    أرجع weights_غوټه;
}

فراغ quantize_q4_0(ثابت عائم * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();
    منطقي is_متناظر = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    إذا (!is_متناظر) {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());
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

    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات());
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>();
    خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    خپلکاری store_zp = [zp](صحيح i, uint8_t v) {
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
                   int64_t بلاک_دننه_واټن = 0) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات()) + بلاک_دننه_واټن * qk;
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>() + بلاک_دننه_واټن;
    منطقي is_متناظر = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    إذا (!is_متناظر) {
        خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات()) + بلاک_دننه_واټن;
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
        خپلکاری * signed_weights = reinterpret_cast<int8_t *>(weights);
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
                   int64_t بلاک_دننه_واټن = 0) {
    أكد(k % qk == 0);
    ثابت صحيح nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    خپلکاری * weights = static_cast<uint8_t *>(weights_arr.معلومات()) + بلاک_دننه_واټن * qk;
    خپلکاری * scales = scales_arr.معلومات<ov::element_type_traits<ov::element::f16>::value_type>() + بلاک_دننه_واټن;
    خپلکاری * zp = static_cast<uint8_t *>(zp_arr.معلومات()) + بلاک_دننه_واټن;
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
std::shared_ptr<ov::Node> ایستل_quantized_weights(ثابت ggml_tensor * tensor,
                                                    ثابت فراغ * معلومات,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    منطقي use_bias = خطأ) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor لنډمهاله_tensor = *tensor;
    لنډمهاله_tensor.معلومات = const_cast<فراغ *>(معلومات);

    إذا (tensor->type == GGML_TYPE_MXFP4) {
        ایستل_mxfp4_معلومات(&لنډمهاله_tensor, weights, scales);
        خپلکاری پایله = make_mxfp4_weights(weights, scales).get_غوټه_shared_ptr();
        پایله->سېټ_friendly_name(tensor->name);
        أرجع پایله;
    }

    // Determine block size based on tensor type
    int64_t weights_فی_بلاک;
    منطقي is_u4;
    بدل (tensor->type) {
    حالة GGML_TYPE_Q4_0:
    حالة GGML_TYPE_Q4_1:
    حالة GGML_TYPE_Q4_K:
        is_u4 = ریښتیا;
        weights_فی_بلاک = 32;
        اكسر;
    حالة GGML_TYPE_Q8_0:
    حالة GGML_TYPE_Q5_1:
    حالة GGML_TYPE_Q5_K:
        is_u4 = خطأ;
        weights_فی_بلاک = 32;
        اكسر;
    حالة GGML_TYPE_Q6_K:
        is_u4 = خطأ;
        weights_فی_بلاک = 16;
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
        ایستل_q4_0_معلومات(&لنډمهاله_tensor, weights, scales, zp);
        اكسر;
    حالة GGML_TYPE_Q4_1:
        ایستل_q4_1_معلومات(&لنډمهاله_tensor, weights, scales, zp, use_bias);
        اكسر;
    حالة GGML_TYPE_Q4_K:
        ایستل_q4_k_معلومات(&لنډمهاله_tensor, weights, scales, zp, use_bias);
        اكسر;
    حالة GGML_TYPE_Q5_1:
        ایستل_q5_1_معلومات(&لنډمهاله_tensor, weights, scales, zp, use_bias);
        اكسر;
    حالة GGML_TYPE_Q8_0:
        ایستل_q8_0_معلومات(&لنډمهاله_tensor, weights, scales, zp);
        اكسر;
    حالة GGML_TYPE_Q6_K:
        ایستل_q6_k_معلومات(&لنډمهاله_tensor, weights, scales, zp);
        اكسر;
    حالة GGML_TYPE_Q5_K:
        ایستل_q5_k_معلومات(&لنډمهاله_tensor, weights, scales, zp, use_bias);
        اكسر;
    افتراضي:
        ارم std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> وزن_غوټه;
    إذا (is_u4) {
        وزن_غوټه = make_int4_weights(weights, scales, zp, weights_فی_بلاک, use_bias, for_gather_matmul);
    } وإلا {
        وزن_غوټه = make_int8_weights(weights, scales, zp, weights_فی_بلاک, use_bias, for_gather_matmul);
    }

    خپلکاری پایله = وزن_غوټه.get_غوټه_shared_ptr();
    پایله->سېټ_friendly_name(tensor->name);
    أرجع پایله;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_منزل_buffers(ثابت ggml_tensor * tensor,
                                                ثابت فراغ * معلومات,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t بلاک_اندازه_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    ثابت int64_t ne0 = tensor->ne[0];                 // elements per row
    ثابت int64_t n_لیکې = n_elements / ne0;
    ثابت خپلکاری * type_traits = ggml_get_type_traits(tensor->type);
    ثابت size_t src_لیکه_بایټونه = ggml_لیکه_اندازه(tensor->type, ne0);

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
    ثابت منطقي بهیر_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(بلاک_اندازه_3 > 0 && ne0 % بلاک_اندازه_3 != 0);

    إذا (!بهیر_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<عائم> weights_f32(n_elements);
        type_traits->منزل_float(معلومات, weights_f32.معلومات(), n_elements);
        إذا (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.معلومات(), weights.معلومات(), n_elements);
            خپلکاری پایله = std::make_shared<ov::op::v0::Constant>(weights);
            پایله->سېټ_friendly_name(tensor->name);
            أرجع پایله;
        }
        إذا (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.معلومات(), weights, scales, zp, n_elements, بلاک_اندازه_3);
        } وإلا إذا (is_u4) {
            quantize_q4_0(weights_f32.معلومات(), weights, scales, zp, n_elements, بلاک_اندازه_3);
        } وإلا إذا (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.معلومات(), weights, scales, zp, n_elements, بلاک_اندازه_3);
        } وإلا {
            quantize_q8_0(weights_f32.معلومات(), weights, scales, zp, n_elements, بلاک_اندازه_3);
        }
    } وإلا {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        ثابت int64_t CHUNK_لیکې = std::min<int64_t>(n_لیکې, 256);
        std::vector<عائم> scratch(CHUNK_لیکې * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        خپلکاری * f16_بنسټ = static_cast<uint8_t *>(weights.معلومات());
        لكل (int64_t r0 = 0; r0 < n_لیکې; r0 += CHUNK_لیکې) {
            ثابت int64_t لیکې = std::min(CHUNK_لیکې, n_لیکې - r0);
            ثابت int64_t elems = لیکې * ne0;
            ثابت خپلکاری * src = static_cast<ثابت uint8_t *>(معلومات) + r0 * src_لیکه_بایټونه;
            type_traits->منزل_float(src, scratch.معلومات(), elems);

            إذا (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.معلومات(), f16_بنسټ + (r0 * ne0) * حجم(uint16_t), elems);
            } وإلا {
                ثابت int64_t بلاک_دننه_واټن = (r0 * ne0) / بلاک_اندازه_3;
                إذا (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.معلومات(), weights, scales, zp, elems, بلاک_اندازه_3, بلاک_دننه_واټن);
                } وإلا {
                    quantize_q8_0(scratch.معلومات(), weights, scales, zp, elems, بلاک_اندازه_3, بلاک_دننه_واټن);
                }
            }
        }
        إذا (requant_type == ExtraQuantType::F16) {
            خپلکاری پایله = std::make_shared<ov::op::v0::Constant>(weights);
            پایله->سېټ_friendly_name(tensor->name);
            أرجع پایله;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> وزن_غوټه;
    إذا (is_u4) {
        وزن_غوټه = make_int4_weights(weights, scales, zp, بلاک_اندازه_3);
    } وإلا {
        وزن_غوټه = make_int8_weights(weights, scales, zp, بلاک_اندازه_3);
    }

    خپلکاری پایله = وزن_غوټه.get_غوټه_shared_ptr();
    پایله->سېټ_friendly_name(tensor->name);
    أرجع پایله;
}
}  // namespace

Ov_وزن process_وزن_tensor(ثابت ggml_tensor * tensor, ثابت فراغ * معلومات, فراغ * output_بنسټ_ptr, منطقي use_bias) {
    GGML_ASSERT(tensor != تش_اشاره);
    GGML_ASSERT(معلومات != تش_اشاره);

    Ov_وزن پایله;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape غوټه_shape = (tensor->ne[2] > 1) ?
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

        إذا (output_بنسټ_ptr && output_بنسټ_ptr != معلومات) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_بایټونه = ggml_nbytes(tensor);
            حافظه_نقل(output_بنسټ_ptr, معلومات, tensor_بایټونه);
            پایله.weights = ov::Tensor(element_type, غوټه_shape, output_بنسټ_ptr);
        } وإلا {
            پایله.weights = ov::Tensor(element_type, غوټه_shape, معلومات);
        }
        پایله.وزن_غوټه = std::make_shared<ov::op::v0::Constant>(پایله.weights);
        أرجع پایله;
    }

    // Handle quantized weights
    إذا (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    پایله.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    ثابت خپلکاری & layout = پایله.layout;
    إذا (layout.ټول_اندازه == 0) {
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
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_بلاک_اندازه),
                                  MXFP4_بلاک_بایټونه};
        ثابت size_t tensor_بایټونه = ggml_nbytes(tensor);
        إذا (output_بنسټ_ptr) {
            خپلکاری * buf_بنسټ = static_cast<uint8_t *>(output_بنسټ_ptr);
            حافظه_نقل(buf_بنسټ + layout.weights_دننه_واټن, معلومات, tensor_بایټونه);
            پایله.weights = ov::Tensor(ov::element::u8, packed_shape, buf_بنسټ + layout.weights_دننه_واټن);
        } وإلا {
            پایله.weights = ov::Tensor(ov::element::u8, packed_shape);
            حافظه_نقل(پایله.weights.معلومات(), معلومات, tensor_بایټونه);
        }
        پایله.وزن_غوټه = make_mxfp4_moe_packed_weights(پایله.weights).get_غوټه_shared_ptr();
        پایله.وزن_غوټه->سېټ_friendly_name(tensor->name);
        أرجع پایله;
    }

    إذا (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_بنسټ_ptr = تش_اشاره;
    }

    // F16 requant path - no separate scales/zp needed in result
    إذا (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        إذا (output_بنسټ_ptr) {
            پایله.weights = ov::Tensor(ov::element::f16, غوټه_shape,
                                        static_cast<uint8_t *>(output_بنسټ_ptr) + layout.weights_دننه_واټن);
        } وإلا {
            پایله.weights = ov::Tensor(ov::element::f16, غوټه_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        پایله.وزن_غوټه =
            requantize_منزل_buffers(tensor, معلومات, ExtraQuantType::F16, 0, پایله.weights, dummy_scales, dummy_zp);
        أرجع پایله;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type وزن_type;
    إذا (tensor->type == GGML_TYPE_MXFP4) {
        وزن_type = ov::element::f4e2m1;
    } وإلا إذا (layout.is_متناظر) {
        وزن_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } وإلا {
        وزن_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = غوټه_shape;
    scale_shape.back() /= layout.weights_فی_بلاک;

    إذا (tensor->type == GGML_TYPE_MXFP4) {
        إذا (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            غوټه_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } وإلا {
            غوټه_shape.clear();
            لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                غوټه_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = غوټه_shape;
        scale_shape.back() /= layout.weights_فی_بلاک;
    }

    ثابت ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    إذا (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    إذا (output_بنسټ_ptr) {
        uint8_t * buf_بنسټ = static_cast<uint8_t *>(output_بنسټ_ptr);
        پایله.weights = ov::Tensor(وزن_type, غوټه_shape, buf_بنسټ + layout.weights_دننه_واټن);
        پایله.scales = ov::Tensor(scale_type, scale_shape, buf_بنسټ + layout.scales_دننه_واټن);
        إذا (!layout.is_متناظر) {
            پایله.zp = ov::Tensor(zp_type, scale_shape, buf_بنسټ + layout.zp_دننه_واټن);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } وإلا {
        پایله.weights = ov::Tensor(وزن_type, غوټه_shape);
        پایله.scales = ov::Tensor(scale_type, scale_shape);
        إذا (!layout.is_متناظر) {
            پایله.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    إذا (layout.is_requant && layout.requant_type.has_value()) {
        پایله.وزن_غوټه = requantize_منزل_buffers(tensor, معلومات, layout.requant_type.value(), layout.weights_فی_بلاک,
                                                   پایله.weights, پایله.scales, پایله.zp);
    } وإلا {
        پایله.وزن_غوټه =
            ایستل_quantized_weights(tensor, معلومات, پایله.weights, پایله.scales, پایله.zp, use_bias);
    }

    أرجع پایله;
}
