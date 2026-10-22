#అన్తర్భావ "ggml-quants.h"

#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-openvino-extra.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <limits>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/except.hpp>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/core/parallel.hpp>
#అన్తర్భావ <openvino/core/shape.hpp>
#అన్తర్భావ <openvino/core/type/element_type.hpp>
#అన్తర్భావ <openvino/core/type/element_type_traits.hpp>
#అన్తర్భావ <openvino/core/type/float16.hpp>
#అన్తర్భావ <openvino/core/type/float4_e2m1.hpp>
#అన్తర్భావ <openvino/core/type/float8_e8m0.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/util/attr_types.hpp>
#అన్తర్భావ <openvino/pass/constant_folding.hpp>
#అన్తర్భావ <openvino/runtime/tensor.hpp>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>

// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
నామాకాశ ov::op::util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
తార్కిక get_single_విలువ(స్థిరరాశి std::shared_ptr<ov::op::v0::Constant> & const_కణుపు,
                      దశాంశం & value,
                      తార్కిక check_విలువ_range = నిజం);
}  // namespace ov::op::util

నామాకాశ {
శూన్య unpack_32_4(స్థిరరాశి uint8_t * దత్తాంశం, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    కొరకు (పూర్ణాంకం j = 0; j < 16; ++j) {
        uint8_t x = (దత్తాంశం[j] & 0x0F);
        uint8_t y = (దత్తాంశం[j] >> 4);
        ఒకవేళ (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

constexpr size_t MXFP4_ఖండం_కొలత = 32;
constexpr size_t MXFP4_ఖండం_QS_కొలత = MXFP4_ఖండం_కొలత / 2;
constexpr size_t MXFP4_ఖండం_బైట్లు = పరిమాణం(uint8_t) + MXFP4_ఖండం_QS_కొలత;

శూన్య pack_32_mxfp4_for_openvino(స్థిరరాశి uint8_t * దత్తాంశం, uint8_t * dst) {
    కొరకు (పూర్ణాంకం j = 0; j < static_cast<పూర్ణాంకం>(MXFP4_ఖండం_QS_కొలత); j += 2) {
        స్థిరరాశి uint8_t v0 = దత్తాంశం[j] & 0x0F;
        స్థిరరాశి uint8_t v1 = (దత్తాంశం[j + 1] & 0x0F) << 4;
        స్థిరరాశి uint8_t v16 = దత్తాంశం[j] >> 4;
        స్థిరరాశి uint8_t v17 = దత్తాంశం[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_ఖండం_కొలత / 4 + j / 2] = v16 | v17;
    }
}

శూన్య వెలికితీయి_mxfp4_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    స్థిరరాశి స్వయంచాలక * దత్తాంశం = static_cast<స్థిరరాశి uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    స్థిరరాశి size_t n_ఖండాలు = scales_arr.get_కొలత();

    ov::parallel_for(n_ఖండాలు, [&](size_t i) {
        స్థిరరాశి uint8_t * ఖండం = దత్తాంశం + i * MXFP4_ఖండం_బైట్లు;
        pack_32_mxfp4_for_openvino(ఖండం + పరిమాణం(uint8_t), weights + i * MXFP4_ఖండం_QS_కొలత);
        scales[i] = ov::float8_e8m0::from_బిట్లు(ఖండం[0]);
    });
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
శూన్య వెలికితీయి_q4_0_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 18;  // 2 bytes scale, 32x0.5 byte weights

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    తార్కిక is_సౌష్ఠవ = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    ఒకవేళ (!is_సౌష్ఠవ) {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            scales[i] = ov::float16::from_బిట్లు(*((uint16_t *) (దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం)));
            // Pack two 4-bit zero points per byte
            ఒకవేళ (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } లేకపోతే {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం + 2, weights + i * 16);
        });
    } లేకపోతే {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            scales[i] = ov::float16::from_బిట్లు(*((uint16_t *) (దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం)));
            unpack_32_4(దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            కొరకు (పూర్ణాంకం j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
శూన్య వెలికితీయి_q4_1_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       తార్కిక use_bias) {
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    ఒకవేళ (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        స్వయంచాలక * bias = zp_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            దశాంశం scale = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) (దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం))));
            దశాంశం min = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) (దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం + 4, weights + i * 16);
        });
    } లేకపోతే {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            దశాంశం scale = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) (దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం))));
            దశాంశం min = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) (దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            ఒకవేళ (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } లేకపోతే {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q5_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: w*d + m.
// Weights are stored as u8 (5-bit values do not fit u4), matching make_int8_weights.
శూన్య వెలికితీయి_q5_1_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       తార్కిక use_bias) {
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    స్థిరరాశి పూర్ణాంకం qk = 32;

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());  // u8 weights, one byte per weight
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    // Read a 16-bit little-endian value without aliasing/const-qual violations.
    స్వయంచాలక read_u16 = [](స్థిరరాశి uint8_t * p) {
        uint16_t v;
        జ్ఞాపకం_నకలు(&v, p, పరిమాణం(v));
        తిరిగి v;
    };

    స్వయంచాలక unpack_ఖండం = [&](స్థిరరాశి uint8_t * ఖండం, uint8_t * dst) {
        uint32_t qh;
        జ్ఞాపకం_నకలు(&qh, ఖండం + 4, పరిమాణం(uint32_t));
        స్థిరరాశి uint8_t * qs = ఖండం + 8;
        కొరకు (పూర్ణాంకం j = 0; j < qk / 2; ++j) {
            స్థిరరాశి uint8_t lo = qs[j] & 0x0F;
            స్థిరరాశి uint8_t hi = qs[j] >> 4;
            స్థిరరాశి uint8_t బిట్టు_lo = (qh >> j) & 1;
            స్థిరరాశి uint8_t బిట్టు_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (బిట్టు_lo << 4);           // first 16 weights
            dst[j + qk / 2] = hi | (బిట్టు_hi << 4);  // last 16 weights
        }
    };

    ఒకవేళ (use_bias) {
        // Store bias (min) directly as f16: dequant w*d + m
        స్వయంచాలక * bias = zp_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            స్థిరరాశి uint8_t * ఖండం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;
            దశాంశం scale = static_cast<దశాంశం>(ov::float16::from_బిట్లు(read_u16(ఖండం)));
            దశాంశం min = static_cast<దశాంశం>(ov::float16::from_బిట్లు(read_u16(ఖండం + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_ఖండం(ఖండం, weights + i * qk);
        });
    } లేకపోతే {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());  // u8 zero points
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            స్థిరరాశి uint8_t * ఖండం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;
            దశాంశం scale = static_cast<దశాంశం>(ov::float16::from_బిట్లు(read_u16(ఖండం)));
            దశాంశం min = static_cast<దశాంశం>(ov::float16::from_బిట్లు(read_u16(ఖండం + 2)));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (dequant: (w - zp) * s == w*s + min)
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_ఖండం(ఖండం, weights + i * qk);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
శూన్య వెలికితీయి_q8_0_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    స్థిరరాశి uint64_t weights_చొప్పున_ఖండం = 32;
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 34;  // 2 bytes scale, 32x1 byte weights

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    తార్కిక is_సౌష్ఠవ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ఒకవేళ (!is_సౌష్ఠవ) {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            uint8_t * ఖండం_దత్తాంశం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;
            scales[i] = ov::float16::from_బిట్లు(*(uint16_t *) ఖండం_దత్తాంశం);
            zp[i] = 128;
            కొరకు (size_t j = 0; j < weights_చొప్పున_ఖండం; ++j) {
                uint8_t x = ఖండం_దత్తాంశం[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_చొప్పున_ఖండం + j] = x;
            }
        });
    } లేకపోతే {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_కొలత(), [&](size_t i) {
            uint8_t * ఖండం_దత్తాంశం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;
            scales[i] = ov::float16::from_బిట్లు(*(uint16_t *) ఖండం_దత్తాంశం);
            // Copy int8 weights as-is (the tensor element type is i8)
            జ్ఞాపకం_నకలు(weights + i * weights_చొప్పున_ఖండం, ఖండం_దత్తాంశం + 2, weights_చొప్పున_ఖండం);
        });
    }
}

శూన్య unpack_256_4(స్థిరరాశి uint8_t * దత్తాంశం, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    కొరకు (size_t i = 0; i < 4; ++i) {
        కొరకు (పూర్ణాంకం j = 0; j < 32; ++j) {
            uint8_t x = (దత్తాంశం[i * 32 + j] & 0x0F);
            uint8_t y = (దత్తాంశం[i * 32 + j] >> 4);
            ఒకవేళ (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

శూన్య వెలికితీయి_q4_k_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       తార్కిక use_bias) {
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 2 + 2 + 12 + 128;
    స్థిరరాశి uint64_t n_super_ఖండం = tensor->nb[3] / బైట్లు_చొప్పున_ఖండం;

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    స్వయంచాలక * zp_u4 = use_bias ? శూన్యసూచిక : static_cast<uint8_t *>(zp_arr.దత్తాంశం());
    స్వయంచాలక * bias_f16 = use_bias ? zp_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>() : శూన్యసూచిక;

    ov::parallel_for(n_super_ఖండం, [&](size_t i) {
        uint8_t * ఖండం_దత్తాంశం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;

        // Extract scale factors and offsets
        దశాంశం scale_scales = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) ఖండం_దత్తాంశం)));
        దశాంశం scale_mins = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) ఖండం_దత్తాంశం + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = ఖండం_దత్తాంశం + 4;

        // Calculate scales
        దశాంశం scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<దశాంశం>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<దశాంశం>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<దశాంశం>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<దశాంశం>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<దశాంశం>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<దశాంశం>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<దశాంశం>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<దశాంశం>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        దశాంశం min_vals[8];
        min_vals[0] = scale_mins * static_cast<దశాంశం>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<దశాంశం>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<దశాంశం>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<దశాంశం>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<దశాంశం>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<దశాంశం>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<దశాంశం>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<దశాంశం>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            ఒకవేళ (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } లేకపోతే {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                ఒకవేళ (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } లేకపోతే {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(ఖండం_దత్తాంశం + 16, weights + i * 128);
    });
}

శూన్య వెలికితీయి_q6_k_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 128 + 64 + 16 + 2;
    స్థిరరాశి uint64_t n_super_ఖండం = tensor->nb[3] / బైట్లు_చొప్పున_ఖండం;

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    తార్కిక is_సౌష్ఠవ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ఒకవేళ (!is_సౌష్ఠవ) {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());
        ov::parallel_for(n_super_ఖండం, [&](size_t i) {
            uint8_t * ఖండం_దత్తాంశం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;
            దశాంశం scale_గుణకం = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) ఖండం_దత్తాంశం + 104)));
            కొరకు (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_గుణకం * static_cast<దశాంశం>(*((int8_t *) (ఖండం_దత్తాంశం + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = ఖండం_దత్తాంశం;
            uint8_t * qh = ఖండం_దత్తాంశం + 128;
            కొరకు (int64_t j = 0; j < 32; ++j) {
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
    } లేకపోతే {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_ఖండం, [&](size_t i) {
            uint8_t * ఖండం_దత్తాంశం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;
            దశాంశం scale_గుణకం = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) ఖండం_దత్తాంశం + 104)));
            కొరకు (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_గుణకం * static_cast<దశాంశం>(*((int8_t *) (ఖండం_దత్తాంశం + 128 + 64 + j))));
            }
            uint8_t * ql = ఖండం_దత్తాంశం;
            uint8_t * qh = ఖండం_దత్తాంశం + 128;
            స్వయంచాలక * signed_weights = reinterpret_cast<int8_t *>(weights);
            కొరకు (int64_t j = 0; j < 32; ++j) {
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

అంతఃవరుస శూన్య get_scale_min_k4(పూర్ణాంకం j, స్థిరరాశి uint8_t * q, uint8_t * d, uint8_t * m) {
    ఒకవేళ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } లేకపోతే {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

శూన్య వెలికితీయి_q5_k_దత్తాంశం(స్థిరరాశి ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       తార్కిక use_bias) {
    స్థిరరాశి uint64_t బైట్లు_చొప్పున_ఖండం = 4 + 12 + 32 + 128;
    స్థిరరాశి uint64_t n_super_ఖండం = tensor->nb[3] / బైట్లు_చొప్పున_ఖండం;

    స్వయంచాలక * దత్తాంశం = static_cast<uint8_t *>(tensor->దత్తాంశం);
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    స్వయంచాలక * zp_u8 = use_bias ? శూన్యసూచిక : static_cast<uint8_t *>(zp_arr.దత్తాంశం());
    స్వయంచాలక * bias_f16 = use_bias ? zp_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>() : శూన్యసూచిక;

    ov::parallel_for(n_super_ఖండం, [&](size_t i) {
        uint8_t * ఖండం_దత్తాంశం = దత్తాంశం + i * బైట్లు_చొప్పున_ఖండం;

        స్థిరరాశి దశాంశం d = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) ఖండం_దత్తాంశం)));
        స్థిరరాశి దశాంశం min_గుణకం = static_cast<దశాంశం>(ov::float16::from_బిట్లు(*((uint16_t *) ఖండం_దత్తాంశం + 1)));

        స్థిరరాశి uint8_t * scales_దత్తాంశం = ఖండం_దత్తాంశం + 4;   // 12 bytes of scales
        స్థిరరాశి uint8_t * qh = ఖండం_దత్తాంశం + 4 + 12;       // 32 bytes of high bits
        స్థిరరాశి uint8_t * ql = ఖండం_దత్తాంశం + 4 + 12 + 32;  // 128 bytes of low bits

        పూర్ణాంకం is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        కొరకు (పూర్ణాంకం j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_దత్తాంశం, &sc, &m);
            స్థిరరాశి దశాంశం d1 = d * sc;
            స్థిరరాశి దశాంశం m1 = min_గుణకం * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_దత్తాంశం, &sc, &m);
            స్థిరరాశి దశాంశం d2 = d * sc;
            స్థిరరాశి దశాంశం m2 = min_గుణకం * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            ఒకవేళ (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } లేకపోతే {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
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
ov::Output<ov::Node> make_int8_weights(ov::Tensor & బరువు,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t సమూహం_కొలత = GGML_QUANTIZATION_సమూహం_కొలత,
                                       తార్కిక use_bias = అబద్ధం,
                                       తార్కిక for_gather_matmul = అబద్ధం) {
    ov::Shape orig_shape = బరువు.get_shape();
    తార్కిక is_signed = (బరువు.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    స్వయంచాలక scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= సమూహం_కొలత;
    packed_shape.push_back(సమూహం_కొలత);
    స్థిరరాశి size_t సమూహం_dim = packed_shape.size() - 2;

    ఒకవేళ (packed_shape[సమూహం_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + సమూహం_dim);
    } లేకపోతే {
        scale_shape.push_back(1);
        scales.సమితి_shape(scale_shape);
        ఒకవేళ (!is_signed && zp.get_కొలత() > 0) {
            స్వయంచాలక zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.సమితి_shape(zp_shape);
        }
    }

    స్వయంచాలక scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ఫలితం;
    ఒకవేళ (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        స్వయంచాలక weights_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(బరువు.దత్తాంశం()), శూన్యసూచిక);
        weights_కణుపు->get_rt_info()["__gguf_tensor_holder"] = బరువు;
        స్వయంచాలక weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_కణుపు, ov::element::f16);
        స్వయంచాలక mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
        ఫలితం = mul;
    } లేకపోతే {
        // Unsigned path
        స్వయంచాలక weights_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(బరువు.దత్తాంశం()), శూన్యసూచిక);
        weights_కణుపు->get_rt_info()["__gguf_tensor_holder"] = బరువు;
        స్వయంచాలక weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_కణుపు, ov::element::f16);

        ఒకవేళ (use_bias && zp.get_కొలత() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            స్వయంచాలక * bias_zp_దత్తాంశం = zp.దత్తాంశం<ov::float16>();
            స్థిరరాశి స్వయంచాలక * scale_దత్తాంశం = scales.దత్తాంశం<ov::float16>();
            స్థిరరాశి size_t n = zp.get_కొలత();
            కొరకు (size_t i = 0; i < n; i++) {
                దశాంశం s = static_cast<దశాంశం>(scale_దత్తాంశం[i]);
                దశాంశం b = static_cast<దశాంశం>(bias_zp_దత్తాంశం[i]);
                bias_zp_దత్తాంశం[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            స్వయంచాలక zero_బిందువు_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            స్వయంచాలక w_zp =
                std::make_shared<ov::op::v1::వ్యవకలనం>(weights_f16, zero_బిందువు_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
            ఫలితం = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
        } లేకపోతే {
            // Zero point path: (w - zp) * s
            స్వయంచాలక zero_బిందువు = std::make_shared<ov::op::v0::Constant>(zp);
            దశాంశం zp_విలువ;
            ఒకవేళ (ov::op::util::get_single_విలువ(zero_బిందువు, zp_విలువ)) {
                zero_బిందువు = ov::op::v0::Constant::create(zero_బిందువు->get_element_type(), {}, {zp_విలువ});
            }
            స్వయంచాలక zero_బిందువు_f16 = std::make_shared<ov::op::v0::Convert>(zero_బిందువు, ov::element::f16);
            స్వయంచాలక w_zp =
                std::make_shared<ov::op::v1::వ్యవకలనం>(weights_f16, zero_బిందువు_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
            స్వయంచాలక mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
            ఫలితం = mul;
        }
    }

    ఒకవేళ (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        స్వయంచాలక final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(ఫలితం, final_shape, అబద్ధం);
        ఫలితం = reshaped;
    }

    ఒకవేళ (for_gather_matmul) {
        తిరిగి ఫలితం;
    }
    తిరిగి std::make_shared<ov::op::v0::Convert>(ఫలితం, ov::element::f32);
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
ov::Output<ov::Node> make_int4_weights(ov::Tensor & బరువు,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t సమూహం_కొలత = GGML_QUANTIZATION_సమూహం_కొలత,
                                       తార్కిక use_bias = అబద్ధం,
                                       తార్కిక for_gather_matmul = అబద్ధం) {
    ov::Shape orig_బరువు_shape = బరువు.get_shape();
    తార్కిక is_signed = (బరువు.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor. Group the innermost (last) dimension: for 2D weights
    // [rows, cols] this yields [rows, cols/group_size, group_size]; for 3D MoE experts
    // [n_expert, rows, cols] this yields [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_బరువు_shape;
    packed_shape.back() /= సమూహం_కొలత;
    packed_shape.push_back(సమూహం_కొలత);
    స్థిరరాశి size_t సమూహం_dim = packed_shape.size() - 2;

    ఒకవేళ (packed_shape[సమూహం_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + సమూహం_dim);
    } లేకపోతే {
        scale_shape.push_back(1);
        scales.సమితి_shape(scale_shape);
        ఒకవేళ (!is_signed && zp.get_కొలత() > 0) {
            స్వయంచాలక zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.సమితి_shape(zp_shape);
        }
    }

    స్వయంచాలక scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> ఫలితం;
    ఒకవేళ (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        స్వయంచాలక weights_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(బరువు.దత్తాంశం()), శూన్యసూచిక);
        weights_కణుపు->get_rt_info()["__gguf_tensor_holder"] = బరువు;
        స్వయంచాలక weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_కణుపు, ov::element::f16);
        స్వయంచాలక mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
        ఫలితం = mul;
    } లేకపోతే {
        // Unsigned path
        స్వయంచాలక weights_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(బరువు.దత్తాంశం()), శూన్యసూచిక);
        weights_కణుపు->get_rt_info()["__gguf_tensor_holder"] = బరువు;
        స్వయంచాలక weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_కణుపు, ov::element::f16);

        ఒకవేళ (use_bias && zp.get_కొలత() > 0) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s with an exact f16
            // zp = -bias/scale. Equivalent to w*s + bias but matches CompressedWeightsBlock's
            // pattern so for_gather_matmul weights still fuse into GatherMatmulCompressed, and
            // avoids the round(min/scale) error of an integer zp. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            స్వయంచాలక * bias_zp_దత్తాంశం = zp.దత్తాంశం<ov::float16>();
            స్థిరరాశి స్వయంచాలక * scale_దత్తాంశం = scales.దత్తాంశం<ov::float16>();
            స్థిరరాశి size_t n = zp.get_కొలత();
            కొరకు (size_t i = 0; i < n; i++) {
                దశాంశం s = static_cast<దశాంశం>(scale_దత్తాంశం[i]);
                దశాంశం b = static_cast<దశాంశం>(bias_zp_దత్తాంశం[i]);
                bias_zp_దత్తాంశం[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            స్వయంచాలక zero_points_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            స్వయంచాలక w_zp =
                std::make_shared<ov::op::v1::వ్యవకలనం>(weights_f16, zero_points_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
            ఫలితం = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
        } లేకపోతే {
            // Zero point path: (w - zp) * s
            స్వయంచాలక zero_points_కణుపు = std::make_shared<ov::op::v0::Constant>(zp);
            దశాంశం zp_విలువ;
            ఒకవేళ (ov::op::util::get_single_విలువ(zero_points_కణుపు, zp_విలువ)) {
                zero_points_కణుపు = ov::op::v0::Constant::create(zero_points_కణుపు->get_element_type(), {}, {zp_విలువ});
            }
            స్వయంచాలక zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_కణుపు, ov::element::f16);
            స్వయంచాలక w_zp =
                std::make_shared<ov::op::v1::వ్యవకలనం>(weights_f16, zero_points_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
            స్వయంచాలక mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::Auto_ప్రసారం_Type::NUMPY);
            ఫలితం = mul;
        }
    }

    ఒకవేళ (packed_shape.size() != orig_బరువు_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        స్వయంచాలక final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_బరువు_shape.size()},
                                                                  orig_బరువు_shape);
        స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(ఫలితం, final_shape, అబద్ధం);
        ఫలితం = reshaped;
    }

    ఒకవేళ (for_gather_matmul) {
        తిరిగి ఫలితం;
    }
    తిరిగి std::make_shared<ov::op::v0::Convert>(ఫలితం, ov::element::f32);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & బరువు, ov::Tensor & scales) {
    స్థిరరాశి ov::Shape final_shape = బరువు.get_shape();
    GGML_ASSERT(!final_shape.ఖాళీలు());
    GGML_ASSERT(final_shape.back() % MXFP4_ఖండం_కొలత == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_ఖండం_కొలత;
    packed_shape.push_back(MXFP4_ఖండం_కొలత);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.సమితి_shape(scale_shape);

    స్వయంచాలక weights_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(బరువు.దత్తాంశం()), శూన్యసూచిక);
    weights_కణుపు->get_rt_info()["__gguf_tensor_holder"] = బరువు;
    స్వయంచాలక weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_కణుపు, ov::element::f32);

    స్వయంచాలక scales_కణుపు = std::make_shared<ov::op::v0::Constant>(scales);
    స్వయంచాలక scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_కణుపు, ov::element::f32);
    ov::Output<ov::Node> ఫలితం =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::Auto_ప్రసారం_Type::NUMPY);

    స్వయంచాలక final_shape_కణుపు =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    తిరిగి std::make_shared<ov::op::v1::Reshape>(ఫలితం, final_shape_కణుపు, అబద్ధం);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & బరువు) {
    స్వయంచాలక weights_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::u8, బరువు.get_shape(),
                                                               static_cast<uint8_t *>(బరువు.దత్తాంశం()), శూన్యసూచిక);
    weights_కణుపు->get_rt_info()["__gguf_tensor_holder"] = బరువు;
    weights_కణుపు->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = నిజం;
    తిరిగి weights_కణుపు;
}

శూన్య quantize_q4_0(స్థిరరాశి దశాంశం * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    ప్రతిజ్ఞా(k % qk == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / qk;

    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();
    తార్కిక is_సౌష్ఠవ = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    ఒకవేళ (!is_సౌష్ఠవ) {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());
        కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
            దశాంశం amax = 0.0f;
            దశాంశం max = 0.0f;
            కొరకు (పూర్ణాంకం j = 0; j < qk; j++) {
                స్థిరరాశి దశాంశం v = x[i * qk + j];
                ఒకవేళ (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            స్థిరరాశి దశాంశం d = max / -8;
            ఒకవేళ (d == 0) {
                scales[i] = ov::float16(1.0f);
                ఒకవేళ (i % 2 == 0) {
                    zp[i / 2] = 8;
                } లేకపోతే {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                కొనసాగు;
            }
            స్థిరరాశి దశాంశం id = 1.0f / d;
            scales[i] = ov::float16(d);
            ఒకవేళ (i % 2 == 0) {
                zp[i / 2] = 8;
            } లేకపోతే {
                zp[i / 2] |= (8 << 4);
            }
            కొరకు (పూర్ణాంకం j = 0; j < qk / 2; ++j) {
                స్థిరరాశి దశాంశం x0 = x[i * qk + 2 * j] * id;
                స్థిరరాశి దశాంశం x1 = x[i * qk + 2 * j + 1] * id;
                స్థిరరాశి uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                స్థిరరాశి uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } లేకపోతే {
        // Symmetric: produce signed i4 values in [-8, 7]
        కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
            దశాంశం amax = 0.0f;
            దశాంశం max = 0.0f;
            కొరకు (పూర్ణాంకం j = 0; j < qk; j++) {
                స్థిరరాశి దశాంశం v = x[i * qk + j];
                ఒకవేళ (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            స్థిరరాశి దశాంశం d = max / -8;
            ఒకవేళ (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                కొనసాగు;
            }
            స్థిరరాశి దశాంశం id = 1.0f / d;
            scales[i] = ov::float16(d);
            కొరకు (పూర్ణాంకం j = 0; j < qk / 2; ++j) {
                స్థిరరాశి దశాంశం x0 = x[i * qk + 2 * j] * id;
                స్థిరరాశి దశాంశం x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (పూర్ణాంకం) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (పూర్ణాంకం) roundf(x1)));
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
శూన్య quantize_q4_1_asym(స్థిరరాశి దశాంశం * x,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
                        ov::Tensor & zp_arr,
                        int64_t k,
                        int64_t qk) {
    ప్రతిజ్ఞా(k % qk == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / qk;

    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం());
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>();
    స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం());

    // u4 zero points are packed two per byte, low nibble first, indexed by group -- the same
    // convention as the unsigned branch of quantize_q4_0.
    స్వయంచాలక store_zp = [zp](పూర్ణాంకం i, uint8_t v) {
        ఒకవేళ (i % 2 == 0) {
            zp[i / 2] = v & 0x0F;
        } లేకపోతే {
            zp[i / 2] |= (uint8_t) ((v & 0x0F) << 4);
        }
    };

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        దశాంశం vmin = x[i * qk];
        దశాంశం vmax = x[i * qk];
        కొరకు (పూర్ణాంకం j = 1; j < qk; j++) {
            స్థిరరాశి దశాంశం v = x[i * qk + j];
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        // Include 0 in the range so an all-positive or all-negative group still represents zero
        // exactly -- these are weights, so an exact zero matters.
        vmin = std::min(vmin, 0.0f);
        vmax = std::max(vmax, 0.0f);

        స్థిరరాశి దశాంశం d = (vmax - vmin) / 15.0f;
        ఒకవేళ (d == 0.0f) {
            scales[i] = ov::float16(1.0f);
            store_zp(i, 0);
            memset(weights + i * qk / 2, 0, qk / 2);
            కొనసాగు;
        }
        స్థిరరాశి దశాంశం id = 1.0f / d;

        // The zero point is itself a 4-bit integer, so round it and dequantize as (q - zq) * d.
        స్థిరరాశి పూర్ణాంకం zq = std::max(0, std::min(15, (పూర్ణాంకం) lroundf(-vmin * id)));
        scales[i] = ov::float16(d);
        store_zp(i, (uint8_t) zq);

        కొరకు (పూర్ణాంకం j = 0; j < qk / 2; ++j) {
            స్థిరరాశి దశాంశం x0 = x[i * qk + 2 * j] * id;
            స్థిరరాశి దశాంశం x1 = x[i * qk + 2 * j + 1] * id;
            స్థిరరాశి uint8_t q0 = (uint8_t) std::max(0, std::min(15, (పూర్ణాంకం) lroundf(x0) + zq));
            స్థిరరాశి uint8_t q1 = (uint8_t) std::max(0, std::min(15, (పూర్ణాంకం) lroundf(x1) + zq));
            weights[i * qk / 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

శూన్య quantize_q8_0(స్థిరరాశి దశాంశం * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t ఖండం_స్థానభ్రంశం = 0) {
    ప్రతిజ్ఞా(k % qk == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం()) + ఖండం_స్థానభ్రంశం * qk;
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>() + ఖండం_స్థానభ్రంశం;
    తార్కిక is_సౌష్ఠవ = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    ఒకవేళ (!is_సౌష్ఠవ) {
        స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం()) + ఖండం_స్థానభ్రంశం;
        కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
            దశాంశం amax = 0.0f;
            కొరకు (పూర్ణాంకం j = 0; j < qk; j++) {
                స్థిరరాశి దశాంశం v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            స్థిరరాశి దశాంశం d = amax / 127.0f;
            స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            కొరకు (పూర్ణాంకం j = 0; j < qk; ++j) {
                స్థిరరాశి దశాంశం x0 = x[i * qk + j] * id;
                స్థిరరాశి int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } లేకపోతే {
        // Symmetric: store signed int8 values directly
        స్వయంచాలక * signed_weights = reinterpret_cast<int8_t *>(weights);
        కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
            దశాంశం amax = 0.0f;
            కొరకు (పూర్ణాంకం j = 0; j < qk; j++) {
                స్థిరరాశి దశాంశం v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            స్థిరరాశి దశాంశం d = amax / 127.0f;
            స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            కొరకు (పూర్ణాంకం j = 0; j < qk; ++j) {
                స్థిరరాశి దశాంశం x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

శూన్య quantize_q8_1(స్థిరరాశి దశాంశం * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t ఖండం_స్థానభ్రంశం = 0) {
    ప్రతిజ్ఞా(k % qk == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    స్వయంచాలక * weights = static_cast<uint8_t *>(weights_arr.దత్తాంశం()) + ఖండం_స్థానభ్రంశం * qk;
    స్వయంచాలక * scales = scales_arr.దత్తాంశం<ov::element_type_traits<ov::element::f16>::value_type>() + ఖండం_స్థానభ్రంశం;
    స్వయంచాలక * zp = static_cast<uint8_t *>(zp_arr.దత్తాంశం()) + ఖండం_స్థానభ్రంశం;
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        దశాంశం min = std::numeric_limits<దశాంశం>::max();
        దశాంశం max = std::numeric_limits<దశాంశం>::lowest();

        కొరకు (పూర్ణాంకం j = 0; j < qk; j++) {
            స్థిరరాశి దశాంశం v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        స్థిరరాశి దశాంశం d = (max - min) / ((1 << 8) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        కొరకు (పూర్ణాంకం j = 0; j < qk; ++j) {
            స్థిరరాశి దశాంశం x0 = (x[i * qk + j] - min) * id;
            స్థిరరాశి uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> వెలికితీయి_quantized_weights(స్థిరరాశి ggml_tensor * tensor,
                                                    స్థిరరాశి శూన్య * దత్తాంశం,  // Source data pointer (may differ from tensor->data)
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    // Use an exact f16 zero point (vs. a rounded integer one); always
                                                    // used for for_gather_matmul (3D MoE expert) weights regardless of
                                                    // this flag, and also settable explicitly for test-backend-ops.
                                                    తార్కిక use_bias = అబద్ధం) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor తాత్కాలిక_tensor = *tensor;
    తాత్కాలిక_tensor.దత్తాంశం = const_cast<శూన్య *>(దత్తాంశం);

    ఒకవేళ (tensor->type == GGML_TYPE_MXFP4) {
        వెలికితీయి_mxfp4_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales);
        స్వయంచాలక ఫలితం = make_mxfp4_weights(weights, scales).get_కణుపు_shared_ptr();
        ఫలితం->సమితి_friendly_name(tensor->name);
        తిరిగి ఫలితం;
    }

    // Determine block size based on tensor type
    int64_t weights_చొప్పున_ఖండం;
    తార్కిక is_u4;
    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_Q4_0:
    స్థితి GGML_TYPE_Q4_1:
    స్థితి GGML_TYPE_Q4_K:
        is_u4 = నిజం;
        weights_చొప్పున_ఖండం = 32;
        ఆపు;
    స్థితి GGML_TYPE_Q8_0:
    స్థితి GGML_TYPE_Q5_1:
    స్థితి GGML_TYPE_Q5_K:
        is_u4 = అబద్ధం;
        weights_చొప్పున_ఖండం = 32;
        ఆపు;
    స్థితి GGML_TYPE_Q6_K:
        is_u4 = అబద్ధం;
        weights_చొప్పున_ఖండం = 16;
        ఆపు;
    అప్రమేయ:
        విసురు std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    స్థిరరాశి తార్కిక for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_Q4_0:
        వెలికితీయి_q4_0_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp);
        ఆపు;
    స్థితి GGML_TYPE_Q4_1:
        వెలికితీయి_q4_1_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp, use_bias);
        ఆపు;
    స్థితి GGML_TYPE_Q4_K:
        వెలికితీయి_q4_k_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp, use_bias);
        ఆపు;
    స్థితి GGML_TYPE_Q5_1:
        వెలికితీయి_q5_1_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp, use_bias);
        ఆపు;
    స్థితి GGML_TYPE_Q8_0:
        వెలికితీయి_q8_0_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp);
        ఆపు;
    స్థితి GGML_TYPE_Q6_K:
        వెలికితీయి_q6_k_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp);
        ఆపు;
    స్థితి GGML_TYPE_Q5_K:
        వెలికితీయి_q5_k_దత్తాంశం(&తాత్కాలిక_tensor, weights, scales, zp, use_bias);
        ఆపు;
    అప్రమేయ:
        విసురు std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> బరువు_కణుపు;
    ఒకవేళ (is_u4) {
        బరువు_కణుపు = make_int4_weights(weights, scales, zp, weights_చొప్పున_ఖండం, use_bias, for_gather_matmul);
    } లేకపోతే {
        బరువు_కణుపు = make_int8_weights(weights, scales, zp, weights_చొప్పున_ఖండం, use_bias, for_gather_matmul);
    }

    స్వయంచాలక ఫలితం = బరువు_కణుపు.get_కణుపు_shared_ptr();
    ఫలితం->సమితి_friendly_name(tensor->name);
    తిరిగి ఫలితం;
}

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_గమ్యం_buffers(స్థిరరాశి ggml_tensor * tensor,
                                                స్థిరరాశి శూన్య * దత్తాంశం,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t ఖండం_కొలత_3,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    స్థిరరాశి int64_t ne0 = tensor->ne[0];                 // elements per row
    స్థిరరాశి int64_t n_అడ్డువరుసలు = n_elements / ne0;
    స్థిరరాశి స్వయంచాలక * type_traits = ggml_get_type_traits(tensor->type);
    స్థిరరాశి size_t src_అడ్డువరుస_బైట్లు = ggml_అడ్డువరుస_కొలత(tensor->type, ne0);

    తార్కిక is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128 ||
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
    స్థిరరాశి తార్కిక ప్రవాహం_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(ఖండం_కొలత_3 > 0 && ne0 % ఖండం_కొలత_3 != 0);

    ఒకవేళ (!ప్రవాహం_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<దశాంశం> weights_f32(n_elements);
        type_traits->గమ్యం_float(దత్తాంశం, weights_f32.దత్తాంశం(), n_elements);
        ఒకవేళ (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.దత్తాంశం(), weights.దత్తాంశం(), n_elements);
            స్వయంచాలక ఫలితం = std::make_shared<ov::op::v0::Constant>(weights);
            ఫలితం->సమితి_friendly_name(tensor->name);
            తిరిగి ఫలితం;
        }
        ఒకవేళ (requant_type == ExtraQuantType::Q4_1_64) {
            quantize_q4_1_asym(weights_f32.దత్తాంశం(), weights, scales, zp, n_elements, ఖండం_కొలత_3);
        } లేకపోతే ఒకవేళ (is_u4) {
            quantize_q4_0(weights_f32.దత్తాంశం(), weights, scales, zp, n_elements, ఖండం_కొలత_3);
        } లేకపోతే ఒకవేళ (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.దత్తాంశం(), weights, scales, zp, n_elements, ఖండం_కొలత_3);
        } లేకపోతే {
            quantize_q8_0(weights_f32.దత్తాంశం(), weights, scales, zp, n_elements, ఖండం_కొలత_3);
        }
    } లేకపోతే {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        స్థిరరాశి int64_t CHUNK_అడ్డువరుసలు = std::min<int64_t>(n_అడ్డువరుసలు, 256);
        std::vector<దశాంశం> scratch(CHUNK_అడ్డువరుసలు * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        స్వయంచాలక * f16_పునాది = static_cast<uint8_t *>(weights.దత్తాంశం());
        కొరకు (int64_t r0 = 0; r0 < n_అడ్డువరుసలు; r0 += CHUNK_అడ్డువరుసలు) {
            స్థిరరాశి int64_t అడ్డువరుసలు = std::min(CHUNK_అడ్డువరుసలు, n_అడ్డువరుసలు - r0);
            స్థిరరాశి int64_t elems = అడ్డువరుసలు * ne0;
            స్థిరరాశి స్వయంచాలక * src = static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం) + r0 * src_అడ్డువరుస_బైట్లు;
            type_traits->గమ్యం_float(src, scratch.దత్తాంశం(), elems);

            ఒకవేళ (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.దత్తాంశం(), f16_పునాది + (r0 * ne0) * పరిమాణం(uint16_t), elems);
            } లేకపోతే {
                స్థిరరాశి int64_t ఖండం_స్థానభ్రంశం = (r0 * ne0) / ఖండం_కొలత_3;
                ఒకవేళ (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.దత్తాంశం(), weights, scales, zp, elems, ఖండం_కొలత_3, ఖండం_స్థానభ్రంశం);
                } లేకపోతే {
                    quantize_q8_0(scratch.దత్తాంశం(), weights, scales, zp, elems, ఖండం_కొలత_3, ఖండం_స్థానభ్రంశం);
                }
            }
        }
        ఒకవేళ (requant_type == ExtraQuantType::F16) {
            స్వయంచాలక ఫలితం = std::make_shared<ov::op::v0::Constant>(weights);
            ఫలితం->సమితి_friendly_name(tensor->name);
            తిరిగి ఫలితం;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> బరువు_కణుపు;
    ఒకవేళ (is_u4) {
        బరువు_కణుపు = make_int4_weights(weights, scales, zp, ఖండం_కొలత_3);
    } లేకపోతే {
        బరువు_కణుపు = make_int8_weights(weights, scales, zp, ఖండం_కొలత_3);
    }

    స్వయంచాలక ఫలితం = బరువు_కణుపు.get_కణుపు_shared_ptr();
    ఫలితం->సమితి_friendly_name(tensor->name);
    తిరిగి ఫలితం;
}
}  // namespace

Ov_బరువు process_బరువు_tensor(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, శూన్య * output_పునాది_ptr, తార్కిక use_bias) {
    GGML_ASSERT(tensor != శూన్యసూచిక);
    GGML_ASSERT(దత్తాంశం != శూన్యసూచిక);

    Ov_బరువు ఫలితం;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape కణుపు_shape = (tensor->ne[2] > 1) ?
                               ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0])} :
                               ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    ఒకవేళ (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_F32:
            element_type = ov::element::f32;
            ఆపు;
        స్థితి GGML_TYPE_F16:
            element_type = ov::element::f16;
            ఆపు;
        స్థితి GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            ఆపు;
        అప్రమేయ:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        ఒకవేళ (output_పునాది_ptr && output_పునాది_ptr != దత్తాంశం) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_బైట్లు = ggml_nbytes(tensor);
            జ్ఞాపకం_నకలు(output_పునాది_ptr, దత్తాంశం, tensor_బైట్లు);
            ఫలితం.weights = ov::Tensor(element_type, కణుపు_shape, output_పునాది_ptr);
        } లేకపోతే {
            ఫలితం.weights = ov::Tensor(element_type, కణుపు_shape, దత్తాంశం);
        }
        ఫలితం.బరువు_కణుపు = std::make_shared<ov::op::v0::Constant>(ఫలితం.weights);
        తిరిగి ఫలితం;
    }

    // Handle quantized weights
    ఒకవేళ (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    ఫలితం.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    స్థిరరాశి స్వయంచాలక & layout = ఫలితం.layout;
    ఒకవేళ (layout.మొత్తం_కొలత == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    స్థిరరాశి తార్కిక for_gather_matmul = tensor->ne[2] > 1;
    స్థిరరాశి తార్కిక zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    స్థిరరాశి తార్కిక is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    ఒకవేళ (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = {static_cast<size_t>(tensor->ne[3]),
                                  static_cast<size_t>(tensor->ne[2]),
                                  static_cast<size_t>(tensor->ne[1]),
                                  static_cast<size_t>(tensor->ne[0] / MXFP4_ఖండం_కొలత),
                                  MXFP4_ఖండం_బైట్లు};
        స్థిరరాశి size_t tensor_బైట్లు = ggml_nbytes(tensor);
        ఒకవేళ (output_పునాది_ptr) {
            స్వయంచాలక * buf_పునాది = static_cast<uint8_t *>(output_పునాది_ptr);
            జ్ఞాపకం_నకలు(buf_పునాది + layout.weights_స్థానభ్రంశం, దత్తాంశం, tensor_బైట్లు);
            ఫలితం.weights = ov::Tensor(ov::element::u8, packed_shape, buf_పునాది + layout.weights_స్థానభ్రంశం);
        } లేకపోతే {
            ఫలితం.weights = ov::Tensor(ov::element::u8, packed_shape);
            జ్ఞాపకం_నకలు(ఫలితం.weights.దత్తాంశం(), దత్తాంశం, tensor_బైట్లు);
        }
        ఫలితం.బరువు_కణుపు = make_mxfp4_moe_packed_weights(ఫలితం.weights).get_కణుపు_shared_ptr();
        ఫలితం.బరువు_కణుపు->సమితి_friendly_name(tensor->name);
        తిరిగి ఫలితం;
    }

    ఒకవేళ (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_పునాది_ptr = శూన్యసూచిక;
    }

    // F16 requant path - no separate scales/zp needed in result
    ఒకవేళ (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        ఒకవేళ (output_పునాది_ptr) {
            ఫలితం.weights = ov::Tensor(ov::element::f16, కణుపు_shape,
                                        static_cast<uint8_t *>(output_పునాది_ptr) + layout.weights_స్థానభ్రంశం);
        } లేకపోతే {
            ఫలితం.weights = ov::Tensor(ov::element::f16, కణుపు_shape);
        }
        // Not used for F16:
        ov::Tensor dummy_scales;
        ov::Tensor dummy_zp;
        ఫలితం.బరువు_కణుపు =
            requantize_గమ్యం_buffers(tensor, దత్తాంశం, ExtraQuantType::F16, 0, ఫలితం.weights, dummy_scales, dummy_zp);
        తిరిగి ఫలితం;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type బరువు_type;
    ఒకవేళ (tensor->type == GGML_TYPE_MXFP4) {
        బరువు_type = ov::element::f4e2m1;
    } లేకపోతే ఒకవేళ (layout.is_సౌష్ఠవ) {
        బరువు_type = layout.is_u4 ? ov::element::i4 : ov::element::i8;
    } లేకపోతే {
        బరువు_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    }
    ov::Shape scale_shape = కణుపు_shape;
    scale_shape.back() /= layout.weights_చొప్పున_ఖండం;

    ఒకవేళ (tensor->type == GGML_TYPE_MXFP4) {
        ఒకవేళ (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
            కణుపు_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
        } లేకపోతే {
            కణుపు_shape.clear();
            కొరకు (పూర్ణాంకం i = GGML_MAX_DIMS - 1; i >= 0; --i) {
                కణుపు_shape.push_back(static_cast<size_t>(tensor->ne[i]));
            }
        }

        scale_shape = కణుపు_shape;
        scale_shape.back() /= layout.weights_చొప్పున_ఖండం;
    }

    స్థిరరాశి ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
    ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    ఒకవేళ (zp_is_f16) {
        zp_type = ov::element::f16;
    }

    ఒకవేళ (output_పునాది_ptr) {
        uint8_t * buf_పునాది = static_cast<uint8_t *>(output_పునాది_ptr);
        ఫలితం.weights = ov::Tensor(బరువు_type, కణుపు_shape, buf_పునాది + layout.weights_స్థానభ్రంశం);
        ఫలితం.scales = ov::Tensor(scale_type, scale_shape, buf_పునాది + layout.scales_స్థానభ్రంశం);
        ఒకవేళ (!layout.is_సౌష్ఠవ) {
            ఫలితం.zp = ov::Tensor(zp_type, scale_shape, buf_పునాది + layout.zp_స్థానభ్రంశం);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } లేకపోతే {
        ఫలితం.weights = ov::Tensor(బరువు_type, కణుపు_shape);
        ఫలితం.scales = ov::Tensor(scale_type, scale_shape);
        ఒకవేళ (!layout.is_సౌష్ఠవ) {
            ఫలితం.zp = ov::Tensor(zp_type, scale_shape);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    ఒకవేళ (layout.is_requant && layout.requant_type.has_value()) {
        ఫలితం.బరువు_కణుపు = requantize_గమ్యం_buffers(tensor, దత్తాంశం, layout.requant_type.value(), layout.weights_చొప్పున_ఖండం,
                                                   ఫలితం.weights, ఫలితం.scales, ఫలితం.zp);
    } లేకపోతే {
        ఫలితం.బరువు_కణుపు =
            వెలికితీయి_quantized_weights(tensor, దత్తాంశం, ఫలితం.weights, ఫలితం.scales, ఫలితం.zp, use_bias);
    }

    తిరిగి ఫలితం;
}
