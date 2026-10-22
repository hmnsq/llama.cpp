#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/interpolate.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_upscale(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    स्वतः आगत = process_view_आगत_new(context, 0);

    अचर स्वतः आगत_shape = context.get_आगत_shape(0).to_shape();
    अचर स्वतः output_shape = context.get_output_shape().to_shape();

    यदि (आगत_shape == output_shape) {
        फर्काऊ rename_outputs_with_suffix({आगत}, context.get_name());
    }

    ov::Output<ov::Node> res = आगत;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    लागि (माप_प्रकार axis = 0; axis < 2; ++axis) {
        अचर माप_प्रकार in_dim = आगत_shape[axis];
        अचर माप_प्रकार out_dim = output_shape[axis];
        यदि (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            लागि (माप_प्रकार i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            स्वतः indices_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            स्वतः axis_गाँठो = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_गाँठो, axis_गाँठो);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    यदि (आगत_shape[2] != output_shape[2] || आगत_shape[3] != output_shape[3]) {
        अचर int32_t * op_params = context.get_output_op_params();
        अचर int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        अचर पूर्णाङ्क op_case = context.get_op_case();
        अचर तार्किक align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        अचर तार्किक antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        यदि (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            अचर माप_प्रकार H_in = आगत_shape[2];
            अचर माप_प्रकार W_in = आगत_shape[3];
            अचर माप_प्रकार H_out = output_shape[2];
            अचर माप_प्रकार W_out = output_shape[3];

            अचर भिन्न pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<भिन्न> Wy_data(H_out * H_in, 0.0f);
            अचर भिन्न sf1 = static_cast<भिन्न>(H_out) / H_in;
            अचर भिन्न support1 = std::max(1.0f, 1.0f / sf1);
            अचर भिन्न invscale1 = 1.0f / support1;

            लागि (माप_प्रकार i1 = 0; i1 < H_out; ++i1) {
                अचर भिन्न y = (static_cast<भिन्न>(i1) + pixel_offset) / sf1;
                अचर स्वतः y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                अचर माप_प्रकार y_min = y_start > 0 ? static_cast<माप_प्रकार>(y_start) : 0;
                अचर स्वतः y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                अचर माप_प्रकार y_max = y_end > 0 ? std::min<माप_प्रकार>(static_cast<माप_प्रकार>(y_end), H_in) : 0;

                भिन्न total_weight = 0.0f;
                लागि (माप_प्रकार sy = y_min; sy < y_max; ++sy) {
                    भिन्न diff = std::असल((static_cast<भिन्न>(sy) - y + pixel_offset) * invscale1);
                    भिन्न weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                यदि (total_weight > 0.0f) {
                    लागि (माप_प्रकार sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<भिन्न> Wx_data(W_in * W_out, 0.0f);
            अचर भिन्न sf0 = static_cast<भिन्न>(W_out) / W_in;
            अचर भिन्न support0 = std::max(1.0f, 1.0f / sf0);
            अचर भिन्न invscale0 = 1.0f / support0;

            लागि (माप_प्रकार i0 = 0; i0 < W_out; ++i0) {
                अचर भिन्न x = (static_cast<भिन्न>(i0) + pixel_offset) / sf0;
                अचर स्वतः x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                अचर माप_प्रकार x_min = x_start > 0 ? static_cast<माप_प्रकार>(x_start) : 0;
                अचर स्वतः x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                अचर माप_प्रकार x_max = x_end > 0 ? std::min<माप_प्रकार>(static_cast<माप_प्रकार>(x_end), W_in) : 0;

                भिन्न total_weight = 0.0f;
                लागि (माप_प्रकार sx = x_min; sx < x_max; ++sx) {
                    भिन्न diff = std::असल((static_cast<भिन्न>(sx) - x + pixel_offset) * invscale0);
                    भिन्न weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                यदि (total_weight > 0.0f) {
                    लागि (माप_प्रकार sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            स्वतः Wy_गाँठो = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            स्वतः Wx_गाँठो = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            स्वतः res_y = std::make_shared<ov::op::v0::MatMul>(Wy_गाँठो, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_गाँठो);
        } अन्यथा {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            विकल्प (op_case) {
            पक्ष 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                रोक;
            पक्ष 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                रोक;
            पक्ष 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                रोक;
            स्वभाव:
                FRONT_END_OP_CONVERSION_CHECK(असत्य, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<भिन्न> scales_vec = {
                static_cast<भिन्न>(output_shape[2]) / static_cast<भिन्न>(आगत_shape[2]),
                static_cast<भिन्न>(output_shape[3]) / static_cast<भिन्न>(आगत_shape[3]),
            };

            स्वतः target_shape_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            स्वतः scales_गाँठो = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            स्वतः axes_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_गाँठो, scales_गाँठो, axes_गाँठो, attrs);
        }
    }

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
