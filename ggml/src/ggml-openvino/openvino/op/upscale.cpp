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

OutputVector translate_upscale(ध्रुव गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto आदान = process_view_आदान_new(context, 0);

    ध्रुव auto आदान_shape = context.get_आदान_shape(0).to_shape();
    ध्रुव auto output_shape = context.get_output_shape().to_shape();

    यदि (आदान_shape == output_shape) {
        प्रत्यागम rename_outputs_with_suffix({आदान}, context.get_name());
    }

    ov::Output<ov::Node> res = आदान;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    प्रति (size_t axis = 0; axis < 2; ++axis) {
        ध्रुव size_t in_dim = आदान_shape[axis];
        ध्रुव size_t out_dim = output_shape[axis];
        यदि (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            प्रति (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_गाठ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_गाठ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_गाठ, axis_गाठ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    यदि (आदान_shape[2] != output_shape[2] || आदान_shape[3] != output_shape[3]) {
        ध्रुव int32_t * op_params = context.get_output_op_params();
        ध्रुव int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ध्रुव पूर्णाङ्क op_case = context.get_op_case();
        ध्रुव तार्किक align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ध्रुव तार्किक antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        यदि (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ध्रुव size_t H_in = आदान_shape[2];
            ध्रुव size_t W_in = आदान_shape[3];
            ध्रुव size_t H_out = output_shape[2];
            ध्रुव size_t W_out = output_shape[3];

            ध्रुव भिन्न pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<भिन्न> Wy_data(H_out * H_in, 0.0f);
            ध्रुव भिन्न sf1 = static_cast<भिन्न>(H_out) / H_in;
            ध्रुव भिन्न support1 = std::max(1.0f, 1.0f / sf1);
            ध्रुव भिन्न invscale1 = 1.0f / support1;

            प्रति (size_t i1 = 0; i1 < H_out; ++i1) {
                ध्रुव भिन्न y = (static_cast<भिन्न>(i1) + pixel_offset) / sf1;
                ध्रुव auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ध्रुव size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ध्रुव auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ध्रुव size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                भिन्न total_weight = 0.0f;
                प्रति (size_t sy = y_min; sy < y_max; ++sy) {
                    भिन्न diff = std::abs((static_cast<भिन्न>(sy) - y + pixel_offset) * invscale1);
                    भिन्न weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                यदि (total_weight > 0.0f) {
                    प्रति (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<भिन्न> Wx_data(W_in * W_out, 0.0f);
            ध्रुव भिन्न sf0 = static_cast<भिन्न>(W_out) / W_in;
            ध्रुव भिन्न support0 = std::max(1.0f, 1.0f / sf0);
            ध्रुव भिन्न invscale0 = 1.0f / support0;

            प्रति (size_t i0 = 0; i0 < W_out; ++i0) {
                ध्रुव भिन्न x = (static_cast<भिन्न>(i0) + pixel_offset) / sf0;
                ध्रुव auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ध्रुव size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ध्रुव auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ध्रुव size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                भिन्न total_weight = 0.0f;
                प्रति (size_t sx = x_min; sx < x_max; ++sx) {
                    भिन्न diff = std::abs((static_cast<भिन्न>(sx) - x + pixel_offset) * invscale0);
                    भिन्न weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                यदि (total_weight > 0.0f) {
                    प्रति (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_गाठ = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_गाठ = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_गाठ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_गाठ);
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
                भङ्ग;
            पक्ष 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                भङ्ग;
            पक्ष 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                भङ्ग;
            स्वभाव:
                FRONT_END_OP_CONVERSION_CHECK(असत्य, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<भिन्न> scales_vec = {
                static_cast<भिन्न>(output_shape[2]) / static_cast<भिन्न>(आदान_shape[2]),
                static_cast<भिन्न>(output_shape[3]) / static_cast<भिन्न>(आदान_shape[3]),
            };

            auto target_shape_गाठ = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_गाठ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_गाठ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_गाठ, scales_गाठ, axes_गाठ, attrs);
        }
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
