#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"
#समावेश "ggml.h"

#समावेश <cmath>
#समावेश <cstddef>
#समावेश <memory>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/interpolate.hpp>
#समावेश <openvino/op/matmul.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_upscale(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    स्वतः निवेश = process_view_निवेश_new(context, 0);

    स्थिर स्वतः निवेश_shape = context.get_निवेश_shape(0).to_shape();
    स्थिर स्वतः output_shape = context.get_output_shape().to_shape();

    अगर (निवेश_shape == output_shape) {
        वापस rename_outputs_with_suffix({निवेश}, context.get_name());
    }

    ov::Output<ov::Node> res = निवेश;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    क्रम (माप_प्रकार axis = 0; axis < 2; ++axis) {
        स्थिर माप_प्रकार in_dim = निवेश_shape[axis];
        स्थिर माप_प्रकार out_dim = output_shape[axis];
        अगर (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            क्रम (माप_प्रकार i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            स्वतः indices_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            स्वतः axis_गाँठ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_गाँठ, axis_गाँठ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    अगर (निवेश_shape[2] != output_shape[2] || निवेश_shape[3] != output_shape[3]) {
        स्थिर int32_t * op_params = context.get_output_op_params();
        स्थिर int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        स्थिर पूर्णांक op_case = context.get_op_case();
        स्थिर bool align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        स्थिर bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        अगर (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            स्थिर माप_प्रकार H_in = निवेश_shape[2];
            स्थिर माप_प्रकार W_in = निवेश_shape[3];
            स्थिर माप_प्रकार H_out = output_shape[2];
            स्थिर माप_प्रकार W_out = output_shape[3];

            स्थिर भग्न pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<भग्न> Wy_तथ्य(H_out * H_in, 0.0f);
            स्थिर भग्न sf1 = static_cast<भग्न>(H_out) / H_in;
            स्थिर भग्न support1 = std::max(1.0f, 1.0f / sf1);
            स्थिर भग्न invscale1 = 1.0f / support1;

            क्रम (माप_प्रकार i1 = 0; i1 < H_out; ++i1) {
                स्थिर भग्न y = (static_cast<भग्न>(i1) + pixel_offset) / sf1;
                स्थिर स्वतः y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                स्थिर माप_प्रकार y_min = y_start > 0 ? static_cast<माप_प्रकार>(y_start) : 0;
                स्थिर स्वतः y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                स्थिर माप_प्रकार y_max = y_end > 0 ? std::min<माप_प्रकार>(static_cast<माप_प्रकार>(y_end), H_in) : 0;

                भग्न योग_weight = 0.0f;
                क्रम (माप_प्रकार sy = y_min; sy < y_max; ++sy) {
                    भग्न diff = std::असल((static_cast<भग्न>(sy) - y + pixel_offset) * invscale1);
                    भग्न weight = std::max(1.0f - diff, 0.0f);
                    Wy_तथ्य[i1 * H_in + sy] = weight;
                    योग_weight += weight;
                }
                अगर (योग_weight > 0.0f) {
                    क्रम (माप_प्रकार sy = y_min; sy < y_max; ++sy) {
                        Wy_तथ्य[i1 * H_in + sy] /= योग_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<भग्न> Wx_तथ्य(W_in * W_out, 0.0f);
            स्थिर भग्न sf0 = static_cast<भग्न>(W_out) / W_in;
            स्थिर भग्न support0 = std::max(1.0f, 1.0f / sf0);
            स्थिर भग्न invscale0 = 1.0f / support0;

            क्रम (माप_प्रकार i0 = 0; i0 < W_out; ++i0) {
                स्थिर भग्न x = (static_cast<भग्न>(i0) + pixel_offset) / sf0;
                स्थिर स्वतः x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                स्थिर माप_प्रकार x_min = x_start > 0 ? static_cast<माप_प्रकार>(x_start) : 0;
                स्थिर स्वतः x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                स्थिर माप_प्रकार x_max = x_end > 0 ? std::min<माप_प्रकार>(static_cast<माप_प्रकार>(x_end), W_in) : 0;

                भग्न योग_weight = 0.0f;
                क्रम (माप_प्रकार sx = x_min; sx < x_max; ++sx) {
                    भग्न diff = std::असल((static_cast<भग्न>(sx) - x + pixel_offset) * invscale0);
                    भग्न weight = std::max(1.0f - diff, 0.0f);
                    Wx_तथ्य[sx * W_out + i0] = weight;
                    योग_weight += weight;
                }
                अगर (योग_weight > 0.0f) {
                    क्रम (माप_प्रकार sx = x_min; sx < x_max; ++sx) {
                        Wx_तथ्य[sx * W_out + i0] /= योग_weight;
                    }
                }
            }

            स्वतः Wy_गाँठ = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_तथ्य);
            स्वतः Wx_गाँठ = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_तथ्य);

            स्वतः res_y = std::make_shared<ov::op::v0::MatMul>(Wy_गाँठ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_गाँठ);
        } अन्यथा {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            चयन (op_case) {
            हाल 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                अवरोध;
            हाल 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                अवरोध;
            हाल 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                अवरोध;
            शेष:
                FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<भग्न> scales_vec = {
                static_cast<भग्न>(output_shape[2]) / static_cast<भग्न>(निवेश_shape[2]),
                static_cast<भग्न>(output_shape[3]) / static_cast<भग्न>(निवेश_shape[3]),
            };

            स्वतः target_shape_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            स्वतः scales_गाँठ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            स्वतः axes_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_गाँठ, scales_गाँठ, axes_गाँठ, attrs);
        }
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
