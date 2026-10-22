#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/interpolate.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/matmul.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_upscale(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);

    ଅଚଳ auto ନିବେଶ_shape = context.get_ନିବେଶ_shape(0).to_shape();
    ଅଚଳ auto output_shape = context.get_output_shape().to_shape();

    ଯଦି (ନିବେଶ_shape == output_shape) {
        ଫେରନ୍ତୁ rename_outputs_with_suffix({ନିବେଶ}, context.get_name());
    }

    ov::Output<ov::Node> res = ନିବେଶ;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    ପାଇଁ (size_t axis = 0; axis < 2; ++axis) {
        ଅଚଳ size_t in_dim = ନିବେଶ_shape[axis];
        ଅଚଳ size_t out_dim = output_shape[axis];
        ଯଦି (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            ପାଇଁ (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ଗଣ୍ଠି, axis_ଗଣ୍ଠି);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    ଯଦି (ନିବେଶ_shape[2] != output_shape[2] || ନିବେଶ_shape[3] != output_shape[3]) {
        ଅଚଳ int32_t * op_params = context.get_output_op_params();
        ଅଚଳ int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = context.get_op_case();
        ଅଚଳ ତାର୍କିକ align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ଅଚଳ ତାର୍କିକ antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        ଯଦି (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ଅଚଳ size_t H_in = ନିବେଶ_shape[2];
            ଅଚଳ size_t W_in = ନିବେଶ_shape[3];
            ଅଚଳ size_t H_out = output_shape[2];
            ଅଚଳ size_t W_out = output_shape[3];

            ଅଚଳ ଦଶମିକ pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<ଦଶମିକ> Wy_data(H_out * H_in, 0.0f);
            ଅଚଳ ଦଶମିକ sf1 = static_cast<ଦଶମିକ>(H_out) / H_in;
            ଅଚଳ ଦଶମିକ support1 = std::max(1.0f, 1.0f / sf1);
            ଅଚଳ ଦଶମିକ invscale1 = 1.0f / support1;

            ପାଇଁ (size_t i1 = 0; i1 < H_out; ++i1) {
                ଅଚଳ ଦଶମିକ y = (static_cast<ଦଶମିକ>(i1) + pixel_offset) / sf1;
                ଅଚଳ auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ଅଚଳ size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ଅଚଳ auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ଅଚଳ size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                ଦଶମିକ total_weight = 0.0f;
                ପାଇଁ (size_t sy = y_min; sy < y_max; ++sy) {
                    ଦଶମିକ diff = std::abs((static_cast<ଦଶମିକ>(sy) - y + pixel_offset) * invscale1);
                    ଦଶମିକ weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                ଯଦି (total_weight > 0.0f) {
                    ପାଇଁ (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<ଦଶମିକ> Wx_data(W_in * W_out, 0.0f);
            ଅଚଳ ଦଶମିକ sf0 = static_cast<ଦଶମିକ>(W_out) / W_in;
            ଅଚଳ ଦଶମିକ support0 = std::max(1.0f, 1.0f / sf0);
            ଅଚଳ ଦଶମିକ invscale0 = 1.0f / support0;

            ପାଇଁ (size_t i0 = 0; i0 < W_out; ++i0) {
                ଅଚଳ ଦଶମିକ x = (static_cast<ଦଶମିକ>(i0) + pixel_offset) / sf0;
                ଅଚଳ auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ଅଚଳ size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ଅଚଳ auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ଅଚଳ size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                ଦଶମିକ total_weight = 0.0f;
                ପାଇଁ (size_t sx = x_min; sx < x_max; ++sx) {
                    ଦଶମିକ diff = std::abs((static_cast<ଦଶମିକ>(sx) - x + pixel_offset) * invscale0);
                    ଦଶମିକ weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                ଯଦି (total_weight > 0.0f) {
                    ପାଇଁ (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ଗଣ୍ଠି, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ଗଣ୍ଠି);
        } ଅନ୍ୟଥା {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            ବାଛ (op_case) {
            ସ୍ଥିତି 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                FRONT_END_OP_CONVERSION_CHECK(ମିଥ୍ୟା, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<ଦଶମିକ> scales_vec = {
                static_cast<ଦଶମିକ>(output_shape[2]) / static_cast<ଦଶମିକ>(ନିବେଶ_shape[2]),
                static_cast<ଦଶମିକ>(output_shape[3]) / static_cast<ଦଶମିକ>(ନିବେଶ_shape[3]),
            };

            auto target_shape_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_ଗଣ୍ଠି, scales_ଗଣ୍ଠି, axes_ଗଣ୍ଠି, attrs);
        }
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
