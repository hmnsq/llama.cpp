#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/interpolate.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_upscale(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಧ್ರುವ auto ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(0).to_shape();
    ಧ್ರುವ auto output_shape = context.get_output_shape().to_shape();

    ಯದಿ (ಒಳಹರಿವು_shape == output_shape) {
        ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({ಒಳಹರಿವು}, context.get_name());
    }

    ov::Output<ov::Node> res = ಒಳಹರಿವು;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    ಪ್ರತಿ (size_t axis = 0; axis < 2; ++axis) {
        ಧ್ರುವ size_t in_dim = ಒಳಹರಿವು_shape[axis];
        ಧ್ರುವ size_t out_dim = output_shape[axis];
        ಯದಿ (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            ಪ್ರತಿ (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ಗಂಟು, axis_ಗಂಟು);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    ಯದಿ (ಒಳಹರಿವು_shape[2] != output_shape[2] || ಒಳಹರಿವು_shape[3] != output_shape[3]) {
        ಧ್ರುವ int32_t * op_params = context.get_output_op_params();
        ಧ್ರುವ int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ op_case = context.get_op_case();
        ಧ್ರುವ ತಾರ್ಕಿಕ align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ಧ್ರುವ ತಾರ್ಕಿಕ antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        ಯದಿ (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ಧ್ರುವ size_t H_in = ಒಳಹರಿವು_shape[2];
            ಧ್ರುವ size_t W_in = ಒಳಹರಿವು_shape[3];
            ಧ್ರುವ size_t H_out = output_shape[2];
            ಧ್ರುವ size_t W_out = output_shape[3];

            ಧ್ರುವ ಭಿನ್ನ pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<ಭಿನ್ನ> Wy_data(H_out * H_in, 0.0f);
            ಧ್ರುವ ಭಿನ್ನ sf1 = static_cast<ಭಿನ್ನ>(H_out) / H_in;
            ಧ್ರುವ ಭಿನ್ನ support1 = std::max(1.0f, 1.0f / sf1);
            ಧ್ರುವ ಭಿನ್ನ invscale1 = 1.0f / support1;

            ಪ್ರತಿ (size_t i1 = 0; i1 < H_out; ++i1) {
                ಧ್ರುವ ಭಿನ್ನ y = (static_cast<ಭಿನ್ನ>(i1) + pixel_offset) / sf1;
                ಧ್ರುವ auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ಧ್ರುವ size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ಧ್ರುವ auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ಧ್ರುವ size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                ಭಿನ್ನ total_weight = 0.0f;
                ಪ್ರತಿ (size_t sy = y_min; sy < y_max; ++sy) {
                    ಭಿನ್ನ diff = std::abs((static_cast<ಭಿನ್ನ>(sy) - y + pixel_offset) * invscale1);
                    ಭಿನ್ನ weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                ಯದಿ (total_weight > 0.0f) {
                    ಪ್ರತಿ (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<ಭಿನ್ನ> Wx_data(W_in * W_out, 0.0f);
            ಧ್ರುವ ಭಿನ್ನ sf0 = static_cast<ಭಿನ್ನ>(W_out) / W_in;
            ಧ್ರುವ ಭಿನ್ನ support0 = std::max(1.0f, 1.0f / sf0);
            ಧ್ರುವ ಭಿನ್ನ invscale0 = 1.0f / support0;

            ಪ್ರತಿ (size_t i0 = 0; i0 < W_out; ++i0) {
                ಧ್ರುವ ಭಿನ್ನ x = (static_cast<ಭಿನ್ನ>(i0) + pixel_offset) / sf0;
                ಧ್ರುವ auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ಧ್ರುವ size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ಧ್ರುವ auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ಧ್ರುವ size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                ಭಿನ್ನ total_weight = 0.0f;
                ಪ್ರತಿ (size_t sx = x_min; sx < x_max; ++sx) {
                    ಭಿನ್ನ diff = std::abs((static_cast<ಭಿನ್ನ>(sx) - x + pixel_offset) * invscale0);
                    ಭಿನ್ನ weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                ಯದಿ (total_weight > 0.0f) {
                    ಪ್ರತಿ (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ಗಂಟು, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ಗಂಟು);
        } ಅನ್ಯಥಾ {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            ವಿಕಲ್ಪ (op_case) {
            ಪಕ್ಷ 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                ಭಙ್ಗ;
            ಪಕ್ಷ 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                ಭಙ್ಗ;
            ಪಕ್ಷ 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                FRONT_END_OP_CONVERSION_CHECK(ಅಸತ್ಯ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<ಭಿನ್ನ> scales_vec = {
                static_cast<ಭಿನ್ನ>(output_shape[2]) / static_cast<ಭಿನ್ನ>(ಒಳಹರಿವು_shape[2]),
                static_cast<ಭಿನ್ನ>(output_shape[3]) / static_cast<ಭಿನ್ನ>(ಒಳಹರಿವು_shape[3]),
            };

            auto target_shape_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_ಗಂಟು, scales_ಗಂಟು, axes_ಗಂಟು, attrs);
        }
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
