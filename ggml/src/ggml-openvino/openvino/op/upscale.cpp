#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/interpolate.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_upscale(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);

    மாறிலி auto உள்ளீடு_shape = context.get_உள்ளீடு_shape(0).to_shape();
    மாறிலி auto output_shape = context.get_output_shape().to_shape();

    எனில் (உள்ளீடு_shape == output_shape) {
        திரும்பு rename_outputs_with_suffix({உள்ளீடு}, context.get_name());
    }

    ov::Output<ov::Node> res = உள்ளீடு;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    க்கு (size_t axis = 0; axis < 2; ++axis) {
        மாறிலி size_t in_dim = உள்ளீடு_shape[axis];
        மாறிலி size_t out_dim = output_shape[axis];
        எனில் (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            க்கு (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_முனை = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_முனை = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_முனை, axis_முனை);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    எனில் (உள்ளீடு_shape[2] != output_shape[2] || உள்ளீடு_shape[3] != output_shape[3]) {
        மாறிலி int32_t * op_params = context.get_output_op_params();
        மாறிலி int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        மாறிலி முழு op_case = context.get_op_case();
        மாறிலி தருக்க align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        மாறிலி தருக்க antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        எனில் (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            மாறிலி size_t H_in = உள்ளீடு_shape[2];
            மாறிலி size_t W_in = உள்ளீடு_shape[3];
            மாறிலி size_t H_out = output_shape[2];
            மாறிலி size_t W_out = output_shape[3];

            மாறிலி தசம pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<தசம> Wy_data(H_out * H_in, 0.0f);
            மாறிலி தசம sf1 = static_cast<தசம>(H_out) / H_in;
            மாறிலி தசம support1 = std::max(1.0f, 1.0f / sf1);
            மாறிலி தசம invscale1 = 1.0f / support1;

            க்கு (size_t i1 = 0; i1 < H_out; ++i1) {
                மாறிலி தசம y = (static_cast<தசம>(i1) + pixel_offset) / sf1;
                மாறிலி auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                மாறிலி size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                மாறிலி auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                மாறிலி size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                தசம total_weight = 0.0f;
                க்கு (size_t sy = y_min; sy < y_max; ++sy) {
                    தசம diff = std::abs((static_cast<தசம>(sy) - y + pixel_offset) * invscale1);
                    தசம weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                எனில் (total_weight > 0.0f) {
                    க்கு (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<தசம> Wx_data(W_in * W_out, 0.0f);
            மாறிலி தசம sf0 = static_cast<தசம>(W_out) / W_in;
            மாறிலி தசம support0 = std::max(1.0f, 1.0f / sf0);
            மாறிலி தசம invscale0 = 1.0f / support0;

            க்கு (size_t i0 = 0; i0 < W_out; ++i0) {
                மாறிலி தசம x = (static_cast<தசம>(i0) + pixel_offset) / sf0;
                மாறிலி auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                மாறிலி size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                மாறிலி auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                மாறிலி size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                தசம total_weight = 0.0f;
                க்கு (size_t sx = x_min; sx < x_max; ++sx) {
                    தசம diff = std::abs((static_cast<தசம>(sx) - x + pixel_offset) * invscale0);
                    தசம weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                எனில் (total_weight > 0.0f) {
                    க்கு (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_முனை = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_முனை = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_முனை, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_முனை);
        } இல்லையெனில் {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            தேர்வு (op_case) {
            நிலை 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                நிறுத்து;
            நிலை 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                நிறுத்து;
            நிலை 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                நிறுத்து;
            இயல்பு:
                FRONT_END_OP_CONVERSION_CHECK(பொய், "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<தசம> scales_vec = {
                static_cast<தசம>(output_shape[2]) / static_cast<தசம>(உள்ளீடு_shape[2]),
                static_cast<தசம>(output_shape[3]) / static_cast<தசம>(உள்ளீடு_shape[3]),
            };

            auto target_shape_முனை = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_முனை = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_முனை = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_முனை, scales_முனை, axes_முனை, attrs);
        }
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
