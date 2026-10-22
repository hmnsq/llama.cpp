#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml.h"

#شامل <cmath>
#شامل <cstddef>
#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/interpolate.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <vector>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_upscale(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto درون_داد = process_view_درون_داد_new(context, 0);

    ثابت auto درون_داد_shape = context.get_درون_داد_shape(0).to_shape();
    ثابت auto output_shape = context.get_output_shape().to_shape();

    اگر (درون_داد_shape == output_shape) {
        بازگشت rename_outputs_with_suffix({درون_داد}, context.get_name());
    }

    ov::Output<ov::Node> res = درون_داد;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    برای (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t in_dim = درون_داد_shape[axis];
        ثابت size_t out_dim = output_shape[axis];
        اگر (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            برای (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_گره = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_گره = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_گره, axis_گره);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    اگر (درون_داد_shape[2] != output_shape[2] || درون_داد_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ثابت صحیح op_case = context.get_op_case();
        ثابت منطقی align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقی antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        اگر (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_in = درون_داد_shape[2];
            ثابت size_t W_in = درون_داد_shape[3];
            ثابت size_t H_out = output_shape[2];
            ثابت size_t W_out = output_shape[3];

            ثابت اعشاری pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<اعشاری> Wy_data(H_out * H_in, 0.0f);
            ثابت اعشاری sf1 = static_cast<اعشاری>(H_out) / H_in;
            ثابت اعشاری support1 = std::max(1.0f, 1.0f / sf1);
            ثابت اعشاری invscale1 = 1.0f / support1;

            برای (size_t i1 = 0; i1 < H_out; ++i1) {
                ثابت اعشاری y = (static_cast<اعشاری>(i1) + pixel_offset) / sf1;
                ثابت auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ثابت size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ثابت auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ثابت size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                اعشاری total_weight = 0.0f;
                برای (size_t sy = y_min; sy < y_max; ++sy) {
                    اعشاری diff = std::abs((static_cast<اعشاری>(sy) - y + pixel_offset) * invscale1);
                    اعشاری weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                اگر (total_weight > 0.0f) {
                    برای (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<اعشاری> Wx_data(W_in * W_out, 0.0f);
            ثابت اعشاری sf0 = static_cast<اعشاری>(W_out) / W_in;
            ثابت اعشاری support0 = std::max(1.0f, 1.0f / sf0);
            ثابت اعشاری invscale0 = 1.0f / support0;

            برای (size_t i0 = 0; i0 < W_out; ++i0) {
                ثابت اعشاری x = (static_cast<اعشاری>(i0) + pixel_offset) / sf0;
                ثابت auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ثابت size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ثابت auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ثابت size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                اعشاری total_weight = 0.0f;
                برای (size_t sx = x_min; sx < x_max; ++sx) {
                    اعشاری diff = std::abs((static_cast<اعشاری>(sx) - x + pixel_offset) * invscale0);
                    اعشاری weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                اگر (total_weight > 0.0f) {
                    برای (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_گره = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_گره = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_گره, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_گره);
        } وگرنه {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            انتخاب (op_case) {
            حالت 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                بشکن;
            حالت 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                بشکن;
            حالت 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                بشکن;
            پیشفرض:
                FRONT_END_OP_CONVERSION_CHECK(نادرست, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<اعشاری> scales_vec = {
                static_cast<اعشاری>(output_shape[2]) / static_cast<اعشاری>(درون_داد_shape[2]),
                static_cast<اعشاری>(output_shape[3]) / static_cast<اعشاری>(درون_داد_shape[3]),
            };

            auto target_shape_گره = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_گره = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_گره = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_گره, scales_گره, axes_گره, attrs);
        }
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
