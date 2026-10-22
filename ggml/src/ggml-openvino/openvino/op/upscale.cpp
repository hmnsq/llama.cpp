#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <cmath>
#ضمن <cstddef>
#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/interpolate.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_upscale(ثابت گرێ_Context & context) {
    ژمارە_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto تێخستن = process_view_تێخستن_new(context, 0);

    ثابت auto تێخستن_shape = context.get_تێخستن_shape(0).to_shape();
    ثابت auto output_shape = context.get_output_shape().to_shape();

    إذا (تێخستن_shape == output_shape) {
        أرجع rename_outputs_with_suffix({تێخستن}, context.get_name());
    }

    ov::Output<ov::Node> res = تێخستن;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    لكل (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t in_dim = تێخستن_shape[axis];
        ثابت size_t out_dim = output_shape[axis];
        إذا (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            لكل (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_گرێ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_گرێ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_گرێ, axis_گرێ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    إذا (تێخستن_shape[2] != output_shape[2] || تێخستن_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ثابت صحيح op_case = context.get_op_case();
        ثابت منطقي align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقي antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        إذا (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_in = تێخستن_shape[2];
            ثابت size_t W_in = تێخستن_shape[3];
            ثابت size_t H_out = output_shape[2];
            ثابت size_t W_out = output_shape[3];

            ثابت عائم pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<عائم> Wy_data(H_out * H_in, 0.0f);
            ثابت عائم sf1 = static_cast<عائم>(H_out) / H_in;
            ثابت عائم support1 = std::max(1.0f, 1.0f / sf1);
            ثابت عائم invscale1 = 1.0f / support1;

            لكل (size_t i1 = 0; i1 < H_out; ++i1) {
                ثابت عائم y = (static_cast<عائم>(i1) + pixel_offset) / sf1;
                ثابت auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ثابت size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ثابت auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ثابت size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                عائم total_weight = 0.0f;
                لكل (size_t sy = y_min; sy < y_max; ++sy) {
                    عائم diff = std::abs((static_cast<عائم>(sy) - y + pixel_offset) * invscale1);
                    عائم weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                إذا (total_weight > 0.0f) {
                    لكل (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<عائم> Wx_data(W_in * W_out, 0.0f);
            ثابت عائم sf0 = static_cast<عائم>(W_out) / W_in;
            ثابت عائم support0 = std::max(1.0f, 1.0f / sf0);
            ثابت عائم invscale0 = 1.0f / support0;

            لكل (size_t i0 = 0; i0 < W_out; ++i0) {
                ثابت عائم x = (static_cast<عائم>(i0) + pixel_offset) / sf0;
                ثابت auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ثابت size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ثابت auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ثابت size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                عائم total_weight = 0.0f;
                لكل (size_t sx = x_min; sx < x_max; ++sx) {
                    عائم diff = std::abs((static_cast<عائم>(sx) - x + pixel_offset) * invscale0);
                    عائم weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                إذا (total_weight > 0.0f) {
                    لكل (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_گرێ = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_گرێ = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_گرێ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_گرێ);
        } وإلا {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            بدل (op_case) {
            حالة 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                اكسر;
            حالة 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                اكسر;
            حالة 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                اكسر;
            افتراضي:
                FRONT_END_OP_CONVERSION_CHECK(خطأ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<عائم> scales_vec = {
                static_cast<عائم>(output_shape[2]) / static_cast<عائم>(تێخستن_shape[2]),
                static_cast<عائم>(output_shape[3]) / static_cast<عائم>(تێخستن_shape[3]),
            };

            auto target_shape_گرێ = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_گرێ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_گرێ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_گرێ, scales_گرێ, axes_گرێ, attrs);
        }
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
