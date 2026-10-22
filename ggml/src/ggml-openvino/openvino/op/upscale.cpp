#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"
#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/gather.hpp>
#অন্তর্ভাৱ <openvino/op/interpolate.hpp>
#অন্তর্ভাৱ <openvino/op/matmul.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_upscale(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);

    ধ্রুৱ auto প্ৰবেশ_shape = context.get_প্ৰবেশ_shape(0).to_shape();
    ধ্রুৱ auto output_shape = context.get_output_shape().to_shape();

    যদি (প্ৰবেশ_shape == output_shape) {
        ঘূৰাই rename_outputs_with_suffix({প্ৰবেশ}, context.get_name());
    }

    ov::Output<ov::Node> res = প্ৰবেশ;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    বাবে (size_t axis = 0; axis < 2; ++axis) {
        ধ্রুৱ size_t in_dim = প্ৰবেশ_shape[axis];
        ধ্রুৱ size_t out_dim = output_shape[axis];
        যদি (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            বাবে (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_গ্ৰন্থি, axis_গ্ৰন্থি);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    যদি (প্ৰবেশ_shape[2] != output_shape[2] || প্ৰবেশ_shape[3] != output_shape[3]) {
        ধ্রুৱ int32_t * op_params = context.get_output_op_params();
        ধ্রুৱ int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ধ্রুৱ পূর্ণাঙ্ক op_case = context.get_op_case();
        ধ্রুৱ তার্কিক align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ধ্রুৱ তার্কিক antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        যদি (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ধ্রুৱ size_t H_in = প্ৰবেশ_shape[2];
            ধ্রুৱ size_t W_in = প্ৰবেশ_shape[3];
            ধ্রুৱ size_t H_out = output_shape[2];
            ধ্রুৱ size_t W_out = output_shape[3];

            ধ্রুৱ ভিন্ন pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<ভিন্ন> Wy_data(H_out * H_in, 0.0f);
            ধ্রুৱ ভিন্ন sf1 = static_cast<ভিন্ন>(H_out) / H_in;
            ধ্রুৱ ভিন্ন support1 = std::max(1.0f, 1.0f / sf1);
            ধ্রুৱ ভিন্ন invscale1 = 1.0f / support1;

            বাবে (size_t i1 = 0; i1 < H_out; ++i1) {
                ধ্রুৱ ভিন্ন y = (static_cast<ভিন্ন>(i1) + pixel_offset) / sf1;
                ধ্রুৱ auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ধ্রুৱ size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ধ্রুৱ auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ধ্রুৱ size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                ভিন্ন total_weight = 0.0f;
                বাবে (size_t sy = y_min; sy < y_max; ++sy) {
                    ভিন্ন diff = std::abs((static_cast<ভিন্ন>(sy) - y + pixel_offset) * invscale1);
                    ভিন্ন weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                যদি (total_weight > 0.0f) {
                    বাবে (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<ভিন্ন> Wx_data(W_in * W_out, 0.0f);
            ধ্রুৱ ভিন্ন sf0 = static_cast<ভিন্ন>(W_out) / W_in;
            ধ্রুৱ ভিন্ন support0 = std::max(1.0f, 1.0f / sf0);
            ধ্রুৱ ভিন্ন invscale0 = 1.0f / support0;

            বাবে (size_t i0 = 0; i0 < W_out; ++i0) {
                ধ্রুৱ ভিন্ন x = (static_cast<ভিন্ন>(i0) + pixel_offset) / sf0;
                ধ্রুৱ auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ধ্রুৱ size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ধ্রুৱ auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ধ্রুৱ size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                ভিন্ন total_weight = 0.0f;
                বাবে (size_t sx = x_min; sx < x_max; ++sx) {
                    ভিন্ন diff = std::abs((static_cast<ভিন্ন>(sx) - x + pixel_offset) * invscale0);
                    ভিন্ন weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                যদি (total_weight > 0.0f) {
                    বাবে (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_গ্ৰন্থি, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_গ্ৰন্থি);
        } অন্যথা {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            ৱিকল্প (op_case) {
            পক্ষ 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                ভঙ্গ;
            পক্ষ 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                ভঙ্গ;
            পক্ষ 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                ভঙ্গ;
            স্ৱভাৱ:
                FRONT_END_OP_CONVERSION_CHECK(মিছা, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<ভিন্ন> scales_vec = {
                static_cast<ভিন্ন>(output_shape[2]) / static_cast<ভিন্ন>(প্ৰবেশ_shape[2]),
                static_cast<ভিন্ন>(output_shape[3]) / static_cast<ভিন্ন>(প্ৰবেশ_shape[3]),
            };

            auto target_shape_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_গ্ৰন্থি, scales_গ্ৰন্থি, axes_গ্ৰন্থি, attrs);
        }
    }

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
