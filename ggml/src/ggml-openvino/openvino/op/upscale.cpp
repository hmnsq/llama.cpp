#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/interpolate.hpp>
#অন্তর্ভাব <openvino/op/matmul.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_upscale(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);

    ধ্রুবক auto প্রবেশ_shape = context.get_প্রবেশ_shape(0).to_shape();
    ধ্রুবক auto output_shape = context.get_output_shape().to_shape();

    যদি (প্রবেশ_shape == output_shape) {
        ফেরত rename_outputs_with_suffix({প্রবেশ}, context.get_name());
    }

    ov::Output<ov::Node> res = প্রবেশ;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    জন্য (size_t axis = 0; axis < 2; ++axis) {
        ধ্রুবক size_t in_dim = প্রবেশ_shape[axis];
        ধ্রুবক size_t out_dim = output_shape[axis];
        যদি (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            জন্য (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_গ্রন্থি, axis_গ্রন্থি);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    যদি (প্রবেশ_shape[2] != output_shape[2] || প্রবেশ_shape[3] != output_shape[3]) {
        ধ্রুবক int32_t * op_params = context.get_output_op_params();
        ধ্রুবক int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        ধ্রুবক পূর্ণসংখ্যা op_case = context.get_op_case();
        ধ্রুবক যৌক্তিক align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ধ্রুবক যৌক্তিক antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        যদি (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ধ্রুবক size_t H_in = প্রবেশ_shape[2];
            ধ্রুবক size_t W_in = প্রবেশ_shape[3];
            ধ্রুবক size_t H_out = output_shape[2];
            ধ্রুবক size_t W_out = output_shape[3];

            ধ্রুবক দশমিক pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<দশমিক> Wy_তথ্য(H_out * H_in, 0.0f);
            ধ্রুবক দশমিক sf1 = static_cast<দশমিক>(H_out) / H_in;
            ধ্রুবক দশমিক support1 = std::max(1.0f, 1.0f / sf1);
            ধ্রুবক দশমিক invscale1 = 1.0f / support1;

            জন্য (size_t i1 = 0; i1 < H_out; ++i1) {
                ধ্রুবক দশমিক y = (static_cast<দশমিক>(i1) + pixel_offset) / sf1;
                ধ্রুবক auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                ধ্রুবক size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ধ্রুবক auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                ধ্রুবক size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                দশমিক যোগ_weight = 0.0f;
                জন্য (size_t sy = y_min; sy < y_max; ++sy) {
                    দশমিক diff = std::abs((static_cast<দশমিক>(sy) - y + pixel_offset) * invscale1);
                    দশমিক weight = std::max(1.0f - diff, 0.0f);
                    Wy_তথ্য[i1 * H_in + sy] = weight;
                    যোগ_weight += weight;
                }
                যদি (যোগ_weight > 0.0f) {
                    জন্য (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_তথ্য[i1 * H_in + sy] /= যোগ_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<দশমিক> Wx_তথ্য(W_in * W_out, 0.0f);
            ধ্রুবক দশমিক sf0 = static_cast<দশমিক>(W_out) / W_in;
            ধ্রুবক দশমিক support0 = std::max(1.0f, 1.0f / sf0);
            ধ্রুবক দশমিক invscale0 = 1.0f / support0;

            জন্য (size_t i0 = 0; i0 < W_out; ++i0) {
                ধ্রুবক দশমিক x = (static_cast<দশমিক>(i0) + pixel_offset) / sf0;
                ধ্রুবক auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                ধ্রুবক size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ধ্রুবক auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                ধ্রুবক size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                দশমিক যোগ_weight = 0.0f;
                জন্য (size_t sx = x_min; sx < x_max; ++sx) {
                    দশমিক diff = std::abs((static_cast<দশমিক>(sx) - x + pixel_offset) * invscale0);
                    দশমিক weight = std::max(1.0f - diff, 0.0f);
                    Wx_তথ্য[sx * W_out + i0] = weight;
                    যোগ_weight += weight;
                }
                যদি (যোগ_weight > 0.0f) {
                    জন্য (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_তথ্য[sx * W_out + i0] /= যোগ_weight;
                    }
                }
            }

            auto Wy_গ্রন্থি = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_তথ্য);
            auto Wx_গ্রন্থি = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_তথ্য);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_গ্রন্থি, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_গ্রন্থি);
        } অন্যথা {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            নির্বাচন (op_case) {
            ক্ষেত্র 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                বিরতি;
            ক্ষেত্র 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                বিরতি;
            ক্ষেত্র 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                বিরতি;
            পূর্বনির্ধারিত:
                FRONT_END_OP_CONVERSION_CHECK(মিথ্যা, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<দশমিক> scales_vec = {
                static_cast<দশমিক>(output_shape[2]) / static_cast<দশমিক>(প্রবেশ_shape[2]),
                static_cast<দশমিক>(output_shape[3]) / static_cast<দশমিক>(প্রবেশ_shape[3]),
            };

            auto target_shape_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_গ্রন্থি = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_গ্রন্থি, scales_গ্রন্থি, axes_গ্রন্থি, attrs);
        }
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
