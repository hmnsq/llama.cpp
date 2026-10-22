#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/interpolate.hpp>
#અન્તર્ભાવ <openvino/op/matmul.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_upscale(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto આગત = process_view_આગત_new(context, 0);

    અચલ auto આગત_shape = context.get_આગત_shape(0).to_shape();
    અચલ auto output_shape = context.get_output_shape().to_shape();

    જો (આગત_shape == output_shape) {
        પાછા rename_outputs_with_suffix({આગત}, context.get_name());
    }

    ov::Output<ov::Node> res = આગત;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    માટે (size_t axis = 0; axis < 2; ++axis) {
        અચલ size_t in_dim = આગત_shape[axis];
        અચલ size_t out_dim = output_shape[axis];
        જો (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            માટે (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ગાંઠ, axis_ગાંઠ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    જો (આગત_shape[2] != output_shape[2] || આગત_shape[3] != output_shape[3]) {
        અચલ int32_t * op_params = context.get_output_op_params();
        અચલ int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        અચલ પૂર્ણાંક op_case = context.get_op_case();
        અચલ તાર્કિક align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        અચલ તાર્કિક antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        જો (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            અચલ size_t H_in = આગત_shape[2];
            અચલ size_t W_in = આગત_shape[3];
            અચલ size_t H_out = output_shape[2];
            અચલ size_t W_out = output_shape[3];

            અચલ દશાંશ pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<દશાંશ> Wy_data(H_out * H_in, 0.0f);
            અચલ દશાંશ sf1 = static_cast<દશાંશ>(H_out) / H_in;
            અચલ દશાંશ support1 = std::max(1.0f, 1.0f / sf1);
            અચલ દશાંશ invscale1 = 1.0f / support1;

            માટે (size_t i1 = 0; i1 < H_out; ++i1) {
                અચલ દશાંશ y = (static_cast<દશાંશ>(i1) + pixel_offset) / sf1;
                અચલ auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                અચલ size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                અચલ auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                અચલ size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                દશાંશ total_weight = 0.0f;
                માટે (size_t sy = y_min; sy < y_max; ++sy) {
                    દશાંશ diff = std::abs((static_cast<દશાંશ>(sy) - y + pixel_offset) * invscale1);
                    દશાંશ weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                જો (total_weight > 0.0f) {
                    માટે (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<દશાંશ> Wx_data(W_in * W_out, 0.0f);
            અચલ દશાંશ sf0 = static_cast<દશાંશ>(W_out) / W_in;
            અચલ દશાંશ support0 = std::max(1.0f, 1.0f / sf0);
            અચલ દશાંશ invscale0 = 1.0f / support0;

            માટે (size_t i0 = 0; i0 < W_out; ++i0) {
                અચલ દશાંશ x = (static_cast<દશાંશ>(i0) + pixel_offset) / sf0;
                અચલ auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                અચલ size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                અચલ auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                અચલ size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                દશાંશ total_weight = 0.0f;
                માટે (size_t sx = x_min; sx < x_max; ++sx) {
                    દશાંશ diff = std::abs((static_cast<દશાંશ>(sx) - x + pixel_offset) * invscale0);
                    દશાંશ weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                જો (total_weight > 0.0f) {
                    માટે (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_ગાંઠ = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_ગાંઠ = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ગાંઠ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ગાંઠ);
        } નહીંતર {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            પસંદગી (op_case) {
            સ્થિતિ 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                તોડો;
            સ્થિતિ 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                તોડો;
            સ્થિતિ 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                તોડો;
            મૂળભૂત:
                FRONT_END_OP_CONVERSION_CHECK(ખોટું, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<દશાંશ> scales_vec = {
                static_cast<દશાંશ>(output_shape[2]) / static_cast<દશાંશ>(આગત_shape[2]),
                static_cast<દશાંશ>(output_shape[3]) / static_cast<દશાંશ>(આગત_shape[3]),
            };

            auto target_shape_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_ગાંઠ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_ગાંઠ, scales_ગાંઠ, axes_ગાંઠ, attrs);
        }
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
