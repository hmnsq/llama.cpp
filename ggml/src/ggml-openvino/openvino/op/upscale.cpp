#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"
#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/interpolate.hpp>
#അന്തര്ഭാവ <openvino/op/matmul.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_upscale(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto നിവേശം = process_view_നിവേശം_new(context, 0);

    സ്ഥിരം auto നിവേശം_shape = context.get_നിവേശം_shape(0).to_shape();
    സ്ഥിരം auto output_shape = context.get_output_shape().to_shape();

    എങ്കിൽ (നിവേശം_shape == output_shape) {
        തിരികെ rename_outputs_with_suffix({നിവേശം}, context.get_name());
    }

    ov::Output<ov::Node> res = നിവേശം;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    വേണ്ടി (size_t axis = 0; axis < 2; ++axis) {
        സ്ഥിരം size_t in_dim = നിവേശം_shape[axis];
        സ്ഥിരം size_t out_dim = output_shape[axis];
        എങ്കിൽ (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            വേണ്ടി (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_കണ്ണി, axis_കണ്ണി);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    എങ്കിൽ (നിവേശം_shape[2] != output_shape[2] || നിവേശം_shape[3] != output_shape[3]) {
        സ്ഥിരം int32_t * op_params = context.get_output_op_params();
        സ്ഥിരം int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        സ്ഥിരം പൂർണ്ണസംഖ്യ op_case = context.get_op_case();
        സ്ഥിരം തർക്കം align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        സ്ഥിരം തർക്കം antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        എങ്കിൽ (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            സ്ഥിരം size_t H_in = നിവേശം_shape[2];
            സ്ഥിരം size_t W_in = നിവേശം_shape[3];
            സ്ഥിരം size_t H_out = output_shape[2];
            സ്ഥിരം size_t W_out = output_shape[3];

            സ്ഥിരം ദശാംശം pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<ദശാംശം> Wy_data(H_out * H_in, 0.0f);
            സ്ഥിരം ദശാംശം sf1 = static_cast<ദശാംശം>(H_out) / H_in;
            സ്ഥിരം ദശാംശം support1 = std::max(1.0f, 1.0f / sf1);
            സ്ഥിരം ദശാംശം invscale1 = 1.0f / support1;

            വേണ്ടി (size_t i1 = 0; i1 < H_out; ++i1) {
                സ്ഥിരം ദശാംശം y = (static_cast<ദശാംശം>(i1) + pixel_offset) / sf1;
                സ്ഥിരം auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                സ്ഥിരം size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                സ്ഥിരം auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                സ്ഥിരം size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                ദശാംശം total_weight = 0.0f;
                വേണ്ടി (size_t sy = y_min; sy < y_max; ++sy) {
                    ദശാംശം diff = std::abs((static_cast<ദശാംശം>(sy) - y + pixel_offset) * invscale1);
                    ദശാംശം weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                എങ്കിൽ (total_weight > 0.0f) {
                    വേണ്ടി (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<ദശാംശം> Wx_data(W_in * W_out, 0.0f);
            സ്ഥിരം ദശാംശം sf0 = static_cast<ദശാംശം>(W_out) / W_in;
            സ്ഥിരം ദശാംശം support0 = std::max(1.0f, 1.0f / sf0);
            സ്ഥിരം ദശാംശം invscale0 = 1.0f / support0;

            വേണ്ടി (size_t i0 = 0; i0 < W_out; ++i0) {
                സ്ഥിരം ദശാംശം x = (static_cast<ദശാംശം>(i0) + pixel_offset) / sf0;
                സ്ഥിരം auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                സ്ഥിരം size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                സ്ഥിരം auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                സ്ഥിരം size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                ദശാംശം total_weight = 0.0f;
                വേണ്ടി (size_t sx = x_min; sx < x_max; ++sx) {
                    ദശാംശം diff = std::abs((static_cast<ദശാംശം>(sx) - x + pixel_offset) * invscale0);
                    ദശാംശം weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                എങ്കിൽ (total_weight > 0.0f) {
                    വേണ്ടി (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_കണ്ണി = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_കണ്ണി = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_കണ്ണി, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_കണ്ണി);
        } അല്ലെങ്കിൽ {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            തിരഞ്ഞെടുപ്പ് (op_case) {
            അവസ്ഥ 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                നിർത്തുക;
            അവസ്ഥ 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                നിർത്തുക;
            അവസ്ഥ 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                FRONT_END_OP_CONVERSION_CHECK(അസത്യം, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<ദശാംശം> scales_vec = {
                static_cast<ദശാംശം>(output_shape[2]) / static_cast<ദശാംശം>(നിവേശം_shape[2]),
                static_cast<ദശാംശം>(output_shape[3]) / static_cast<ദശാംശം>(നിവേശം_shape[3]),
            };

            auto target_shape_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_കണ്ണി = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_കണ്ണി, scales_കണ്ണി, axes_കണ്ണി, attrs);
        }
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
