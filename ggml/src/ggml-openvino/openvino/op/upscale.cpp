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

Output_ಸದಿಶ translate_upscale(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಬಳಸಿ Interpolate = ov::op::v4::Interpolate;
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape().ಗಮ್ಯ_shape();

    ಆದರೆ (ಒಳಹರಿವು_shape == output_shape) {
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({ಒಳಹರಿವು}, context.get_name());
    }

    ov::Output<ov::Node> res = ಒಳಹರಿವು;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    ಗಾಗಿ (size_t axis = 0; axis < 2; ++axis) {
        ಸ್ಥಿರಾಂಕ size_t ಒಳ_dim = ಒಳಹರಿವು_shape[axis];
        ಸ್ಥಿರಾಂಕ size_t ಹೊರಹೋಗು_dim = output_shape[axis];
        ಆದರೆ (ಒಳ_dim != ಹೊರಹೋಗು_dim) {
            std::vector<int64_t> indices(ಹೊರಹೋಗು_dim);
            ಗಾಗಿ (size_t i = 0; i < ಹೊರಹೋಗು_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * ಒಳ_dim) / ಹೊರಹೋಗು_dim);
            }
            ಸ್ವಯಂಚಾಲಿತ indices_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            ಸ್ವಯಂಚಾಲಿತ axis_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ಗಂಟು, axis_ಗಂಟು);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    ಆದರೆ (ಒಳಹರಿವು_shape[2] != output_shape[2] || ಒಳಹರಿವು_shape[3] != output_shape[3]) {
        ಸ್ಥಿರಾಂಕ int32_t * op_params = context.get_output_op_params();
        ಸ್ಥಿರಾಂಕ int32_t ಬಹುಲಕ_flags = op_params != ಶೂನ್ಯಸೂಚಕ ? op_params[0] : 0;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ align_corners = (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ antialias = (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        ಆದರೆ (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ಸ್ಥಿರಾಂಕ size_t H_ಒಳ = ಒಳಹರಿವು_shape[2];
            ಸ್ಥಿರಾಂಕ size_t W_ಒಳ = ಒಳಹರಿವು_shape[3];
            ಸ್ಥಿರಾಂಕ size_t H_ಹೊರಹೋಗು = output_shape[2];
            ಸ್ಥಿರಾಂಕ size_t W_ಹೊರಹೋಗು = output_shape[3];

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<ದಶಮಾಂಶ> Wy_ದತ್ತಾಂಶ(H_ಹೊರಹೋಗು * H_ಒಳ, 0.0f);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf1 = static_cast<ದಶಮಾಂಶ>(H_ಹೊರಹೋಗು) / H_ಒಳ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ support1 = std::max(1.0f, 1.0f / sf1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ invscale1 = 1.0f / support1;

            ಗಾಗಿ (size_t i1 = 0; i1 < H_ಹೊರಹೋಗು; ++i1) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = (static_cast<ದಶಮಾಂಶ>(i1) + pixel_ಸ್ಥಾನಾಂತರ) / sf1;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ y_ಆರಂಭ = static_cast<int64_t>(y - support1 + pixel_ಸ್ಥಾನಾಂತರ);
                ಸ್ಥಿರಾಂಕ size_t y_min = y_ಆರಂಭ > 0 ? static_cast<size_t>(y_ಆರಂಭ) : 0;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ y_ಕೊನೆ = static_cast<int64_t>(y + support1 + pixel_ಸ್ಥಾನಾಂತರ);
                ಸ್ಥಿರಾಂಕ size_t y_max = y_ಕೊನೆ > 0 ? std::min<size_t>(static_cast<size_t>(y_ಕೊನೆ), H_ಒಳ) : 0;

                ದಶಮಾಂಶ ಒಟ್ಟು_ತೂಕ = 0.0f;
                ಗಾಗಿ (size_t sy = y_min; sy < y_max; ++sy) {
                    ದಶಮಾಂಶ diff = std::ನಿರಪೇಕ್ಷ((static_cast<ದಶಮಾಂಶ>(sy) - y + pixel_ಸ್ಥಾನಾಂತರ) * invscale1);
                    ದಶಮಾಂಶ ತೂಕ = std::max(1.0f - diff, 0.0f);
                    Wy_ದತ್ತಾಂಶ[i1 * H_ಒಳ + sy] = ತೂಕ;
                    ಒಟ್ಟು_ತೂಕ += ತೂಕ;
                }
                ಆದರೆ (ಒಟ್ಟು_ತೂಕ > 0.0f) {
                    ಗಾಗಿ (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_ದತ್ತಾಂಶ[i1 * H_ಒಳ + sy] /= ಒಟ್ಟು_ತೂಕ;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<ದಶಮಾಂಶ> Wx_ದತ್ತಾಂಶ(W_ಒಳ * W_ಹೊರಹೋಗು, 0.0f);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf0 = static_cast<ದಶಮಾಂಶ>(W_ಹೊರಹೋಗು) / W_ಒಳ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ support0 = std::max(1.0f, 1.0f / sf0);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ invscale0 = 1.0f / support0;

            ಗಾಗಿ (size_t i0 = 0; i0 < W_ಹೊರಹೋಗು; ++i0) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = (static_cast<ದಶಮಾಂಶ>(i0) + pixel_ಸ್ಥಾನಾಂತರ) / sf0;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ x_ಆರಂಭ = static_cast<int64_t>(x - support0 + pixel_ಸ್ಥಾನಾಂತರ);
                ಸ್ಥಿರಾಂಕ size_t x_min = x_ಆರಂಭ > 0 ? static_cast<size_t>(x_ಆರಂಭ) : 0;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ x_ಕೊನೆ = static_cast<int64_t>(x + support0 + pixel_ಸ್ಥಾನಾಂತರ);
                ಸ್ಥಿರಾಂಕ size_t x_max = x_ಕೊನೆ > 0 ? std::min<size_t>(static_cast<size_t>(x_ಕೊನೆ), W_ಒಳ) : 0;

                ದಶಮಾಂಶ ಒಟ್ಟು_ತೂಕ = 0.0f;
                ಗಾಗಿ (size_t sx = x_min; sx < x_max; ++sx) {
                    ದಶಮಾಂಶ diff = std::ನಿರಪೇಕ್ಷ((static_cast<ದಶಮಾಂಶ>(sx) - x + pixel_ಸ್ಥಾನಾಂತರ) * invscale0);
                    ದಶಮಾಂಶ ತೂಕ = std::max(1.0f - diff, 0.0f);
                    Wx_ದತ್ತಾಂಶ[sx * W_ಹೊರಹೋಗು + i0] = ತೂಕ;
                    ಒಟ್ಟು_ತೂಕ += ತೂಕ;
                }
                ಆದರೆ (ಒಟ್ಟು_ತೂಕ > 0.0f) {
                    ಗಾಗಿ (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_ದತ್ತಾಂಶ[sx * W_ಹೊರಹೋಗು + i0] /= ಒಟ್ಟು_ತೂಕ;
                    }
                }
            }

            ಸ್ವಯಂಚಾಲಿತ Wy_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, {H_ಹೊರಹೋಗು, H_ಒಳ}, Wy_ದತ್ತಾಂಶ);
            ಸ್ವಯಂಚಾಲಿತ Wx_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, {W_ಒಳ, W_ಹೊರಹೋಗು}, Wx_ದತ್ತಾಂಶ);

            ಸ್ವಯಂಚಾಲಿತ res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ಗಂಟು, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ಗಂಟು);
        } ಇಲ್ಲದಿದ್ದರೆ {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_ಬಹುಲಕ = Interpolate::Shape_Calc_ಬಹುಲಕ::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_ಕೊನೆ = {0, 0, 0, 0};

            ಆಯ್ಕೆ (op_case) {
            ಸ್ಥಿತಿ 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_ಬಹುಲಕ::ಹತ್ತಿರದ;
                attrs.ಹತ್ತಿರದ_ಬಹುಲಕ = Interpolate::ಹತ್ತಿರದ_ಬಹುಲಕ_2::FLOOR;
                attrs.coordinate_transformation_ಬಹುಲಕ = Interpolate::Coordinate_Transform_ಬಹುಲಕ::ASYMMETRIC;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_ಬಹುಲಕ::ರೇಖೀಯ;
                attrs.coordinate_transformation_ಬಹುಲಕ = align_corners
                    ? Interpolate::Coordinate_Transform_ಬಹುಲಕ::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_ಬಹುಲಕ::HALF_PIXEL;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_ಬಹುಲಕ::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_ಬಹುಲಕ = align_corners
                    ? Interpolate::Coordinate_Transform_ಬಹುಲಕ::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_ಬಹುಲಕ::HALF_PIXEL;
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಸುಳ್ಳು, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> ಗುರಿ_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<ದಶಮಾಂಶ> scales_vec = {
                static_cast<ದಶಮಾಂಶ>(output_shape[2]) / static_cast<ದಶಮಾಂಶ>(ಒಳಹರಿವು_shape[2]),
                static_cast<ದಶಮಾಂಶ>(output_shape[3]) / static_cast<ದಶಮಾಂಶ>(ಒಳಹರಿವು_shape[3]),
            };

            ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_shape_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {2}, ಗುರಿ_shape_vec);
            ಸ್ವಯಂಚಾಲಿತ scales_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            ಸ್ವಯಂಚಾಲಿತ axes_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, ಗುರಿ_shape_ಗಂಟು, scales_ಗಂಟು, axes_ಗಂಟು, attrs);
        }
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
