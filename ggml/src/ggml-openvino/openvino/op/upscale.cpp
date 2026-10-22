#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/interpolate.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_upscale(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਵਰਤੋਂ Interpolate = ov::op::v4::Interpolate;
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
    ਅਚਲ ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();

    ਜੇ (ਨਿਵੇਸ਼_shape == output_shape) {
        ਵਾਪਸ rename_outputs_with_suffix({ਨਿਵੇਸ਼}, context.get_name());
    }

    ov::Output<ov::Node> res = ਨਿਵੇਸ਼;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    ਲਈ (size_t axis = 0; axis < 2; ++axis) {
        ਅਚਲ size_t ਵਿੱਚ_dim = ਨਿਵੇਸ਼_shape[axis];
        ਅਚਲ size_t ਬਾਹਰਲਾ_dim = output_shape[axis];
        ਜੇ (ਵਿੱਚ_dim != ਬਾਹਰਲਾ_dim) {
            std::vector<int64_t> indices(ਬਾਹਰਲਾ_dim);
            ਲਈ (size_t i = 0; i < ਬਾਹਰਲਾ_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * ਵਿੱਚ_dim) / ਬਾਹਰਲਾ_dim);
            }
            ਸਵੈਚਾਲਿਤ indices_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            ਸਵੈਚਾਲਿਤ axis_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ਗੰਢ, axis_ਗੰਢ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    ਜੇ (ਨਿਵੇਸ਼_shape[2] != output_shape[2] || ਨਿਵੇਸ਼_shape[3] != output_shape[3]) {
        ਅਚਲ int32_t * op_params = context.get_output_op_params();
        ਅਚਲ int32_t ਬਹੁਲਕ_flags = op_params != ਖਾਲੀ_ਸੂਚਕ ? op_params[0] : 0;

        ਅਚਲ ਪੂਰਨ op_case = context.get_op_case();
        ਅਚਲ ਤਾਰਕਿਕ align_corners = (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ਅਚਲ ਤਾਰਕਿਕ antialias = (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        ਜੇ (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ਅਚਲ size_t H_ਵਿੱਚ = ਨਿਵੇਸ਼_shape[2];
            ਅਚਲ size_t W_ਵਿੱਚ = ਨਿਵੇਸ਼_shape[3];
            ਅਚਲ size_t H_ਬਾਹਰਲਾ = output_shape[2];
            ਅਚਲ size_t W_ਬਾਹਰਲਾ = output_shape[3];

            ਅਚਲ ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<ਦਸ਼ਮਲਵ> Wy_ਅੰਕੜੇ(H_ਬਾਹਰਲਾ * H_ਵਿੱਚ, 0.0f);
            ਅਚਲ ਦਸ਼ਮਲਵ sf1 = static_cast<ਦਸ਼ਮਲਵ>(H_ਬਾਹਰਲਾ) / H_ਵਿੱਚ;
            ਅਚਲ ਦਸ਼ਮਲਵ support1 = std::max(1.0f, 1.0f / sf1);
            ਅਚਲ ਦਸ਼ਮਲਵ invscale1 = 1.0f / support1;

            ਲਈ (size_t i1 = 0; i1 < H_ਬਾਹਰਲਾ; ++i1) {
                ਅਚਲ ਦਸ਼ਮਲਵ y = (static_cast<ਦਸ਼ਮਲਵ>(i1) + pixel_ਵਿੱਥ) / sf1;
                ਅਚਲ ਸਵੈਚਾਲਿਤ y_ਸ਼ੁਰੂ = static_cast<int64_t>(y - support1 + pixel_ਵਿੱਥ);
                ਅਚਲ size_t y_min = y_ਸ਼ੁਰੂ > 0 ? static_cast<size_t>(y_ਸ਼ੁਰੂ) : 0;
                ਅਚਲ ਸਵੈਚਾਲਿਤ y_ਸਿਰਾ = static_cast<int64_t>(y + support1 + pixel_ਵਿੱਥ);
                ਅਚਲ size_t y_max = y_ਸਿਰਾ > 0 ? std::min<size_t>(static_cast<size_t>(y_ਸਿਰਾ), H_ਵਿੱਚ) : 0;

                ਦਸ਼ਮਲਵ ਕੁੱਲ_ਭਾਰ = 0.0f;
                ਲਈ (size_t sy = y_min; sy < y_max; ++sy) {
                    ਦਸ਼ਮਲਵ diff = std::ਨਿਰਪੇਖ((static_cast<ਦਸ਼ਮਲਵ>(sy) - y + pixel_ਵਿੱਥ) * invscale1);
                    ਦਸ਼ਮਲਵ ਭਾਰ = std::max(1.0f - diff, 0.0f);
                    Wy_ਅੰਕੜੇ[i1 * H_ਵਿੱਚ + sy] = ਭਾਰ;
                    ਕੁੱਲ_ਭਾਰ += ਭਾਰ;
                }
                ਜੇ (ਕੁੱਲ_ਭਾਰ > 0.0f) {
                    ਲਈ (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_ਅੰਕੜੇ[i1 * H_ਵਿੱਚ + sy] /= ਕੁੱਲ_ਭਾਰ;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<ਦਸ਼ਮਲਵ> Wx_ਅੰਕੜੇ(W_ਵਿੱਚ * W_ਬਾਹਰਲਾ, 0.0f);
            ਅਚਲ ਦਸ਼ਮਲਵ sf0 = static_cast<ਦਸ਼ਮਲਵ>(W_ਬਾਹਰਲਾ) / W_ਵਿੱਚ;
            ਅਚਲ ਦਸ਼ਮਲਵ support0 = std::max(1.0f, 1.0f / sf0);
            ਅਚਲ ਦਸ਼ਮਲਵ invscale0 = 1.0f / support0;

            ਲਈ (size_t i0 = 0; i0 < W_ਬਾਹਰਲਾ; ++i0) {
                ਅਚਲ ਦਸ਼ਮਲਵ x = (static_cast<ਦਸ਼ਮਲਵ>(i0) + pixel_ਵਿੱਥ) / sf0;
                ਅਚਲ ਸਵੈਚਾਲਿਤ x_ਸ਼ੁਰੂ = static_cast<int64_t>(x - support0 + pixel_ਵਿੱਥ);
                ਅਚਲ size_t x_min = x_ਸ਼ੁਰੂ > 0 ? static_cast<size_t>(x_ਸ਼ੁਰੂ) : 0;
                ਅਚਲ ਸਵੈਚਾਲਿਤ x_ਸਿਰਾ = static_cast<int64_t>(x + support0 + pixel_ਵਿੱਥ);
                ਅਚਲ size_t x_max = x_ਸਿਰਾ > 0 ? std::min<size_t>(static_cast<size_t>(x_ਸਿਰਾ), W_ਵਿੱਚ) : 0;

                ਦਸ਼ਮਲਵ ਕੁੱਲ_ਭਾਰ = 0.0f;
                ਲਈ (size_t sx = x_min; sx < x_max; ++sx) {
                    ਦਸ਼ਮਲਵ diff = std::ਨਿਰਪੇਖ((static_cast<ਦਸ਼ਮਲਵ>(sx) - x + pixel_ਵਿੱਥ) * invscale0);
                    ਦਸ਼ਮਲਵ ਭਾਰ = std::max(1.0f - diff, 0.0f);
                    Wx_ਅੰਕੜੇ[sx * W_ਬਾਹਰਲਾ + i0] = ਭਾਰ;
                    ਕੁੱਲ_ਭਾਰ += ਭਾਰ;
                }
                ਜੇ (ਕੁੱਲ_ਭਾਰ > 0.0f) {
                    ਲਈ (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_ਅੰਕੜੇ[sx * W_ਬਾਹਰਲਾ + i0] /= ਕੁੱਲ_ਭਾਰ;
                    }
                }
            }

            ਸਵੈਚਾਲਿਤ Wy_ਗੰਢ = ov::op::v0::Constant::create(ov::element::f32, {H_ਬਾਹਰਲਾ, H_ਵਿੱਚ}, Wy_ਅੰਕੜੇ);
            ਸਵੈਚਾਲਿਤ Wx_ਗੰਢ = ov::op::v0::Constant::create(ov::element::f32, {W_ਵਿੱਚ, W_ਬਾਹਰਲਾ}, Wx_ਅੰਕੜੇ);

            ਸਵੈਚਾਲਿਤ res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ਗੰਢ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ਗੰਢ);
        } ਨਹੀਂਤਾਂ {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_ਬਹੁਲਕ = Interpolate::Shape_Calc_ਬਹੁਲਕ::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_ਸਿਰਾ = {0, 0, 0, 0};

            ਚੋਣ (op_case) {
            ਹਾਲਤ 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_ਬਹੁਲਕ::ਸਭ_ਤੋਂ_ਨੇੜੇ;
                attrs.ਸਭ_ਤੋਂ_ਨੇੜੇ_ਬਹੁਲਕ = Interpolate::ਸਭ_ਤੋਂ_ਨੇੜੇ_ਬਹੁਲਕ_2::FLOOR;
                attrs.coordinate_transformation_ਬਹੁਲਕ = Interpolate::Coordinate_Transform_ਬਹੁਲਕ::ASYMMETRIC;
                ਰੋਕੋ;
            ਹਾਲਤ 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_ਬਹੁਲਕ::ਰੇਖਿਕ;
                attrs.coordinate_transformation_ਬਹੁਲਕ = align_corners
                    ? Interpolate::Coordinate_Transform_ਬਹੁਲਕ::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_ਬਹੁਲਕ::HALF_PIXEL;
                ਰੋਕੋ;
            ਹਾਲਤ 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_ਬਹੁਲਕ::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_ਬਹੁਲਕ = align_corners
                    ? Interpolate::Coordinate_Transform_ਬਹੁਲਕ::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_ਬਹੁਲਕ::HALF_PIXEL;
                ਰੋਕੋ;
            ਮੂਲ:
                FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਝੂਠ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> ਨਿਸ਼ਾਨਾ_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<ਦਸ਼ਮਲਵ> scales_vec = {
                static_cast<ਦਸ਼ਮਲਵ>(output_shape[2]) / static_cast<ਦਸ਼ਮਲਵ>(ਨਿਵੇਸ਼_shape[2]),
                static_cast<ਦਸ਼ਮਲਵ>(output_shape[3]) / static_cast<ਦਸ਼ਮਲਵ>(ਨਿਵੇਸ਼_shape[3]),
            };

            ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_shape_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {2}, ਨਿਸ਼ਾਨਾ_shape_vec);
            ਸਵੈਚਾਲਿਤ scales_ਗੰਢ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            ਸਵੈਚਾਲਿਤ axes_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, ਨਿਸ਼ਾਨਾ_shape_ਗੰਢ, scales_ਗੰਢ, axes_ਗੰਢ, attrs);
        }
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
