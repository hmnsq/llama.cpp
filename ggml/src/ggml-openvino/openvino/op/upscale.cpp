#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/interpolate.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_upscale(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    उपयुज्य Interpolate = ov::op::v4::Interpolate;
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);

    ध्रुव स्वचालित निवेशः_shape = context.get_निवेशः_shape(0).गन्तव्यम्_shape();
    ध्रुव स्वचालित output_shape = context.get_output_shape().गन्तव्यम्_shape();

    यदि (निवेशः_shape == output_shape) {
        प्रत्यागम rename_outputs_with_suffix({निवेशः}, context.get_name());
    }

    ov::Output<ov::Node> res = निवेशः;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    प्रति (size_t axis = 0; axis < 2; ++axis) {
        ध्रुव size_t अन्तर्_dim = निवेशः_shape[axis];
        ध्रुव size_t बहिर्गमनम्_dim = output_shape[axis];
        यदि (अन्तर्_dim != बहिर्गमनम्_dim) {
            std::vector<int64_t> indices(बहिर्गमनम्_dim);
            प्रति (size_t i = 0; i < बहिर्गमनम्_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * अन्तर्_dim) / बहिर्गमनम्_dim);
            }
            स्वचालित indices_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            स्वचालित axis_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ग्रन्थिः, axis_ग्रन्थिः);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    यदि (निवेशः_shape[2] != output_shape[2] || निवेशः_shape[3] != output_shape[3]) {
        ध्रुव int32_t * op_params = context.get_output_op_params();
        ध्रुव int32_t बहुलकम्_flags = op_params != शून्यसूचकः ? op_params[0] : 0;

        ध्रुव पूर्णाङ्क op_case = context.get_op_case();
        ध्रुव तार्किक align_corners = (बहुलकम्_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ध्रुव तार्किक antialias = (बहुलकम्_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        यदि (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ध्रुव size_t H_अन्तर् = निवेशः_shape[2];
            ध्रुव size_t W_अन्तर् = निवेशः_shape[3];
            ध्रुव size_t H_बहिर्गमनम् = output_shape[2];
            ध्रुव size_t W_बहिर्गमनम् = output_shape[3];

            ध्रुव भिन्न pixel_विस्थापनम् = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<भिन्न> Wy_दत्त(H_बहिर्गमनम् * H_अन्तर्, 0.0f);
            ध्रुव भिन्न sf1 = static_cast<भिन्न>(H_बहिर्गमनम्) / H_अन्तर्;
            ध्रुव भिन्न support1 = std::max(1.0f, 1.0f / sf1);
            ध्रुव भिन्न invscale1 = 1.0f / support1;

            प्रति (size_t i1 = 0; i1 < H_बहिर्गमनम्; ++i1) {
                ध्रुव भिन्न y = (static_cast<भिन्न>(i1) + pixel_विस्थापनम्) / sf1;
                ध्रुव स्वचालित y_start = static_cast<int64_t>(y - support1 + pixel_विस्थापनम्);
                ध्रुव size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                ध्रुव स्वचालित y_अन्तः = static_cast<int64_t>(y + support1 + pixel_विस्थापनम्);
                ध्रुव size_t y_max = y_अन्तः > 0 ? std::min<size_t>(static_cast<size_t>(y_अन्तः), H_अन्तर्) : 0;

                भिन्न योग_weight = 0.0f;
                प्रति (size_t sy = y_min; sy < y_max; ++sy) {
                    भिन्न diff = std::निरपेक्षम्((static_cast<भिन्न>(sy) - y + pixel_विस्थापनम्) * invscale1);
                    भिन्न weight = std::max(1.0f - diff, 0.0f);
                    Wy_दत्त[i1 * H_अन्तर् + sy] = weight;
                    योग_weight += weight;
                }
                यदि (योग_weight > 0.0f) {
                    प्रति (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_दत्त[i1 * H_अन्तर् + sy] /= योग_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<भिन्न> Wx_दत्त(W_अन्तर् * W_बहिर्गमनम्, 0.0f);
            ध्रुव भिन्न sf0 = static_cast<भिन्न>(W_बहिर्गमनम्) / W_अन्तर्;
            ध्रुव भिन्न support0 = std::max(1.0f, 1.0f / sf0);
            ध्रुव भिन्न invscale0 = 1.0f / support0;

            प्रति (size_t i0 = 0; i0 < W_बहिर्गमनम्; ++i0) {
                ध्रुव भिन्न x = (static_cast<भिन्न>(i0) + pixel_विस्थापनम्) / sf0;
                ध्रुव स्वचालित x_start = static_cast<int64_t>(x - support0 + pixel_विस्थापनम्);
                ध्रुव size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                ध्रुव स्वचालित x_अन्तः = static_cast<int64_t>(x + support0 + pixel_विस्थापनम्);
                ध्रुव size_t x_max = x_अन्तः > 0 ? std::min<size_t>(static_cast<size_t>(x_अन्तः), W_अन्तर्) : 0;

                भिन्न योग_weight = 0.0f;
                प्रति (size_t sx = x_min; sx < x_max; ++sx) {
                    भिन्न diff = std::निरपेक्षम्((static_cast<भिन्न>(sx) - x + pixel_विस्थापनम्) * invscale0);
                    भिन्न weight = std::max(1.0f - diff, 0.0f);
                    Wx_दत्त[sx * W_बहिर्गमनम् + i0] = weight;
                    योग_weight += weight;
                }
                यदि (योग_weight > 0.0f) {
                    प्रति (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_दत्त[sx * W_बहिर्गमनम् + i0] /= योग_weight;
                    }
                }
            }

            स्वचालित Wy_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, {H_बहिर्गमनम्, H_अन्तर्}, Wy_दत्त);
            स्वचालित Wx_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, {W_अन्तर्, W_बहिर्गमनम्}, Wx_दत्त);

            स्वचालित res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ग्रन्थिः, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ग्रन्थिः);
        } अन्यथा {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_बहुलकम् = Interpolate::Shape_Calc_बहुलकम्::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_अन्तः = {0, 0, 0, 0};

            विकल्प (op_case) {
            पक्ष 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_बहुलकम्::निकटतमः;
                attrs.निकटतमः_बहुलकम् = Interpolate::निकटतमः_बहुलकम्_2::FLOOR;
                attrs.coordinate_transformation_बहुलकम् = Interpolate::Coordinate_Transform_बहुलकम्::ASYMMETRIC;
                भङ्ग;
            पक्ष 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_बहुलकम्::रैखिकम्;
                attrs.coordinate_transformation_बहुलकम् = align_corners
                    ? Interpolate::Coordinate_Transform_बहुलकम्::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_बहुलकम्::HALF_PIXEL;
                भङ्ग;
            पक्ष 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_बहुलकम्::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_बहुलकम् = align_corners
                    ? Interpolate::Coordinate_Transform_बहुलकम्::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_बहुलकम्::HALF_PIXEL;
                भङ्ग;
            स्वभाव:
                FRONT_अन्तः_OP_CONVERSION_CHECK(असत्य, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<भिन्न> scales_vec = {
                static_cast<भिन्न>(output_shape[2]) / static_cast<भिन्न>(निवेशः_shape[2]),
                static_cast<भिन्न>(output_shape[3]) / static_cast<भिन्न>(निवेशः_shape[3]),
            };

            स्वचालित target_shape_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            स्वचालित scales_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            स्वचालित axes_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_ग्रन्थिः, scales_ग्रन्थिः, axes_ग्रन्थिः, attrs);
        }
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
