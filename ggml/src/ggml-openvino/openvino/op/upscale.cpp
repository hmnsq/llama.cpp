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

Output_سمتیہ translate_upscale(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    ورتوں Interpolate = ov::op::v4::Interpolate;
    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ثابت خودکار داخلہ_shape = context.get_داخلہ_shape(0).منزل_shape();
    ثابت خودکار output_shape = context.get_output_shape().منزل_shape();

    إذا (داخلہ_shape == output_shape) {
        أرجع rename_outputs_with_suffix({داخلہ}, context.get_name());
    }

    ov::Output<ov::Node> res = داخلہ;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    لكل (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t وچ_dim = داخلہ_shape[axis];
        ثابت size_t باہرلا_dim = output_shape[axis];
        إذا (وچ_dim != باہرلا_dim) {
            std::vector<int64_t> indices(باہرلا_dim);
            لكل (size_t i = 0; i < باہرلا_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * وچ_dim) / باہرلا_dim);
            }
            خودکار indices_گرہ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            خودکار axis_گرہ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_گرہ, axis_گرہ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    إذا (داخلہ_shape[2] != output_shape[2] || داخلہ_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t منوال_flags = op_params != خالی_سوچک ? op_params[0] : 0;

        ثابت صحيح op_case = context.get_op_case();
        ثابت منطقي align_corners = (منوال_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقي antialias = (منوال_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        إذا (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_وچ = داخلہ_shape[2];
            ثابت size_t W_وچ = داخلہ_shape[3];
            ثابت size_t H_باہرلا = output_shape[2];
            ثابت size_t W_باہرلا = output_shape[3];

            ثابت عائم pixel_وتھ = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<عائم> Wy_انکڑے(H_باہرلا * H_وچ, 0.0f);
            ثابت عائم sf1 = static_cast<عائم>(H_باہرلا) / H_وچ;
            ثابت عائم support1 = std::max(1.0f, 1.0f / sf1);
            ثابت عائم invscale1 = 1.0f / support1;

            لكل (size_t i1 = 0; i1 < H_باہرلا; ++i1) {
                ثابت عائم y = (static_cast<عائم>(i1) + pixel_وتھ) / sf1;
                ثابت خودکار y_شروع = static_cast<int64_t>(y - support1 + pixel_وتھ);
                ثابت size_t y_min = y_شروع > 0 ? static_cast<size_t>(y_شروع) : 0;
                ثابت خودکار y_سرا = static_cast<int64_t>(y + support1 + pixel_وتھ);
                ثابت size_t y_max = y_سرا > 0 ? std::min<size_t>(static_cast<size_t>(y_سرا), H_وچ) : 0;

                عائم کل_بھار = 0.0f;
                لكل (size_t sy = y_min; sy < y_max; ++sy) {
                    عائم diff = std::مطلق_قدر((static_cast<عائم>(sy) - y + pixel_وتھ) * invscale1);
                    عائم بھار = std::max(1.0f - diff, 0.0f);
                    Wy_انکڑے[i1 * H_وچ + sy] = بھار;
                    کل_بھار += بھار;
                }
                إذا (کل_بھار > 0.0f) {
                    لكل (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_انکڑے[i1 * H_وچ + sy] /= کل_بھار;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<عائم> Wx_انکڑے(W_وچ * W_باہرلا, 0.0f);
            ثابت عائم sf0 = static_cast<عائم>(W_باہرلا) / W_وچ;
            ثابت عائم support0 = std::max(1.0f, 1.0f / sf0);
            ثابت عائم invscale0 = 1.0f / support0;

            لكل (size_t i0 = 0; i0 < W_باہرلا; ++i0) {
                ثابت عائم x = (static_cast<عائم>(i0) + pixel_وتھ) / sf0;
                ثابت خودکار x_شروع = static_cast<int64_t>(x - support0 + pixel_وتھ);
                ثابت size_t x_min = x_شروع > 0 ? static_cast<size_t>(x_شروع) : 0;
                ثابت خودکار x_سرا = static_cast<int64_t>(x + support0 + pixel_وتھ);
                ثابت size_t x_max = x_سرا > 0 ? std::min<size_t>(static_cast<size_t>(x_سرا), W_وچ) : 0;

                عائم کل_بھار = 0.0f;
                لكل (size_t sx = x_min; sx < x_max; ++sx) {
                    عائم diff = std::مطلق_قدر((static_cast<عائم>(sx) - x + pixel_وتھ) * invscale0);
                    عائم بھار = std::max(1.0f - diff, 0.0f);
                    Wx_انکڑے[sx * W_باہرلا + i0] = بھار;
                    کل_بھار += بھار;
                }
                إذا (کل_بھار > 0.0f) {
                    لكل (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_انکڑے[sx * W_باہرلا + i0] /= کل_بھار;
                    }
                }
            }

            خودکار Wy_گرہ = ov::op::v0::Constant::create(ov::element::f32, {H_باہرلا, H_وچ}, Wy_انکڑے);
            خودکار Wx_گرہ = ov::op::v0::Constant::create(ov::element::f32, {W_وچ, W_باہرلا}, Wx_انکڑے);

            خودکار res_y = std::make_shared<ov::op::v0::MatMul>(Wy_گرہ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_گرہ);
        } وإلا {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_منوال = Interpolate::Shape_Calc_منوال::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_سرا = {0, 0, 0, 0};

            بدل (op_case) {
            حالة 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_منوال::سبھ_توں_نیڑے;
                attrs.سبھ_توں_نیڑے_منوال = Interpolate::سبھ_توں_نیڑے_منوال_2::FLOOR;
                attrs.coordinate_transformation_منوال = Interpolate::Coordinate_Transform_منوال::ASYMMETRIC;
                اكسر;
            حالة 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_منوال::خطی;
                attrs.coordinate_transformation_منوال = align_corners
                    ? Interpolate::Coordinate_Transform_منوال::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_منوال::HALF_PIXEL;
                اكسر;
            حالة 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_منوال::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_منوال = align_corners
                    ? Interpolate::Coordinate_Transform_منوال::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_منوال::HALF_PIXEL;
                اكسر;
            افتراضي:
                FRONT_سرا_OP_CONVERSION_CHECK(خطأ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> نشانہ_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<عائم> scales_vec = {
                static_cast<عائم>(output_shape[2]) / static_cast<عائم>(داخلہ_shape[2]),
                static_cast<عائم>(output_shape[3]) / static_cast<عائم>(داخلہ_shape[3]),
            };

            خودکار نشانہ_shape_گرہ = ov::op::v0::Constant::create(ov::element::i64, {2}, نشانہ_shape_vec);
            خودکار scales_گرہ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            خودکار axes_گرہ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, نشانہ_shape_گرہ, scales_گرہ, axes_گرہ, attrs);
        }
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
