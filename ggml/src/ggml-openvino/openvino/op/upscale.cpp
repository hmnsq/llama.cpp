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

Output_ویکټور translate_upscale(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    کارول Interpolate = ov::op::v4::Interpolate;
    خپلکاری ننوت = process_view_ننوت_new(context, 0);

    ثابت خپلکاری ننوت_shape = context.get_ننوت_shape(0).منزل_shape();
    ثابت خپلکاری output_shape = context.get_output_shape().منزل_shape();

    إذا (ننوت_shape == output_shape) {
        أرجع rename_outputs_with_suffix({ننوت}, context.get_name());
    }

    ov::Output<ov::Node> res = ننوت;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    لكل (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t دننه_dim = ننوت_shape[axis];
        ثابت size_t بهر_dim = output_shape[axis];
        إذا (دننه_dim != بهر_dim) {
            std::vector<int64_t> indices(بهر_dim);
            لكل (size_t i = 0; i < بهر_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * دننه_dim) / بهر_dim);
            }
            خپلکاری indices_غوټه = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            خپلکاری axis_غوټه = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_غوټه, axis_غوټه);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    إذا (ننوت_shape[2] != output_shape[2] || ننوت_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t ډېرپېښ_flags = op_params != تش_اشاره ? op_params[0] : 0;

        ثابت صحيح op_case = context.get_op_case();
        ثابت منطقي align_corners = (ډېرپېښ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقي antialias = (ډېرپېښ_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        إذا (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_دننه = ننوت_shape[2];
            ثابت size_t W_دننه = ننوت_shape[3];
            ثابت size_t H_بهر = output_shape[2];
            ثابت size_t W_بهر = output_shape[3];

            ثابت عائم pixel_دننه_واټن = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<عائم> Wy_معلومات(H_بهر * H_دننه, 0.0f);
            ثابت عائم sf1 = static_cast<عائم>(H_بهر) / H_دننه;
            ثابت عائم support1 = std::max(1.0f, 1.0f / sf1);
            ثابت عائم invscale1 = 1.0f / support1;

            لكل (size_t i1 = 0; i1 < H_بهر; ++i1) {
                ثابت عائم y = (static_cast<عائم>(i1) + pixel_دننه_واټن) / sf1;
                ثابت خپلکاری y_پیل = static_cast<int64_t>(y - support1 + pixel_دننه_واټن);
                ثابت size_t y_min = y_پیل > 0 ? static_cast<size_t>(y_پیل) : 0;
                ثابت خپلکاری y_پای_ټکی = static_cast<int64_t>(y + support1 + pixel_دننه_واټن);
                ثابت size_t y_max = y_پای_ټکی > 0 ? std::min<size_t>(static_cast<size_t>(y_پای_ټکی), H_دننه) : 0;

                عائم ټول_وزن = 0.0f;
                لكل (size_t sy = y_min; sy < y_max; ++sy) {
                    عائم diff = std::مطلق_قیمت((static_cast<عائم>(sy) - y + pixel_دننه_واټن) * invscale1);
                    عائم وزن = std::max(1.0f - diff, 0.0f);
                    Wy_معلومات[i1 * H_دننه + sy] = وزن;
                    ټول_وزن += وزن;
                }
                إذا (ټول_وزن > 0.0f) {
                    لكل (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_معلومات[i1 * H_دننه + sy] /= ټول_وزن;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<عائم> Wx_معلومات(W_دننه * W_بهر, 0.0f);
            ثابت عائم sf0 = static_cast<عائم>(W_بهر) / W_دننه;
            ثابت عائم support0 = std::max(1.0f, 1.0f / sf0);
            ثابت عائم invscale0 = 1.0f / support0;

            لكل (size_t i0 = 0; i0 < W_بهر; ++i0) {
                ثابت عائم x = (static_cast<عائم>(i0) + pixel_دننه_واټن) / sf0;
                ثابت خپلکاری x_پیل = static_cast<int64_t>(x - support0 + pixel_دننه_واټن);
                ثابت size_t x_min = x_پیل > 0 ? static_cast<size_t>(x_پیل) : 0;
                ثابت خپلکاری x_پای_ټکی = static_cast<int64_t>(x + support0 + pixel_دننه_واټن);
                ثابت size_t x_max = x_پای_ټکی > 0 ? std::min<size_t>(static_cast<size_t>(x_پای_ټکی), W_دننه) : 0;

                عائم ټول_وزن = 0.0f;
                لكل (size_t sx = x_min; sx < x_max; ++sx) {
                    عائم diff = std::مطلق_قیمت((static_cast<عائم>(sx) - x + pixel_دننه_واټن) * invscale0);
                    عائم وزن = std::max(1.0f - diff, 0.0f);
                    Wx_معلومات[sx * W_بهر + i0] = وزن;
                    ټول_وزن += وزن;
                }
                إذا (ټول_وزن > 0.0f) {
                    لكل (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_معلومات[sx * W_بهر + i0] /= ټول_وزن;
                    }
                }
            }

            خپلکاری Wy_غوټه = ov::op::v0::Constant::create(ov::element::f32, {H_بهر, H_دننه}, Wy_معلومات);
            خپلکاری Wx_غوټه = ov::op::v0::Constant::create(ov::element::f32, {W_دننه, W_بهر}, Wx_معلومات);

            خپلکاری res_y = std::make_shared<ov::op::v0::MatMul>(Wy_غوټه, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_غوټه);
        } وإلا {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_ډېرپېښ = Interpolate::Shape_Calc_ډېرپېښ::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_پای_ټکی = {0, 0, 0, 0};

            بدل (op_case) {
            حالة 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_ډېرپېښ::تر_ټولو_نږدې;
                attrs.تر_ټولو_نږدې_ډېرپېښ = Interpolate::تر_ټولو_نږدې_ډېرپېښ_2::FLOOR;
                attrs.coordinate_transformation_ډېرپېښ = Interpolate::Coordinate_Transform_ډېرپېښ::ASYMMETRIC;
                اكسر;
            حالة 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_ډېرپېښ::خطي;
                attrs.coordinate_transformation_ډېرپېښ = align_corners
                    ? Interpolate::Coordinate_Transform_ډېرپېښ::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_ډېرپېښ::HALF_PIXEL;
                اكسر;
            حالة 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_ډېرپېښ::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_ډېرپېښ = align_corners
                    ? Interpolate::Coordinate_Transform_ډېرپېښ::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_ډېرپېښ::HALF_PIXEL;
                اكسر;
            افتراضي:
                FRONT_پای_ټکی_OP_CONVERSION_CHECK(خطأ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> موخه_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<عائم> scales_vec = {
                static_cast<عائم>(output_shape[2]) / static_cast<عائم>(ننوت_shape[2]),
                static_cast<عائم>(output_shape[3]) / static_cast<عائم>(ننوت_shape[3]),
            };

            خپلکاری موخه_shape_غوټه = ov::op::v0::Constant::create(ov::element::i64, {2}, موخه_shape_vec);
            خپلکاری scales_غوټه = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            خپلکاری axes_غوټه = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, موخه_shape_غوټه, scales_غوټه, axes_غوټه, attrs);
        }
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
