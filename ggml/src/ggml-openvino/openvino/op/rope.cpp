#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/split.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/op/variadic_split.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ویکټور translate_rope(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 2, 3);

    صحيح op_case = context.get_op_case();

    ov::Output<Node> res;

    خپلکاری معلومات_غوټه = process_view_ننوت_new(context, 0).get_غوټه_shared_ptr();
    خپلکاری output_shape = context.get_output_shape().منزل_shape();
    int32_t * op_params = context.get_output_op_params();
    ثابت صحيح mode = op_case;
    ثابت int64_t سر_dim = static_cast<int64_t>(output_shape[3]);
    ثابت int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ثابت int64_t n_dims = configured_n_dims == 0 ? سر_dim : configured_n_dims;
    ثابت int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr صحيح TYPE_نورمال = 0;
    constexpr صحيح TYPE_NEOX = 1;
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    Output<Node> cos_theta_غوټه;
    Output<Node> sin_theta_غوټه;
    إذا (context.has_ننوت("rope_cos")) {
        cos_theta_غوټه = context.get_ننوت("rope_cos");
        sin_theta_غوټه = context.get_ننوت("rope_sin");
    } وإلا {
        std::string cache_کلی = "rope_sin_cos";
        لكل (صحيح i = 0; i < 15; i++) {
            cache_کلی += "_" + std::to_string(op_params[i]);
        }
        إذا (context.get_ننوت_اندازه() == 3) {
            cache_کلی += "_ff_" + context.get_ننوت_نومونه()[2];
        }
        إذا (context.has_ننوت(cache_کلی + "_cos")) {
            cos_theta_غوټه = context.get_ننوت(cache_کلی + "_cos");
            sin_theta_غوټه = context.get_ننوت(cache_کلی + "_sin");
        } وإلا {
            خپلکاری inp_pos = context.get_ننوت(1).get_غوټه_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_وزن;
            إذا (context.get_ننوت_اندازه() == 3) {
                rope_freqs_وزن = context.get_ننوت(2).get_غوټه_shared_ptr();
            }
            خپلکاری sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_وزن, mode, خطأ, سر_dim);
            sin_theta_غوټه = sin_cos.لومړی;
            cos_theta_غوټه = sin_cos.second;
            context.put_shared(cache_کلی + "_cos", cos_theta_غوټه);
            context.put_shared(cache_کلی + "_sin", sin_theta_غوټه);
        }
    }

    خپلکاری output_type = context.get_output_type();
    إذا (معلومات_غوټه->get_element_type() != ov::element::f32) {
        معلومات_غوټه = std::make_shared<ov::op::v0::Convert>(معلومات_غوټه, ov::element::f32);
    }

    ثابت int64_t ټول_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(n_dims > 0 && ټول_rope_dims + n_offs <= سر_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    إذا (context.is_stateful()) {
        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        خپلکاری r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        معلومات_غوټه = std::make_shared<ov::op::v1::Reshape>(معلومات_غوټه, r4_shape, خطأ);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    إذا (mode == TYPE_نورمال) {
        خپلکاری axis_وروستی = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خپلکاری ګام_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        ثابت int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ثابت int64_t half = n_dims / 2;
        خپلکاری rot_پیل = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        خپلکاری rot_پای_ټکی = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        خپلکاری rot_معلومات = std::make_shared<ov::op::v8::ټوټه>(معلومات_غوټه, rot_پیل, rot_پای_ټکی, ګام_one, axis_وروستی);

        خپلکاری neg_one_f = ov::op::v0::Constant::create(معلومات_غوټه->get_element_type(), ov::Shape{}, {-1.0f});

        خپلکاری paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        خپلکاری x_paired = std::make_shared<ov::op::v1::Reshape>(rot_معلومات, paired_shape, خطأ);

        خپلکاری وېشل_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        خپلکاری معلومات_وېشل = std::make_shared<ov::op::v1::وېشل_2>(x_paired, وېشل_axis, 2);
        Output<Node> x0 = معلومات_وېشل->outputs()[0];
        Output<Node> x1 = معلومات_وېشل->outputs()[1];

        خپلکاری x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        خپلکاری x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{x1_neg, x0}, -1);

        خپلکاری flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        خپلکاری x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, خطأ);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        خپلکاری expand_cos_sin = [&](ثابت Output<Node>& cs) {
            خپلکاری cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            خپلکاری bcast_موخه = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            خپلکاری bcast = std::make_shared<ov::op::v3::خپرونه>(
                cs_unsq, bcast_موخه, ov::op::خپرونه_Type::BIDIRECTIONAL);
            خپلکاری flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            أرجع std::make_shared<ov::op::v1::Reshape>(bcast, flat, ریښتیا);
        };
        Output<Node> cos_ډک = expand_cos_sin(cos_theta_غوټه);
        Output<Node> sin_ډک = expand_cos_sin(sin_theta_غوټه);

        خپلکاری y1 = std::make_shared<ov::op::v1::Multiply>(rot_معلومات, cos_ډک);
        خپلکاری y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_ډک);
        خپلکاری rotated = std::make_shared<ov::op::v1::جمع_کړه>(y1, y2);

        ov::Output_ویکټور concat_parts;
        إذا (n_offs > 0) {
            خپلکاری سر_پیل = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خپلکاری سر_پای_ټکی = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خپلکاری سر = std::make_shared<ov::op::v8::ټوټه>(معلومات_غوټه, سر_پیل, سر_پای_ټکی, ګام_one, axis_وروستی);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        إذا (n_offs + n_dims < سر_dim) {
            خپلکاری لکۍ_پیل = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            خپلکاری لکۍ_پای_ټکی = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خپلکاری لکۍ = std::make_shared<ov::op::v8::ټوټه>(معلومات_غوټه, لکۍ_پیل, لکۍ_پای_ټکی, ګام_one, axis_وروستی);
            concat_parts.push_back(لکۍ);
        }
        إذا (concat_parts.size() == 1) {
            res = rotated;
        } وإلا {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } وإلا إذا (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ثابت int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ثابت int64_t rot_dims = 2 * half;

        إذا (mode != TYPE_NEOX) {
            خپلکاری cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_غوټه = std::make_shared<ov::op::v1::Reshape>(cos_theta_غوټه, cos_sin_shape, ریښتیا);
            sin_theta_غوټه = std::make_shared<ov::op::v1::Reshape>(sin_theta_غوټه, cos_sin_shape, ریښتیا);
        }

        خپلکاری axis_وروستی = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خپلکاری ګام_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_معلومات = معلومات_غوټه;
        إذا (n_offs > 0 || n_offs + rot_dims < سر_dim) {
            خپلکاری rot_پیل = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خپلکاری rot_پای_ټکی = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_معلومات = std::make_shared<ov::op::v8::ټوټه>(معلومات_غوټه, rot_پیل, rot_پای_ټکی, ګام_one, axis_وروستی);
        }

        خپلکاری neg_one_f = ov::op::v0::Constant::create(معلومات_غوټه->get_element_type(), ov::Shape{}, {-1.0f});

        خپلکاری وېشل_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        خپلکاری وېشل_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        خپلکاری معلومات_وېشل = std::make_shared<ov::op::v1::Variadic_وېشل>(rot_معلومات, وېشل_axis, وېشل_lengths);
        Output<Node> x1 = معلومات_وېشل->outputs()[0];
        Output<Node> x2 = معلومات_وېشل->outputs()[1];

        خپلکاری x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        خپلکاری x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{x2_neg, x1}, -1);

        خپلکاری cos_ډک = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{cos_theta_غوټه, cos_theta_غوټه}, -1);
        خپلکاری sin_ډک = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{sin_theta_غوټه, sin_theta_غوټه}, -1);

        خپلکاری y1 = std::make_shared<ov::op::v1::Multiply>(rot_معلومات, cos_ډک);
        خپلکاری y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_ډک);
        خپلکاری rotated = std::make_shared<ov::op::v1::جمع_کړه>(y1, y2);

        ov::Output_ویکټور concat_parts;
        إذا (n_offs > 0) {
            خپلکاری سر_پیل = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خپلکاری سر_پای_ټکی = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خپلکاری سر = std::make_shared<ov::op::v8::ټوټه>(معلومات_غوټه, سر_پیل, سر_پای_ټکی, ګام_one, axis_وروستی);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        إذا (n_offs + rot_dims < سر_dim) {
            خپلکاری لکۍ_پیل = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            خپلکاری لکۍ_پای_ټکی = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خپلکاری لکۍ = std::make_shared<ov::op::v8::ټوټه>(معلومات_غوټه, لکۍ_پیل, لکۍ_پای_ټکی, ګام_one, axis_وروستی);
            concat_parts.push_back(لکۍ);
        }
        إذا (concat_parts.size() == 1) {
            res = rotated;
        } وإلا {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
