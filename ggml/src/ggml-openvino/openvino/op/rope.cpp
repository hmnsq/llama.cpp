#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstdint>
#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/split.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/op/variadic_split.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_rope(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 3);

    صحیح op_case = context.get_op_case();

    ov::Output<Node> res;

    خودکار ڈیٹا_گرہ = process_view_داخلہ_new(context, 0).get_گرہ_shared_ptr();
    خودکار output_shape = context.get_output_shape().مقصد_shape();
    int32_t * op_params = context.get_output_op_params();
    ثابت صحیح mode = op_case;
    ثابت int64_t سر_dim = static_cast<int64_t>(output_shape[3]);
    ثابت int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ثابت int64_t n_dims = configured_n_dims == 0 ? سر_dim : configured_n_dims;
    ثابت int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr صحیح TYPE_معمول = 0;
    constexpr صحیح TYPE_NEOX = 1;
    constexpr صحیح TYPE_IMROPE = 2;
    constexpr صحیح TYPE_VISION = 3;
    constexpr صحیح TYPE_MROPE = 4;

    Output<Node> cos_theta_گرہ;
    Output<Node> sin_theta_گرہ;
    اگر (context.has_داخلہ("rope_cos")) {
        cos_theta_گرہ = context.get_داخلہ("rope_cos");
        sin_theta_گرہ = context.get_داخلہ("rope_sin");
    } ورنہ {
        std::string cache_کلید = "rope_sin_cos";
        برائے (صحیح i = 0; i < 15; i++) {
            cache_کلید += "_" + std::to_string(op_params[i]);
        }
        اگر (context.get_داخلہ_سائز() == 3) {
            cache_کلید += "_ff_" + context.get_داخلہ_نام()[2];
        }
        اگر (context.has_داخلہ(cache_کلید + "_cos")) {
            cos_theta_گرہ = context.get_داخلہ(cache_کلید + "_cos");
            sin_theta_گرہ = context.get_داخلہ(cache_کلید + "_sin");
        } ورنہ {
            خودکار inp_pos = context.get_داخلہ(1).get_گرہ_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_وزن;
            اگر (context.get_داخلہ_سائز() == 3) {
                rope_freqs_وزن = context.get_داخلہ(2).get_گرہ_shared_ptr();
            }
            خودکار sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_وزن, mode, جھوٹ, سر_dim);
            sin_theta_گرہ = sin_cos.پہلا;
            cos_theta_گرہ = sin_cos.second;
            context.put_shared(cache_کلید + "_cos", cos_theta_گرہ);
            context.put_shared(cache_کلید + "_sin", sin_theta_گرہ);
        }
    }

    خودکار output_type = context.get_output_type();
    اگر (ڈیٹا_گرہ->get_element_type() != ov::element::f32) {
        ڈیٹا_گرہ = std::make_shared<ov::op::v0::Convert>(ڈیٹا_گرہ, ov::element::f32);
    }

    ثابت int64_t کل_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_انتہا_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_انتہا_OP_CONVERSION_CHECK(n_dims > 0 && کل_rope_dims + n_offs <= سر_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    اگر (context.is_stateful()) {
        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        خودکار r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        ڈیٹا_گرہ = std::make_shared<ov::op::v1::Reshape>(ڈیٹا_گرہ, r4_shape, جھوٹ);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    اگر (mode == TYPE_معمول) {
        خودکار axis_آخری = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار قدم_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        ثابت int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ثابت int64_t half = n_dims / 2;
        خودکار rot_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        خودکار rot_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        خودکار rot_ڈیٹا = std::make_shared<ov::op::v8::قاش>(ڈیٹا_گرہ, rot_آغاز, rot_انتہا, قدم_one, axis_آخری);

        خودکار neg_one_f = ov::op::v0::Constant::create(ڈیٹا_گرہ->get_element_type(), ov::Shape{}, {-1.0f});

        خودکار paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        خودکار x_paired = std::make_shared<ov::op::v1::Reshape>(rot_ڈیٹا, paired_shape, جھوٹ);

        خودکار بانٹو_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        خودکار ڈیٹا_بانٹو = std::make_shared<ov::op::v1::بانٹو_2>(x_paired, بانٹو_axis, 2);
        Output<Node> x0 = ڈیٹا_بانٹو->outputs()[0];
        Output<Node> x1 = ڈیٹا_بانٹو->outputs()[1];

        خودکار x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        خودکار x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{x1_neg, x0}, -1);

        خودکار flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        خودکار x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, جھوٹ);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        خودکار expand_cos_sin = [&](ثابت Output<Node>& cs) {
            خودکار cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            خودکار bcast_ہدف = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            خودکار bcast = std::make_shared<ov::op::v3::نشر>(
                cs_unsq, bcast_ہدف, ov::op::نشر_Type::BIDIRECTIONAL);
            خودکار flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            واپس std::make_shared<ov::op::v1::Reshape>(bcast, flat, سچ);
        };
        Output<Node> cos_بھرا = expand_cos_sin(cos_theta_گرہ);
        Output<Node> sin_بھرا = expand_cos_sin(sin_theta_گرہ);

        خودکار y1 = std::make_shared<ov::op::v1::Multiply>(rot_ڈیٹا, cos_بھرا);
        خودکار y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_بھرا);
        خودکار rotated = std::make_shared<ov::op::v1::جوڑو>(y1, y2);

        ov::Output_سمتیہ concat_parts;
        اگر (n_offs > 0) {
            خودکار سر_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودکار سر_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودکار سر = std::make_shared<ov::op::v8::قاش>(ڈیٹا_گرہ, سر_آغاز, سر_انتہا, قدم_one, axis_آخری);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        اگر (n_offs + n_dims < سر_dim) {
            خودکار دم_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            خودکار دم_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خودکار دم = std::make_shared<ov::op::v8::قاش>(ڈیٹا_گرہ, دم_آغاز, دم_انتہا, قدم_one, axis_آخری);
            concat_parts.push_back(دم);
        }
        اگر (concat_parts.size() == 1) {
            res = rotated;
        } ورنہ {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } ورنہ اگر (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ثابت int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ثابت int64_t rot_dims = 2 * half;

        اگر (mode != TYPE_NEOX) {
            خودکار cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_گرہ = std::make_shared<ov::op::v1::Reshape>(cos_theta_گرہ, cos_sin_shape, سچ);
            sin_theta_گرہ = std::make_shared<ov::op::v1::Reshape>(sin_theta_گرہ, cos_sin_shape, سچ);
        }

        خودکار axis_آخری = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار قدم_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_ڈیٹا = ڈیٹا_گرہ;
        اگر (n_offs > 0 || n_offs + rot_dims < سر_dim) {
            خودکار rot_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودکار rot_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_ڈیٹا = std::make_shared<ov::op::v8::قاش>(ڈیٹا_گرہ, rot_آغاز, rot_انتہا, قدم_one, axis_آخری);
        }

        خودکار neg_one_f = ov::op::v0::Constant::create(ڈیٹا_گرہ->get_element_type(), ov::Shape{}, {-1.0f});

        خودکار بانٹو_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        خودکار بانٹو_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        خودکار ڈیٹا_بانٹو = std::make_shared<ov::op::v1::Variadic_بانٹو>(rot_ڈیٹا, بانٹو_axis, بانٹو_lengths);
        Output<Node> x1 = ڈیٹا_بانٹو->outputs()[0];
        Output<Node> x2 = ڈیٹا_بانٹو->outputs()[1];

        خودکار x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        خودکار x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{x2_neg, x1}, -1);

        خودکار cos_بھرا = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{cos_theta_گرہ, cos_theta_گرہ}, -1);
        خودکار sin_بھرا = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{sin_theta_گرہ, sin_theta_گرہ}, -1);

        خودکار y1 = std::make_shared<ov::op::v1::Multiply>(rot_ڈیٹا, cos_بھرا);
        خودکار y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_بھرا);
        خودکار rotated = std::make_shared<ov::op::v1::جوڑو>(y1, y2);

        ov::Output_سمتیہ concat_parts;
        اگر (n_offs > 0) {
            خودکار سر_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودکار سر_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودکار سر = std::make_shared<ov::op::v8::قاش>(ڈیٹا_گرہ, سر_آغاز, سر_انتہا, قدم_one, axis_آخری);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        اگر (n_offs + rot_dims < سر_dim) {
            خودکار دم_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            خودکار دم_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خودکار دم = std::make_shared<ov::op::v8::قاش>(ڈیٹا_گرہ, دم_آغاز, دم_انتہا, قدم_one, axis_آخری);
            concat_parts.push_back(دم);
        }
        اگر (concat_parts.size() == 1) {
            res = rotated;
        } ورنہ {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
