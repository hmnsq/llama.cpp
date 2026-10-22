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

Output_وکتور translate_rope(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 3);

    صحيح op_case = context.get_op_case();

    ov::Output<Node> res;

    خودکار داده_گره = process_view_درون_داد_new(context, 0).get_گره_shared_ptr();
    خودکار output_shape = context.get_output_shape().مقصد_shape();
    int32_t * op_params = context.get_output_op_params();
    ثابت صحيح mode = op_case;
    ثابت int64_t سر_dim = static_cast<int64_t>(output_shape[3]);
    ثابت int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ثابت int64_t n_dims = configured_n_dims == 0 ? سر_dim : configured_n_dims;
    ثابت int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr صحيح TYPE_نرمال = 0;
    constexpr صحيح TYPE_NEOX = 1;
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    Output<Node> cos_theta_گره;
    Output<Node> sin_theta_گره;
    إذا (context.has_درون_داد("rope_cos")) {
        cos_theta_گره = context.get_درون_داد("rope_cos");
        sin_theta_گره = context.get_درون_داد("rope_sin");
    } وإلا {
        std::string cache_کلید = "rope_sin_cos";
        لكل (صحيح i = 0; i < 15; i++) {
            cache_کلید += "_" + std::to_string(op_params[i]);
        }
        إذا (context.get_درون_داد_اندازه() == 3) {
            cache_کلید += "_ff_" + context.get_درون_داد_نامها()[2];
        }
        إذا (context.has_درون_داد(cache_کلید + "_cos")) {
            cos_theta_گره = context.get_درون_داد(cache_کلید + "_cos");
            sin_theta_گره = context.get_درون_داد(cache_کلید + "_sin");
        } وإلا {
            خودکار inp_pos = context.get_درون_داد(1).get_گره_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_وزن;
            إذا (context.get_درون_داد_اندازه() == 3) {
                rope_freqs_وزن = context.get_درون_داد(2).get_گره_shared_ptr();
            }
            خودکار sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_وزن, mode, خطأ, سر_dim);
            sin_theta_گره = sin_cos.اولی;
            cos_theta_گره = sin_cos.second;
            context.put_shared(cache_کلید + "_cos", cos_theta_گره);
            context.put_shared(cache_کلید + "_sin", sin_theta_گره);
        }
    }

    خودکار output_type = context.get_output_type();
    إذا (داده_گره->get_element_type() != ov::element::f32) {
        داده_گره = std::make_shared<ov::op::v0::Convert>(داده_گره, ov::element::f32);
    }

    ثابت int64_t مجموع_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_انتها_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_انتها_OP_CONVERSION_CHECK(n_dims > 0 && مجموع_rope_dims + n_offs <= سر_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    إذا (context.is_stateful()) {
        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        خودکار r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        داده_گره = std::make_shared<ov::op::v1::Reshape>(داده_گره, r4_shape, خطأ);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    إذا (mode == TYPE_نرمال) {
        خودکار axis_آخرین = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار گام_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        ثابت int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ثابت int64_t half = n_dims / 2;
        خودکار rot_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        خودکار rot_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        خودکار rot_داده = std::make_shared<ov::op::v8::برش>(داده_گره, rot_شروع, rot_انتها, گام_one, axis_آخرین);

        خودکار neg_one_f = ov::op::v0::Constant::create(داده_گره->get_element_type(), ov::Shape{}, {-1.0f});

        خودکار paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        خودکار x_paired = std::make_shared<ov::op::v1::Reshape>(rot_داده, paired_shape, خطأ);

        خودکار تقسیم_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        خودکار داده_تقسیم = std::make_shared<ov::op::v1::تقسیم_2>(x_paired, تقسیم_axis, 2);
        Output<Node> x0 = داده_تقسیم->outputs()[0];
        Output<Node> x1 = داده_تقسیم->outputs()[1];

        خودکار x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        خودکار x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{x1_neg, x0}, -1);

        خودکار flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        خودکار x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, خطأ);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        خودکار expand_cos_sin = [&](ثابت Output<Node>& cs) {
            خودکار cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            خودکار bcast_هدف = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            خودکار bcast = std::make_shared<ov::op::v3::پخش>(
                cs_unsq, bcast_هدف, ov::op::پخش_Type::BIDIRECTIONAL);
            خودکار flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            أرجع std::make_shared<ov::op::v1::Reshape>(bcast, flat, درست);
        };
        Output<Node> cos_پر = expand_cos_sin(cos_theta_گره);
        Output<Node> sin_پر = expand_cos_sin(sin_theta_گره);

        خودکار y1 = std::make_shared<ov::op::v1::Multiply>(rot_داده, cos_پر);
        خودکار y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_پر);
        خودکار rotated = std::make_shared<ov::op::v1::جمع_کن>(y1, y2);

        ov::Output_وکتور concat_parts;
        إذا (n_offs > 0) {
            خودکار سر_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودکار سر_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودکار سر = std::make_shared<ov::op::v8::برش>(داده_گره, سر_شروع, سر_انتها, گام_one, axis_آخرین);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        إذا (n_offs + n_dims < سر_dim) {
            خودکار دم_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            خودکار دم_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خودکار دم = std::make_shared<ov::op::v8::برش>(داده_گره, دم_شروع, دم_انتها, گام_one, axis_آخرین);
            concat_parts.push_back(دم);
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
            خودکار cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_گره = std::make_shared<ov::op::v1::Reshape>(cos_theta_گره, cos_sin_shape, درست);
            sin_theta_گره = std::make_shared<ov::op::v1::Reshape>(sin_theta_گره, cos_sin_shape, درست);
        }

        خودکار axis_آخرین = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار گام_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_داده = داده_گره;
        إذا (n_offs > 0 || n_offs + rot_dims < سر_dim) {
            خودکار rot_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودکار rot_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_داده = std::make_shared<ov::op::v8::برش>(داده_گره, rot_شروع, rot_انتها, گام_one, axis_آخرین);
        }

        خودکار neg_one_f = ov::op::v0::Constant::create(داده_گره->get_element_type(), ov::Shape{}, {-1.0f});

        خودکار تقسیم_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        خودکار تقسیم_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        خودکار داده_تقسیم = std::make_shared<ov::op::v1::Variadic_تقسیم>(rot_داده, تقسیم_axis, تقسیم_lengths);
        Output<Node> x1 = داده_تقسیم->outputs()[0];
        Output<Node> x2 = داده_تقسیم->outputs()[1];

        خودکار x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        خودکار x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{x2_neg, x1}, -1);

        خودکار cos_پر = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{cos_theta_گره, cos_theta_گره}, -1);
        خودکار sin_پر = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{sin_theta_گره, sin_theta_گره}, -1);

        خودکار y1 = std::make_shared<ov::op::v1::Multiply>(rot_داده, cos_پر);
        خودکار y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_پر);
        خودکار rotated = std::make_shared<ov::op::v1::جمع_کن>(y1, y2);

        ov::Output_وکتور concat_parts;
        إذا (n_offs > 0) {
            خودکار سر_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودکار سر_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودکار سر = std::make_shared<ov::op::v8::برش>(داده_گره, سر_شروع, سر_انتها, گام_one, axis_آخرین);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        إذا (n_offs + rot_dims < سر_dim) {
            خودکار دم_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            خودکار دم_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خودکار دم = std::make_shared<ov::op::v8::برش>(داده_گره, دم_شروع, دم_انتها, گام_one, axis_آخرین);
            concat_parts.push_back(دم);
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
