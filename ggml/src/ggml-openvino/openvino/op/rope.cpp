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

Output_ويڪٽر translate_rope(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 2, 3);

    صحيح op_case = context.get_op_case();

    ov::Output<Node> res;

    خودڪار معلومات_ڳنڍ = process_view_داخلا_new(context, 0).get_ڳنڍ_shared_ptr();
    خودڪار output_shape = context.get_output_shape().منزل_shape();
    int32_t * op_params = context.get_output_op_params();
    ثابت صحيح mode = op_case;
    ثابت int64_t سر_dim = static_cast<int64_t>(output_shape[3]);
    ثابت int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ثابت int64_t n_dims = configured_n_dims == 0 ? سر_dim : configured_n_dims;
    ثابت int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr صحيح TYPE_عام = 0;
    constexpr صحيح TYPE_NEOX = 1;
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    Output<Node> cos_theta_ڳنڍ;
    Output<Node> sin_theta_ڳنڍ;
    إذا (context.has_داخلا("rope_cos")) {
        cos_theta_ڳنڍ = context.get_داخلا("rope_cos");
        sin_theta_ڳنڍ = context.get_داخلا("rope_sin");
    } وإلا {
        std::string cache_ڪنجي = "rope_sin_cos";
        لكل (صحيح i = 0; i < 15; i++) {
            cache_ڪنجي += "_" + std::to_string(op_params[i]);
        }
        إذا (context.get_داخلا_سائيز() == 3) {
            cache_ڪنجي += "_ff_" + context.get_داخلا_نالا()[2];
        }
        إذا (context.has_داخلا(cache_ڪنجي + "_cos")) {
            cos_theta_ڳنڍ = context.get_داخلا(cache_ڪنجي + "_cos");
            sin_theta_ڳنڍ = context.get_داخلا(cache_ڪنجي + "_sin");
        } وإلا {
            خودڪار inp_pos = context.get_داخلا(1).get_ڳنڍ_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_وزن;
            إذا (context.get_داخلا_سائيز() == 3) {
                rope_freqs_وزن = context.get_داخلا(2).get_ڳنڍ_shared_ptr();
            }
            خودڪار sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_وزن, mode, خطأ, سر_dim);
            sin_theta_ڳنڍ = sin_cos.پهريون;
            cos_theta_ڳنڍ = sin_cos.second;
            context.put_shared(cache_ڪنجي + "_cos", cos_theta_ڳنڍ);
            context.put_shared(cache_ڪنجي + "_sin", sin_theta_ڳنڍ);
        }
    }

    خودڪار output_type = context.get_output_type();
    إذا (معلومات_ڳنڍ->get_element_type() != ov::element::f32) {
        معلومات_ڳنڍ = std::make_shared<ov::op::v0::Convert>(معلومات_ڳنڍ, ov::element::f32);
    }

    ثابت int64_t ڪل_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(n_dims > 0 && ڪل_rope_dims + n_offs <= سر_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    إذا (context.is_stateful()) {
        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        خودڪار r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        معلومات_ڳنڍ = std::make_shared<ov::op::v1::Reshape>(معلومات_ڳنڍ, r4_shape, خطأ);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    إذا (mode == TYPE_عام) {
        خودڪار axis_آخري = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودڪار قدم_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        ثابت int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ثابت int64_t half = n_dims / 2;
        خودڪار rot_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        خودڪار rot_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        خودڪار rot_معلومات = std::make_shared<ov::op::v8::ٽڪرو>(معلومات_ڳنڍ, rot_شروعات, rot_پڇاڙي, قدم_one, axis_آخري);

        خودڪار neg_one_f = ov::op::v0::Constant::create(معلومات_ڳنڍ->get_element_type(), ov::Shape{}, {-1.0f});

        خودڪار paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        خودڪار x_paired = std::make_shared<ov::op::v1::Reshape>(rot_معلومات, paired_shape, خطأ);

        خودڪار ورهايو_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        خودڪار معلومات_ورهايو = std::make_shared<ov::op::v1::ورهايو_2>(x_paired, ورهايو_axis, 2);
        Output<Node> x0 = معلومات_ورهايو->outputs()[0];
        Output<Node> x1 = معلومات_ورهايو->outputs()[1];

        خودڪار x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        خودڪار x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{x1_neg, x0}, -1);

        خودڪار flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        خودڪار x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, خطأ);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        خودڪار expand_cos_sin = [&](ثابت Output<Node>& cs) {
            خودڪار cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            خودڪار bcast_هدف = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            خودڪار bcast = std::make_shared<ov::op::v3::نشر>(
                cs_unsq, bcast_هدف, ov::op::نشر_Type::BIDIRECTIONAL);
            خودڪار flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            أرجع std::make_shared<ov::op::v1::Reshape>(bcast, flat, سچ);
        };
        Output<Node> cos_ڀريل = expand_cos_sin(cos_theta_ڳنڍ);
        Output<Node> sin_ڀريل = expand_cos_sin(sin_theta_ڳنڍ);

        خودڪار y1 = std::make_shared<ov::op::v1::Multiply>(rot_معلومات, cos_ڀريل);
        خودڪار y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_ڀريل);
        خودڪار rotated = std::make_shared<ov::op::v1::جوڙيو>(y1, y2);

        ov::Output_ويڪٽر concat_parts;
        إذا (n_offs > 0) {
            خودڪار سر_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودڪار سر_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودڪار سر = std::make_shared<ov::op::v8::ٽڪرو>(معلومات_ڳنڍ, سر_شروعات, سر_پڇاڙي, قدم_one, axis_آخري);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        إذا (n_offs + n_dims < سر_dim) {
            خودڪار پڇ_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            خودڪار پڇ_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خودڪار پڇ = std::make_shared<ov::op::v8::ٽڪرو>(معلومات_ڳنڍ, پڇ_شروعات, پڇ_پڇاڙي, قدم_one, axis_آخري);
            concat_parts.push_back(پڇ);
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
            خودڪار cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_ڳنڍ = std::make_shared<ov::op::v1::Reshape>(cos_theta_ڳنڍ, cos_sin_shape, سچ);
            sin_theta_ڳنڍ = std::make_shared<ov::op::v1::Reshape>(sin_theta_ڳنڍ, cos_sin_shape, سچ);
        }

        خودڪار axis_آخري = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودڪار قدم_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_معلومات = معلومات_ڳنڍ;
        إذا (n_offs > 0 || n_offs + rot_dims < سر_dim) {
            خودڪار rot_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودڪار rot_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_معلومات = std::make_shared<ov::op::v8::ٽڪرو>(معلومات_ڳنڍ, rot_شروعات, rot_پڇاڙي, قدم_one, axis_آخري);
        }

        خودڪار neg_one_f = ov::op::v0::Constant::create(معلومات_ڳنڍ->get_element_type(), ov::Shape{}, {-1.0f});

        خودڪار ورهايو_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        خودڪار ورهايو_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        خودڪار معلومات_ورهايو = std::make_shared<ov::op::v1::Variadic_ورهايو>(rot_معلومات, ورهايو_axis, ورهايو_lengths);
        Output<Node> x1 = معلومات_ورهايو->outputs()[0];
        Output<Node> x2 = معلومات_ورهايو->outputs()[1];

        خودڪار x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        خودڪار x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{x2_neg, x1}, -1);

        خودڪار cos_ڀريل = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{cos_theta_ڳنڍ, cos_theta_ڳنڍ}, -1);
        خودڪار sin_ڀريل = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{sin_theta_ڳنڍ, sin_theta_ڳنڍ}, -1);

        خودڪار y1 = std::make_shared<ov::op::v1::Multiply>(rot_معلومات, cos_ڀريل);
        خودڪار y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_ڀريل);
        خودڪار rotated = std::make_shared<ov::op::v1::جوڙيو>(y1, y2);

        ov::Output_ويڪٽر concat_parts;
        إذا (n_offs > 0) {
            خودڪار سر_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودڪار سر_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            خودڪار سر = std::make_shared<ov::op::v8::ٽڪرو>(معلومات_ڳنڍ, سر_شروعات, سر_پڇاڙي, قدم_one, axis_آخري);
            concat_parts.push_back(سر);
        }
        concat_parts.push_back(rotated);
        إذا (n_offs + rot_dims < سر_dim) {
            خودڪار پڇ_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            خودڪار پڇ_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_dim});
            خودڪار پڇ = std::make_shared<ov::op::v8::ٽڪرو>(معلومات_ڳنڍ, پڇ_شروعات, پڇ_پڇاڙي, قدم_one, axis_آخري);
            concat_parts.push_back(پڇ);
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
