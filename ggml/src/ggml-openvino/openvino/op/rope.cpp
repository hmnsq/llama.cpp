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

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_rope(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 3);

    صحیح op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_گره = process_view_درون_داد_new(context, 0).get_گره_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    ثابت صحیح mode = op_case;
    ثابت int64_t head_dim = static_cast<int64_t>(output_shape[3]);
    ثابت int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ثابت int64_t n_dims = configured_n_dims == 0 ? head_dim : configured_n_dims;
    ثابت int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr صحیح TYPE_NORMAL = 0;
    constexpr صحیح TYPE_NEOX = 1;
    constexpr صحیح TYPE_IMROPE = 2;
    constexpr صحیح TYPE_VISION = 3;
    constexpr صحیح TYPE_MROPE = 4;

    Output<Node> cos_theta_گره;
    Output<Node> sin_theta_گره;
    اگر (context.has_درون_داد("rope_cos")) {
        cos_theta_گره = context.get_درون_داد("rope_cos");
        sin_theta_گره = context.get_درون_داد("rope_sin");
    } وگرنه {
        std::string cache_کلید = "rope_sin_cos";
        برای (صحیح i = 0; i < 15; i++) {
            cache_کلید += "_" + std::to_string(op_params[i]);
        }
        اگر (context.get_درون_داد_size() == 3) {
            cache_کلید += "_ff_" + context.get_درون_داد_names()[2];
        }
        اگر (context.has_درون_داد(cache_کلید + "_cos")) {
            cos_theta_گره = context.get_درون_داد(cache_کلید + "_cos");
            sin_theta_گره = context.get_درون_داد(cache_کلید + "_sin");
        } وگرنه {
            auto inp_pos = context.get_درون_داد(1).get_گره_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_weight;
            اگر (context.get_درون_داد_size() == 3) {
                rope_freqs_weight = context.get_درون_داد(2).get_گره_shared_ptr();
            }
            auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode, نادرست, head_dim);
            sin_theta_گره = sin_cos.first;
            cos_theta_گره = sin_cos.second;
            context.put_shared(cache_کلید + "_cos", cos_theta_گره);
            context.put_shared(cache_کلید + "_sin", sin_theta_گره);
        }
    }

    auto output_type = context.get_output_type();
    اگر (data_گره->get_element_type() != ov::element::f32) {
        data_گره = std::make_shared<ov::op::v0::Convert>(data_گره, ov::element::f32);
    }

    ثابت int64_t total_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_END_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_END_OP_CONVERSION_CHECK(n_dims > 0 && total_rope_dims + n_offs <= head_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    اگر (context.is_stateful()) {
        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        auto r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        data_گره = std::make_shared<ov::op::v1::Reshape>(data_گره, r4_shape, نادرست);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    اگر (mode == TYPE_NORMAL) {
        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ثابت int64_t batch = static_cast<int64_t>(output_shape[0]);
        ثابت int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ثابت int64_t half = n_dims / 2;
        auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        auto rot_data = std::make_shared<ov::op::v8::Slice>(data_گره, rot_start, rot_end, step_one, axis_last);

        auto neg_one_f = ov::op::v0::Constant::create(data_گره->get_element_type(), ov::Shape{}, {-1.0f});

        auto paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(rot_data, paired_shape, نادرست);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = data_split->outputs()[0];
        Output<Node> x1 = data_split->outputs()[1];

        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        auto flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        auto x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, نادرست);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        auto expand_cos_sin = [&](ثابت Output<Node>& cs) {
            auto cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            auto bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            بازگشت std::make_shared<ov::op::v1::Reshape>(bcast, flat, درست);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_گره);
        Output<Node> sin_full = expand_cos_sin(sin_theta_گره);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        اگر (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_گره, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        اگر (n_offs + n_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_گره, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        اگر (concat_parts.size() == 1) {
            res = rotated;
        } وگرنه {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } وگرنه اگر (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ثابت int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ثابت int64_t rot_dims = 2 * half;

        اگر (mode != TYPE_NEOX) {
            auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_گره = std::make_shared<ov::op::v1::Reshape>(cos_theta_گره, cos_sin_shape, درست);
            sin_theta_گره = std::make_shared<ov::op::v1::Reshape>(sin_theta_گره, cos_sin_shape, درست);
        }

        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_data = data_گره;
        اگر (n_offs > 0 || n_offs + rot_dims < head_dim) {
            auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_data = std::make_shared<ov::op::v8::Slice>(data_گره, rot_start, rot_end, step_one, axis_last);
        }

        auto neg_one_f = ov::op::v0::Constant::create(data_گره->get_element_type(), ov::Shape{}, {-1.0f});

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        auto data_split = std::make_shared<ov::op::v1::VariadicSplit>(rot_data, split_axis, split_lengths);
        Output<Node> x1 = data_split->outputs()[0];
        Output<Node> x2 = data_split->outputs()[1];

        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        auto x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);

        auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_گره, cos_theta_گره}, -1);
        auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_گره, sin_theta_گره}, -1);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        اگر (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_گره, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        اگر (n_offs + rot_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_گره, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        اگر (concat_parts.size() == 1) {
            res = rotated;
        } وگرنه {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
