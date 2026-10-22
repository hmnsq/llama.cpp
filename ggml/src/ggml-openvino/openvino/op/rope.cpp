#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/split.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/variadic_split.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_rope(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 3);

    ਪੂਰਨ op_case = context.get_op_case();

    ov::Output<Node> res;

    ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_ਗੰਢ = process_view_ਨਿਵੇਸ਼_new(context, 0).get_ਗੰਢ_shared_ptr();
    ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
    int32_t * op_params = context.get_output_op_params();
    ਅਚਲ ਪੂਰਨ mode = op_case;
    ਅਚਲ int64_t ਸਿਰ_dim = static_cast<int64_t>(output_shape[3]);
    ਅਚਲ int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ਅਚਲ int64_t n_dims = configured_n_dims == 0 ? ਸਿਰ_dim : configured_n_dims;
    ਅਚਲ int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr ਪੂਰਨ TYPE_ਸਾਧਾਰਨ = 0;
    constexpr ਪੂਰਨ TYPE_NEOX = 1;
    constexpr ਪੂਰਨ TYPE_IMROPE = 2;
    constexpr ਪੂਰਨ TYPE_VISION = 3;
    constexpr ਪੂਰਨ TYPE_MROPE = 4;

    Output<Node> cos_theta_ਗੰਢ;
    Output<Node> sin_theta_ਗੰਢ;
    ਜੇ (context.has_ਨਿਵੇਸ਼("rope_cos")) {
        cos_theta_ਗੰਢ = context.get_ਨਿਵੇਸ਼("rope_cos");
        sin_theta_ਗੰਢ = context.get_ਨਿਵੇਸ਼("rope_sin");
    } ਨਹੀਂਤਾਂ {
        std::string cache_ਕੁੰਜੀ = "rope_sin_cos";
        ਲਈ (ਪੂਰਨ i = 0; i < 15; i++) {
            cache_ਕੁੰਜੀ += "_" + std::to_string(op_params[i]);
        }
        ਜੇ (context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 3) {
            cache_ਕੁੰਜੀ += "_ff_" + context.get_ਨਿਵੇਸ਼_ਨਾਂ()[2];
        }
        ਜੇ (context.has_ਨਿਵੇਸ਼(cache_ਕੁੰਜੀ + "_cos")) {
            cos_theta_ਗੰਢ = context.get_ਨਿਵੇਸ਼(cache_ਕੁੰਜੀ + "_cos");
            sin_theta_ਗੰਢ = context.get_ਨਿਵੇਸ਼(cache_ਕੁੰਜੀ + "_sin");
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ inp_pos = context.get_ਨਿਵੇਸ਼(1).get_ਗੰਢ_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_ਭਾਰ;
            ਜੇ (context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 3) {
                rope_freqs_ਭਾਰ = context.get_ਨਿਵੇਸ਼(2).get_ਗੰਢ_shared_ptr();
            }
            ਸਵੈਚਾਲਿਤ sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_ਭਾਰ, mode, ਝੂਠ, ਸਿਰ_dim);
            sin_theta_ਗੰਢ = sin_cos.ਪਹਿਲਾ;
            cos_theta_ਗੰਢ = sin_cos.second;
            context.put_shared(cache_ਕੁੰਜੀ + "_cos", cos_theta_ਗੰਢ);
            context.put_shared(cache_ਕੁੰਜੀ + "_sin", sin_theta_ਗੰਢ);
        }
    }

    ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (ਅੰਕੜੇ_ਗੰਢ->get_element_type() != ov::element::f32) {
        ਅੰਕੜੇ_ਗੰਢ = std::make_shared<ov::op::v0::Convert>(ਅੰਕੜੇ_ਗੰਢ, ov::element::f32);
    }

    ਅਚਲ int64_t ਕੁੱਲ_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(n_dims > 0 && ਕੁੱਲ_rope_dims + n_offs <= ਸਿਰ_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    ਜੇ (context.is_stateful()) {
        ਅਚਲ int64_t batch = static_cast<int64_t>(output_shape[0]);
        ਸਵੈਚਾਲਿਤ r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        ਅੰਕੜੇ_ਗੰਢ = std::make_shared<ov::op::v1::Reshape>(ਅੰਕੜੇ_ਗੰਢ, r4_shape, ਝੂਠ);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    ਜੇ (mode == TYPE_ਸਾਧਾਰਨ) {
        ਸਵੈਚਾਲਿਤ axis_ਆਖਰੀ = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ਸਵੈਚਾਲਿਤ ਕਦਮ_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ਅਚਲ int64_t batch = static_cast<int64_t>(output_shape[0]);
        ਅਚਲ int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ਅਚਲ int64_t half = n_dims / 2;
        ਸਵੈਚਾਲਿਤ rot_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        ਸਵੈਚਾਲਿਤ rot_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        ਸਵੈਚਾਲਿਤ rot_ਅੰਕੜੇ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ_ਗੰਢ, rot_ਸ਼ੁਰੂ, rot_ਸਿਰਾ, ਕਦਮ_one, axis_ਆਖਰੀ);

        ਸਵੈਚਾਲਿਤ neg_one_f = ov::op::v0::Constant::create(ਅੰਕੜੇ_ਗੰਢ->get_element_type(), ov::Shape{}, {-1.0f});

        ਸਵੈਚਾਲਿਤ paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        ਸਵੈਚਾਲਿਤ x_paired = std::make_shared<ov::op::v1::Reshape>(rot_ਅੰਕੜੇ, paired_shape, ਝੂਠ);

        ਸਵੈਚਾਲਿਤ ਵੰਡੋ_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_ਵੰਡੋ = std::make_shared<ov::op::v1::ਵੰਡੋ_2>(x_paired, ਵੰਡੋ_axis, 2);
        Output<Node> x0 = ਅੰਕੜੇ_ਵੰਡੋ->outputs()[0];
        Output<Node> x1 = ਅੰਕੜੇ_ਵੰਡੋ->outputs()[1];

        ਸਵੈਚਾਲਿਤ x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        ਸਵੈਚਾਲਿਤ x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{x1_neg, x0}, -1);

        ਸਵੈਚਾਲਿਤ flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        ਸਵੈਚਾਲਿਤ x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, ਝੂਠ);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        ਸਵੈਚਾਲਿਤ expand_cos_sin = [&](ਅਚਲ Output<Node>& cs) {
            ਸਵੈਚਾਲਿਤ cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            ਸਵੈਚਾਲਿਤ bcast_ਨਿਸ਼ਾਨਾ = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            ਸਵੈਚਾਲਿਤ bcast = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(
                cs_unsq, bcast_ਨਿਸ਼ਾਨਾ, ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
            ਸਵੈਚਾਲਿਤ flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            ਵਾਪਸ std::make_shared<ov::op::v1::Reshape>(bcast, flat, ਸੱਚ);
        };
        Output<Node> cos_ਭਰਿਆ = expand_cos_sin(cos_theta_ਗੰਢ);
        Output<Node> sin_ਭਰਿਆ = expand_cos_sin(sin_theta_ਗੰਢ);

        ਸਵੈਚਾਲਿਤ y1 = std::make_shared<ov::op::v1::Multiply>(rot_ਅੰਕੜੇ, cos_ਭਰਿਆ);
        ਸਵੈਚਾਲਿਤ y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_ਭਰਿਆ);
        ਸਵੈਚਾਲਿਤ rotated = std::make_shared<ov::op::v1::ਜੋੜੋ>(y1, y2);

        ov::Output_ਸਦਿਸ਼ concat_parts;
        ਜੇ (n_offs > 0) {
            ਸਵੈਚਾਲਿਤ ਸਿਰ_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            ਸਵੈਚਾਲਿਤ ਸਿਰ_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            ਸਵੈਚਾਲਿਤ ਸਿਰ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ_ਗੰਢ, ਸਿਰ_ਸ਼ੁਰੂ, ਸਿਰ_ਸਿਰਾ, ਕਦਮ_one, axis_ਆਖਰੀ);
            concat_parts.push_back(ਸਿਰ);
        }
        concat_parts.push_back(rotated);
        ਜੇ (n_offs + n_dims < ਸਿਰ_dim) {
            ਸਵੈਚਾਲਿਤ ਪੂਛ_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            ਸਵੈਚਾਲਿਤ ਪੂਛ_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰ_dim});
            ਸਵੈਚਾਲਿਤ ਪੂਛ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ_ਗੰਢ, ਪੂਛ_ਸ਼ੁਰੂ, ਪੂਛ_ਸਿਰਾ, ਕਦਮ_one, axis_ਆਖਰੀ);
            concat_parts.push_back(ਪੂਛ);
        }
        ਜੇ (concat_parts.size() == 1) {
            res = rotated;
        } ਨਹੀਂਤਾਂ {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ਅਚਲ int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ਅਚਲ int64_t rot_dims = 2 * half;

        ਜੇ (mode != TYPE_NEOX) {
            ਸਵੈਚਾਲਿਤ cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_ਗੰਢ = std::make_shared<ov::op::v1::Reshape>(cos_theta_ਗੰਢ, cos_sin_shape, ਸੱਚ);
            sin_theta_ਗੰਢ = std::make_shared<ov::op::v1::Reshape>(sin_theta_ਗੰਢ, cos_sin_shape, ਸੱਚ);
        }

        ਸਵੈਚਾਲਿਤ axis_ਆਖਰੀ = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ਸਵੈਚਾਲਿਤ ਕਦਮ_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_ਅੰਕੜੇ = ਅੰਕੜੇ_ਗੰਢ;
        ਜੇ (n_offs > 0 || n_offs + rot_dims < ਸਿਰ_dim) {
            ਸਵੈਚਾਲਿਤ rot_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            ਸਵੈਚਾਲਿਤ rot_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_ਅੰਕੜੇ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ_ਗੰਢ, rot_ਸ਼ੁਰੂ, rot_ਸਿਰਾ, ਕਦਮ_one, axis_ਆਖਰੀ);
        }

        ਸਵੈਚਾਲਿਤ neg_one_f = ov::op::v0::Constant::create(ਅੰਕੜੇ_ਗੰਢ->get_element_type(), ov::Shape{}, {-1.0f});

        ਸਵੈਚਾਲਿਤ ਵੰਡੋ_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        ਸਵੈਚਾਲਿਤ ਵੰਡੋ_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_ਵੰਡੋ = std::make_shared<ov::op::v1::Variadic_ਵੰਡੋ>(rot_ਅੰਕੜੇ, ਵੰਡੋ_axis, ਵੰਡੋ_lengths);
        Output<Node> x1 = ਅੰਕੜੇ_ਵੰਡੋ->outputs()[0];
        Output<Node> x2 = ਅੰਕੜੇ_ਵੰਡੋ->outputs()[1];

        ਸਵੈਚਾਲਿਤ x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        ਸਵੈਚਾਲਿਤ x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{x2_neg, x1}, -1);

        ਸਵੈਚਾਲਿਤ cos_ਭਰਿਆ = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{cos_theta_ਗੰਢ, cos_theta_ਗੰਢ}, -1);
        ਸਵੈਚਾਲਿਤ sin_ਭਰਿਆ = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{sin_theta_ਗੰਢ, sin_theta_ਗੰਢ}, -1);

        ਸਵੈਚਾਲਿਤ y1 = std::make_shared<ov::op::v1::Multiply>(rot_ਅੰਕੜੇ, cos_ਭਰਿਆ);
        ਸਵੈਚਾਲਿਤ y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_ਭਰਿਆ);
        ਸਵੈਚਾਲਿਤ rotated = std::make_shared<ov::op::v1::ਜੋੜੋ>(y1, y2);

        ov::Output_ਸਦਿਸ਼ concat_parts;
        ਜੇ (n_offs > 0) {
            ਸਵੈਚਾਲਿਤ ਸਿਰ_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            ਸਵੈਚਾਲਿਤ ਸਿਰ_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            ਸਵੈਚਾਲਿਤ ਸਿਰ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ_ਗੰਢ, ਸਿਰ_ਸ਼ੁਰੂ, ਸਿਰ_ਸਿਰਾ, ਕਦਮ_one, axis_ਆਖਰੀ);
            concat_parts.push_back(ਸਿਰ);
        }
        concat_parts.push_back(rotated);
        ਜੇ (n_offs + rot_dims < ਸਿਰ_dim) {
            ਸਵੈਚਾਲਿਤ ਪੂਛ_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            ਸਵੈਚਾਲਿਤ ਪੂਛ_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰ_dim});
            ਸਵੈਚਾਲਿਤ ਪੂਛ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ_ਗੰਢ, ਪੂਛ_ਸ਼ੁਰੂ, ਪੂਛ_ਸਿਰਾ, ਕਦਮ_one, axis_ਆਖਰੀ);
            concat_parts.push_back(ਪੂਛ);
        }
        ਜੇ (concat_parts.size() == 1) {
            res = rotated;
        } ਨਹੀਂਤਾਂ {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
