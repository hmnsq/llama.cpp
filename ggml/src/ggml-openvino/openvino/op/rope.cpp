#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cstdint>
#समावेश <memory>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/split.hpp>
#समावेश <openvino/op/subtract.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <openvino/op/variadic_split.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 3);

    पूर्णांक op_case = context.get_op_case();

    ov::Output<Node> res;

    स्वतः तथ्य_गाँठ = process_view_निवेश_new(context, 0).get_गाँठ_shared_ptr();
    स्वतः output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    स्थिर पूर्णांक mode = op_case;
    स्थिर int64_t शीर्ष_dim = static_cast<int64_t>(output_shape[3]);
    स्थिर int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    स्थिर int64_t n_dims = configured_n_dims == 0 ? शीर्ष_dim : configured_n_dims;
    स्थिर int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr पूर्णांक TYPE_NORMAL = 0;
    constexpr पूर्णांक TYPE_NEOX = 1;
    constexpr पूर्णांक TYPE_IMROPE = 2;
    constexpr पूर्णांक TYPE_VISION = 3;
    constexpr पूर्णांक TYPE_MROPE = 4;

    Output<Node> cos_theta_गाँठ;
    Output<Node> sin_theta_गाँठ;
    अगर (context.has_निवेश("rope_cos")) {
        cos_theta_गाँठ = context.get_निवेश("rope_cos");
        sin_theta_गाँठ = context.get_निवेश("rope_sin");
    } अन्यथा {
        std::string cache_कुंजी = "rope_sin_cos";
        क्रम (पूर्णांक i = 0; i < 15; i++) {
            cache_कुंजी += "_" + std::to_string(op_params[i]);
        }
        अगर (context.get_निवेश_आकार() == 3) {
            cache_कुंजी += "_ff_" + context.get_निवेश_names()[2];
        }
        अगर (context.has_निवेश(cache_कुंजी + "_cos")) {
            cos_theta_गाँठ = context.get_निवेश(cache_कुंजी + "_cos");
            sin_theta_गाँठ = context.get_निवेश(cache_कुंजी + "_sin");
        } अन्यथा {
            स्वतः inp_pos = context.get_निवेश(1).get_गाँठ_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_weight;
            अगर (context.get_निवेश_आकार() == 3) {
                rope_freqs_weight = context.get_निवेश(2).get_गाँठ_shared_ptr();
            }
            स्वतः sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode, false, शीर्ष_dim);
            sin_theta_गाँठ = sin_cos.first;
            cos_theta_गाँठ = sin_cos.second;
            context.put_shared(cache_कुंजी + "_cos", cos_theta_गाँठ);
            context.put_shared(cache_कुंजी + "_sin", sin_theta_गाँठ);
        }
    }

    स्वतः output_type = context.get_output_type();
    अगर (तथ्य_गाँठ->get_element_type() != ov::element::f32) {
        तथ्य_गाँठ = std::make_shared<ov::op::v0::Convert>(तथ्य_गाँठ, ov::element::f32);
    }

    स्थिर int64_t योग_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_END_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_END_OP_CONVERSION_CHECK(n_dims > 0 && योग_rope_dims + n_offs <= शीर्ष_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    अगर (context.is_stateful()) {
        स्थिर int64_t batch = static_cast<int64_t>(output_shape[0]);
        स्वतः r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        तथ्य_गाँठ = std::make_shared<ov::op::v1::Reshape>(तथ्य_गाँठ, r4_shape, false);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    अगर (mode == TYPE_NORMAL) {
        स्वतः axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वतः step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        स्थिर int64_t batch = static_cast<int64_t>(output_shape[0]);
        स्थिर int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        स्थिर int64_t half = n_dims / 2;
        स्वतः rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        स्वतः rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        स्वतः rot_तथ्य = std::make_shared<ov::op::v8::Slice>(तथ्य_गाँठ, rot_start, rot_end, step_one, axis_last);

        स्वतः neg_one_f = ov::op::v0::Constant::create(तथ्य_गाँठ->get_element_type(), ov::Shape{}, {-1.0f});

        स्वतः paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        स्वतः x_paired = std::make_shared<ov::op::v1::Reshape>(rot_तथ्य, paired_shape, false);

        स्वतः split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        स्वतः तथ्य_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = तथ्य_split->outputs()[0];
        Output<Node> x1 = तथ्य_split->outputs()[1];

        स्वतः x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        स्वतः x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        स्वतः flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        स्वतः x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, false);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        स्वतः expand_cos_sin = [&](स्थिर Output<Node>& cs) {
            स्वतः cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            स्वतः bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            स्वतः bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            स्वतः flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            वापस std::make_shared<ov::op::v1::Reshape>(bcast, flat, true);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_गाँठ);
        Output<Node> sin_full = expand_cos_sin(sin_theta_गाँठ);

        स्वतः y1 = std::make_shared<ov::op::v1::Multiply>(rot_तथ्य, cos_full);
        स्वतः y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        स्वतः rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        अगर (n_offs > 0) {
            स्वतः शीर्ष_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            स्वतः शीर्ष_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            स्वतः शीर्ष = std::make_shared<ov::op::v8::Slice>(तथ्य_गाँठ, शीर्ष_start, शीर्ष_end, step_one, axis_last);
            concat_parts.push_back(शीर्ष);
        }
        concat_parts.push_back(rotated);
        अगर (n_offs + n_dims < शीर्ष_dim) {
            स्वतः पुच्छ_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            स्वतः पुच्छ_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {शीर्ष_dim});
            स्वतः पुच्छ = std::make_shared<ov::op::v8::Slice>(तथ्य_गाँठ, पुच्छ_start, पुच्छ_end, step_one, axis_last);
            concat_parts.push_back(पुच्छ);
        }
        अगर (concat_parts.size() == 1) {
            res = rotated;
        } अन्यथा {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } अन्यथा अगर (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        स्थिर int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        स्थिर int64_t rot_dims = 2 * half;

        अगर (mode != TYPE_NEOX) {
            स्वतः cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_गाँठ = std::make_shared<ov::op::v1::Reshape>(cos_theta_गाँठ, cos_sin_shape, true);
            sin_theta_गाँठ = std::make_shared<ov::op::v1::Reshape>(sin_theta_गाँठ, cos_sin_shape, true);
        }

        स्वतः axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वतः step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_तथ्य = तथ्य_गाँठ;
        अगर (n_offs > 0 || n_offs + rot_dims < शीर्ष_dim) {
            स्वतः rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            स्वतः rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_तथ्य = std::make_shared<ov::op::v8::Slice>(तथ्य_गाँठ, rot_start, rot_end, step_one, axis_last);
        }

        स्वतः neg_one_f = ov::op::v0::Constant::create(तथ्य_गाँठ->get_element_type(), ov::Shape{}, {-1.0f});

        स्वतः split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        स्वतः split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        स्वतः तथ्य_split = std::make_shared<ov::op::v1::VariadicSplit>(rot_तथ्य, split_axis, split_lengths);
        Output<Node> x1 = तथ्य_split->outputs()[0];
        Output<Node> x2 = तथ्य_split->outputs()[1];

        स्वतः x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        स्वतः x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);

        स्वतः cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_गाँठ, cos_theta_गाँठ}, -1);
        स्वतः sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_गाँठ, sin_theta_गाँठ}, -1);

        स्वतः y1 = std::make_shared<ov::op::v1::Multiply>(rot_तथ्य, cos_full);
        स्वतः y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_full);
        स्वतः rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        अगर (n_offs > 0) {
            स्वतः शीर्ष_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            स्वतः शीर्ष_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            स्वतः शीर्ष = std::make_shared<ov::op::v8::Slice>(तथ्य_गाँठ, शीर्ष_start, शीर्ष_end, step_one, axis_last);
            concat_parts.push_back(शीर्ष);
        }
        concat_parts.push_back(rotated);
        अगर (n_offs + rot_dims < शीर्ष_dim) {
            स्वतः पुच्छ_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            स्वतः पुच्छ_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {शीर्ष_dim});
            स्वतः पुच्छ = std::make_shared<ov::op::v8::Slice>(तथ्य_गाँठ, पुच्छ_start, पुच्छ_end, step_one, axis_last);
            concat_parts.push_back(पुच्छ);
        }
        अगर (concat_parts.size() == 1) {
            res = rotated;
        } अन्यथा {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
