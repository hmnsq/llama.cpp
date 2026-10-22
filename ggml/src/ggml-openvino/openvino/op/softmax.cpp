#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/broadcast.hpp>
#અન્તર્ભાવ <openvino/frontend/exception.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/softmax.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

સ્થિર તાર્કિક is_static_one(અચલ ov::Dimension & dim) {
    પાછા dim.is_static() && dim.get_length() == 1;
}

સ્થિર તાર્કિક same_static_dim(અચલ ov::Dimension & lhs, અચલ ov::Dimension & rhs) {
    પાછા lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

સ્થિર તાર્કિક is_attention_sinks_આગત_shape(અચલ ov::PartialShape & candidate, અચલ ov::PartialShape & logits_shape) {
    જો (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        પાછા ખોટું;
    }

    પાછા is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 3);

    દશાંશ scale = 1.0f;
    દશાંશ max_bias = 0.0f;
    memcpy(&scale, (દશાંશ *) context.get_output_op_params() + 0, કદ(દશાંશ));
    memcpy(&max_bias, (દશાંશ *) context.get_output_op_params() + 1, કદ(દશાંશ));

    ov::Output<ov::Node> logits = context.get_આગત(0);
    અચલ તાર્કિક second_આગત_is_sinks =
        context.get_આગત_size() == 2 && is_attention_sinks_આગત_shape(context.get_આગત_shape(1), context.get_output_shape());
    અચલ તાર્કિક has_mask = context.get_આગત_size() > 1 && !second_આગત_is_sinks;
    અચલ તાર્કિક has_sinks = second_આગત_is_sinks || context.get_આગત_size() > 2;
    અચલ size_t sinks_આગત_idx = second_આગત_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    જો (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<દશાંશ>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    જો (has_mask) {
        ov::Output<ov::Node> mask = context.get_આગત(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        જો (context.get_આગત_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        જો (context.has_આગત(mask_name)) {
            mask = context.get_આગત(mask_name);
        }

        જો (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        જો (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            અચલ uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            અચલ uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<દશાંશ>(n_head))));
            અચલ દશાંશ m0 = std::pow(2.0f, -(max_bias) / static_cast<દશાંશ>(n_head_log2));
            અચલ દશાંશ m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<દશાંશ>(n_head_log2));

            std::vector<દશાંશ> slopes(n_head);
            માટે (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<દશાંશ>(h + 1)) :
                                              std::pow(m1, static_cast<દશાંશ>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_ગાંઠ =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            જો (slope_ગાંઠ.get_element_type() != mask.get_element_type()) {
                slope_ગાંઠ = std::make_shared<ov::op::v0::Convert>(slope_ગાંઠ, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_ગાંઠ, slope_shape, ખોટું);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_આગત = logits;
    જો (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_આગત(sinks_આગત_idx);
        જો (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, ખોટું);

        auto logits_shape = std::make_shared<ov::op::v3::ShapeOf>(logits, ov::element::i64);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        auto shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        auto sink_prefix_shape = std::make_shared<ov::op::v8::Slice>(logits_shape, zero, three, one, shape_axis);
        auto sink_last_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto sink_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{sink_prefix_shape, sink_last_dim}, 0);
        auto sink_column = std::make_shared<ov::op::v3::Broadcast>(sinks_4d, sink_broadcast_shape,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        softmax_આગત = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_આગત, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        પાછા rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_આગત, -1);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
