#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/reduce_sum.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_add(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    જો (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_size = context.get_view_આગત_size(1);
        auto base_name = context.get_view_આગત_src_name(1, view_size - 1);
        auto base = context.get_આગત(base_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        અચલ auto base_rank = base.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(base_rank.is_static() && base_rank.get_length() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        અચલ int64_t rank = base_rank.get_length();

        auto reduced = std::make_shared<ov::op::v1::ReduceSum>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), ખોટું);
        auto res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        પાછા rename_outputs_with_suffix({res}, context.get_name());
    }

    auto આગત_0 = process_view_આગત_new(context, 0);
    auto આગત_1 = process_view_આગત_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    auto output_type = context.get_output_type();
    જો (આગત_0.get_element_type() != આગત_1.get_element_type()) {
        જો (આગત_0.get_element_type() != ov::element::f32) {
            આગત_0 = std::make_shared<ov::op::v0::Convert>(આગત_0, ov::element::f32);
        }
        જો (આગત_1.get_element_type() != ov::element::f32) {
            આગત_1 = std::make_shared<ov::op::v0::Convert>(આગત_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(આગત_0, આગત_1);
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
