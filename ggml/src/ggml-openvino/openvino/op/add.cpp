#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/unsqueeze.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_add(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    اگر (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_size = context.get_view_درون_داد_size(1);
        auto base_name = context.get_view_درون_داد_src_name(1, view_size - 1);
        auto base = context.get_درون_داد(base_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ثابت auto base_rank = base.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(base_rank.is_static() && base_rank.get_length() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ثابت int64_t rank = base_rank.get_length();

        auto reduced = std::make_shared<ov::op::v1::ReduceSum>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), نادرست);
        auto res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        بازگشت rename_outputs_with_suffix({res}, context.get_name());
    }

    auto درون_داد_0 = process_view_درون_داد_new(context, 0);
    auto درون_داد_1 = process_view_درون_داد_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    auto output_type = context.get_output_type();
    اگر (درون_داد_0.get_element_type() != درون_داد_1.get_element_type()) {
        اگر (درون_داد_0.get_element_type() != ov::element::f32) {
            درون_داد_0 = std::make_shared<ov::op::v0::Convert>(درون_داد_0, ov::element::f32);
        }
        اگر (درون_داد_1.get_element_type() != ov::element::f32) {
            درون_داد_1 = std::make_shared<ov::op::v0::Convert>(درون_داد_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(درون_داد_0, درون_داد_1);
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
