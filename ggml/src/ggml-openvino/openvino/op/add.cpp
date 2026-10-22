#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_sum.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_ਜੋੜੋ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਜੇ (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_ਮਾਪ = context.get_view_ਨਿਵੇਸ਼_ਮਾਪ(1);
        ਸਵੈਚਾਲਿਤ ਅਧਾਰ_name = context.get_view_ਨਿਵੇਸ਼_src_name(1, view_ਮਾਪ - 1);
        ਸਵੈਚਾਲਿਤ ਅਧਾਰ = context.get_ਨਿਵੇਸ਼(ਅਧਾਰ_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਅਧਾਰ_rank = ਅਧਾਰ.get_partial_shape().rank();
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਅਧਾਰ_rank.is_static() && ਅਧਾਰ_rank.get_ਲੰਮਾਈ() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ਅਚਲ int64_t rank = ਅਧਾਰ_rank.get_ਲੰਮਾਈ();

        ਸਵੈਚਾਲਿਤ reduced = std::make_shared<ov::op::v1::Reduce_ਜੋੜ>(
            ਅਧਾਰ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), ਝੂਠ);
        ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_0 = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_1 = process_view_ਨਿਵੇਸ਼_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (ਨਿਵੇਸ਼_0.get_element_type() != ਨਿਵੇਸ਼_1.get_element_type()) {
        ਜੇ (ਨਿਵੇਸ਼_0.get_element_type() != ov::element::f32) {
            ਨਿਵੇਸ਼_0 = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼_0, ov::element::f32);
        }
        ਜੇ (ਨਿਵੇਸ਼_1.get_element_type() != ov::element::f32) {
            ਨਿਵੇਸ਼_1 = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::ਜੋੜੋ>(ਨਿਵੇਸ਼_0, ਨਿਵੇਸ਼_1);
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
