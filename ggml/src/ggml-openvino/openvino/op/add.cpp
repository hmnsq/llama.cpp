#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_add(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 2, 2);

    यदि (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        माप_प्रकार view_size = context.get_view_आगत_size(1);
        स्वतः base_name = context.get_view_आगत_src_name(1, view_size - 1);
        स्वतः base = context.get_आगत(base_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        अचर स्वतः base_rank = base.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(base_rank.is_static() && base_rank.get_length() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        अचर int64_t rank = base_rank.get_length();

        स्वतः reduced = std::make_shared<ov::op::v1::ReduceSum>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), असत्य);
        स्वतः res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
    }

    स्वतः आगत_0 = process_view_आगत_new(context, 0);
    स्वतः आगत_1 = process_view_आगत_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    स्वतः output_type = context.get_output_type();
    यदि (आगत_0.get_element_type() != आगत_1.get_element_type()) {
        यदि (आगत_0.get_element_type() != ov::element::f32) {
            आगत_0 = std::make_shared<ov::op::v0::Convert>(आगत_0, ov::element::f32);
        }
        यदि (आगत_1.get_element_type() != ov::element::f32) {
            आगत_1 = std::make_shared<ov::op::v0::Convert>(आगत_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(आगत_0, आगत_1);
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
