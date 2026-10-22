#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/add.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/op/reduce_sum.hpp>
#অন্তর্ভাৱ <openvino/op/unsqueeze.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_add(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    যদি (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_size = context.get_view_প্ৰবেশ_size(1);
        auto base_name = context.get_view_প্ৰবেশ_src_name(1, view_size - 1);
        auto base = context.get_প্ৰবেশ(base_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ধ্রুৱ auto base_rank = base.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(base_rank.is_static() && base_rank.get_length() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ধ্রুৱ int64_t rank = base_rank.get_length();

        auto reduced = std::make_shared<ov::op::v1::ReduceSum>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), মিছা);
        auto res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
    }

    auto প্ৰবেশ_0 = process_view_প্ৰবেশ_new(context, 0);
    auto প্ৰবেশ_1 = process_view_প্ৰবেশ_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    auto output_type = context.get_output_type();
    যদি (প্ৰবেশ_0.get_element_type() != প্ৰবেশ_1.get_element_type()) {
        যদি (প্ৰবেশ_0.get_element_type() != ov::element::f32) {
            প্ৰবেশ_0 = std::make_shared<ov::op::v0::Convert>(প্ৰবেশ_0, ov::element::f32);
        }
        যদি (প্ৰবেশ_1.get_element_type() != ov::element::f32) {
            প্ৰবেশ_1 = std::make_shared<ov::op::v0::Convert>(প্ৰবেশ_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(প্ৰবেশ_0, প্ৰবেশ_1);
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
