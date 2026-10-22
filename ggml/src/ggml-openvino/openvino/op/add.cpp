#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reduce_sum.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_add(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ଯଦି (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_size = context.get_view_ନିବେଶ_size(1);
        auto base_name = context.get_view_ନିବେଶ_src_name(1, view_size - 1);
        auto base = context.get_ନିବେଶ(base_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ଅଚଳ auto base_rank = base.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(base_rank.is_static() && base_rank.get_length() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ଅଚଳ int64_t rank = base_rank.get_length();

        auto reduced = std::make_shared<ov::op::v1::ReduceSum>(
            base, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), ମିଥ୍ୟା);
        auto res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
    }

    auto ନିବେଶ_0 = process_view_ନିବେଶ_new(context, 0);
    auto ନିବେଶ_1 = process_view_ନିବେଶ_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    auto output_type = context.get_output_type();
    ଯଦି (ନିବେଶ_0.get_element_type() != ନିବେଶ_1.get_element_type()) {
        ଯଦି (ନିବେଶ_0.get_element_type() != ov::element::f32) {
            ନିବେଶ_0 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_0, ov::element::f32);
        }
        ଯଦି (ନିବେଶ_1.get_element_type() != ov::element::f32) {
            ନିବେଶ_1 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(ନିବେଶ_0, ନିବେଶ_1);
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
