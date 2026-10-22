#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/unsqueeze.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_جوڙيو(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 2, 2);

    إذا (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_سائيز = context.get_view_داخلا_سائيز(1);
        خودڪار بنياد_name = context.get_view_داخلا_src_name(1, view_سائيز - 1);
        خودڪار بنياد = context.get_داخلا(بنياد_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        ثابت خودڪار بنياد_rank = بنياد.get_partial_shape().rank();
        FRONT_پڇاڙي_OP_CONVERSION_CHECK(بنياد_rank.is_static() && بنياد_rank.get_ڊيگهائي() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        ثابت int64_t rank = بنياد_rank.get_ڊيگهائي();

        خودڪار reduced = std::make_shared<ov::op::v1::Reduce_جوڙ>(
            بنياد, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), خطأ);
        خودڪار res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    خودڪار داخلا_0 = process_view_داخلا_new(context, 0);
    خودڪار داخلا_1 = process_view_داخلا_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    خودڪار output_type = context.get_output_type();
    إذا (داخلا_0.get_element_type() != داخلا_1.get_element_type()) {
        إذا (داخلا_0.get_element_type() != ov::element::f32) {
            داخلا_0 = std::make_shared<ov::op::v0::Convert>(داخلا_0, ov::element::f32);
        }
        إذا (داخلا_1.get_element_type() != ov::element::f32) {
            داخلا_1 = std::make_shared<ov::op::v0::Convert>(داخلا_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::جوڙيو>(داخلا_0, داخلا_1);
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
