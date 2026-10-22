#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/reduce_sum.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_కూడు(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    ఒకవేళ (context.get_op_case() == 1) {
        // MoE expert-plane sum (see is_moe_expert_sum_add): input 1 is a VIEW plane of the
        // shared base tensor `experts` = [n_embd, n_expert_used, n_tokens, 1] (ggml order) ->
        // [1, n_tokens, n_expert_used, n_embd] (OV order). The whole ADD chain is equivalent to
        // reducing the expert axis (OV axis 2) of that base, so bypass the chain and the
        // per-plane Slices entirely.
        size_t view_కొలత = context.get_view_ఆదానం_కొలత(1);
        స్వయంచాలక పునాది_name = context.get_view_ఆదానం_src_name(1, view_కొలత - 1);
        స్వయంచాలక పునాది = context.get_ఆదానం(పునాది_name);

        // Stateful models drop the leading batch dim, so the base is rank 3 and both axes
        // below shift down by one. Take them from the actual rank: the expert axis is always
        // second from last, and the token axis is re-added just before it.
        స్థిరరాశి స్వయంచాలక పునాది_rank = పునాది.get_partial_shape().rank();
        FRONT_చివర_OP_CONVERSION_CHECK(పునాది_rank.is_static() && పునాది_rank.get_నిడివి() >= 3,
                                      "MoE expert sum needs a static rank of at least 3");
        స్థిరరాశి int64_t rank = పునాది_rank.get_నిడివి();

        స్వయంచాలక reduced = std::make_shared<ov::op::v1::Reduce_కూడిక>(
            పునాది, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {rank - 2}), అబద్ధం);
        స్వయంచాలక res = std::make_shared<ov::op::v0::Unsqueeze>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 3}));
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    స్వయంచాలక ఆదానం_0 = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక ఆదానం_1 = process_view_ఆదానం_new(context, 1);
    // opset1::Add needs matching types (e.g. fused ADD_ADD mixes f16/f32); add in f32, cast once.
    స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (ఆదానం_0.get_element_type() != ఆదానం_1.get_element_type()) {
        ఒకవేళ (ఆదానం_0.get_element_type() != ov::element::f32) {
            ఆదానం_0 = std::make_shared<ov::op::v0::Convert>(ఆదానం_0, ov::element::f32);
        }
        ఒకవేళ (ఆదానం_1.get_element_type() != ov::element::f32) {
            ఆదానం_1 = std::make_shared<ov::op::v0::Convert>(ఆదానం_1, ov::element::f32);
        }
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::కూడు>(ఆదానం_0, ఆదానం_1);
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
