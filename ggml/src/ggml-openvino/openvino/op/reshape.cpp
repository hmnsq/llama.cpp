#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_reshape(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    ఒకవేళ (context.get_ఆదానం(0).get_partial_shape().is_static() &&
        context.get_ఆదానం_shape(0) == context.get_output_shape()) {
        తిరిగి {context.get_ఆదానం(0)};
    }

    స్థిరరాశి పూర్ణాంకం op_case = context.get_op_case();
    స్థిరరాశి స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_కణుపు;
    ఎంపిక (op_case) {
    స్థితి 0:
        ఆపు;
    స్థితి 1:
    స్థితి 9:
        shape[1] = -1;
        ఒకవేళ (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        ఆపు;
    స్థితి 2:
    స్థితి 3:
        shape[2] = -1;
        ఒకవేళ (op_case == 3) {
            shape[3] = 1;
        }
        ఆపు;
    స్థితి 4:
        తిరిగి {context.get_ఆదానం(0).get_కణుపు_shared_ptr()->ఆదానం_విలువ(0)};
    స్థితి 5:
    స్థితి 7:
        shape = {1, 1, -1, shape[3]};
        ఒకవేళ (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        ఆపు;
    స్థితి 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        ఒకవేళ (context.has_ఆదానం("s_copy_active_slot_len")) {
            స్వయంచాలక n_స్థానం_చురుకైన_len = context.get_ఆదానం("s_copy_active_slot_len");
            స్వయంచాలక emb_కొలత = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            స్వయంచాలక neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_కణుపు =
                std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{one, n_స్థానం_చురుకైన_len, neg_one, emb_కొలత}, 0);
        } లేకపోతే {
            shape = {1, 1, -1, shape[3]};
        }
        ఆపు;
    స్థితి 8:
        shape[0] = -1;
        ఆపు;
    అప్రమేయ:
        FRONT_చివర_OP_CONVERSION_CHECK(అబద్ధం, "Unsupported RESHAPE case: ", op_case);
    }
    ఒకవేళ (!new_shape_కణుపు) {
        new_shape_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    స్వయంచాలక res = std::make_shared<ov::op::v1::Reshape>(context.get_ఆదానం(0), new_shape_కణుపు, అబద్ధం);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
