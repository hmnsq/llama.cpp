#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/topk.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_argsort(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);

    స్థిరరాశి int32_t క్రమం = context.get_output_op_params()[0];

    ov::op::v11::శిఖరం_K::బహుళకం_2 mode;
    ఎంపిక (క్రమం) {
    స్థితి GGML_క్రమీకరించు_క్రమం_ASC:
        mode = ov::op::v11::శిఖరం_K::బహుళకం_2::MIN;
        ఆపు;
    స్థితి GGML_క్రమీకరించు_క్రమం_DESC:
        mode = ov::op::v11::శిఖరం_K::బహుళకం_2::MAX;
        ఆపు;
    అప్రమేయ:
        FRONT_చివర_OP_CONVERSION_CHECK(అబద్ధం, "Unsupported GGML_OP_ARGSORT order: ", క్రమం);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    స్థిరరాశి పూర్ణాంకం axis = (context.is_stateful() && ఆదానం.get_partial_shape().rank() == 3) ? 2 : 3;

    స్వయంచాలక k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(ఆదానం.get_కణుపు_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    స్వయంచాలక topk = std::make_shared<ov::op::v11::శిఖరం_K>(ఆదానం, k, axis, mode, ov::op::v11::శిఖరం_K::క్రమీకరించు_Type::క్రమీకరించు_విలువలు,
                                                    context.get_output_type(), అబద్ధం);

    తిరిగి rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
