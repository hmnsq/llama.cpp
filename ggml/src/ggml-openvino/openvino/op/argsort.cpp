#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <openvino/frontend/exception.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/topk.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_argsort(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto આગત = process_view_આગત_new(context, 0);

    અચલ int32_t order = context.get_output_op_params()[0];

    ov::op::v11::ટોચ_K::Mode mode;
    પસંદગી (order) {
    સ્થિતિ GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::ટોચ_K::Mode::MIN;
        તોડો;
    સ્થિતિ GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::ટોચ_K::Mode::MAX;
        તોડો;
    મૂળભૂત:
        FRONT_END_OP_CONVERSION_CHECK(ખોટું, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    અચલ પૂર્ણાંક axis = (context.is_stateful() && આગત.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(આગત.get_ગાંઠ_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::ટોચ_K>(આગત, k, axis, mode, ov::op::v11::ટોચ_K::SortType::SORT_કિંમતો,
                                                    context.get_output_type(), ખોટું);

    પાછા rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
