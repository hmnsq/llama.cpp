#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/reduce_sum.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_sum_rows(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto આગત = process_view_આગત_new(context, 0);
    auto res = std::make_shared<ov::op::v1::ReduceSum>(
        આગત, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), સાચું);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
