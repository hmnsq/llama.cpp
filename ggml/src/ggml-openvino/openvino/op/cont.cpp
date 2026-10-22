
#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_cont(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto src_shape = context.get_આગત_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    જો (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto આગત = process_view_આગત_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        આગત, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), ખોટું);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
