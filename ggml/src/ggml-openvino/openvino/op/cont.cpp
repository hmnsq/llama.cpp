
#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_cont(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto src_shape = context.get_உள்ளீடு_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    எனில் (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        உள்ளீடு, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), பொய்);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
