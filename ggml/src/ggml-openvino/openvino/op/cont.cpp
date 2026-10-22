
#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_cont(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto src_shape = context.get_প্রবেশ_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    যদি (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        প্রবেশ, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), অসত্য);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
