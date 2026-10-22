#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/mvn.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_norm(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্ৰবেশ_গ্ৰন্থি = process_view_প্ৰবেশ_new(context, 0);
    ভিন্ন eps;
    memcpy(&eps, context.get_output_op_params(), পরিমাণ(ভিন্ন));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(প্ৰবেশ_গ্ৰন্থি, axes, সঁচা, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
