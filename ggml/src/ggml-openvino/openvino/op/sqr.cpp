#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/sqrt.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_sqr(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(প্ৰবেশ, প্ৰবেশ);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(প্ৰবেশ);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
