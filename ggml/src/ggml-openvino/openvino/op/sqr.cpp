#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sqrt.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_sqr(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(ନିବେଶ, ନିବେଶ);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(ନିବେଶ);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
