#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/frontend/exception.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_concat(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ଅଚଳ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    ଅଚଳ auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ଅଚଳ auto rank = output_shape.rank().get_length();
    ଅଚଳ int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto ନିବେଶ_0 = process_view_ନିବେଶ_new(context, 0);
    auto ନିବେଶ_1 = process_view_ନିବେଶ_new(context, 1);
    ଅଚଳ auto output_type = context.get_output_type();

    ଯଦି (ନିବେଶ_0.get_element_type() != output_type) {
        ନିବେଶ_0 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_0, output_type);
    }
    ଯଦି (ନିବେଶ_1.get_element_type() != output_type) {
        ନିବେଶ_1 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_1, output_type);
    }

    ଅଚଳ auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{ନିବେଶ_0, ନିବେଶ_1}, axis);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
