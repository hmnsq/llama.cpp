#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_sum_rows(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v1::ReduceSum>(
        आदान, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सत्य);

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
