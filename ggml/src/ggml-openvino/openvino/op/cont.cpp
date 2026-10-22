
#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_cont(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    auto src_shape = context.get_निवेशः_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    जइ (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto निवेशः = process_view_निवेशः_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        निवेशः, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), मिच्छ);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
