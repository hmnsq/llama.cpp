#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/transpose.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_transpose(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    स्वतः निवेश_stride = context.get_निवेश_stride(0);
    स्वतः output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<माप_प्रकार, पूर्णाङ्क>> output_stride_dims;
    std::vector<std::pair<माप_प्रकार, पूर्णाङ्क>> निवेश_stride_dims;

    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        निवेश_stride_dims.push_back({निवेश_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(निवेश_stride_dims.rbegin(), निवेश_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        पूर्णाङ्क output_dim = output_stride_dims[i].second;
        पूर्णाङ्क निवेश_dim = निवेश_stride_dims[i].second;
        permute_order[output_dim] = निवेश_dim;
    }

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    स्वतः res = std::make_shared<ov::op::v1::Transpose>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
