#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/transpose.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_transpose(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto निवेशः_stride = context.get_निवेशः_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, पूर्णाङ्क>> output_stride_dims;
    std::vector<std::pair<size_t, पूर्णाङ्क>> निवेशः_stride_dims;

    अत्थाय (पूर्णाङ्क i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        निवेशः_stride_dims.push_back({निवेशः_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(निवेशः_stride_dims.rbegin(), निवेशः_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    अत्थाय (पूर्णाङ्क i = 0; i < 4; ++i) {
        पूर्णाङ्क output_dim = output_stride_dims[i].second;
        पूर्णाङ्क निवेशः_dim = निवेशः_stride_dims[i].second;
        permute_order[output_dim] = निवेशः_dim;
    }

    auto निवेशः = process_view_निवेशः_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        निवेशः, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    पटिनिवत्त rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
