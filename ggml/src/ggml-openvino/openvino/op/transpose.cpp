#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <openvino/op/transpose.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_transpose(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto આગત_stride = context.get_આગત_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, પૂર્ણાંક>> output_stride_dims;
    std::vector<std::pair<size_t, પૂર્ણાંક>> આગત_stride_dims;

    માટે (પૂર્ણાંક i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        આગત_stride_dims.push_back({આગત_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(આગત_stride_dims.rbegin(), આગત_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    માટે (પૂર્ણાંક i = 0; i < 4; ++i) {
        પૂર્ણાંક output_dim = output_stride_dims[i].second;
        પૂર્ણાંક આગત_dim = આગત_stride_dims[i].second;
        permute_order[output_dim] = આગત_dim;
    }

    auto આગત = process_view_આગત_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        આગત, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
