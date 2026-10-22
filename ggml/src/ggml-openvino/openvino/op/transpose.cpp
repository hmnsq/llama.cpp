#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/transpose.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_transpose(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto درون_داد_stride = context.get_درون_داد_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, صحیح>> output_stride_dims;
    std::vector<std::pair<size_t, صحیح>> درون_داد_stride_dims;

    برای (صحیح i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        درون_داد_stride_dims.push_back({درون_داد_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(درون_داد_stride_dims.rbegin(), درون_داد_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    برای (صحیح i = 0; i < 4; ++i) {
        صحیح output_dim = output_stride_dims[i].second;
        صحیح درون_داد_dim = درون_داد_stride_dims[i].second;
        permute_order[output_dim] = درون_داد_dim;
    }

    auto درون_داد = process_view_درون_داد_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        درون_داد, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
