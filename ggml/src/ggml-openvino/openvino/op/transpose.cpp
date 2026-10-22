#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/transpose.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_transpose(ثابت نود_Context & context) {
    اڠک_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto ماسوقن_stride = context.get_ماسوقن_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, صحيح>> output_stride_dims;
    std::vector<std::pair<size_t, صحيح>> ماسوقن_stride_dims;

    لكل (صحيح i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        ماسوقن_stride_dims.push_back({ماسوقن_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(ماسوقن_stride_dims.rbegin(), ماسوقن_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    لكل (صحيح i = 0; i < 4; ++i) {
        صحيح output_dim = output_stride_dims[i].second;
        صحيح ماسوقن_dim = ماسوقن_stride_dims[i].second;
        permute_order[output_dim] = ماسوقن_dim;
    }

    auto ماسوقن = process_view_ماسوقن_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        ماسوقن, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
