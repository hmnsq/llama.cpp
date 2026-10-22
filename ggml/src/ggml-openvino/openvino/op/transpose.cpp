#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/transpose.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_transpose(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    خودکار درون_داد_stride = context.get_درون_داد_stride(0);
    خودکار output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, صحيح>> output_stride_dims;
    std::vector<std::pair<size_t, صحيح>> درون_داد_stride_dims;

    لكل (صحيح i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        درون_داد_stride_dims.push_back({درون_داد_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(درون_داد_stride_dims.rbegin(), درون_داد_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> جایگشت_ترتیب(4);
    لكل (صحيح i = 0; i < 4; ++i) {
        صحيح output_dim = output_stride_dims[i].second;
        صحيح درون_داد_dim = درون_داد_stride_dims[i].second;
        جایگشت_ترتیب[output_dim] = درون_داد_dim;
    }

    خودکار درون_داد = process_view_درون_داد_new(context, 0);

    خودکار res = std::make_shared<ov::op::v1::Transpose>(
        درون_داد, ov::op::v0::Constant::create(ov::element::i64, {4}, جایگشت_ترتیب));
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
