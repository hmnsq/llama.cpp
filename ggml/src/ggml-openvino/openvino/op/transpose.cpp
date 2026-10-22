#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/transpose.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_transpose(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    خودڪار داخلا_stride = context.get_داخلا_stride(0);
    خودڪار output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, صحيح>> output_stride_dims;
    std::vector<std::pair<size_t, صحيح>> داخلا_stride_dims;

    لكل (صحيح i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        داخلا_stride_dims.push_back({داخلا_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(داخلا_stride_dims.rbegin(), داخلا_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> ترتيب_بدلايو_ترتيب(4);
    لكل (صحيح i = 0; i < 4; ++i) {
        صحيح output_dim = output_stride_dims[i].second;
        صحيح داخلا_dim = داخلا_stride_dims[i].second;
        ترتيب_بدلايو_ترتيب[output_dim] = داخلا_dim;
    }

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);

    خودڪار res = std::make_shared<ov::op::v1::Transpose>(
        داخلا_2, ov::op::v0::Constant::create(ov::element::i64, {4}, ترتيب_بدلايو_ترتيب));
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
