#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/transpose.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_transpose(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto உள்ளீடு_stride = context.get_உள்ளீடு_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, முழு>> output_stride_dims;
    std::vector<std::pair<size_t, முழு>> உள்ளீடு_stride_dims;

    க்கு (முழு i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        உள்ளீடு_stride_dims.push_back({உள்ளீடு_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(உள்ளீடு_stride_dims.rbegin(), உள்ளீடு_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    க்கு (முழு i = 0; i < 4; ++i) {
        முழு output_dim = output_stride_dims[i].second;
        முழு உள்ளீடு_dim = உள்ளீடு_stride_dims[i].second;
        permute_order[output_dim] = உள்ளீடு_dim;
    }

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        உள்ளீடு, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
