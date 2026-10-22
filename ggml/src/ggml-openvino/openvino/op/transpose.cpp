#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/transpose.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ویکټور translate_transpose(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    خپلکاری ننوت_stride = context.get_ننوت_stride(0);
    خپلکاری output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, صحيح>> output_stride_dims;
    std::vector<std::pair<size_t, صحيح>> ننوت_stride_dims;

    لكل (صحيح i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        ننوت_stride_dims.push_back({ننوت_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(ننوت_stride_dims.rbegin(), ننوت_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> ترتیب_بدلول_ترتیب(4);
    لكل (صحيح i = 0; i < 4; ++i) {
        صحيح output_dim = output_stride_dims[i].second;
        صحيح ننوت_dim = ننوت_stride_dims[i].second;
        ترتیب_بدلول_ترتیب[output_dim] = ننوت_dim;
    }

    خپلکاری ننوت = process_view_ننوت_new(context, 0);

    خپلکاری res = std::make_shared<ov::op::v1::Transpose>(
        ننوت, ov::op::v0::Constant::create(ov::element::i64, {4}, ترتیب_بدلول_ترتیب));
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
