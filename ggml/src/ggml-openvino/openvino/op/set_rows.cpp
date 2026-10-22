#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cassert>
#شامل <cstdint>
#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/core/type.hpp>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/scatter_elements_update.hpp>
#شامل <openvino/op/scatter_update.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_سیٹ_صفیں(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 3, 3);

    خودکار ڈیٹا = process_view_داخلہ_new(context, 0);
    خودکار indices = process_view_داخلہ_new(context, 1);
    خودکار dst = process_view_داخلہ_new(context, 2);

    ڈیٹا = std::make_shared<ov::op::v0::Convert>(ڈیٹا, context.get_output_type());

    ثابت خودکار indices_shape = context.get_داخلہ_shape(1);
    ثابت منطقی multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_لمبائی() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_لمبائی() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_لمبائی() > 1));

    خودکار axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    اگر (context.is_stateful()) {
        صحیح concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_لمبائی();
        int64_t dim3 = dst.get_partial_shape()[3].get_لمبائی();
        ڈیٹا = std::make_shared<ov::op::v1::Reshape>(
            ڈیٹا, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), جھوٹ);
        res = std::make_shared<ov::op::v0::Concat>(Output_سمتیہ{dst, ڈیٹا}, concat_axis);
    } ورنہ اگر (multidim_indices) {
        خودکار updates_shape = std::make_shared<ov::op::v3::Shape_کا>(ڈیٹا, ov::element::i64);

        خودکار indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(Output_سمتیہ{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        خودکار indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, جھوٹ);
        خودکار broadcasted_indices = std::make_shared<ov::op::v3::نشر>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, ڈیٹا, axes);
    } ورنہ {
        خودکار صف_سائز = context.get_داخلہ_shape(2)[3].get_لمبائی();
        خودکار ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        خودکار ڈیٹا_reshaped = std::make_shared<ov::op::v1::Reshape>(
            ڈیٹا,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) صف_سائز}),
            جھوٹ);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, ڈیٹا_reshaped, axes);
    }

    خودکار dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_گرہ_shared_ptr());
    اگر (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        خودکار dst_shape_partial = dst_reshape->get_داخلہ_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_لمبائی(), dst_shape_partial[1].get_لمبائی(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_لمبائی() : -1,
                                          dst_shape_partial[3].get_لمبائی()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    جھوٹ);
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
