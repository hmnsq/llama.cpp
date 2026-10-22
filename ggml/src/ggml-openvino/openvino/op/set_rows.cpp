#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cassert>
#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/scatter_elements_update.hpp>
#ضمن <openvino/op/scatter_update.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_سمتیہ translate_سیٹ_صفاں(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 3, 3);

    خودکار انکڑے = process_view_داخلہ_new(context, 0);
    خودکار indices = process_view_داخلہ_new(context, 1);
    خودکار dst = process_view_داخلہ_new(context, 2);

    انکڑے = std::make_shared<ov::op::v0::Convert>(انکڑے, context.get_output_type());

    ثابت خودکار indices_shape = context.get_داخلہ_shape(1);
    ثابت منطقي multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_لمائی() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_لمائی() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_لمائی() > 1));

    خودکار axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    إذا (context.is_stateful()) {
        صحيح concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_لمائی();
        int64_t dim3 = dst.get_partial_shape()[3].get_لمائی();
        انکڑے = std::make_shared<ov::op::v1::Reshape>(
            انکڑے, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), خطأ);
        res = std::make_shared<ov::op::v0::Concat>(Output_سمتیہ{dst, انکڑے}, concat_axis);
    } وإلا إذا (multidim_indices) {
        خودکار updates_shape = std::make_shared<ov::op::v3::Shape_دا>(انکڑے, ov::element::i64);

        خودکار indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(Output_سمتیہ{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        خودکار indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, خطأ);
        خودکار broadcasted_indices = std::make_shared<ov::op::v3::نشر>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, انکڑے, axes);
    } وإلا {
        خودکار صف_ناپ = context.get_داخلہ_shape(2)[3].get_لمائی();
        خودکار ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        خودکار انکڑے_reshaped = std::make_shared<ov::op::v1::Reshape>(
            انکڑے,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) صف_ناپ}),
            خطأ);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, انکڑے_reshaped, axes);
    }

    خودکار dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_گرہ_shared_ptr());
    إذا (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        خودکار dst_shape_partial = dst_reshape->get_داخلہ_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_لمائی(), dst_shape_partial[1].get_لمائی(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_لمائی() : -1,
                                          dst_shape_partial[3].get_لمائی()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    خطأ);
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
