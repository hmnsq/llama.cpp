#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cassert>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/core/type.hpp>
#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/scatter_elements_update.hpp>
#అన్తర్భావ <openvino/op/scatter_update.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_సమితి_అడ్డువరుసలు(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 3, 3);

    స్వయంచాలక దత్తాంశం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక indices = process_view_ఆదానం_new(context, 1);
    స్వయంచాలక dst = process_view_ఆదానం_new(context, 2);

    దత్తాంశం = std::make_shared<ov::op::v0::Convert>(దత్తాంశం, context.get_output_type());

    స్థిరరాశి స్వయంచాలక indices_shape = context.get_ఆదానం_shape(1);
    స్థిరరాశి తార్కిక multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_నిడివి() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_నిడివి() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_నిడివి() > 1));

    స్వయంచాలక axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    ఒకవేళ (context.is_stateful()) {
        పూర్ణాంకం concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_నిడివి();
        int64_t dim3 = dst.get_partial_shape()[3].get_నిడివి();
        దత్తాంశం = std::make_shared<ov::op::v1::Reshape>(
            దత్తాంశం, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), అబద్ధం);
        res = std::make_shared<ov::op::v0::Concat>(Output_సదిశ{dst, దత్తాంశం}, concat_axis);
    } లేకపోతే ఒకవేళ (multidim_indices) {
        స్వయంచాలక updates_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(దత్తాంశం, ov::element::i64);

        స్వయంచాలక indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(Output_సదిశ{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        స్వయంచాలక indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, అబద్ధం);
        స్వయంచాలక broadcasted_indices = std::make_shared<ov::op::v3::ప్రసారం>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, దత్తాంశం, axes);
    } లేకపోతే {
        స్వయంచాలక అడ్డువరుస_కొలత = context.get_ఆదానం_shape(2)[3].get_నిడివి();
        స్వయంచాలక ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        స్వయంచాలక దత్తాంశం_reshaped = std::make_shared<ov::op::v1::Reshape>(
            దత్తాంశం,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) అడ్డువరుస_కొలత}),
            అబద్ధం);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, దత్తాంశం_reshaped, axes);
    }

    స్వయంచాలక dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_కణుపు_shared_ptr());
    ఒకవేళ (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        స్వయంచాలక dst_shape_partial = dst_reshape->get_ఆదానం_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_నిడివి(), dst_shape_partial[1].get_నిడివి(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_నిడివి() : -1,
                                          dst_shape_partial[3].get_నిడివి()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    అబద్ధం);
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
