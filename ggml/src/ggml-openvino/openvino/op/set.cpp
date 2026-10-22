#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/range.hpp>
#అన్తర్భావ <openvino/op/reduce_prod.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/scatter_update.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_సదిశ translate_సమితి(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    స్వయంచాలక dst = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక src = process_view_ఆదానం_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    స్థిరరాశి స్వయంచాలక dst_stride = context.get_ఆదానం_stride(0);
    FRONT_చివర_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    స్థిరరాశి స్వయంచాలక * op_params = reinterpret_cast<స్థిరరాశి uint32_t *>(context.get_output_op_params());
    స్థిరరాశి size_t స్థానభ్రంశం = static_cast<size_t>(op_params[3]);

    స్థిరరాశి size_t elem_కొలత = dst_stride.back();
    FRONT_చివర_OP_CONVERSION_CHECK(elem_కొలత != 0 && స్థానభ్రంశం % elem_కొలత == 0,
                                  "SET offset must be aligned to destination element size");

    స్థిరరాశి int64_t స్థానభ్రంశం_elems = static_cast<int64_t>(స్థానభ్రంశం / elem_కొలత);

    స్వయంచాలక dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        అబద్ధం);

    స్వయంచాలక src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        అబద్ధం);

    స్వయంచాలక src_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(src_flat, ov::element::i64);
    స్వయంచాలక src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        అబద్ధం);

    స్వయంచాలక start = ov::op::v0::Constant::create(ov::element::i64, {}, {స్థానభ్రంశం_elems});
    స్వయంచాలక stop = std::make_shared<ov::op::v1::కూడు>(start, src_len);
    స్వయంచాలక అడుగు = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    స్వయంచాలక indices = std::make_shared<ov::op::v4::Range>(start, stop, అడుగు, ov::element::i64);
    స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    స్వయంచాలక updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    స్వయంచాలక dst_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(dst, ov::element::i64);
    స్వయంచాలక res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, అబద్ధం);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
