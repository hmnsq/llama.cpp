#ضمن "kv_state_seq_axis.h"

#ضمن <memory>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/op/assign.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/read_value.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

ثابت std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ساكن ثابت std::vector<int64_t> perm{0, 2, 1, 3};
    أرجع perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
منطقي can_move_seq_axis(ثابت ov::PartialShape & shape) {
    أرجع shape.rank().is_static() && shape.rank().get_اوږدتیا() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> سمون_kv_append(ثابت std::shared_ptr<ov::op::v6::ټاکل_2> & ټاکل) {
    خپلکاری concat = ov::as_type_ptr<ov::op::v0::Concat>(ټاکل->get_ننوت_غوټه_shared_ptr(0));
    إذا (!concat || concat->get_ننوت_اندازه() != 2 || concat->get_axis() != 1) {
        أرجع تش_اشاره;
    }
    خپلکاری read_ارزښت = ov::as_type_ptr<ov::op::v6::Read_ارزښت>(concat->get_ننوت_غوټه_shared_ptr(0));
    إذا (!read_ارزښت || read_ارزښت->get_variable() != ټاکل->get_variable()) {
        أرجع تش_اشاره;
    }
    إذا (!can_move_seq_axis(read_ارزښت->get_output_partial_shape(0))) {
        أرجع تش_اشاره;
    }
    أرجع concat;
}

}  // namespace

منطقي KVStateSeqAxis::پرلپسې_on_model(ثابت std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::ټاکل_2>> assigns;
    لكل (ثابت خپلکاری & op : model->get_ops()) {
        إذا (خپلکاری ټاکل = ov::as_type_ptr<ov::op::v6::ټاکل_2>(op)) {
            assigns.push_back(ټاکل);
        }
    }

    منطقي بدل_شوی = خطأ;
    لكل (ثابت خپلکاری & ټاکل : assigns) {
        خپلکاری concat = سمون_kv_append(ټاکل);
        إذا (!concat) {
            استمر;
        }
        خپلکاری read_ارزښت = ov::as_type_ptr<ov::op::v6::Read_ارزښت>(concat->get_ننوت_غوټه_shared_ptr(0));

        خپلکاری variable = read_ارزښت->get_variable();
        خپلکاری info = variable->get_info();
        ثابت خپلکاری & shape = info.معلومات_shape;
        info.معلومات_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_ارزښت->validate_and_infer_types();

        خپلکاری readers = concat->output(0).get_موخه_inputs();

        خپلکاری new_لیکې = concat->ننوت_ارزښت(1);
        خپلکاری perm_دننه = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->سېټ_دلیل(1, std::make_shared<ov::op::v1::Transpose>(new_لیکې, perm_دننه));
        concat->سېټ_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        لكل (ثابت خپلکاری & reader : readers) {
            خپلکاری * غوټه = reader.get_غوټه();
            إذا (ov::is_type<ov::op::v6::ټاکل_2>(غوټه)) {
                استمر;
            }
            منطقي dropped = خطأ;
            إذا (خپلکاری * transpose = ov::as_type<ov::op::v1::Transpose>(غوټه)) {
                خپلکاری ترتیب = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_ننوت_غوټه_shared_ptr(1));
                إذا (ترتیب && ترتیب->cast_ویکټور<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = ریښتیا;
                }
            }
            إذا (!dropped) {
                خپلکاری perm_بهر = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_سرچینه_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_بهر));
            }
        }
        بدل_شوی = ریښتیا;
    }

    أرجع بدل_شوی;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
