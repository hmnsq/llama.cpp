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
    أرجع shape.rank().is_static() && shape.rank().get_لمائی() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> میل_kv_append(ثابت std::shared_ptr<ov::op::v6::سونپو_2> & سونپو) {
    خودکار concat = ov::as_type_ptr<ov::op::v0::Concat>(سونپو->get_داخلہ_گرہ_shared_ptr(0));
    إذا (!concat || concat->get_داخلہ_ناپ() != 2 || concat->get_axis() != 1) {
        أرجع خالی_سوچک;
    }
    خودکار read_قدر = ov::as_type_ptr<ov::op::v6::Read_قدر>(concat->get_داخلہ_گرہ_shared_ptr(0));
    إذا (!read_قدر || read_قدر->get_variable() != سونپو->get_variable()) {
        أرجع خالی_سوچک;
    }
    إذا (!can_move_seq_axis(read_قدر->get_output_partial_shape(0))) {
        أرجع خالی_سوچک;
    }
    أرجع concat;
}

}  // namespace

منطقي KVStateSeqAxis::لگاتار_on_model(ثابت std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::سونپو_2>> assigns;
    لكل (ثابت خودکار & op : model->get_ops()) {
        إذا (خودکار سونپو = ov::as_type_ptr<ov::op::v6::سونپو_2>(op)) {
            assigns.push_back(سونپو);
        }
    }

    منطقي بدلیا = خطأ;
    لكل (ثابت خودکار & سونپو : assigns) {
        خودکار concat = میل_kv_append(سونپو);
        إذا (!concat) {
            استمر;
        }
        خودکار read_قدر = ov::as_type_ptr<ov::op::v6::Read_قدر>(concat->get_داخلہ_گرہ_shared_ptr(0));

        خودکار variable = read_قدر->get_variable();
        خودکار info = variable->get_info();
        ثابت خودکار & shape = info.انکڑے_shape;
        info.انکڑے_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_قدر->validate_and_infer_types();

        خودکار readers = concat->output(0).get_نشانہ_inputs();

        خودکار new_صفاں = concat->داخلہ_قدر(1);
        خودکار perm_وچ = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->سیٹ_دلیل(1, std::make_shared<ov::op::v1::Transpose>(new_صفاں, perm_وچ));
        concat->سیٹ_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        لكل (ثابت خودکار & reader : readers) {
            خودکار * گرہ = reader.get_گرہ();
            إذا (ov::is_type<ov::op::v6::سونپو_2>(گرہ)) {
                استمر;
            }
            منطقي dropped = خطأ;
            إذا (خودکار * transpose = ov::as_type<ov::op::v1::Transpose>(گرہ)) {
                خودکار ترتیب = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_داخلہ_گرہ_shared_ptr(1));
                إذا (ترتیب && ترتیب->cast_سمتیہ<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = سچ;
                }
            }
            إذا (!dropped) {
                خودکار perm_باہرلا = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_سروت_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_باہرلا));
            }
        }
        بدلیا = سچ;
    }

    أرجع بدلیا;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
