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
    أرجع shape.rank().is_static() && shape.rank().get_ڊيگهائي() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> ميلاپ_kv_append(ثابت std::shared_ptr<ov::op::v6::سونپيو_2> & سونپيو) {
    خودڪار concat = ov::as_type_ptr<ov::op::v0::Concat>(سونپيو->get_داخلا_ڳنڍ_shared_ptr(0));
    إذا (!concat || concat->get_داخلا_سائيز() != 2 || concat->get_axis() != 1) {
        أرجع خالي_اشارو;
    }
    خودڪار read_قدر = ov::as_type_ptr<ov::op::v6::Read_قدر>(concat->get_داخلا_ڳنڍ_shared_ptr(0));
    إذا (!read_قدر || read_قدر->get_variable() != سونپيو->get_variable()) {
        أرجع خالي_اشارو;
    }
    إذا (!can_move_seq_axis(read_قدر->get_output_partial_shape(0))) {
        أرجع خالي_اشارو;
    }
    أرجع concat;
}

}  // namespace

منطقي KVStateSeqAxis::لاڳيتو_on_model(ثابت std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::سونپيو_2>> assigns;
    لكل (ثابت خودڪار & op : model->get_ops()) {
        إذا (خودڪار سونپيو = ov::as_type_ptr<ov::op::v6::سونپيو_2>(op)) {
            assigns.push_back(سونپيو);
        }
    }

    منطقي بدليو = خطأ;
    لكل (ثابت خودڪار & سونپيو : assigns) {
        خودڪار concat = ميلاپ_kv_append(سونپيو);
        إذا (!concat) {
            استمر;
        }
        خودڪار read_قدر = ov::as_type_ptr<ov::op::v6::Read_قدر>(concat->get_داخلا_ڳنڍ_shared_ptr(0));

        خودڪار variable = read_قدر->get_variable();
        خودڪار info = variable->get_info();
        ثابت خودڪار & shape = info.معلومات_shape;
        info.معلومات_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_قدر->validate_and_infer_types();

        خودڪار readers = concat->output(0).get_هدف_inputs();

        خودڪار new_صفون = concat->داخلا_قدر(1);
        خودڪار perm_اندريون = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->سيٽ_دليل(1, std::make_shared<ov::op::v1::Transpose>(new_صفون, perm_اندريون));
        concat->سيٽ_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        لكل (ثابت خودڪار & reader : readers) {
            خودڪار * ڳنڍ = reader.get_ڳنڍ();
            إذا (ov::is_type<ov::op::v6::سونپيو_2>(ڳنڍ)) {
                استمر;
            }
            منطقي dropped = خطأ;
            إذا (خودڪار * transpose = ov::as_type<ov::op::v1::Transpose>(ڳنڍ)) {
                خودڪار ترتيب = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_داخلا_ڳنڍ_shared_ptr(1));
                إذا (ترتيب && ترتيب->cast_ويڪٽر<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = سچ;
                }
            }
            إذا (!dropped) {
                خودڪار perm_ٻاهريون = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_ذريعو_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_ٻاهريون));
            }
        }
        بدليو = سچ;
    }

    أرجع بدليو;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
