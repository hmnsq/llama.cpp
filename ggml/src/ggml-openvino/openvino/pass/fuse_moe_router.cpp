#ضمن "fuse_moe_router.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & ارزښتونه) {
    خپلکاری غوټه = ov::as_type_ptr<ov::op::v0::Constant>(output.get_غوټه_shared_ptr());
    أرجع غوټه && غوټه->get_element_type().is_integral_شمېره() && غوټه->cast_ویکټور<int64_t>() == ارزښتونه;
}

}  // namespace

Fuse_Moe_لارښود::Fuse_Moe_لارښود() {
    کارول نطاق ov::pass::pattern;
    کارول نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    خپلکاری logits = wrap_type<v0::MatMul>();
    خپلکاری softmax = wrap_type<v8::Softmax>({logits});
    خپلکاری topk = wrap_type<v11::سر_K>({softmax, any_ننوت()});
    topk->سېټ_output_اندازه(2);
    خپلکاری ids = wrap_type<v8::ټوټه>({topk->output(1), any_ننوت(), any_ننوت(), any_ننوت(), any_ننوت()});
    خپلکاری ids_2d = wrap_type<v0::Squeeze>({ids, any_ننوت()});
    خپلکاری probs = wrap_type<v1::Reshape>({softmax, any_ننوت()});
    خپلکاری معلومات = wrap_type<v0::Squeeze>({probs, any_ننوت()});
    خپلکاری معلومات_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_د>({معلومات}), any_ننوت(), any_ننوت()});
    خپلکاری ids_شمېر = wrap_type<v8::Gather>({wrap_type<v3::Shape_د>({ids_2d}), any_ننوت(), any_ننوت()});
    خپلکاری موخه = wrap_type<v0::Concat>({معلومات_batch, ids_شمېر});
    خپلکاری خپرونه_2 = wrap_type<v3::خپرونه>({ids_2d, موخه});
    خپلکاری gather = wrap_type<v8::Gather>({معلومات, خپرونه_2, any_ننوت()});
    خپلکاری weights_4d = wrap_type<v0::Unsqueeze>({gather, any_ننوت()});
    خپلکاری weights = wrap_type<v1::Reshape>({weights_4d, any_ننوت()});
    خپلکاری مجموعه = wrap_type<v1::Reduce_مجموعه>({weights, any_ننوت()});
    خپلکاری clamp = wrap_type<v0::Clamp>({مجموعه});
    خپلکاری tile = wrap_type<v0::Tile>({clamp, any_ننوت()});
    خپلکاری norm = wrap_type<v1::Divide>({weights, tile});

    ثابت خپلکاری callback = [=](Matcher & m) {
        ثابت خپلکاری & pm = m.get_نمونه_ارزښت_map();
        ثابت خپلکاری غوټه = [&](ثابت std::shared_ptr<ov::Node> & p) { أرجع pm.at(p).get_غوټه_shared_ptr(); };
        ثابت خپلکاری ننوت_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            أرجع constant_is(غوټه(p)->ننوت_ارزښت(i), v);
        };
        خپلکاری mm = ov::as_type_ptr<v0::MatMul>(غوټه(logits));
        خپلکاری sm = ov::as_type_ptr<v8::Softmax>(غوټه(softmax));
        خپلکاری tk = ov::as_type_ptr<v11::سر_K>(غوټه(topk));
        خپلکاری reduce = ov::as_type_ptr<v1::Reduce_مجموعه>(غوټه(مجموعه));
        خپلکاری حد = ov::as_type_ptr<v0::Clamp>(غوټه(clamp));
        ثابت خپلکاری shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_ننوت_partial_shape(0).rank() != 4 ||
            mm->get_ننوت_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_ډېرپېښ() != v11::سر_K::ډېرپېښ_2::MAX ||
            tk->get_ترتیبول_type() != v11::سر_K::ترتیبول_Type::ترتیبول_ارزښتونه || tk->get_stable() ||
            tk->get_شاخص_element_type() != ov::element::i32 ||
            !tk->output(0).get_موخه_inputs().تشه()) {
            أرجع خطأ;
        }
        ثابت int64_t experts = shape[3].get_اوږدتیا();
        خپلکاری end = ov::as_type_ptr<v0::Constant>(غوټه(ids)->get_ننوت_غوټه_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_شمېره() || ov::shape_اندازه(end->get_shape()) != 1) {
            أرجع خطأ;
        }
        ثابت int64_t k = end->cast_ویکټور<int64_t>()[0];
        ثابت خپلکاری sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_اوږدتیا() < k ||
            !ننوت_is(probs, 1, {1, -1, experts, 1}) ||
            !ننوت_is(معلومات, 1, {0}) || !ننوت_is(ids_2d, 1, {0, 1}) ||
            !ننوت_is(weights_4d, 1, {0}) || !ننوت_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(غوټه(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(غوټه(weights))->get_special_zero() ||
            !ننوت_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(غوټه(gather))->get_batch_dims() != 1 ||
            !ننوت_is(معلومات_batch, 1, {0}) || !ننوت_is(معلومات_batch, 2, {0}) ||
            !ننوت_is(ids_شمېر, 1, {1}) || !ننوت_is(ids_شمېر, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(غوټه(معلومات_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(غوټه(ids_شمېر))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(غوټه(موخه))->get_axis() != 0 ||
            ov::as_type_ptr<v3::خپرونه>(غوټه(خپرونه_2))->get_خپرونه_spec().m_type != ov::op::خپرونه_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!ننوت_is(مجموعه, 1, {-1}) && !ننوت_is(مجموعه, 1, {3})) ||
            !ننوت_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(غوټه(norm))->get_autob().m_type != ov::op::Auto_خپرونه_Type::NUMPY) {
            أرجع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(حد->get_min() <= 0.5 * مزدوج(k) / experts && حد->get_max() >= 2.0)) {
            أرجع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت خپلکاری & ننوت : tk->output(1).get_موخه_inputs()) {
            خپلکاری ټوټه_2 = ov::as_type_ptr<v8::ټوټه>(ننوت.get_غوټه()->shared_from_this());
            إذا (!ټوټه_2 || ننوت.get_شاخص() != 0 || ټوټه_2->get_ننوت_اندازه() != 5 ||
                !constant_is(ټوټه_2->ننوت_ارزښت(1), {0}) || !constant_is(ټوټه_2->ننوت_ارزښت(2), {k}) ||
                !constant_is(ټوټه_2->ننوت_ارزښت(3), {1}) ||
                (!constant_is(ټوټه_2->ننوت_ارزښت(4), {3}) && !constant_is(ټوټه_2->ننوت_ارزښت(4), {-1}))) {
                أرجع خطأ;
            }
            slices.push_back(ټوټه_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        خپلکاری axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        خپلکاری hidden = std::make_shared<v0::Squeeze>(mm->ننوت_ارزښت(0), axis0);
        خپلکاری routing = std::make_shared<v0::MatMul>(hidden, mm->ننوت_ارزښت(1), خطأ, mm->get_transpose_b());
        خپلکاری probabilities = std::make_shared<v8::Softmax>(routing, -1);
        خپلکاری selected = std::make_shared<v11::سر_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::سر_K::ډېرپېښ_2::MAX, v11::سر_K::ترتیبول_Type::ترتیبول_ارزښتونه, tk->get_شاخص_element_type(), خطأ);
        خپلکاری ټول = std::make_shared<v1::Reduce_مجموعه>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ریښتیا);
        خپلکاری normalized = std::make_shared<v1::Divide>(selected->output(0), ټول);
        خپلکاری weights_بهر = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        خپلکاری ids_بهر = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, غوټه(norm)}, {hidden, routing, probabilities, selected, ټول, normalized, weights_بهر, ids_بهر});
        ov::replace_غوټه(غوټه(norm), weights_بهر);
        لكل (ثابت خپلکاری & ټوټه_2 : slices) {
            ov::replace_غوټه(ټوټه_2, ids_بهر);
        }
        أرجع ریښتیا;
    };
    ثبتوونکی_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
