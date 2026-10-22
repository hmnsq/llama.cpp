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

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & مقادیر) {
    خودکار گره = ov::as_type_ptr<ov::op::v0::Constant>(output.get_گره_shared_ptr());
    أرجع گره && گره->get_element_type().is_integral_عدد() && گره->cast_وکتور<int64_t>() == مقادیر;
}

}  // namespace

Fuse_Moe_مسیریاب::Fuse_Moe_مسیریاب() {
    استفاده نطاق ov::pass::pattern;
    استفاده نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    خودکار logits = wrap_type<v0::MatMul>();
    خودکار softmax = wrap_type<v8::Softmax>({logits});
    خودکار topk = wrap_type<v11::بالاترین_K>({softmax, any_درون_داد()});
    topk->مجموعه_output_اندازه(2);
    خودکار ids = wrap_type<v8::برش>({topk->output(1), any_درون_داد(), any_درون_داد(), any_درون_داد(), any_درون_داد()});
    خودکار ids_2d = wrap_type<v0::Squeeze>({ids, any_درون_داد()});
    خودکار probs = wrap_type<v1::Reshape>({softmax, any_درون_داد()});
    خودکار داده = wrap_type<v0::Squeeze>({probs, any_درون_داد()});
    خودکار داده_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_از>({داده}), any_درون_داد(), any_درون_داد()});
    خودکار ids_شمار = wrap_type<v8::Gather>({wrap_type<v3::Shape_از>({ids_2d}), any_درون_داد(), any_درون_داد()});
    خودکار هدف = wrap_type<v0::Concat>({داده_batch, ids_شمار});
    خودکار پخش_2 = wrap_type<v3::پخش>({ids_2d, هدف});
    خودکار gather = wrap_type<v8::Gather>({داده, پخش_2, any_درون_داد()});
    خودکار weights_4d = wrap_type<v0::Unsqueeze>({gather, any_درون_داد()});
    خودکار weights = wrap_type<v1::Reshape>({weights_4d, any_درون_داد()});
    خودکار جمع = wrap_type<v1::Reduce_جمع>({weights, any_درون_داد()});
    خودکار clamp = wrap_type<v0::Clamp>({جمع});
    خودکار tile = wrap_type<v0::Tile>({clamp, any_درون_داد()});
    خودکار norm = wrap_type<v1::Divide>({weights, tile});

    ثابت خودکار callback = [=](Matcher & m) {
        ثابت خودکار & pm = m.get_الگو_مقدار_map();
        ثابت خودکار گره = [&](ثابت std::shared_ptr<ov::Node> & p) { أرجع pm.at(p).get_گره_shared_ptr(); };
        ثابت خودکار درون_داد_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            أرجع constant_is(گره(p)->درون_داد_مقدار(i), v);
        };
        خودکار mm = ov::as_type_ptr<v0::MatMul>(گره(logits));
        خودکار sm = ov::as_type_ptr<v8::Softmax>(گره(softmax));
        خودکار tk = ov::as_type_ptr<v11::بالاترین_K>(گره(topk));
        خودکار reduce = ov::as_type_ptr<v1::Reduce_جمع>(گره(جمع));
        خودکار حد = ov::as_type_ptr<v0::Clamp>(گره(clamp));
        ثابت خودکار shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_درون_داد_partial_shape(0).rank() != 4 ||
            mm->get_درون_داد_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_نما() != v11::بالاترین_K::نما_2::MAX ||
            tk->get_مرتب_سازی_type() != v11::بالاترین_K::مرتب_سازی_Type::مرتب_سازی_مقادیر || tk->get_stable() ||
            tk->get_شاخص_element_type() != ov::element::i32 ||
            !tk->output(0).get_هدف_inputs().خالی()) {
            أرجع خطأ;
        }
        ثابت int64_t experts = shape[3].get_درازی();
        خودکار end = ov::as_type_ptr<v0::Constant>(گره(ids)->get_درون_داد_گره_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_عدد() || ov::shape_اندازه(end->get_shape()) != 1) {
            أرجع خطأ;
        }
        ثابت int64_t k = end->cast_وکتور<int64_t>()[0];
        ثابت خودکار sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_درازی() < k ||
            !درون_داد_is(probs, 1, {1, -1, experts, 1}) ||
            !درون_داد_is(داده, 1, {0}) || !درون_داد_is(ids_2d, 1, {0, 1}) ||
            !درون_داد_is(weights_4d, 1, {0}) || !درون_داد_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(گره(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(گره(weights))->get_special_zero() ||
            !درون_داد_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(گره(gather))->get_batch_dims() != 1 ||
            !درون_داد_is(داده_batch, 1, {0}) || !درون_داد_is(داده_batch, 2, {0}) ||
            !درون_داد_is(ids_شمار, 1, {1}) || !درون_داد_is(ids_شمار, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(گره(داده_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(گره(ids_شمار))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(گره(هدف))->get_axis() != 0 ||
            ov::as_type_ptr<v3::پخش>(گره(پخش_2))->get_پخش_spec().m_type != ov::op::پخش_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!درون_داد_is(جمع, 1, {-1}) && !درون_داد_is(جمع, 1, {3})) ||
            !درون_داد_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(گره(norm))->get_autob().m_type != ov::op::Auto_پخش_Type::NUMPY) {
            أرجع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(حد->get_min() <= 0.5 * مزدوج(k) / experts && حد->get_max() >= 2.0)) {
            أرجع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت خودکار & درون_داد : tk->output(1).get_هدف_inputs()) {
            خودکار برش_2 = ov::as_type_ptr<v8::برش>(درون_داد.get_گره()->shared_from_this());
            إذا (!برش_2 || درون_داد.get_شاخص() != 0 || برش_2->get_درون_داد_اندازه() != 5 ||
                !constant_is(برش_2->درون_داد_مقدار(1), {0}) || !constant_is(برش_2->درون_داد_مقدار(2), {k}) ||
                !constant_is(برش_2->درون_داد_مقدار(3), {1}) ||
                (!constant_is(برش_2->درون_داد_مقدار(4), {3}) && !constant_is(برش_2->درون_داد_مقدار(4), {-1}))) {
                أرجع خطأ;
            }
            slices.push_back(برش_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        خودکار axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        خودکار hidden = std::make_shared<v0::Squeeze>(mm->درون_داد_مقدار(0), axis0);
        خودکار routing = std::make_shared<v0::MatMul>(hidden, mm->درون_داد_مقدار(1), خطأ, mm->get_transpose_b());
        خودکار probabilities = std::make_shared<v8::Softmax>(routing, -1);
        خودکار selected = std::make_shared<v11::بالاترین_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::بالاترین_K::نما_2::MAX, v11::بالاترین_K::مرتب_سازی_Type::مرتب_سازی_مقادیر, tk->get_شاخص_element_type(), خطأ);
        خودکار مجموع = std::make_shared<v1::Reduce_جمع>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);
        خودکار normalized = std::make_shared<v1::Divide>(selected->output(0), مجموع);
        خودکار weights_بیرون = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        خودکار ids_بیرون = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, گره(norm)}, {hidden, routing, probabilities, selected, مجموع, normalized, weights_بیرون, ids_بیرون});
        ov::replace_گره(گره(norm), weights_بیرون);
        لكل (ثابت خودکار & برش_2 : slices) {
            ov::replace_گره(برش_2, ids_بیرون);
        }
        أرجع درست;
    };
    ثبات_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
