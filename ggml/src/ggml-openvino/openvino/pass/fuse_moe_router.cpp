#ਅਨ੍ਤਰ੍ਭਾਵ "fuse_moe_router.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/rt_info.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/clamp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_sum.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/softmax.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/tile.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/topk.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

ਨਾਮਾਕਾਸ਼ {

ਤਾਰਕਿਕ constant_is(ਅਚਲ ov::Output<ov::Node> & output, ਅਚਲ std::vector<int64_t> & ਮੁੱਲਾਂ) {
    ਸਵੈਚਾਲਿਤ ਗੰਢ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ਗੰਢ_shared_ptr());
    ਵਾਪਸ ਗੰਢ && ਗੰਢ->get_element_type().is_integral_ਸੰਖਿਆ() && ਗੰਢ->cast_ਸਦਿਸ਼<int64_t>() == ਮੁੱਲਾਂ;
}

}  // namespace

Fuse_Moe_ਰਾਊਟਰ::Fuse_Moe_ਰਾਊਟਰ() {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ ov::pass::pattern;
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    ਸਵੈਚਾਲਿਤ logits = wrap_type<v0::MatMul>();
    ਸਵੈਚਾਲਿਤ softmax = wrap_type<v8::Softmax>({logits});
    ਸਵੈਚਾਲਿਤ topk = wrap_type<v11::ਸਿਖਰ_K>({softmax, any_ਨਿਵੇਸ਼()});
    topk->ਸੈੱਟ_output_ਮਾਪ(2);
    ਸਵੈਚਾਲਿਤ ids = wrap_type<v8::ਟੁਕੜਾ>({topk->output(1), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ ids_2d = wrap_type<v0::Squeeze>({ids, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ probs = wrap_type<v1::Reshape>({softmax, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ = wrap_type<v0::Squeeze>({probs, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_ਦਾ>({ਅੰਕੜੇ}), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ ids_ਗਿਣਤੀ = wrap_type<v8::Gather>({wrap_type<v3::Shape_ਦਾ>({ids_2d}), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ = wrap_type<v0::Concat>({ਅੰਕੜੇ_batch, ids_ਗਿਣਤੀ});
    ਸਵੈਚਾਲਿਤ ਪ੍ਰਸਾਰਣ_2 = wrap_type<v3::ਪ੍ਰਸਾਰਣ>({ids_2d, ਨਿਸ਼ਾਨਾ});
    ਸਵੈਚਾਲਿਤ gather = wrap_type<v8::Gather>({ਅੰਕੜੇ, ਪ੍ਰਸਾਰਣ_2, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ weights_4d = wrap_type<v0::Unsqueeze>({gather, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ weights = wrap_type<v1::Reshape>({weights_4d, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ ਜੋੜ = wrap_type<v1::Reduce_ਜੋੜ>({weights, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ clamp = wrap_type<v0::Clamp>({ਜੋੜ});
    ਸਵੈਚਾਲਿਤ tile = wrap_type<v0::Tile>({clamp, any_ਨਿਵੇਸ਼()});
    ਸਵੈਚਾਲਿਤ norm = wrap_type<v1::Divide>({weights, tile});

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [=](Matcher & m) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ & pm = m.get_ਨਮੂਨਾ_ਮੁੱਲ_map();
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਗੰਢ = [&](ਅਚਲ std::shared_ptr<ov::Node> & p) { ਵਾਪਸ pm.at(p).get_ਗੰਢ_shared_ptr(); };
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_is = [&](ਅਚਲ std::shared_ptr<ov::Node> & p, size_t i, ਅਚਲ std::vector<int64_t> & v) {
            ਵਾਪਸ constant_is(ਗੰਢ(p)->ਨਿਵੇਸ਼_ਮੁੱਲ(i), v);
        };
        ਸਵੈਚਾਲਿਤ mm = ov::as_type_ptr<v0::MatMul>(ਗੰਢ(logits));
        ਸਵੈਚਾਲਿਤ sm = ov::as_type_ptr<v8::Softmax>(ਗੰਢ(softmax));
        ਸਵੈਚਾਲਿਤ tk = ov::as_type_ptr<v11::ਸਿਖਰ_K>(ਗੰਢ(topk));
        ਸਵੈਚਾਲਿਤ reduce = ov::as_type_ptr<v1::Reduce_ਜੋੜ>(ਗੰਢ(ਜੋੜ));
        ਸਵੈਚਾਲਿਤ ਹੱਦਬੰਦੀ = ov::as_type_ptr<v0::Clamp>(ਗੰਢ(clamp));
        ਅਚਲ ਸਵੈਚਾਲਿਤ shape = mm->get_output_partial_shape(0);
        ਜੇ (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_ਨਿਵੇਸ਼_partial_shape(0).rank() != 4 ||
            mm->get_ਨਿਵੇਸ਼_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_ਬਹੁਲਕ() != v11::ਸਿਖਰ_K::ਬਹੁਲਕ_2::MAX ||
            tk->get_ਛਾਂਟੋ_type() != v11::ਸਿਖਰ_K::ਛਾਂਟੋ_Type::ਛਾਂਟੋ_ਮੁੱਲਾਂ || tk->get_stable() ||
            tk->get_ਸੂਚਕਾਂਕ_element_type() != ov::element::i32 ||
            !tk->output(0).get_ਨਿਸ਼ਾਨਾ_inputs().ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ int64_t experts = shape[3].get_ਲੰਮਾਈ();
        ਸਵੈਚਾਲਿਤ end = ov::as_type_ptr<v0::Constant>(ਗੰਢ(ids)->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(2));
        ਜੇ (!end || !end->get_element_type().is_integral_ਸੰਖਿਆ() || ov::shape_ਮਾਪ(end->get_shape()) != 1) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ int64_t k = end->cast_ਸਦਿਸ਼<int64_t>()[0];
        ਅਚਲ ਸਵੈਚਾਲਿਤ sorted_shape = tk->get_output_partial_shape(1);
        ਜੇ (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_ਲੰਮਾਈ() < k ||
            !ਨਿਵੇਸ਼_is(probs, 1, {1, -1, experts, 1}) ||
            !ਨਿਵੇਸ਼_is(ਅੰਕੜੇ, 1, {0}) || !ਨਿਵੇਸ਼_is(ids_2d, 1, {0, 1}) ||
            !ਨਿਵੇਸ਼_is(weights_4d, 1, {0}) || !ਨਿਵੇਸ਼_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ਗੰਢ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ਗੰਢ(weights))->get_special_zero() ||
            !ਨਿਵੇਸ਼_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ਗੰਢ(gather))->get_batch_dims() != 1 ||
            !ਨਿਵੇਸ਼_is(ਅੰਕੜੇ_batch, 1, {0}) || !ਨਿਵੇਸ਼_is(ਅੰਕੜੇ_batch, 2, {0}) ||
            !ਨਿਵੇਸ਼_is(ids_ਗਿਣਤੀ, 1, {1}) || !ਨਿਵੇਸ਼_is(ids_ਗਿਣਤੀ, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ਗੰਢ(ਅੰਕੜੇ_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ਗੰਢ(ids_ਗਿਣਤੀ))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ਗੰਢ(ਨਿਸ਼ਾਨਾ))->get_axis() != 0 ||
            ov::as_type_ptr<v3::ਪ੍ਰਸਾਰਣ>(ਗੰਢ(ਪ੍ਰਸਾਰਣ_2))->get_ਪ੍ਰਸਾਰਣ_spec().m_type != ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!ਨਿਵੇਸ਼_is(ਜੋੜ, 1, {-1}) && !ਨਿਵੇਸ਼_is(ਜੋੜ, 1, {3})) ||
            !ਨਿਵੇਸ਼_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ਗੰਢ(norm))->get_autob().m_type != ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY) {
            ਵਾਪਸ ਝੂਠ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        ਜੇ (!(ਹੱਦਬੰਦੀ->get_min() <= 0.5 * ਦੁੱਗਣਾ(k) / experts && ਹੱਦਬੰਦੀ->get_max() >= 2.0)) {
            ਵਾਪਸ ਝੂਠ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਨਿਵੇਸ਼ : tk->output(1).get_ਨਿਸ਼ਾਨਾ_inputs()) {
            ਸਵੈਚਾਲਿਤ ਟੁਕੜਾ_2 = ov::as_type_ptr<v8::ਟੁਕੜਾ>(ਨਿਵੇਸ਼.get_ਗੰਢ()->shared_from_this());
            ਜੇ (!ਟੁਕੜਾ_2 || ਨਿਵੇਸ਼.get_ਸੂਚਕਾਂਕ() != 0 || ਟੁਕੜਾ_2->get_ਨਿਵੇਸ਼_ਮਾਪ() != 5 ||
                !constant_is(ਟੁਕੜਾ_2->ਨਿਵੇਸ਼_ਮੁੱਲ(1), {0}) || !constant_is(ਟੁਕੜਾ_2->ਨਿਵੇਸ਼_ਮੁੱਲ(2), {k}) ||
                !constant_is(ਟੁਕੜਾ_2->ਨਿਵੇਸ਼_ਮੁੱਲ(3), {1}) ||
                (!constant_is(ਟੁਕੜਾ_2->ਨਿਵੇਸ਼_ਮੁੱਲ(4), {3}) && !constant_is(ਟੁਕੜਾ_2->ਨਿਵੇਸ਼_ਮੁੱਲ(4), {-1}))) {
                ਵਾਪਸ ਝੂਠ;
            }
            slices.push_back(ਟੁਕੜਾ_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        ਸਵੈਚਾਲਿਤ axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        ਸਵੈਚਾਲਿਤ hidden = std::make_shared<v0::Squeeze>(mm->ਨਿਵੇਸ਼_ਮੁੱਲ(0), axis0);
        ਸਵੈਚਾਲਿਤ routing = std::make_shared<v0::MatMul>(hidden, mm->ਨਿਵੇਸ਼_ਮੁੱਲ(1), ਝੂਠ, mm->get_transpose_b());
        ਸਵੈਚਾਲਿਤ probabilities = std::make_shared<v8::Softmax>(routing, -1);
        ਸਵੈਚਾਲਿਤ selected = std::make_shared<v11::ਸਿਖਰ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::ਸਿਖਰ_K::ਬਹੁਲਕ_2::MAX, v11::ਸਿਖਰ_K::ਛਾਂਟੋ_Type::ਛਾਂਟੋ_ਮੁੱਲਾਂ, tk->get_ਸੂਚਕਾਂਕ_element_type(), ਝੂਠ);
        ਸਵੈਚਾਲਿਤ ਕੁੱਲ = std::make_shared<v1::Reduce_ਜੋੜ>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ਸੱਚ);
        ਸਵੈਚਾਲਿਤ normalized = std::make_shared<v1::Divide>(selected->output(0), ਕੁੱਲ);
        ਸਵੈਚਾਲਿਤ weights_ਬਾਹਰਲਾ = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        ਸਵੈਚਾਲਿਤ ids_ਬਾਹਰਲਾ = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ਗੰਢ(norm)}, {hidden, routing, probabilities, selected, ਕੁੱਲ, normalized, weights_ਬਾਹਰਲਾ, ids_ਬਾਹਰਲਾ});
        ov::replace_ਗੰਢ(ਗੰਢ(norm), weights_ਬਾਹਰਲਾ);
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਟੁਕੜਾ_2 : slices) {
            ov::replace_ਗੰਢ(ਟੁਕੜਾ_2, ids_ਬਾਹਰਲਾ);
        }
        ਵਾਪਸ ਸੱਚ;
    };
    ਰਜਿਸਟਰ_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
