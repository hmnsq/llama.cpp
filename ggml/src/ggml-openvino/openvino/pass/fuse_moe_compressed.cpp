#ਅਨ੍ਤਰ੍ਭਾਵ "fuse_moe_compressed.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/rt_info.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gelu.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_sum.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/swish.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/constant_folding.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/optional.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>

#ਅਨ੍ਤਰ੍ਭਾਵ "../op/gather_matmul.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op/moe_compressed.hpp"

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

ਨਾਮਾਕਾਸ਼ {

ਬਣਤਰ dequant_inputs {
    ov::Output<ov::Node> ਭਾਰ;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    ਤਾਰਕਿਕ has_zp = ਝੂਠ;
    ਤਾਰਕਿਕ ok = ਝੂਠ;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ਅਚਲ ov::Output<ov::Node> & b) {
    dequant_inputs res;

    ਸਵੈਚਾਲਿਤ ਗੰਢ = b.get_ਗੰਢ_shared_ptr();
    ਜਦੋਂ (ov::is_type<ov::op::v0::Convert>(ਗੰਢ) || ov::is_type<ov::op::v1::Reshape>(ਗੰਢ)) {
        ਗੰਢ = ਗੰਢ->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0);
    }

    ਸਵੈਚਾਲਿਤ mul = ov::as_type_ptr<ov::op::v1::Multiply>(ਗੰਢ);
    ਜੇ (!mul) {
        ਵਾਪਸ res;
    }
    res.scale = mul->ਨਿਵੇਸ਼_ਮੁੱਲ(1);

    ਸਵੈਚਾਲਿਤ lhs = mul->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0);
    ਜੇ (ਸਵੈਚਾਲਿਤ sub = ov::as_type_ptr<ov::op::v1::ਘਟਾਓ>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        ਸਵੈਚਾਲਿਤ zp_ਗੰਢ = sub->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1);
        ਜਦੋਂ (ov::is_type<ov::op::v0::Convert>(zp_ਗੰਢ)) {
            zp_ਗੰਢ = zp_ਗੰਢ->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0);
        }
        res.zp = zp_ਗੰਢ->output(0);
        res.has_zp = ਸੱਚ;
        lhs = sub->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0);
    }
    ਜਦੋਂ (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0);
    }
    ਜੇ (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        ਵਾਪਸ res;
    }

    res.ਭਾਰ = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.ਭਾਰ.get_partial_shape().is_static();
    ਵਾਪਸ res;
}

size_t logical_k(ਅਚਲ ov::Shape & shape) {
    ਵਾਪਸ shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> ਟੁਕੜਾ_experts_dim1(ਅਚਲ ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    ਸਵੈਚਾਲਿਤ constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_ਗੰਢ_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ਅਚਲ ਸਵੈਚਾਲਿਤ & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ਅਚਲ ਸਵੈਚਾਲਿਤ & type = constant->get_element_type();

    size_t ਪੰਕਤੀ_elems = 1;
    ਲਈ (size_t i = 2; i < shape.size(); ++i) {
        ਪੰਕਤੀ_elems *= shape[i];
    }
    ਅਚਲ size_t ਪੰਕਤੀ_ਬਿੱਟ = ਪੰਕਤੀ_elems * type.bitwidth();
    OPENVINO_ASSERT(ਪੰਕਤੀ_ਬਿੱਟ % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = ਪੰਕਤੀ_ਬਿੱਟ / 8;

    ov::Shape ਬਾਹਰਲਾ_shape = shape;
    ਬਾਹਰਲਾ_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> ਬਾਹਰਲਾ_ਅੰਕੜੇ(shape[0] * static_cast<size_t>(end - begin) * ਪੰਕਤੀ_ਬਾਈਟਾਂ);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * src = static_cast<ਅਚਲ uint8_t *>(constant->get_ਅੰਕੜੇ_ptr());
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = shape[1] * ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ਲਈ (size_t e = 0; e < shape[0]; ++e) {
        std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ_ਅੰਕੜੇ.ਅੰਕੜੇ() + e * static_cast<size_t>(end - begin) * ਪੰਕਤੀ_ਬਾਈਟਾਂ,
                   src + e * src_ਪੰਕਤੀ_ਬਾਈਟਾਂ + static_cast<size_t>(begin) * ਪੰਕਤੀ_ਬਾਈਟਾਂ,
                   static_cast<size_t>(end - begin) * ਪੰਕਤੀ_ਬਾਈਟਾਂ);
    }

    ਵਾਪਸ std::make_shared<ov::op::v0::Constant>(type, ਬਾਹਰਲਾ_shape, ਬਾਹਰਲਾ_ਅੰਕੜੇ.ਅੰਕੜੇ());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    ਸਵੈਚਾਲਿਤ hidden_gate_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ hidden_ਉੱਪਰ_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ a_ਉੱਪਰ_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_ਉੱਪਰ_m, any_ਨਿਵੇਸ਼() }) }),
          any_ਨਿਵੇਸ਼() });

    ਸਵੈਚਾਲਿਤ gate_w_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ਉੱਪਰ_w_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_w_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ids_gate_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ids_ਉੱਪਰ_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ids_ਥੱਲੇ_m = any_ਨਿਵੇਸ਼();

    ਸਵੈਚਾਲਿਤ bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_ਨਿਵੇਸ਼() }), any_ਨਿਵੇਸ਼() }) });

    ਸਵੈਚਾਲਿਤ silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    ਸਵੈਚਾਲਿਤ bgm_ਉੱਪਰ_m = wrap_type<ov::op::internal::GatherMatmul>({ a_ਉੱਪਰ_m, ਉੱਪਰ_w_m, ids_ਉੱਪਰ_m, any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ ਉੱਪਰ_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_ਉੱਪਰ_m, any_ਨਿਵੇਸ਼() }), any_ਨਿਵੇਸ਼() }) });
    ਸਵੈਚਾਲਿਤ swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, ਉੱਪਰ_u_m });

    ਸਵੈਚਾਲਿਤ d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_ਨਿਵੇਸ਼() }) }),
          any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ bgm_ਥੱਲੇ_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, ਥੱਲੇ_w_m, ids_ਥੱਲੇ_m, any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_ਥੱਲੇ_m, any_ਨਿਵੇਸ਼() }), any_ਨਿਵੇਸ਼() }) });

    ਸਵੈਚਾਲਿਤ routing_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ weighted_m = wrap_type<ov::op::v1::Multiply>({ ਥੱਲੇ_u_m, routing_m });
    ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_m = wrap_type<ov::op::v1::Reduce_ਜੋੜ>({ weighted_m, any_ਨਿਵੇਸ਼() });

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [=](Matcher & m) {
        ਸਵੈਚਾਲਿਤ & pm = m.get_ਨਮੂਨਾ_ਮੁੱਲ_map();

        ਅਚਲ ਸਵੈਚਾਲਿਤ gate = unwrap_dequant(pm.at(gate_w_m));
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਉੱਪਰ = unwrap_dequant(pm.at(ਉੱਪਰ_w_m));
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਥੱਲੇ = unwrap_dequant(pm.at(ਥੱਲੇ_w_m));
        ਜੇ (!gate.ok || !ਉੱਪਰ.ok || !ਥੱਲੇ.ok) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ gate_shape = gate.ਭਾਰ.get_shape();
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਉੱਪਰ_shape = ਉੱਪਰ.ਭਾਰ.get_shape();
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਥੱਲੇ_shape = ਥੱਲੇ.ਭਾਰ.get_shape();
        ਜੇ (gate_shape != ਉੱਪਰ_shape || gate_shape.size() < 3 || ਥੱਲੇ_shape.size() < 3) {
            ਵਾਪਸ ਝੂਠ;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        ਜੇ (gate.has_zp != ਥੱਲੇ.has_zp || gate_shape.size() != ਥੱਲੇ_shape.size()) {
            ਵਾਪਸ ਝੂਠ;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        ਜੇ (gate.has_zp) {
            ਸਥਿਰ ਅਚਲ std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            ਜੇ (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(ਥੱਲੇ.zp.get_element_type()) == 0) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        // Config holds a single group_size for all three projections.
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਸਮੂਹ_ਦਾ = [](ਅਚਲ dequant_inputs & w) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ s = w.ਭਾਰ.get_shape();
            ਵਾਪਸ s.size() == 4 ? s[3] : logical_k(s);
        };
        ਜੇ (ਸਮੂਹ_ਦਾ(gate) != ਸਮੂਹ_ਦਾ(ਉੱਪਰ) || ਸਮੂਹ_ਦਾ(gate) != ਸਮੂਹ_ਦਾ(ਥੱਲੇ)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // all three branches must route the same hidden state through the same experts
        ਜੇ (pm.at(hidden_gate_m) != pm.at(hidden_ਉੱਪਰ_m)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਸਵੈਚਾਲਿਤ ids = pm.at(ids_ਥੱਲੇ_m);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ids_pshape = ids.get_partial_shape();
        ਜੇ (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_ਲੰਮਾਈ() - 1].is_dynamic()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ size_t ਸਿਖਰ_k = ids_pshape[ids_pshape.rank().get_ਲੰਮਾਈ() - 1].get_ਲੰਮਾਈ();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        ਸਵੈਚਾਲਿਤ routing = pm.at(routing_m);
        ਅਚਲ ਸਵੈਚਾਲਿਤ routing_pshape = routing.get_partial_shape();
        ਜੇ (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_ਲੰਮਾਈ() != 4 ||
            routing_pshape[3] != 1) {
            ਵਾਪਸ ਝੂਠ;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        ਜੇ (ids_pshape.rank().get_ਲੰਮਾਈ() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ਜੇ (routing.get_partial_shape() != ids.get_partial_shape()) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ size_t ਥੱਲੇ_k = logical_k(ਥੱਲੇ_shape);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਥੱਲੇ_scale_shape = ਥੱਲੇ.scale.get_shape();
        ਅਚਲ size_t ਥੱਲੇ_groups = ਥੱਲੇ_scale_shape.size() >= 3 ? ਥੱਲੇ_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_ਮਾਪ = logical_k(gate_shape);
        config.inter_ਮਾਪ = gate_shape[1];
        config.ਅੰਕ_expert = gate_shape[0];
        config.ਅੰਕ_shared_expert = 0;
        config.ਸਿਖਰ_k = ਸਿਖਰ_k;
        config.ਸਮੂਹ_ਮਾਪ = ਥੱਲੇ_groups <= 1 ? std::numeric_limits<size_t>::max() : ਥੱਲੇ_k / ਥੱਲੇ_groups;
        config.has_batch_dim = ਸੱਚ;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.ਬਾਹਰਲਾ_type = ov::element::dynamic;

        ਸਵੈਚਾਲਿਤ absent_zp = [] {
            ਸਵੈਚਾਲਿਤ zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            ਵਾਪਸ zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ਅਚਲ ਸਵੈਚਾਲਿਤ a_transpose = pm.at(a_gate_m).get_ਗੰਢ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->ਨਿਵੇਸ਼_ਮੁੱਲ(1));

        ਅਚਲ ov::Output_ਸਦਿਸ਼ ਦਲੀਲਾਂ_2 = {
            hidden, routing, ids,
            gate.ਭਾਰ,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            ਉੱਪਰ.ਭਾਰ,    ਉੱਪਰ.scale,     ਉੱਪਰ.has_zp ? ਉੱਪਰ.zp : absent_zp(),
            ਥੱਲੇ.ਭਾਰ,  ਥੱਲੇ.scale,   ਥੱਲੇ.has_zp ? ਥੱਲੇ.zp : absent_zp(),
        };

        ਸਵੈਚਾਲਿਤ moe = std::make_shared<ov::op::internal::MOECompressed>(ਦਲੀਲਾਂ_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> ਨਤੀਜਾ = moe->output(0);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_type = m.get_ਮੇਲ_ਜੜ੍ਹ()->get_output_element_type(0);
        ਜੇ (ਨਤੀਜਾ.get_element_type() != ਜੜ੍ਹ_type) {
            ਨਤੀਜਾ = std::make_shared<ov::op::v0::Convert>(ਨਤੀਜਾ, ਜੜ੍ਹ_type);
        }

        ਨਤੀਜਾ.get_ਗੰਢ_shared_ptr()->ਸੈੱਟ_friendly_name(m.get_ਮੇਲ_ਜੜ੍ਹ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ਨਤੀਜਾ.get_ਗੰਢ_shared_ptr());
        ov::replace_ਗੰਢ(m.get_ਮੇਲ_ਜੜ੍ਹ(), ਨਤੀਜਾ.get_ਗੰਢ_shared_ptr());
        ਰਜਿਸਟਰ_new_ਗੰਢ(moe);
        ਵਾਪਸ ਸੱਚ;
    };

    ਰਜਿਸਟਰ_matcher(std::make_shared<Matcher>(ਜੜ੍ਹ_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_ਉੱਪਰ::Fuse_Moe_Compressed_Fused_Gate_ਉੱਪਰ() {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    ਸਵੈਚਾਲਿਤ hidden_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_ਨਿਵੇਸ਼() });

    ਸਵੈਚਾਲਿਤ gate_ਉੱਪਰ_w_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ids_gate_ਉੱਪਰ_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_ਉੱਪਰ_w_m, ids_gate_ਉੱਪਰ_m, any_ਨਿਵੇਸ਼() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    ਸਵੈਚਾਲਿਤ gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_ਨਿਵੇਸ਼() }) });

    ਸਵੈਚਾਲਿਤ gate_ਟੁਕੜਾ_m = wrap_type<ov::op::v8::ਟੁਕੜਾ>({ gu_u_m, any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ ਉੱਪਰ_ਟੁਕੜਾ_m = wrap_type<ov::op::v8::ਟੁਕੜਾ>({ gu_u_m, any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼(), any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_ਟੁਕੜਾ_m });
    ਸਵੈਚਾਲਿਤ geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, ਉੱਪਰ_ਟੁਕੜਾ_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    ਸਵੈਚਾਲਿਤ d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_ਨਿਵੇਸ਼() }), any_ਨਿਵੇਸ਼() }) }),
          any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_w_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ids_ਥੱਲੇ_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ bgm_ਥੱਲੇ_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, ਥੱਲੇ_w_m, ids_ਥੱਲੇ_m, any_ਨਿਵੇਸ਼() });
    // Same two shapes as the gate/up branch above.
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_ਥੱਲੇ_m }), any_ਨਿਵੇਸ਼() });
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ ਥੱਲੇ_t_m, any_ਨਿਵੇਸ਼() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_scale_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ ਥੱਲੇ_scaled_m = wrap_type<ov::op::v1::Multiply>({ ਥੱਲੇ_u_m, ਥੱਲੇ_scale_m });

    ਸਵੈਚਾਲਿਤ routing_m = any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ weighted_m = wrap_type<ov::op::v1::Multiply>({ ਥੱਲੇ_scaled_m, routing_m });
    ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_m = wrap_type<ov::op::v1::Reduce_ਜੋੜ>({ weighted_m, any_ਨਿਵੇਸ਼() });

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [=](Matcher & m) {
        ਸਵੈਚਾਲਿਤ & pm = m.get_ਨਮੂਨਾ_ਮੁੱਲ_map();

        ਅਚਲ ਸਵੈਚਾਲਿਤ gate_ਉੱਪਰ = unwrap_dequant(pm.at(gate_ਉੱਪਰ_w_m));
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਥੱਲੇ = unwrap_dequant(pm.at(ਥੱਲੇ_w_m));
        ਜੇ (!gate_ਉੱਪਰ.ok || !ਥੱਲੇ.ok) {
            ਵਾਪਸ ਝੂਠ;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        ਜੇ (!gate_ਉੱਪਰ.has_zp || !ਥੱਲੇ.has_zp) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ gate_ਉੱਪਰ_shape = gate_ਉੱਪਰ.ਭਾਰ.get_shape();
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਥੱਲੇ_shape = ਥੱਲੇ.ਭਾਰ.get_shape();
        ਜੇ (gate_ਉੱਪਰ_shape.size() < 3 || ਥੱਲੇ_shape.size() < 3 || gate_ਉੱਪਰ_shape.size() != ਥੱਲੇ_shape.size()) {
            ਵਾਪਸ ਝੂਠ;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ਸਥਿਰ ਅਚਲ std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        ਜੇ (int_types.count(gate_ਉੱਪਰ.ਭਾਰ.get_element_type()) == 0 ||
            int_types.count(ਥੱਲੇ.ਭਾਰ.get_element_type()) == 0 ||
            int_types.count(gate_ਉੱਪਰ.zp.get_element_type()) == 0 ||
            int_types.count(ਥੱਲੇ.zp.get_element_type()) == 0) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ ਸਮੂਹ_ਦਾ = [](ਅਚਲ dequant_inputs & w) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ s = w.ਭਾਰ.get_shape();
            ਵਾਪਸ s.size() == 4 ? s[3] : logical_k(s);
        };
        ਜੇ (ਸਮੂਹ_ਦਾ(gate_ਉੱਪਰ) != ਸਮੂਹ_ਦਾ(ਥੱਲੇ)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਸਵੈਚਾਲਿਤ gelu_ਗੰਢ = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_ਗੰਢ_shared_ptr());
        ਜੇ (!gelu_ਗੰਢ || gelu_ਗੰਢ->get_approximation_ਬਹੁਲਕ() != ov::op::Gelu_Approximation_ਬਹੁਲਕ::ERF) {
            ਵਾਪਸ ਝੂਠ;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        ਸਵੈਚਾਲਿਤ gate_ਟੁਕੜਾ = ov::as_type_ptr<ov::op::v8::ਟੁਕੜਾ>(pm.at(gate_ਟੁਕੜਾ_m).get_ਗੰਢ_shared_ptr());
        ਸਵੈਚਾਲਿਤ ਉੱਪਰ_ਟੁਕੜਾ = ov::as_type_ptr<ov::op::v8::ਟੁਕੜਾ>(pm.at(ਉੱਪਰ_ਟੁਕੜਾ_m).get_ਗੰਢ_shared_ptr());
        ਸਵੈਚਾਲਿਤ gate_ਸਿਰਾ_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_ਟੁਕੜਾ->ਨਿਵੇਸ਼_ਮੁੱਲ(2).get_ਗੰਢ_shared_ptr());
        ਸਵੈਚਾਲਿਤ ਉੱਪਰ_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(ਉੱਪਰ_ਟੁਕੜਾ->ਨਿਵੇਸ਼_ਮੁੱਲ(1).get_ਗੰਢ_shared_ptr());
        ਜੇ (!gate_ਸਿਰਾ_c || !ਉੱਪਰ_begin_c) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ ਸਵੈਚਾਲਿਤ gate_ਸਿਰਾ = gate_ਸਿਰਾ_c->cast_ਸਦਿਸ਼<int64_t>().at(0);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਉੱਪਰ_begin = ਉੱਪਰ_begin_c->cast_ਸਦਿਸ਼<int64_t>().at(0);
        ਜੇ (gate_ਸਿਰਾ != ਉੱਪਰ_begin || gate_ਸਿਰਾ <= 0 || gate_ਸਿਰਾ >= static_cast<int64_t>(gate_ਉੱਪਰ_shape[1])) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ int64_t mid = gate_ਸਿਰਾ;
        ਅਚਲ int64_t ਭਰਿਆ = static_cast<int64_t>(gate_ਉੱਪਰ_shape[1]);

        ਸਵੈਚਾਲਿਤ ids = pm.at(ids_ਥੱਲੇ_m);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ids_pshape = ids.get_partial_shape();
        ਜੇ (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_ਲੰਮਾਈ() - 1].is_dynamic()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ size_t ਸਿਖਰ_k = ids_pshape[ids_pshape.rank().get_ਲੰਮਾਈ() - 1].get_ਲੰਮਾਈ();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ਅਚਲ ਸਵੈਚਾਲਿਤ routing_ਵਿੱਚ = pm.at(routing_m);
        ਅਚਲ ਸਵੈਚਾਲਿਤ routing_pshape = routing_ਵਿੱਚ.get_partial_shape();
        ਅਚਲ ਸਵੈਚਾਲਿਤ routing_rank = routing_pshape.rank();
        ਜੇ (routing_rank.is_dynamic() || (routing_rank.get_ਲੰਮਾਈ() != 4 && routing_rank.get_ਲੰਮਾਈ() != 3) ||
            routing_pshape[routing_rank.get_ਲੰਮਾਈ() - 1] != 1) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ ਤਾਰਕਿਕ batchless = routing_rank.get_ਲੰਮਾਈ() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        ਸਵੈਚਾਲਿਤ ਥੱਲੇ_scale = pm.at(ਥੱਲੇ_scale_m);
        ਜੇ (ਥੱਲੇ_scale.get_partial_shape() != routing_pshape) {
            ਵਾਪਸ ਝੂਠ;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_ਵਿੱਚ, ਥੱਲੇ_scale);
        ਜੇ (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        ਜੇ (ids_pshape.rank().get_ਲੰਮਾਈ() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ਜੇ (routing.get_partial_shape() != ids.get_partial_shape()) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ size_t ਥੱਲੇ_k = logical_k(ਥੱਲੇ_shape);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਥੱਲੇ_scale_shape = ਥੱਲੇ.scale.get_shape();
        ਅਚਲ size_t ਥੱਲੇ_groups = ਥੱਲੇ_scale_shape.size() >= 3 ? ਥੱਲੇ_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ਅਚਲ ov::Output<ov::Node> gate_ਭਾਰ = ਟੁਕੜਾ_experts_dim1(gate_ਉੱਪਰ.ਭਾਰ, 0, mid);
        ਅਚਲ ov::Output<ov::Node> ਉੱਪਰ_ਭਾਰ = ਟੁਕੜਾ_experts_dim1(gate_ਉੱਪਰ.ਭਾਰ, mid, ਭਰਿਆ);
        ਅਚਲ ov::Output<ov::Node> gate_scale = ਟੁਕੜਾ_experts_dim1(gate_ਉੱਪਰ.scale, 0, mid);
        ਅਚਲ ov::Output<ov::Node> ਉੱਪਰ_scale = ਟੁਕੜਾ_experts_dim1(gate_ਉੱਪਰ.scale, mid, ਭਰਿਆ);
        ਅਚਲ ov::Output<ov::Node> gate_zp = ਟੁਕੜਾ_experts_dim1(gate_ਉੱਪਰ.zp, 0, mid);
        ਅਚਲ ov::Output<ov::Node> ਉੱਪਰ_zp = ਟੁਕੜਾ_experts_dim1(gate_ਉੱਪਰ.zp, mid, ਭਰਿਆ);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_ਮਾਪ = logical_k(gate_ਉੱਪਰ_shape);
        config.inter_ਮਾਪ = static_cast<size_t>(mid);
        config.ਅੰਕ_expert = gate_ਉੱਪਰ_shape[0];
        config.ਅੰਕ_shared_expert = 0;
        config.ਸਿਖਰ_k = ਸਿਖਰ_k;
        config.ਸਮੂਹ_ਮਾਪ = ਥੱਲੇ_groups <= 1 ? std::numeric_limits<size_t>::max() : ਥੱਲੇ_k / ਥੱਲੇ_groups;
        config.has_batch_dim = ਸੱਚ;
        config.has_zp = ਸੱਚ;
        config.ਬਾਹਰਲਾ_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ਅਚਲ ਸਵੈਚਾਲਿਤ a_transpose = pm.at(a_m).get_ਗੰਢ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->ਨਿਵੇਸ਼_ਮੁੱਲ(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ਅਚਲ ov::Output_ਸਦਿਸ਼ ਦਲੀਲਾਂ_2 = {
            hidden,      routing,     ids,
            gate_ਭਾਰ, gate_scale,  gate_zp,
            ਉੱਪਰ_ਭਾਰ,   ਉੱਪਰ_scale,    ਉੱਪਰ_zp,
            ਥੱਲੇ.ਭਾਰ, ਥੱਲੇ.scale,  ਥੱਲੇ.zp,
        };

        ਸਵੈਚਾਲਿਤ moe = std::make_shared<ov::op::internal::MOECompressed>(ਦਲੀਲਾਂ_2, config);

        ov::Output<ov::Node> ਨਤੀਜਾ = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        ਜੇ (batchless) {
            ਨਤੀਜਾ = std::make_shared<ov::op::v0::Squeeze>(
                ਨਤੀਜਾ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_type = m.get_ਮੇਲ_ਜੜ੍ਹ()->get_output_element_type(0);
        ਜੇ (ਨਤੀਜਾ.get_element_type() != ਜੜ੍ਹ_type) {
            ਨਤੀਜਾ = std::make_shared<ov::op::v0::Convert>(ਨਤੀਜਾ, ਜੜ੍ਹ_type);
        }
        ਜੇ (ਨਤੀਜਾ.get_partial_shape() != m.get_ਮੇਲ_ਜੜ੍ਹ()->get_output_partial_shape(0)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਨਤੀਜਾ.get_ਗੰਢ_shared_ptr()->ਸੈੱਟ_friendly_name(m.get_ਮੇਲ_ਜੜ੍ਹ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ਨਤੀਜਾ.get_ਗੰਢ_shared_ptr());
        ov::replace_ਗੰਢ(m.get_ਮੇਲ_ਜੜ੍ਹ(), ਨਤੀਜਾ.get_ਗੰਢ_shared_ptr());
        ਰਜਿਸਟਰ_new_ਗੰਢ(moe);
        ਵਾਪਸ ਸੱਚ;
    };

    ਰਜਿਸਟਰ_matcher(
        std::make_shared<Matcher>(ਜੜ੍ਹ_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
