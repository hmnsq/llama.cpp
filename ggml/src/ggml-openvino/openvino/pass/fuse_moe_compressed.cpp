#ضمن "fuse_moe_compressed.h"

#ضمن <cstring>
#ضمن <limits>
#ضمن <set>
#ضمن <memory>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gelu.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/swish.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/constant_folding.hpp>
#ضمن <openvino/pass/pattern/op/optional.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

#ضمن "../op/gather_matmul.hpp"
#ضمن "../op/moe_compressed.hpp"

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

بنية dequant_inputs {
    ov::Output<ov::Node> وزن;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    منطقي has_zp = خطأ;
    منطقي ok = خطأ;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ثابت ov::Output<ov::Node> & b) {
    dequant_inputs res;

    خپلکاری غوټه = b.get_غوټه_shared_ptr();
    طالما (ov::is_type<ov::op::v0::Convert>(غوټه) || ov::is_type<ov::op::v1::Reshape>(غوټه)) {
        غوټه = غوټه->get_ننوت_غوټه_shared_ptr(0);
    }

    خپلکاری mul = ov::as_type_ptr<ov::op::v1::Multiply>(غوټه);
    إذا (!mul) {
        أرجع res;
    }
    res.scale = mul->ننوت_ارزښت(1);

    خپلکاری lhs = mul->get_ننوت_غوټه_shared_ptr(0);
    إذا (خپلکاری sub = ov::as_type_ptr<ov::op::v1::کمول>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        خپلکاری zp_غوټه = sub->get_ننوت_غوټه_shared_ptr(1);
        طالما (ov::is_type<ov::op::v0::Convert>(zp_غوټه)) {
            zp_غوټه = zp_غوټه->get_ننوت_غوټه_shared_ptr(0);
        }
        res.zp = zp_غوټه->output(0);
        res.has_zp = ریښتیا;
        lhs = sub->get_ننوت_غوټه_shared_ptr(0);
    }
    طالما (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_ننوت_غوټه_shared_ptr(0);
    }
    إذا (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        أرجع res;
    }

    res.وزن = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.وزن.get_partial_shape().is_static();
    أرجع res;
}

size_t logical_k(ثابت ov::Shape & shape) {
    أرجع shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> ټوټه_experts_dim1(ثابت ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    خپلکاری constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_غوټه_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ثابت خپلکاری & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ثابت خپلکاری & type = constant->get_element_type();

    size_t لیکه_elems = 1;
    لكل (size_t i = 2; i < shape.size(); ++i) {
        لیکه_elems *= shape[i];
    }
    ثابت size_t لیکه_بټونه = لیکه_elems * type.bitwidth();
    OPENVINO_ASSERT(لیکه_بټونه % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ثابت size_t لیکه_بایټونه = لیکه_بټونه / 8;

    ov::Shape بهر_shape = shape;
    بهر_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> بهر_معلومات(shape[0] * static_cast<size_t>(end - begin) * لیکه_بایټونه);
    ثابت خپلکاری * src = static_cast<ثابت uint8_t *>(constant->get_معلومات_ptr());
    ثابت size_t src_لیکه_بایټونه = shape[1] * لیکه_بایټونه;
    لكل (size_t e = 0; e < shape[0]; ++e) {
        std::حافظه_نقل(بهر_معلومات.معلومات() + e * static_cast<size_t>(end - begin) * لیکه_بایټونه,
                   src + e * src_لیکه_بایټونه + static_cast<size_t>(begin) * لیکه_بایټونه,
                   static_cast<size_t>(end - begin) * لیکه_بایټونه);
    }

    أرجع std::make_shared<ov::op::v0::Constant>(type, بهر_shape, بهر_معلومات.معلومات());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    کارول نطاق ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    خپلکاری hidden_gate_m = any_ننوت();
    خپلکاری a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_ننوت() });
    خپلکاری a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_ننوت() });
    خپلکاری hidden_پورته_m = any_ننوت();
    خپلکاری a_پورته_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_پورته_m, any_ننوت() }) }),
          any_ننوت() });

    خپلکاری gate_w_m = any_ننوت();
    خپلکاری پورته_w_m = any_ننوت();
    خپلکاری ښکته_w_m = any_ننوت();
    خپلکاری ids_gate_m = any_ننوت();
    خپلکاری ids_پورته_m = any_ننوت();
    خپلکاری ids_ښکته_m = any_ننوت();

    خپلکاری bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_ننوت() });
    خپلکاری gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_ننوت() }), any_ننوت() }) });

    خپلکاری silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    خپلکاری bgm_پورته_m = wrap_type<ov::op::internal::GatherMatmul>({ a_پورته_m, پورته_w_m, ids_پورته_m, any_ننوت() });
    خپلکاری پورته_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_پورته_m, any_ننوت() }), any_ننوت() }) });
    خپلکاری swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, پورته_u_m });

    خپلکاری d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_ننوت() }) }),
          any_ننوت() });
    خپلکاری bgm_ښکته_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, ښکته_w_m, ids_ښکته_m, any_ننوت() });
    خپلکاری ښکته_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_ښکته_m, any_ننوت() }), any_ننوت() }) });

    خپلکاری routing_m = any_ننوت();
    خپلکاری weighted_m = wrap_type<ov::op::v1::Multiply>({ ښکته_u_m, routing_m });
    خپلکاری ریښه_m = wrap_type<ov::op::v1::Reduce_مجموعه>({ weighted_m, any_ننوت() });

    ثابت خپلکاری callback = [=](Matcher & m) {
        خپلکاری & pm = m.get_نمونه_ارزښت_map();

        ثابت خپلکاری gate = unwrap_dequant(pm.at(gate_w_m));
        ثابت خپلکاری پورته = unwrap_dequant(pm.at(پورته_w_m));
        ثابت خپلکاری ښکته = unwrap_dequant(pm.at(ښکته_w_m));
        إذا (!gate.ok || !پورته.ok || !ښکته.ok) {
            أرجع خطأ;
        }

        ثابت خپلکاری gate_shape = gate.وزن.get_shape();
        ثابت خپلکاری پورته_shape = پورته.وزن.get_shape();
        ثابت خپلکاری ښکته_shape = ښکته.وزن.get_shape();
        إذا (gate_shape != پورته_shape || gate_shape.size() < 3 || ښکته_shape.size() < 3) {
            أرجع خطأ;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        إذا (gate.has_zp != ښکته.has_zp || gate_shape.size() != ښکته_shape.size()) {
            أرجع خطأ;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        إذا (gate.has_zp) {
            ساكن ثابت std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            إذا (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(ښکته.zp.get_element_type()) == 0) {
                أرجع خطأ;
            }
        }

        // Config holds a single group_size for all three projections.
        ثابت خپلکاری ډله_د = [](ثابت dequant_inputs & w) {
            ثابت خپلکاری s = w.وزن.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (ډله_د(gate) != ډله_د(پورته) || ډله_د(gate) != ډله_د(ښکته)) {
            أرجع خطأ;
        }

        // all three branches must route the same hidden state through the same experts
        إذا (pm.at(hidden_gate_m) != pm.at(hidden_پورته_m)) {
            أرجع خطأ;
        }

        خپلکاری ids = pm.at(ids_ښکته_m);
        ثابت خپلکاری ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_اوږدتیا() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t سر_k = ids_pshape[ids_pshape.rank().get_اوږدتیا() - 1].get_اوږدتیا();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        خپلکاری routing = pm.at(routing_m);
        ثابت خپلکاری routing_pshape = routing.get_partial_shape();
        إذا (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_اوږدتیا() != 4 ||
            routing_pshape[3] != 1) {
            أرجع خطأ;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_اوږدتیا() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t ښکته_k = logical_k(ښکته_shape);
        ثابت خپلکاری ښکته_scale_shape = ښکته.scale.get_shape();
        ثابت size_t ښکته_groups = ښکته_scale_shape.size() >= 3 ? ښکته_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_اندازه = logical_k(gate_shape);
        config.inter_اندازه = gate_shape[1];
        config.عدد_expert = gate_shape[0];
        config.عدد_shared_expert = 0;
        config.سر_k = سر_k;
        config.ډله_اندازه = ښکته_groups <= 1 ? std::numeric_limits<size_t>::max() : ښکته_k / ښکته_groups;
        config.has_batch_dim = ریښتیا;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.بهر_type = ov::element::dynamic;

        خپلکاری absent_zp = [] {
            خپلکاری zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            أرجع zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ثابت خپلکاری a_transpose = pm.at(a_gate_m).get_غوټه_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->ننوت_ارزښت(1));

        ثابت ov::Output_ویکټور دلیلونه_2 = {
            hidden, routing, ids,
            gate.وزن,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            پورته.وزن,    پورته.scale,     پورته.has_zp ? پورته.zp : absent_zp(),
            ښکته.وزن,  ښکته.scale,   ښکته.has_zp ? ښکته.zp : absent_zp(),
        };

        خپلکاری moe = std::make_shared<ov::op::internal::MOECompressed>(دلیلونه_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> پایله = moe->output(0);
        ثابت خپلکاری ریښه_type = m.get_سمون_ریښه()->get_output_element_type(0);
        إذا (پایله.get_element_type() != ریښه_type) {
            پایله = std::make_shared<ov::op::v0::Convert>(پایله, ریښه_type);
        }

        پایله.get_غوټه_shared_ptr()->سېټ_friendly_name(m.get_سمون_ریښه()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), پایله.get_غوټه_shared_ptr());
        ov::replace_غوټه(m.get_سمون_ریښه(), پایله.get_غوټه_shared_ptr());
        ثبتوونکی_new_غوټه(moe);
        أرجع ریښتیا;
    };

    ثبتوونکی_matcher(std::make_shared<Matcher>(ریښه_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_پورته::Fuse_Moe_Compressed_Fused_Gate_پورته() {
    کارول نطاق ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    خپلکاری hidden_m = any_ننوت();
    خپلکاری a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_ننوت() });
    خپلکاری a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_ننوت() });

    خپلکاری gate_پورته_w_m = any_ننوت();
    خپلکاری ids_gate_پورته_m = any_ننوت();
    خپلکاری bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_پورته_w_m, ids_gate_پورته_m, any_ننوت() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    خپلکاری gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_ننوت() });
    خپلکاری gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_ننوت() }) });

    خپلکاری gate_ټوټه_m = wrap_type<ov::op::v8::ټوټه>({ gu_u_m, any_ننوت(), any_ننوت(), any_ننوت(), any_ننوت() });
    خپلکاری پورته_ټوټه_m = wrap_type<ov::op::v8::ټوټه>({ gu_u_m, any_ننوت(), any_ننوت(), any_ننوت(), any_ننوت() });
    خپلکاری gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_ټوټه_m });
    خپلکاری geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, پورته_ټوټه_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    خپلکاری d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_ننوت() }), any_ننوت() }) }),
          any_ننوت() });
    خپلکاری ښکته_w_m = any_ننوت();
    خپلکاری ids_ښکته_m = any_ننوت();
    خپلکاری bgm_ښکته_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, ښکته_w_m, ids_ښکته_m, any_ننوت() });
    // Same two shapes as the gate/up branch above.
    خپلکاری ښکته_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_ښکته_m }), any_ننوت() });
    خپلکاری ښکته_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ ښکته_t_m, any_ننوت() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    خپلکاری ښکته_scale_m = any_ننوت();
    خپلکاری ښکته_scaled_m = wrap_type<ov::op::v1::Multiply>({ ښکته_u_m, ښکته_scale_m });

    خپلکاری routing_m = any_ننوت();
    خپلکاری weighted_m = wrap_type<ov::op::v1::Multiply>({ ښکته_scaled_m, routing_m });
    خپلکاری ریښه_m = wrap_type<ov::op::v1::Reduce_مجموعه>({ weighted_m, any_ننوت() });

    ثابت خپلکاری callback = [=](Matcher & m) {
        خپلکاری & pm = m.get_نمونه_ارزښت_map();

        ثابت خپلکاری gate_پورته = unwrap_dequant(pm.at(gate_پورته_w_m));
        ثابت خپلکاری ښکته = unwrap_dequant(pm.at(ښکته_w_m));
        إذا (!gate_پورته.ok || !ښکته.ok) {
            أرجع خطأ;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        إذا (!gate_پورته.has_zp || !ښکته.has_zp) {
            أرجع خطأ;
        }

        ثابت خپلکاری gate_پورته_shape = gate_پورته.وزن.get_shape();
        ثابت خپلکاری ښکته_shape = ښکته.وزن.get_shape();
        إذا (gate_پورته_shape.size() < 3 || ښکته_shape.size() < 3 || gate_پورته_shape.size() != ښکته_shape.size()) {
            أرجع خطأ;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ساكن ثابت std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        إذا (int_types.count(gate_پورته.وزن.get_element_type()) == 0 ||
            int_types.count(ښکته.وزن.get_element_type()) == 0 ||
            int_types.count(gate_پورته.zp.get_element_type()) == 0 ||
            int_types.count(ښکته.zp.get_element_type()) == 0) {
            أرجع خطأ;
        }

        ثابت خپلکاری ډله_د = [](ثابت dequant_inputs & w) {
            ثابت خپلکاری s = w.وزن.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (ډله_د(gate_پورته) != ډله_د(ښکته)) {
            أرجع خطأ;
        }

        خپلکاری gelu_غوټه = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_غوټه_shared_ptr());
        إذا (!gelu_غوټه || gelu_غوټه->get_approximation_ډېرپېښ() != ov::op::Gelu_Approximation_ډېرپېښ::ERF) {
            أرجع خطأ;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        خپلکاری gate_ټوټه = ov::as_type_ptr<ov::op::v8::ټوټه>(pm.at(gate_ټوټه_m).get_غوټه_shared_ptr());
        خپلکاری پورته_ټوټه = ov::as_type_ptr<ov::op::v8::ټوټه>(pm.at(پورته_ټوټه_m).get_غوټه_shared_ptr());
        خپلکاری gate_پای_ټکی_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_ټوټه->ننوت_ارزښت(2).get_غوټه_shared_ptr());
        خپلکاری پورته_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(پورته_ټوټه->ننوت_ارزښت(1).get_غوټه_shared_ptr());
        إذا (!gate_پای_ټکی_c || !پورته_begin_c) {
            أرجع خطأ;
        }
        ثابت خپلکاری gate_پای_ټکی = gate_پای_ټکی_c->cast_ویکټور<int64_t>().at(0);
        ثابت خپلکاری پورته_begin = پورته_begin_c->cast_ویکټور<int64_t>().at(0);
        إذا (gate_پای_ټکی != پورته_begin || gate_پای_ټکی <= 0 || gate_پای_ټکی >= static_cast<int64_t>(gate_پورته_shape[1])) {
            أرجع خطأ;
        }
        ثابت int64_t mid = gate_پای_ټکی;
        ثابت int64_t ډک = static_cast<int64_t>(gate_پورته_shape[1]);

        خپلکاری ids = pm.at(ids_ښکته_m);
        ثابت خپلکاری ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_اوږدتیا() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t سر_k = ids_pshape[ids_pshape.rank().get_اوږدتیا() - 1].get_اوږدتیا();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ثابت خپلکاری routing_دننه = pm.at(routing_m);
        ثابت خپلکاری routing_pshape = routing_دننه.get_partial_shape();
        ثابت خپلکاری routing_rank = routing_pshape.rank();
        إذا (routing_rank.is_dynamic() || (routing_rank.get_اوږدتیا() != 4 && routing_rank.get_اوږدتیا() != 3) ||
            routing_pshape[routing_rank.get_اوږدتیا() - 1] != 1) {
            أرجع خطأ;
        }
        ثابت منطقي batchless = routing_rank.get_اوږدتیا() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        خپلکاری ښکته_scale = pm.at(ښکته_scale_m);
        إذا (ښکته_scale.get_partial_shape() != routing_pshape) {
            أرجع خطأ;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_دننه, ښکته_scale);
        إذا (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_اوږدتیا() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t ښکته_k = logical_k(ښکته_shape);
        ثابت خپلکاری ښکته_scale_shape = ښکته.scale.get_shape();
        ثابت size_t ښکته_groups = ښکته_scale_shape.size() >= 3 ? ښکته_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ثابت ov::Output<ov::Node> gate_وزن = ټوټه_experts_dim1(gate_پورته.وزن, 0, mid);
        ثابت ov::Output<ov::Node> پورته_وزن = ټوټه_experts_dim1(gate_پورته.وزن, mid, ډک);
        ثابت ov::Output<ov::Node> gate_scale = ټوټه_experts_dim1(gate_پورته.scale, 0, mid);
        ثابت ov::Output<ov::Node> پورته_scale = ټوټه_experts_dim1(gate_پورته.scale, mid, ډک);
        ثابت ov::Output<ov::Node> gate_zp = ټوټه_experts_dim1(gate_پورته.zp, 0, mid);
        ثابت ov::Output<ov::Node> پورته_zp = ټوټه_experts_dim1(gate_پورته.zp, mid, ډک);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_اندازه = logical_k(gate_پورته_shape);
        config.inter_اندازه = static_cast<size_t>(mid);
        config.عدد_expert = gate_پورته_shape[0];
        config.عدد_shared_expert = 0;
        config.سر_k = سر_k;
        config.ډله_اندازه = ښکته_groups <= 1 ? std::numeric_limits<size_t>::max() : ښکته_k / ښکته_groups;
        config.has_batch_dim = ریښتیا;
        config.has_zp = ریښتیا;
        config.بهر_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ثابت خپلکاری a_transpose = pm.at(a_m).get_غوټه_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->ننوت_ارزښت(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ثابت ov::Output_ویکټور دلیلونه_2 = {
            hidden,      routing,     ids,
            gate_وزن, gate_scale,  gate_zp,
            پورته_وزن,   پورته_scale,    پورته_zp,
            ښکته.وزن, ښکته.scale,  ښکته.zp,
        };

        خپلکاری moe = std::make_shared<ov::op::internal::MOECompressed>(دلیلونه_2, config);

        ov::Output<ov::Node> پایله = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        إذا (batchless) {
            پایله = std::make_shared<ov::op::v0::Squeeze>(
                پایله, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ثابت خپلکاری ریښه_type = m.get_سمون_ریښه()->get_output_element_type(0);
        إذا (پایله.get_element_type() != ریښه_type) {
            پایله = std::make_shared<ov::op::v0::Convert>(پایله, ریښه_type);
        }
        إذا (پایله.get_partial_shape() != m.get_سمون_ریښه()->get_output_partial_shape(0)) {
            أرجع خطأ;
        }

        پایله.get_غوټه_shared_ptr()->سېټ_friendly_name(m.get_سمون_ریښه()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), پایله.get_غوټه_shared_ptr());
        ov::replace_غوټه(m.get_سمون_ریښه(), پایله.get_غوټه_shared_ptr());
        ثبتوونکی_new_غوټه(moe);
        أرجع ریښتیا;
    };

    ثبتوونکی_matcher(
        std::make_shared<Matcher>(ریښه_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
