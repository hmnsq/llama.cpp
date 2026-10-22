#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_view(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    सचे (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        पटिनिवत्त {context.get_निवेशः(0)};
    }

    सचे (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ध्रुव std::string & vname = context.get_name();
        सचे (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_निवेशः_shape(0);
            auto dst_ps = context.get_output_shape();
            सचे (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_निवेशः_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                ध्रुव size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ध्रुव auto in_rank = context.get_निवेशः(0).get_partial_shape().rank();
                ध्रुव पूर्णाङ्क axis_shift =
                    in_rank.is_static() ? (पूर्णाङ्क) nd - (पूर्णाङ्क) in_rank.get_length() : 0;
                तार्किक shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                अत्थाय (पूर्णाङ्क a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                सचे (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<तार्किक> used(nd, मिच्छा);
                    std::vector<पूर्णाङ्क> dst_to_src(nd, -1);
                    तार्किक ok = सच्च;
                    अत्थाय (size_t d = 0; d < nd; ++d) {
                        सचे (dd[d] == 1) {
                            अनुवर्त;
                        }
                        पूर्णाङ्क found = -1;
                        अत्थाय (size_t s = 0; s < nd; ++s) {
                            सचे (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (पूर्णाङ्क) s;
                                भङ्ग;
                            }
                        }
                        सचे (found < 0) {
                            ok = मिच्छा;
                            भङ्ग;
                        }
                        used[found] = सच्च;
                        dst_to_src[d] = found;
                    }
                    पूर्णाङ्क dropped = -1;
                    सचे (ok) {
                        अत्थाय (size_t s = 0; s < nd; ++s) {
                            सचे (!used[s] && ss[s] > 1) {
                                सचे (dropped >= 0) {
                                    ok = मिच्छा;
                                    भङ्ग;
                                }
                                dropped = (पूर्णाङ्क) s;
                            }
                        }
                    }
                    सचे (ok && dropped >= 0) {
                        ध्रुव size_t dstr = sst[dropped];
                        ध्रुव int64_t dsz = (int64_t) ss[dropped];
                        सचे (dstr > 0 && voff % dstr == 0) {
                            ध्रुव int64_t sel = (int64_t) (voff / dstr);
                            सचे (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_निवेशः(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_shift}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                ध्रुव int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                पूर्णाङ्क dst_ov_axis = (dyn != -1) ? ((पूर्णाङ्क) nd - 1 - (पूर्णाङ्क) dyn) : -1;
                                पूर्णाङ्क src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (पूर्णाङ्क) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                सचे (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    अत्थाय (पूर्णाङ्क a = axis_shift; a < (पूर्णाङ्क) nd; ++a) {
                                        सचे (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } अञ्ञथा {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, मिच्छा);
                                    पटिनिवत्त rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, मिच्छा);
                                पटिनिवत्त rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        पटिनिवत्त {context.get_निवेशः(0)};
    }

    auto निवेशः = context.get_निवेशः(0);
    auto src_shape = context.get_निवेशः_shape(0);
    auto dst_shape = context.get_output_shape();

    सचे (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        पटिनिवत्त {निवेशः};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    अत्थाय (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        सचे (src_shape[i].is_dynamic()) {
            पटिनिवत्त {निवेशः};
        }
        src_elems *= src_shape[i].get_length();
    }
    अत्थाय (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        सचे (dst_shape[i].is_dynamic()) {
            पटिनिवत्त {निवेशः};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    सचे (dst_elems >= src_elems) {
        पटिनिवत्त {निवेशः};
    }

    auto src_stride = context.get_निवेशः_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    तार्किक same_stride = (src_stride.size() == dst_stride.size());
    सचे (same_stride) {
        अत्थाय (size_t i = 0; i < src_stride.size(); ++i) {
            सचे (src_stride[i] != dst_stride[i]) {
                same_stride = मिच्छा;
                भङ्ग;
            }
        }
    }

    सचे (!same_stride) {
        पटिनिवत्त {निवेशः};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    सचे (dst_ov_shape.size() != ndims) {
        पटिनिवत्त {निवेशः};
    }

    std::vector<पूर्णाङ्क> diff_dims;
    अत्थाय (size_t i = 0; i < ndims; ++i) {
        सचे (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
        }
    }

    सचे (diff_dims.size() != 1) {
        पटिनिवत्त {निवेशः};
    }

    पूर्णाङ्क slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    अत्थाय (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    सचे (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    सचे (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    सचे (begin_val < 0 || end_val > dim_size) {
        पटिनिवत्त {निवेशः};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(निवेशः, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    पटिनिवत्त {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
