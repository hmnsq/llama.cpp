#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_view(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಯದಿ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        ಪ್ರತ್ಯಾಗಮ {context.get_ಒಳಹರಿವು(0)};
    }

    ಯದಿ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ಧ್ರುವ std::string & vname = context.get_name();
        ಯದಿ (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_ಒಳಹರಿವು_shape(0);
            auto dst_ps = context.get_output_shape();
            ಯದಿ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_ಒಳಹರಿವು_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                ಧ್ರುವ size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ಧ್ರುವ auto in_rank = context.get_ಒಳಹರಿವು(0).get_partial_shape().rank();
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis_shift =
                    in_rank.is_static() ? (ಪೂರ್ಣಾಙ್ಕ) nd - (ಪೂರ್ಣಾಙ್ಕ) in_rank.get_length() : 0;
                ತಾರ್ಕಿಕ shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                ಯದಿ (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<ತಾರ್ಕಿಕ> used(nd, ಅಸತ್ಯ);
                    std::vector<ಪೂರ್ಣಾಙ್ಕ> dst_to_src(nd, -1);
                    ತಾರ್ಕಿಕ ok = ಸತ್ಯ;
                    ಪ್ರತಿ (size_t d = 0; d < nd; ++d) {
                        ಯದಿ (dd[d] == 1) {
                            ಅನುವರ್ತ;
                        }
                        ಪೂರ್ಣಾಙ್ಕ found = -1;
                        ಪ್ರತಿ (size_t s = 0; s < nd; ++s) {
                            ಯದಿ (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (ಪೂರ್ಣಾಙ್ಕ) s;
                                ಭಙ್ಗ;
                            }
                        }
                        ಯದಿ (found < 0) {
                            ok = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                        used[found] = ಸತ್ಯ;
                        dst_to_src[d] = found;
                    }
                    ಪೂರ್ಣಾಙ್ಕ dropped = -1;
                    ಯದಿ (ok) {
                        ಪ್ರತಿ (size_t s = 0; s < nd; ++s) {
                            ಯದಿ (!used[s] && ss[s] > 1) {
                                ಯದಿ (dropped >= 0) {
                                    ok = ಅಸತ್ಯ;
                                    ಭಙ್ಗ;
                                }
                                dropped = (ಪೂರ್ಣಾಙ್ಕ) s;
                            }
                        }
                    }
                    ಯದಿ (ok && dropped >= 0) {
                        ಧ್ರುವ size_t dstr = sst[dropped];
                        ಧ್ರುವ int64_t dsz = (int64_t) ss[dropped];
                        ಯದಿ (dstr > 0 && voff % dstr == 0) {
                            ಧ್ರುವ int64_t sel = (int64_t) (voff / dstr);
                            ಯದಿ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_ಒಳಹರಿವು(0),
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
                                ಧ್ರುವ int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                ಪೂರ್ಣಾಙ್ಕ dst_ov_axis = (dyn != -1) ? ((ಪೂರ್ಣಾಙ್ಕ) nd - 1 - (ಪೂರ್ಣಾಙ್ಕ) dyn) : -1;
                                ಪೂರ್ಣಾಙ್ಕ src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (ಪೂರ್ಣಾಙ್ಕ) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                ಯದಿ (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ a = axis_shift; a < (ಪೂರ್ಣಾಙ್ಕ) nd; ++a) {
                                        ಯದಿ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } ಅನ್ಯಥಾ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ಅಸತ್ಯ);
                                    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ಅಸತ್ಯ);
                                ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ {context.get_ಒಳಹರಿವು(0)};
    }

    auto ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(0);
    auto src_shape = context.get_ಒಳಹರಿವು_shape(0);
    auto dst_shape = context.get_output_shape();

    ಯದಿ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    ಪ್ರತಿ (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        ಯದಿ (src_shape[i].is_dynamic()) {
            ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
        }
        src_elems *= src_shape[i].get_length();
    }
    ಪ್ರತಿ (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        ಯದಿ (dst_shape[i].is_dynamic()) {
            ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    ಯದಿ (dst_elems >= src_elems) {
        ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
    }

    auto src_stride = context.get_ಒಳಹರಿವು_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    ತಾರ್ಕಿಕ same_stride = (src_stride.size() == dst_stride.size());
    ಯದಿ (same_stride) {
        ಪ್ರತಿ (size_t i = 0; i < src_stride.size(); ++i) {
            ಯದಿ (src_stride[i] != dst_stride[i]) {
                same_stride = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
    }

    ಯದಿ (!same_stride) {
        ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    ಯದಿ (dst_ov_shape.size() != ndims) {
        ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
    }

    std::vector<ಪೂರ್ಣಾಙ್ಕ> diff_dims;
    ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
        ಯದಿ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<ಪೂರ್ಣಾಙ್ಕ>(i));
        }
    }

    ಯದಿ (diff_dims.size() != 1) {
        ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
    }

    ಪೂರ್ಣಾಙ್ಕ slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    ಪ್ರತಿ (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    ಯದಿ (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    ಯದಿ (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    ಯದಿ (begin_val < 0 || end_val > dim_size) {
        ಪ್ರತ್ಯಾಗಮ {ಒಳಹರಿವು};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    ಪ್ರತ್ಯಾಗಮ {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
