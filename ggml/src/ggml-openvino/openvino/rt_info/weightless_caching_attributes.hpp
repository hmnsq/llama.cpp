// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ଅନ୍ତର୍ଭାଵ <openvino/core/core_visibility.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/runtime_attribute.hpp>

ନାମାକାଶ ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
ଶ୍ରେଣୀ OPENVINO_API WeightlessCacheAttribute : ସର୍ବସାଧାରଣ RuntimeAttribute {
ସର୍ବସାଧାରଣ:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = delete;

    WeightlessCacheAttribute(size_t original_size, size_t bin_offset, ov::element::Type original_dtype)
        : original_size(original_size),
          bin_offset(bin_offset),
          original_dtype(original_dtype) {}

    ତାର୍କିକ is_copyable() ଅଚଳ override;

    size_t original_size;
    size_t bin_offset;
    ov::element::Type original_dtype;
};

}  // namespace ov
