// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#অন্তর্ভাৱ <openvino/core/core_visibility.hpp>
#অন্তর্ভাৱ <openvino/core/node.hpp>
#অন্তর্ভাৱ <openvino/core/runtime_attribute.hpp>

নামাকাশ ov {

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
শ্ৰেণী OPENVINO_API WeightlessCacheAttribute : সার্ৱজনিক RuntimeAttribute {
সার্ৱজনিক:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = delete;

    WeightlessCacheAttribute(size_t original_size, size_t bin_offset, ov::element::Type original_dtype)
        : original_size(original_size),
          bin_offset(bin_offset),
          original_dtype(original_dtype) {}

    তার্কিক is_copyable() ধ্রুৱ override;

    size_t original_size;
    size_t bin_offset;
    ov::element::Type original_dtype;
};

}  // namespace ov
