#ಅನ್ತರ್ಭಾವ "ggml-vulkan-common.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <shared_mutex>

ಸ್ಥಿರ ಶೂನ್ಯ * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    ಪ್ರತ್ಯಾಗಮ _aligned_malloc(size, align);
#ಅನ್ಯಥಾ
    ಶೂನ್ಯ * ptr = nullptr;
    ಯದಿ (posix_memalign(&ptr, align, size) != 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ ptr;
#endif
}

ಸ್ಥಿರ ಶೂನ್ಯ portable_aligned_free(ಶೂನ್ಯ * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#ಅನ್ಯಥಾ
    std::ಮುಕ್ತ(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
    /* .is_host             = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ std::vector<uint32_t> ggml_vk_find_memory_properties(ಧ್ರುವ vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    ಪ್ರತಿ (uint32_t i = 0; i < mem_props->memory_Type_ಎಣಿಕೆ; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        ಯದಿ ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    ಪ್ರತ್ಯಾಗಮ indices;
}

ಸ್ಥಿರ vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, ಧ್ರುವ std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       ಶೂನ್ಯ *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    ಯದಿ (size > device->max_buffer_size) {
        ಕ್ಷಿಪ vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    ಯದಿ (size == 0) {
        buf->size = 0;
        ಪ್ರತ್ಯಾಗಮ buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    ಯದಿ (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    ಯದಿ (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    ಧ್ರುವ vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    ಯದಿ (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    ಯದಿ (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        ಯತ್ನ {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } ಗೃಹಾಣ (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            ಪ್ರತ್ಯಾಗಮ {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        ಪ್ರತಿ (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            ಯದಿ (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                ಅನುವರ್ತ;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            ಯದಿ ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            ಪ್ರತ್ಯಾಗಮ {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        ಯತ್ನ {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        }
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ಧ್ರುವ auto & req_flags = *it;

            ಧ್ರುವ std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            ಯದಿ (memory_type_indices.empty()) {
                ಅನುವರ್ತ;
            }

            ತಾರ್ಕಿಕ done = ಅಸತ್ಯ;

            ಪ್ರತಿ (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                ಯತ್ನ {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = ಸತ್ಯ;
                    ಭಙ್ಗ;
                } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    ಯದಿ (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        ಕ್ಷಿಪ e;
                    }
                }
            }

            ಯದಿ (done) {
                ಭಙ್ಗ;
            }
        }
    }

    ಯದಿ (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        ಕ್ಷಿಪ vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    ಯದಿ (import_ptr) {
        buf->ptr = import_ptr;
    } ಅನ್ಯಥಾ {
        ಯದಿ (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    ಯದಿ (device->buffer_device_address) {
        ಧ್ರುವ vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    ಪ್ರತ್ಯಾಗಮ buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    ಯತ್ನ {
        ಪ್ರತ್ಯಾಗಮ ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        ಕ್ಷಿಪ e;
    }
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    ಯತ್ನ {
        ಯದಿ (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } ಅನ್ಯಥಾ ಯದಿ (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } ಅನ್ಯಥಾ ಯದಿ (device->disable_host_visible_vidmem) {
            ಯದಿ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } ಅನ್ಯಥಾ {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } ಅನ್ಯಥಾ {
            // use rebar if available, otherwise fallback to device only visible memory
            ಯದಿ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } ಅನ್ಯಥಾ {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        ಕ್ಷಿಪ e;
    }

    ಪ್ರತ್ಯಾಗಮ buf;
}

ಶೂನ್ಯ ggml_vk_destroy_buffer(vk_buffer& buf) {
    ಯದಿ (buf == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ಸ್ಥಿರ size_t ggml_vk_host_allocation_find(ಧ್ರುವ ಶೂನ್ಯ * ptr) {
    ಪ್ರತಿ (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ಧ್ರುವ uint8_t * base = (ಧ್ರುವ uint8_t *) vk_instance.host_allocations[i].base;
        ಯದಿ (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }
    ಪ್ರತ್ಯಾಗಮ SIZE_MAX;
}

ಶೂನ್ಯ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    ಯದಿ (vk_instance.multi_device) {
        ಧ್ರುವ size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ಧ್ರುವ size_t alloc_size = (size + align - 1) / align * align;

        ಶೂನ್ಯ * ptr = portable_aligned_malloc(align, alloc_size);
        ಯದಿ (ptr == nullptr) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_buffer buf;
    ಯತ್ನ {
        buf = ggml_vk_create_buffer(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } ಗೃಹಾಣ (vk::SystemError&) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಯದಿ (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    ಪ್ರತ್ಯಾಗಮ buf->ptr;
}

ಶೂನ್ಯ ggml_vk_host_free(ಶೂನ್ಯ * ptr) {
    ಯದಿ (ptr == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ಧ್ರುವ size_t index = ggml_vk_host_allocation_find(ptr);
        ಯದಿ (index == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            ಪ್ರತ್ಯಾಗಮ;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    ಪ್ರತಿ (auto & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    ಯದಿ (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

ಶೂನ್ಯ ggml_vk_host_get(ಧ್ರುವ vk_device& device, ಧ್ರುವ ಶೂನ್ಯ * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ಧ್ರುವ size_t index = ggml_vk_host_allocation_find(ptr);
    ಯದಿ (index == SIZE_MAX) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    ಯದಿ (entry.views[device->idx] == nullptr) {
        ಧ್ರುವ uint32_t bit = 1u << device->idx;
        ಯದಿ (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            ಪ್ರತ್ಯಾಗಮ;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        ಯದಿ (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            ಪ್ರತ್ಯಾಗಮ;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (ಧ್ರುವ uint8_t *) ptr - (ಧ್ರುವ uint8_t *) entry.base;
}

ಶೂನ್ಯ ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    ಯದಿ (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

ಶೂನ್ಯ ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    ಯದಿ (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, ಧ್ರುವ ggml_tensor * tensor, ತಾರ್ಕಿಕ sync_staging = ಅಸತ್ಯ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    ಯದಿ(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    ಧ್ರುವ uint64_t ne0 = tensor->ne[0];
    ಧ್ರುವ uint64_t ne1 = tensor->ne[1];
    ಧ್ರುವ uint64_t ne2 = tensor->ne[2];
    ಧ್ರುವ uint64_t ne3 = tensor->ne[3];
    ಧ್ರುವ uint64_t nb0 = tensor->nb[0];
    ಧ್ರುವ uint64_t nb1 = tensor->nb[1];
    ಧ್ರುವ uint64_t nb2 = tensor->nb[2];
    ಧ್ರುವ uint64_t nb3 = tensor->nb[3];
    ಧ್ರುವ ggml_type type = tensor->type;
    ಧ್ರುವ uint64_t ts = ggml_type_size(type);
    ಧ್ರುವ uint64_t bs = ggml_blck_size(type);

    ಧ್ರುವ uint64_t dstnb0 = ts;
    ಧ್ರುವ uint64_t dstnb1 = dstnb0*(ne0/bs);
    ಧ್ರುವ uint64_t dstnb2 = dstnb1*ne1;
    ಧ್ರುವ uint64_t dstnb3 = dstnb2*ne2;

    ಧ್ರುವ uint64_t ne = ggml_nelements(tensor);

    ಯದಿ (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        ಪ್ರತಿ (uint64_t i3 = 0; i3 < ne3; i3++) {
            ಪ್ರತಿ (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                ಯದಿ (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } ಅನ್ಯಥಾ {
                    ಪ್ರತಿ (uint64_t i1 = 0; i1 < ne1; i1++) {
                        ಯದಿ (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } ಅನ್ಯಥಾ {
                            ಧ್ರುವ uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            ಧ್ರುವ uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            ಪ್ರತಿ (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    ಧ್ರುವ uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    ಪ್ರತಿ (uint64_t i3 = 0; i3 < ne3; i3++) {
        ಪ್ರತಿ (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            ಯದಿ (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ಧ್ರುವ uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } ಅನ್ಯಥಾ {
                ಪ್ರತಿ (uint64_t i1 = 0; i1 < ne1; i1++) {
                    ಯದಿ (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ಧ್ರುವ uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } ಅನ್ಯಥಾ {
                        ಧ್ರುವ uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        ಧ್ರುವ uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        ಪ್ರತಿ (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ಧ್ರುವ uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

ತಾರ್ಕಿಕ ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, ಧ್ರುವ ಶೂನ್ಯ * src, size_t spitch, size_t dpitch, size_t width, size_t height, ತಾರ್ಕಿಕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    ಯದಿ (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        ಯದಿ (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } ಅನ್ಯಥಾ {
            slices.resize(height);
            ಪ್ರತಿ (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    VK_LOG_DEBUG("STAGING");

    ಯದಿ (!sync_staging) {
        // copy was not handled caller needs to fall back
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Staging buffer required
    ಧ್ರುವ size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    ಯದಿ (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } ಅನ್ಯಥಾ {
        slices.resize(height);
        ಪ್ರತಿ (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    ಯದಿ (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (ಧ್ರುವ uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, ಧ್ರುವ ಶೂನ್ಯ * src, size_t size, ತಾರ್ಕಿಕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    ಪ್ರತ್ಯಾಗಮ ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

ಶೂನ್ಯ ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, ಧ್ರುವ ಶೂನ್ಯ * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    ಯದಿ(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        ಯದಿ (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (ಧ್ರುವ uint8_t *) src + i * spitch, width);
            }
        }
    } ಅನ್ಯಥಾ {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        ತಾರ್ಕಿಕ ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, ಸತ್ಯ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ಪ್ರತಿ (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ಪ್ರತಿ (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ಸತ್ಯ, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

ಶೂನ್ಯ ggml_vk_buffer_write(vk_buffer& dst, size_t offset, ಧ್ರುವ ಶೂನ್ಯ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

ತಾರ್ಕಿಕ ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, ಶೂನ್ಯ * dst, size_t spitch, size_t dpitch, size_t width, size_t height, ತಾರ್ಕಿಕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    ಯದಿ (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } ಅನ್ಯಥಾ {
        slices.resize(height);
        ಪ್ರತಿ (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ಯದಿ (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    VK_LOG_DEBUG("STAGING");

    ಯದಿ (!sync_staging) {
        // copy was not handled caller needs to fall back
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Fall back to staging buffer
    ಧ್ರುವ size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    ಯದಿ (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } ಅನ್ಯಥಾ {
        staging_slices.resize(height);
        ಪ್ರತಿ (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    ಯದಿ (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ಧ್ರುವ uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, ಶೂನ್ಯ * dst, size_t size, ತಾರ್ಕಿಕ sync_staging = ಅಸತ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

ಶೂನ್ಯ ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, ಶೂನ್ಯ * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ಧ್ರುವ ತಾರ್ಕಿಕ slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    ಯದಿ(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ಸತ್ಯ, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        ಯದಿ (width == spitch && width == dpitch) {
            memcpy(dst, (ಧ್ರುವ uint8_t *) src->ptr + offset, width * height);
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (ಧ್ರುವ uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } ಅನ್ಯಥಾ {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ತಾರ್ಕಿಕ ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, ಸತ್ಯ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ಸತ್ಯ, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        ಪ್ರತಿ (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

ಶೂನ್ಯ ggml_vk_buffer_read(vk_buffer& src, size_t offset, ಶೂನ್ಯ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

ಶೂನ್ಯ ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

ಶೂನ್ಯ ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    ಯದಿ (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ಸತ್ಯ, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } ಅನ್ಯಥಾ {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

ಶೂನ್ಯ ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    ಯದಿ (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        ಪ್ರತ್ಯಾಗಮ;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

ಶೂನ್ಯ ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    ಯದಿ (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ಸತ್ಯ, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, ಶೂನ್ಯ * ptr, size_t size) {
    ಯದಿ (!device->external_memory_host) {
        ಪ್ರತ್ಯಾಗಮ {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    ಯದಿ (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        ಪ್ರತ್ಯಾಗಮ {};
    }
    ಯದಿ (size & (device->min_imported_host_pointer_alignment - 1)) {
        ಪ್ರತ್ಯಾಗಮ {};
    }

    ಧ್ರುವ vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    ಯತ್ನ {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } ಗೃಹಾಣ (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    ಪ್ರತ್ಯಾಗಮ buf;
}

