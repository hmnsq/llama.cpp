#அந்தர்பாவ "ggml-vulkan-common.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <shared_mutex>

நிலையான வெறுமை * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    திரும்பு _aligned_malloc(size, align);
#இல்லையெனில்
    வெறுமை * ptr = nullptr;
    எனில் (posix_memalign(&ptr, align, size) != 0) {
        திரும்பு nullptr;
    }
    திரும்பு ptr;
#endif
}

நிலையான வெறுமை portable_aligned_free(வெறுமை * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#இல்லையெனில்
    std::விடுவி(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ வெற்று,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ வெற்று,
    /* .is_host             = */ வெற்று,
};

நிலையான std::vector<uint32_t> ggml_vk_find_memory_properties(மாறிலி vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    க்கு (uint32_t i = 0; i < mem_props->memory_Type_மொத்தம்; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        எனில் ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    திரும்பு indices;
}

நிலையான vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, மாறிலி std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       வெறுமை *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    எனில் (size > device->max_buffer_size) {
        எறி vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    எனில் (size == 0) {
        buf->size = 0;
        திரும்பு buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    எனில் (device->buffer_device_address) {
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
    எனில் (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    மாறிலி vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    எனில் (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    எனில் (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        முயல் {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } பிடி (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            திரும்பு {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        க்கு (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            எனில் (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                தொடர்;
            }
            எனில் (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                தொடர்;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            எனில் ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                நிறுத்து;
            }
        }
        எனில் (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            திரும்பு {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        முயல் {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } பிடி (மாறிலி vk::SystemError& e) {
        }
    } இல்லையெனில் {
        க்கு (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            மாறிலி auto & req_flags = *it;

            மாறிலி std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            எனில் (memory_type_indices.empty()) {
                தொடர்;
            }

            தருக்க done = பொய்;

            க்கு (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                முயல் {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = உண்மை;
                    நிறுத்து;
                } பிடி (மாறிலி vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    எனில் (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        எறி e;
                    }
                }
            }

            எனில் (done) {
                நிறுத்து;
            }
        }
    }

    எனில் (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        எறி vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    எனில் (import_ptr) {
        buf->ptr = import_ptr;
    } இல்லையெனில் {
        எனில் (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    எனில் (device->buffer_device_address) {
        மாறிலி vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    திரும்பு buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    முயல் {
        திரும்பு ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } பிடி (மாறிலி vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        எறி e;
    }
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    முயல் {
        எனில் (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } இல்லையெனில் எனில் (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } இல்லையெனில் எனில் (device->disable_host_visible_vidmem) {
            எனில் (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } இல்லையெனில் {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } இல்லையெனில் {
            // use rebar if available, otherwise fallback to device only visible memory
            எனில் (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } இல்லையெனில் {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } பிடி (மாறிலி vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        எறி e;
    }

    திரும்பு buf;
}

வெறுமை ggml_vk_destroy_buffer(vk_buffer& buf) {
    எனில் (buf == nullptr) {
        திரும்பு;
    }

    எனில் (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
நிலையான size_t ggml_vk_host_allocation_find(மாறிலி வெறுமை * ptr) {
    க்கு (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        மாறிலி uint8_t * base = (மாறிலி uint8_t *) vk_instance.host_allocations[i].base;
        எனில் (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            திரும்பு i;
        }
    }
    திரும்பு SIZE_MAX;
}

வெறுமை * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    எனில் (vk_instance.multi_device) {
        மாறிலி size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        மாறிலி size_t alloc_size = (size + align - 1) / align * align;

        வெறுமை * ptr = portable_aligned_malloc(align, alloc_size);
        எனில் (ptr == nullptr) {
            திரும்பு nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        திரும்பு ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_buffer buf;
    முயல் {
        buf = ggml_vk_create_buffer(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } பிடி (vk::SystemError&) {
        திரும்பு nullptr;
    }

    எனில் (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        திரும்பு nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    திரும்பு buf->ptr;
}

வெறுமை ggml_vk_host_free(வெறுமை * ptr) {
    எனில் (ptr == nullptr) {
        திரும்பு;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        மாறிலி size_t index = ggml_vk_host_allocation_find(ptr);
        எனில் (index == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            திரும்பு;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    க்கு (auto & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    எனில் (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

வெறுமை ggml_vk_host_get(மாறிலி vk_device& device, மாறிலி வெறுமை * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    மாறிலி size_t index = ggml_vk_host_allocation_find(ptr);
    எனில் (index == SIZE_MAX) {
        திரும்பு;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    எனில் (entry.views[device->idx] == nullptr) {
        மாறிலி uint32_t bit = 1u << device->idx;
        எனில் (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            திரும்பு;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        எனில் (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            திரும்பு;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (மாறிலி uint8_t *) ptr - (மாறிலி uint8_t *) entry.base;
}

வெறுமை ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    எனில் (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

வெறுமை ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    எனில் (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

நிலையான வெறுமை ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, மாறிலி ggml_tensor * tensor, தருக்க sync_staging = பொய்) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    எனில்(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    மாறிலி uint64_t ne0 = tensor->ne[0];
    மாறிலி uint64_t ne1 = tensor->ne[1];
    மாறிலி uint64_t ne2 = tensor->ne[2];
    மாறிலி uint64_t ne3 = tensor->ne[3];
    மாறிலி uint64_t nb0 = tensor->nb[0];
    மாறிலி uint64_t nb1 = tensor->nb[1];
    மாறிலி uint64_t nb2 = tensor->nb[2];
    மாறிலி uint64_t nb3 = tensor->nb[3];
    மாறிலி ggml_type type = tensor->type;
    மாறிலி uint64_t ts = ggml_type_size(type);
    மாறிலி uint64_t bs = ggml_blck_size(type);

    மாறிலி uint64_t dstnb0 = ts;
    மாறிலி uint64_t dstnb1 = dstnb0*(ne0/bs);
    மாறிலி uint64_t dstnb2 = dstnb1*ne1;
    மாறிலி uint64_t dstnb3 = dstnb2*ne2;

    மாறிலி uint64_t ne = ggml_nelements(tensor);

    எனில் (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        க்கு (uint64_t i3 = 0; i3 < ne3; i3++) {
            க்கு (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                எனில் (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } இல்லையெனில் {
                    க்கு (uint64_t i1 = 0; i1 < ne1; i1++) {
                        எனில் (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } இல்லையெனில் {
                            மாறிலி uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            மாறிலி uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            க்கு (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        திரும்பு;
    }

    எனில் (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    மாறிலி uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    க்கு (uint64_t i3 = 0; i3 < ne3; i3++) {
        க்கு (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            எனில் (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (மாறிலி uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } இல்லையெனில் {
                க்கு (uint64_t i1 = 0; i1 < ne1; i1++) {
                    எனில் (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (மாறிலி uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } இல்லையெனில் {
                        மாறிலி uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        மாறிலி uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        க்கு (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (மாறிலி uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

தருக்க ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, மாறிலி வெறுமை * src, size_t spitch, size_t dpitch, size_t width, size_t height, தருக்க sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    எனில் (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        எனில் (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } இல்லையெனில் {
            slices.resize(height);
            க்கு (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        திரும்பு உண்மை;
    }
    VK_LOG_DEBUG("STAGING");

    எனில் (!sync_staging) {
        // copy was not handled caller needs to fall back
        திரும்பு பொய்;
    }

    // Staging buffer required
    மாறிலி size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    எனில் (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } இல்லையெனில் {
        slices.resize(height);
        க்கு (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    எனில் (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } இல்லையெனில் {
        க்கு (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (மாறிலி uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    திரும்பு உண்மை;
}

தருக்க ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, மாறிலி வெறுமை * src, size_t size, தருக்க sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    திரும்பு ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

வெறுமை ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, மாறிலி வெறுமை * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    எனில்(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        எனில் (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } இல்லையெனில் {
            க்கு (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (மாறிலி uint8_t *) src + i * spitch, width);
            }
        }
    } இல்லையெனில் {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        தருக்க ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, உண்மை);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        க்கு (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        க்கு (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, உண்மை, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

வெறுமை ggml_vk_buffer_write(vk_buffer& dst, size_t offset, மாறிலி வெறுமை * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

தருக்க ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, வெறுமை * dst, size_t spitch, size_t dpitch, size_t width, size_t height, தருக்க sync_staging) {
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
    எனில் (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } இல்லையெனில் {
        slices.resize(height);
        க்கு (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    எனில் (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        திரும்பு உண்மை;
    }
    VK_LOG_DEBUG("STAGING");

    எனில் (!sync_staging) {
        // copy was not handled caller needs to fall back
        திரும்பு பொய்;
    }

    // Fall back to staging buffer
    மாறிலி size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    எனில் (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } இல்லையெனில் {
        staging_slices.resize(height);
        க்கு (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    எனில் (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } இல்லையெனில் {
        க்கு (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (மாறிலி uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, வெறுமை * dst, size_t size, தருக்க sync_staging = பொய்) {
    திரும்பு ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

வெறுமை ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, வெறுமை * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    மாறிலி தருக்க slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    எனில்(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
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
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, உண்மை, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        எனில் (width == spitch && width == dpitch) {
            memcpy(dst, (மாறிலி uint8_t *) src->ptr + offset, width * height);
        } இல்லையெனில் {
            க்கு (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (மாறிலி uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } இல்லையெனில் {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        தருக்க ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, உண்மை);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, உண்மை, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        க்கு (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

வெறுமை ggml_vk_buffer_read(vk_buffer& src, size_t offset, வெறுமை * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

வெறுமை ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

வெறுமை ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    எனில் (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, உண்மை, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } இல்லையெனில் {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

வெறுமை ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    எனில் (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        திரும்பு;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

வெறுமை ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    எனில் (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        திரும்பு;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, உண்மை, UINT64_MAX), "vk_memset waitForFences", dst->device);
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
    /* .reset           = */ வெற்று,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, வெறுமை * ptr, size_t size) {
    எனில் (!device->external_memory_host) {
        திரும்பு {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    எனில் (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        திரும்பு {};
    }
    எனில் (size & (device->min_imported_host_pointer_alignment - 1)) {
        திரும்பு {};
    }

    மாறிலி vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    முயல் {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } பிடி (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    திரும்பு buf;
}

