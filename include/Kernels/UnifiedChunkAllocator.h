#ifndef UNIFIED_CHUNK_ALLOCATOR_H
#define UNIFIED_CHUNK_ALLOCATOR_H

#include <vector>
#include <queue>
#include <new>
#include <cuda_runtime.h>
#include "Kernels/CudaUtils.h"

template <typename T>
class UnifiedChunk {
public:
    static constexpr int CHUNK_SIZE = 512;

    UnifiedChunk() {
        checkCudaError(
            cudaMallocManaged((void**)&um_data, CHUNK_SIZE * sizeof(T)),
            "UnifiedChunk: failed to allocate unified memory"
        );
    }

    T* slotAt(int i) { return &um_data[i]; }

    void prefetchToDevice(int device_id) {
        cudaMemPrefetchAsync(um_data, CHUNK_SIZE * sizeof(T), device_id);
    }

    void freeRaw() { cudaFree(um_data); }

    T* um_data;
};

template <typename T>
class UnifiedChunkAllocator {
public:
    T* allocate() {
        if (!free_slots.empty()) {
            T* ptr = free_slots.front();
            free_slots.pop();
            return ptr;
        }
        int chunk_idx = watermark / UnifiedChunk<T>::CHUNK_SIZE;
        int slot_idx  = watermark % UnifiedChunk<T>::CHUNK_SIZE;
        if (chunk_idx >= (int)chunks.size())
            chunks.push_back(new UnifiedChunk<T>());
        T* ptr = chunks[chunk_idx]->slotAt(slot_idx);
        new (ptr) T();
        watermark++;
        return ptr;
    }

    void deallocate(T* ptr) {
        free_slots.push(ptr);
    }

    void prefetchToDevice(int device_id) {
        for (auto* chunk : chunks)
            chunk->prefetchToDevice(device_id);
    }

    void shutdown() {
        for (int i = 0; i < watermark; i++) {
            int chunk_idx = i / UnifiedChunk<T>::CHUNK_SIZE;
            int slot_idx  = i % UnifiedChunk<T>::CHUNK_SIZE;
            chunks[chunk_idx]->slotAt(slot_idx)->freeMemory();
        }
        for (auto* chunk : chunks) {
            chunk->freeRaw();
            delete chunk;
        }
        chunks.clear();
        while (!free_slots.empty()) free_slots.pop();
        watermark = 0;
    }

private:
    std::vector<UnifiedChunk<T>*> chunks;
    std::queue<T*> free_slots;
    int watermark = 0;
};

#endif
