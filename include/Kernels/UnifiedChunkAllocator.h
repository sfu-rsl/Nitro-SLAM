#ifndef UNIFIED_CHUNK_ALLOCATOR_H
#define UNIFIED_CHUNK_ALLOCATOR_H

#include <vector>
#include <queue>
#include <new>
#include <mutex>
#include <shared_mutex>
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

    // No-op on Jetson: see the note in CudaKeyFrameAllocator::create.
    void prefetchToDevice(int device_id) {
#ifndef DEVICE_JETSON
        cudaMemPrefetchAsync(um_data, CHUNK_SIZE * sizeof(T), device_id);
#else
        (void)device_id;
#endif
    }

    void freeRaw() { cudaFree(um_data); }

    T* um_data;
};

// Thread-safe pool of unified-memory slots for T.
//
// Slots are handed out from a free list first, then from fresh chunks. A slot
// is constructed exactly once (placement new on first use) and recycled as-is
// afterwards, so T is expected to own device buffers that survive recycling and
// are released in T::freeMemory() at shutdown().
//
// The mutex is shared: mutating calls (allocate/deallocate/shutdown) take it
// exclusively, read-only calls (prefetchToDevice/allocatedSlots) take it shared
// so concurrent readers do not serialize against each other.
template <typename T>
class UnifiedChunkAllocator {
public:
    static UnifiedChunkAllocator<T>& instance() {
        static UnifiedChunkAllocator<T> allocator;
        return allocator;
    }

    T* allocate() {
        std::unique_lock<std::shared_timed_mutex> lock(mtx);
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
        if (ptr == nullptr) return;
        std::unique_lock<std::shared_timed_mutex> lock(mtx);
        free_slots.push(ptr);
    }

    void prefetchToDevice(int device_id) {
        std::shared_lock<std::shared_timed_mutex> lock(mtx);
        for (auto* chunk : chunks)
            chunk->prefetchToDevice(device_id);
    }

    // Slots currently handed out, i.e. everything ever constructed minus the
    // ones sitting on the free list.
    int liveSlots() {
        std::shared_lock<std::shared_timed_mutex> lock(mtx);
        return watermark - (int)free_slots.size();
    }

    // Idempotent: a second call finds no chunks left and does nothing.
    void shutdown() {
        std::unique_lock<std::shared_timed_mutex> lock(mtx);
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
    std::shared_timed_mutex mtx;
};

#endif
