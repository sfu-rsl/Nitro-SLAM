#pragma once

namespace compute
{
    template <typename DataType>
    BufferPtr<DataType> loadFromFile(ComputeEngine &engine, const char *path, BufferType buffer_type = BufferType::Host)
    {
        std::ifstream file(path);

        std::string line;
        std::getline(file, line);
        std::istringstream is(line);
        auto start = std::istream_iterator<DataType>(is);
        auto end = std::istream_iterator<DataType>();

        std::vector<DataType> values(start, end);

        return engine.create_buffer<DataType>(values.data(), values.size(), buffer_type);
    }

    // Deprecated. Use SolverSeq instead.
    class SyncObject
    {

    private:
        VulkanComputeEngine *owner;
        std::shared_ptr<SolverSeq> seq_local;
        std::shared_ptr<SolverSeq> seq_device;
        // std::vector<std::shared_ptr<kp::Tensor>> to_sync;

    public:
        // [[deprecated("Use SolverSeq instead")]]
        SyncObject(
            std::shared_ptr<SolverSeq> seq_local,
            std::shared_ptr<SolverSeq> seq_device) : seq_local(seq_local), seq_device(seq_device)
        {
        }

        template <typename DataType>
        void rec(const std::vector<VCBPtr<DataType>> &buffers, const std::vector<std::pair<uint32_t, uint32_t>> &ranges = {})
        {

            seq_device->sync_device<DataType>(buffers, ranges);
            seq_local->sync_local<DataType>(buffers, ranges);
        }

        void sync_device()
        {
            seq_device->execute();
        }

        void sync_local()
        {
            seq_local->execute();
        }
    };

}