#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

namespace hornets_nest {

//using triangle_t = int;
using triangle_t = unsigned long long;
using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;


//==============================================================================

class commonNeigh : public StaticAlgorithm<HornetGraph> {
public:
    commonNeigh(HornetGraph& hornet);
    ~commonNeigh();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void run(const int WORK_FACTOR);
    void init();

    triangle_t countTriangles();
    void writeToFile(char* outPath);

private:
   TwoLevelQueue<vid2_t>        queue;
   load_balancing::BinarySearch load_balancing;
   triangle_t* d_countsPerPair { nullptr };

};

//==============================================================================

} // namespace hornets_nest
