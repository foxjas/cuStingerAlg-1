/**
 * @brief
 * @file
 */

#include "HornetAlg.hpp"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
#include "Static/CommonNeighbors/commonNeigh.cuh"

using namespace timer;
using namespace hornets_nest;

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

bool hasOption(const char* option, int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], option) == 0)
          return true;
  }
  return false;
}


int main(int argc, char* argv[]) {
   
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], PRINT_INFO | SORT);
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    commonNeigh cn(hornet_graph);
    cn.init();
    
    const int work_factor = 9999;
	char* outPath;
    if (argc > 2) {
        outPath = argv[2];
    } 
	bool isTopK = hasOption("--top", argc, argv);

    Timer<DEVICE> TM(5);
    //cudaProfilerStart();
    TM.start();

    cn.run(work_factor, isTopK);

    TM.stop();
    //cudaProfilerStop();
    TM.print("Computation time:");

	if (argc > 2) {	
		cn.writeToFile(outPath);
  	}
    return 0;
}
