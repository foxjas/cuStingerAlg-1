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

// CPU Version - assume sorted index lists. 
int hostSingleIntersection ( vid_t ai, vid_t bi, degree_t alen, 
                            degree_t blen, const vid_t * a, const vid_t * b){

    int32_t ka = 0, kb = 0;
     int32_t out = 0;


    if (!alen || !blen || a[alen-1] < b[0] || b[blen-1] < a[0])
    return 0;

    const vid_t *aptr=a, *aend=a+alen;
    const vid_t *bptr=b, *bend=b+blen;

    while(aptr< aend && bptr<bend){
        if(*aptr==*bptr){
            aptr++, bptr++, out++;
        }
        else if(*aptr<*bptr){
            aptr++;
        }
        else {
            bptr++;
        }
      }  
  
    return out;
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
    
    const int work_factor = 99;
	char* outPath;
    if (argc > 2) {
        outPath = argv[2];
    } 

	bool isVerbose = hasOption("-v", argc, argv);
	bool isTopK = hasOption("--top", argc, argv);

    Timer<DEVICE> TM(5);
    //cudaProfilerStart();
    TM.start();

    cn.run(work_factor, isTopK, isVerbose);

    TM.stop();
    //cudaProfilerStop();
    TM.print("Computation time:");

    /*
    vid_t src = 4002;
    vid_t dst = 34026;
    const eoff_t *off = graph.csr_out_offsets();
    const vid_t *ind = graph.csr_out_edges();
    vid_t src_len = off[src+1]-off[src];
    vid_t dst_len = off[dst+1]-off[dst];
    int commonCount = hostSingleIntersection(src, dst, src_len, dst_len, ind+off[src], ind+off[dst]);
    std::cout << "(" << src << "," << dst << "): " << commonCount << std::endl;
    */
	if (argc > 3) {	
		cn.writeToFile(outPath);
  	}
    return 0;
}
