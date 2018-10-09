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


// CPU Version - assume sorted index lists. 
int hostSingleIntersection (const vid_t ai, const degree_t alen, const vid_t * a,
                            const vid_t bi, const degree_t blen, const vid_t * b){

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

void hostCountCommonNeighbors(const vid_t nv, const vid_t ne, const eoff_t * off,
    const vid_t * ind, int64_t* allTriangles)
{
    int32_t edge=0;
    int64_t sum=0;

    for (vid_t src = 0; src < nv; src++)
    {
        degree_t srcLen=off[src+1]-off[src];
        for(int iter=off[src]; iter<off[src+1]; iter++)
        {
            vid_t dest=ind[iter];
            degree_t destLen=off[dest+1]-off[dest];            
            int64_t tris= hostSingleIntersection (src, srcLen, ind+off[src],
                                                    dest, destLen, ind+off[dest]);
            sum+=tris;
        }
    }    
    *allTriangles=sum;
    //printf("Sequential number of triangles %ld\n",sum);
}


int main(int argc, char* argv[]) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);      
    std::cout << "Number of devices: " << deviceCount << std::endl; 
    //int device = 4;
    //cudaSetDevice(device);
    //struct cudaDeviceProp properties;
    //cudaGetDeviceProperties(&properties, device);
    //std::cout<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<std::endl;
    //std::cout<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<std::endl;
   
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


    Timer<DEVICE> TM(5);
    //cudaProfilerStart();
    TM.start();

    cn.run(work_factor);

    TM.stop();
    //cudaProfilerStop();
    TM.print("Computation time:");

    //triangle_t deviceTriangleCount = cn.countTriangles();
    //printf("Device triangles: %llu\n", deviceTriangleCount);

	if (argc > 2) {	
		cn.writeToFile(outPath);
  	}
    return 0;
}
