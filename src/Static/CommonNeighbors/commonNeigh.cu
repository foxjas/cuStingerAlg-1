
#include <cuda.h>
#include <cuda_runtime.h>
#include <Device/Util/Timer.cuh>
#include "Static/CommonNeighbors/commonNeigh.cuh"
#include <iostream>
#include <fstream>
#include <math.h>
//#include <tuple>
#include <vector>

namespace hornets_nest {

commonNeigh::commonNeigh(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet)

{                                       
}

commonNeigh::~commonNeigh(){
    release();
}
/*
struct OPERATOR_InitPairCommonNeighCounts {
    triangle_t *d_countsPerPair;
    // usage in ForAllEdges preferable, but unclear how to get dst index in adjacency
    OPERATOR (Vertex &vertex) {
        degree_t degree = vertex.degree();
        for (int i=0; i<degree; i++) {
            eoff_t src_offset = d_offsets[vertex.id()];
            //d_countsPerPair[src_offset+i] = 1; // test: total count should = |E|
            d_countsPerPair[src_offset+i] = 0; 
        }
    }
};
*/

/*
 * Search for position of key in array
 */
__device__ __forceinline__
void indexBinarySearch(vid_t* data, vid_t arrLen, vid_t key, vid_t& pos) {
    vid_t low = 0;
    vid_t high = arrLen - 1;
    while (high >= low) {
        vid_t middle = (low + high) / 2;
        if (data[middle] == key) {
             pos = middle;
             return;
        } else if (data[middle] < key) {
            low = middle + 1;
		} else {
            high = middle - 1;
		}
    }
}

struct OPERATOR_AdjIntersectionCountBalanced {
    triangle_t* d_countsPerPair;
    const unsigned int vertex_offset;
    const unsigned int nV;

    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
        int count = 0;
        if (!FLAG) {
            int comp_equals, comp1, comp2, ui_bound, vi_bound;
            //printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
            while (vi_begin <= vi_end && ui_begin <= ui_end) {
                comp_equals = (*ui_begin == *vi_begin);
                count += comp_equals;
                comp1 = (*ui_begin >= *vi_begin);
                comp2 = (*ui_begin <= *vi_begin);
                ui_bound = (ui_begin == ui_end);
                vi_bound = (vi_begin == vi_end);
                // early termination
                if ((ui_bound && comp2) || (vi_bound && comp1))
                    break;
                if ((comp1 && !vi_bound) || ui_bound)
                    vi_begin += 1;
                if ((comp2 && !ui_bound) || vi_bound)
                    ui_begin += 1;
            }
        } else {
            vid_t vi_low, vi_high, vi_mid;
            while (ui_begin <= ui_end) {
                auto search_val = *ui_begin;
                vi_low = 0;
                vi_high = vi_end-vi_begin;
                while (vi_low <= vi_high) {
                    vi_mid = (vi_low+vi_high)/2;
                    auto comp = (*(vi_begin+vi_mid) - search_val);
                    if (!comp) {
                        count += 1;
                        break;
                    }
                    if (comp > 0) {
                        vi_high = vi_mid-1;
                    } else if (comp < 0) {
                        vi_low = vi_mid+1;
                    }
                }
                ui_begin += 1;
            }
        }
        //printf("(%d, %d)\n", u.id(), v.id());
        // NOTE: this will error if u > v
        bool sourceSmaller = u.id() < v.id();
        vid_t u_id = sourceSmaller ? u.id() : v.id();
        vid_t v_id = sourceSmaller ? v.id() : u.id(); 
        eoff_t offset = (u_id-vertex_offset)*nV+v_id;
        atomicAdd(d_countsPerPair+offset, count);
    }
};

/*
typedef std::tuple<vid_t,vid_t,triangle_t> mytuple;
bool decreasingComparator( const mytuple& l, const mytuple& r) {
	return std::get<2>(l)> std::get<2>(r); 
}
*/
typedef struct tuple {
	vid_t u;
	vid_t v;
	triangle_t count;
	tuple(vid_t v1, vid_t v2, triangle_t neighCount) : u(v1), v(v2), count(neighCount) { } 
} mytuple;

bool decreasingComparator( mytuple& l, const mytuple& r) {
	return l.count > r.count; 
}
    
void topK(triangle_t* countsPerPair, vid_t vStart, vid_t vEnd, vid_t nV, int K) {
	//std::vector<mytuple> indexedCounts;
	std::vector<mytuple> indexedCounts;

	for (vid_t u=vStart; u<vEnd; u++) {
		for (vid_t v = u+1; v<nV; v++) {
			mytuple pair_count = mytuple(u, v, countsPerPair[(u-vStart)*nV+v]);
			indexedCounts.push_back(pair_count);
		} 
	}
	std::cout << "indexedCounts size: " << indexedCounts.size() << std::endl;	
	// consider using std::nth_element instead
	std::sort(indexedCounts.begin(), indexedCounts.end(), decreasingComparator);
	for (int i=0; i<K; i++) {
		/*vid_t u = std::get<0>(indexedCounts[i]);
		vid_t v = std::get<1>(indexedCounts[i]);
		triangle_t count = std::get<2>(indexedCounts[i]);
		*/
		mytuple t = indexedCounts[i];
		std::cout << "(" << t.u << "," << t.v << "): " << t.count << std::endl;
	}
}

triangle_t commonNeigh::countTriangles(){

    triangle_t* h_countsPerPair;
    host::allocate(h_countsPerPair, hornet.nE());
    gpu::copyToHost(d_countsPerPair, hornet.nE(), h_countsPerPair);
    triangle_t sum=0;
    for(int i=0; i<hornet.nE(); i++){
        // printf("%d %ld\n", i,outputArray[i]);
        sum+=h_countsPerPair[i];
    }
    free(h_countsPerPair);
    //triangle_t sum=gpu::reduce(hd_triangleData().countsPerPair, hd_triangleData().nv+1);

    return sum;
}

/*
 * Writes common neighbor counts to file
 */
void commonNeigh::writeToFile(char* outPath) {

    triangle_t* h_countsPerPair;
    host::allocate(h_countsPerPair, hornet.nE());
    gpu::copyToHost(d_countsPerPair, hornet.nE(), h_countsPerPair);

	std::ofstream fout;
    fout.open(outPath);
	const eoff_t* offsets = hornet.csr_offsets();
	const vid_t* edges = hornet.csr_edges();
	vid_t dst = -1;
    fout << "# Nodes: " << hornet.nV() << " Edges: " << hornet.nE() << std::endl;
	triangle_t triangles = 0;
    for (vid_t src=0; src<hornet.nV(); src++) {
		for (eoff_t j=offsets[src]; j<offsets[src+1]; j++) {
			dst = edges[j];
			triangles = h_countsPerPair[j];
			fout << src << " " << dst << " " << triangles << std::endl;
		}
    }
    fout.close();
    free(h_countsPerPair);
}

void printResults(triangle_t* countsPerPair, unsigned int vStart, unsigned int vEnd, unsigned int nV) {
    
    for (unsigned int v = vStart; v < vEnd; v++) {
        for (int i=0; i<nV; i++) {
            std::cout << "(" << v << "," << i << "): " << countsPerPair[(v-vStart)*nV+i] << std::endl; 
        }
    }
}

void commonNeigh::reset(){
    //forAllVertices(hornet, OPERATOR_InitTriangleCounts { countsPerPair, hornet.device_csr_offsets() });
}

void commonNeigh::run() {
    return;
}

void commonNeigh::run(const int WORK_FACTOR=1){
  
    using namespace timer;
    const unsigned int nV = hornet.nV(); 
    const unsigned int QUEUE_PAIRS_LIMIT = min(nV*nV, (int)5E8); // allocate memory for pairs up to limit
    std::cout << "QUEUE_PAIRS_LIMIT: " << QUEUE_PAIRS_LIMIT << std::endl;
    vid2_t* vertexPairs = NULL;
    const unsigned int vStepSize = floor((double)QUEUE_PAIRS_LIMIT/nV); // double to avoid underflow from division
    std::cout << "vStepSize: " << vStepSize << std::endl;
    unsigned int vStart = 0;
    unsigned int vEnd = min(vStart + vStepSize, nV); 
    unsigned int queue_size;

    vertexPairs = new vid2_t[QUEUE_PAIRS_LIMIT];
    vid2_t* d_vertexPairs = nullptr; 
    gpu::allocate(d_vertexPairs, QUEUE_PAIRS_LIMIT); // could be smaller
    gpu::allocate(d_countsPerPair, vStepSize*nV);
    cudaMemset(d_countsPerPair, 0, vStepSize*nV*sizeof(triangle_t)); // initialize pair common neighbor counts to 0
    Timer<DEVICE> TM(5);
    while (vStart < nV) {
        std::cout << "vStart: " << vStart << ", " << "vEnd: " << vEnd << std::endl;
       // fill array 
       TM.start();
       // could cut memory requirements by ~half if we enforce (u < v)
       for (unsigned int v = vStart; v < vEnd; v++) {
           for (unsigned int index = 0; index < nV; index++) {
              vertexPairs[(v-vStart)*nV+index] = xlib::make2<vid_t>(v, index);
           }
       }
       queue_size = (vEnd-vStart)*nV;
       std::cout << "queue_size: " << queue_size << std::endl;
       cudaMemcpy(d_vertexPairs, vertexPairs, queue_size*sizeof(vid2_t), cudaMemcpyHostToDevice);
       TM.stop();
       TM.print("Creating pairs:");
       forAllAdjUnions(hornet, d_vertexPairs, queue_size, OPERATOR_AdjIntersectionCountBalanced { d_countsPerPair, vStart, nV }, WORK_FACTOR); 

       triangle_t* h_countsPerPair;
       host::allocate(h_countsPerPair, (vEnd-vStart)*nV);
       gpu::copyToHost(d_countsPerPair, (vEnd-vStart)*nV, h_countsPerPair);
       //printResults(h_countsPerPair, vStart, vEnd, nV);
	   topK(h_countsPerPair, vStart, vEnd, nV, 10);

       vStart = vEnd;
       vEnd = min(vEnd+vStepSize, nV);
       cudaMemset(d_countsPerPair, 0, vStepSize*nV*sizeof(triangle_t)); // initialize pair common neighbor counts to 0
    }

    delete [] vertexPairs;
    gpu::free(d_vertexPairs);
}


void commonNeigh::release(){
    gpu::free(d_countsPerPair);
    d_countsPerPair = nullptr;
}

void commonNeigh::init(){
    //gpu::allocate(countsPerPair, hornet.nE());
    reset();
}

} // namespace hornets_nest
