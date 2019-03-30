
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

const int UNSET = -1; 

commonNeigh::commonNeigh(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet)
{                                       
}

commonNeigh::~commonNeigh(){
    release();
}

/*
 * Fills queue with vertices' ids in a subset range
 */
struct OPERATOR_InitVertexSubset {
    TwoLevelQueue<vid_t> queue;
    unsigned int vStart;

    OPERATOR (int tid) {
        queue.insert(vStart+tid);
    }
};

/*
 * Initializes pairs-related data in "sparse" manner
 * by accessing via length 2 chains
 */
struct OPERATOR_InitPairsData {
	int* d_pairsVisited;	
    count_t *d_countsPerPair;
    unsigned int vStart;
    const unsigned int nV;

	OPERATOR(Vertex& vertex, Edge& edge) {
		vid_t src_id = vertex.id();
		Vertex dst = edge.dst();

		for (int i=dst.degree()-1; i>= 0; i--) {
			vid_t dst_neighb_id = dst.neighbor_id(i); 

            //printf("(%d, %d, %d)\n", src_id, dst.id(), dst_neighb_id);
            // enforcing dst neighbor > src
            // early termination for sorted adjacency
            if (dst_neighb_id <= src_id) 
                break;

            int dst_neigh_offset = (src_id - vStart)*nV + dst_neighb_id;
            //printf("(%d, %d, %d)\n", src_id, dst.id(), dst_neighb_id);
            d_pairsVisited[dst_neigh_offset] = UNSET; 
            d_countsPerPair[dst_neigh_offset] = 0;
		}
    }
};

/*
 * Initializes vertex pairs from all length 2 chains
 * ForAllEdges operator
 * TODO: perf. optimization could be setting value to src_id + nV, in atomicCAS
 */
struct OPERATOR_InitPairsFromChains {
    TwoLevelQueue<vid2_t> uniquePairs;
	int* d_pairsVisited;	
    unsigned int vStart;
    const unsigned int nV;

	OPERATOR(Vertex& vertex, Edge& edge) {
		vid_t src_id = vertex.id();
		Vertex dst = edge.dst();
		for (int i=dst.degree()-1; i>= 0; i--) {
			vid_t dst_neighb_id = dst.neighbor_id(i); 
            // enforcing dst neighbor > src
            // early termination for sorted adjacency
            if (dst_neighb_id <= src_id) 
                break;


			// atomic compare and swap on d_pairsVisited;
            int dst_neigh_offset = (src_id - vStart)*nV + dst_neighb_id;
            int old_val = atomicCAS(d_pairsVisited+dst_neigh_offset, UNSET, (int)src_id);
			if (old_val != src_id) {
				uniquePairs.insert(xlib::make2<vid_t>(src_id, dst_neighb_id));
			}
		}
    }
};

struct OPERATOR_AdjIntersectionCountBalanced {
    count_t* d_countsPerPair;
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

void commonNeigh::reset(){
    //forAllVertices(hornet, OPERATOR_InitTriangleCounts { countsPerPair, hornet.device_csr_offsets() });
}

void commonNeigh::run() {
    return;
}

void commonNeigh::release(){
    gpu::free(d_countsPerPair);
    d_countsPerPair = nullptr;
}

void commonNeigh::init(){
    //gpu::allocate(countsPerPair, hornet.nE());
    reset();
}

/////////////////////////////////////////////////////
// Helpers for CPU-side verification
////////////////////////////////////////////////////
typedef struct tuple {
	vid_t u;
	vid_t v;
	count_t count;
	tuple(vid_t v1, vid_t v2, count_t neighCount) : u(v1), v(v2), count(neighCount) { } 
} mytuple;

bool decreasingComparator( mytuple& l, const mytuple& r) {
	return l.count > r.count; 
}
    
std::vector<mytuple> topK(count_t* countsPerPair, vid_t vStart, vid_t vEnd, vid_t nV, int K, bool verbose=false) {
	//std::vector<mytuple> indexedCounts;
	std::vector<mytuple> indexedCounts;

	for (vid_t u=vStart; u<vEnd; u++) {
		for (vid_t v = u+1; v<nV; v++) {
            count_t commonCount = countsPerPair[(u-vStart)*nV+v];
            // ** undefined behavior if there are < K non-zero pair counts
            if (commonCount > 0) {
			    mytuple pair_count = mytuple(u, v, commonCount);
			    indexedCounts.push_back(pair_count);
            }
		} 
	}
    std::nth_element(indexedCounts.begin(), indexedCounts.begin()+K, indexedCounts.end(), decreasingComparator);
    std::vector<mytuple> top_k(&indexedCounts[0], &indexedCounts[K]);
	std::sort(top_k.begin(), top_k.end(), decreasingComparator);
    if (verbose) {
        for (int i=0; i<K; i++) {
            //mytuple t = indexedCounts[i];
            mytuple t = top_k[i];
            std::cout << "(" << t.u << "," << t.v << "): " << t.count << std::endl;
        }
    }
    return top_k;
}

/*
 * Writes common neighbor counts to file
 */
void commonNeigh::writeToFile(char* outPath) {

    count_t* h_countsPerPair;
    host::allocate(h_countsPerPair, hornet.nE());
    gpu::copyToHost(d_countsPerPair, hornet.nE(), h_countsPerPair);

	std::ofstream fout;
    fout.open(outPath);
	const eoff_t* offsets = hornet.csr_offsets();
	const vid_t* edges = hornet.csr_edges();
	vid_t dst = -1;
    fout << "# Nodes: " << hornet.nV() << " Edges: " << hornet.nE() << std::endl;
	count_t triangles = 0;
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

void printResults(count_t* countsPerPair, unsigned int vStart, unsigned int vEnd, unsigned int nV) {
    
    for (unsigned int v = vStart; v < vEnd; v++) {
        for (int i=0; i<nV; i++) {
            std::cout << "(" << v << "," << i << "): " << countsPerPair[(v-vStart)*nV+i] << std::endl; 
        }
    }
}


/////////////////////////////////////////////////////
// Main logic 
////////////////////////////////////////////////////
void commonNeigh::run(const int WORK_FACTOR=9999, bool isTopK=false, bool verbose=false){
  
    using namespace timer;
    const unsigned int K = 10;
    const unsigned int nV = hornet.nV(); 
    const unsigned int QUEUE_PAIRS_LIMIT = min(nV*nV, (int)5E8); // allocate memory for pairs up to limit
    std::cout << "QUEUE_PAIRS_LIMIT: " << QUEUE_PAIRS_LIMIT << std::endl;
    const unsigned int vStepSize = floor((double)QUEUE_PAIRS_LIMIT/nV); // double to avoid underflow from division
    //std::cout << "vStepSize: " << vStepSize << std::endl;
    unsigned int vStart = 0;
    unsigned int vEnd = min(vStart + vStepSize, nV); 
    unsigned int queue_size;
    
    Timer<DEVICE> TM(5);
    queue.initialize(static_cast<size_t>(vStepSize*nV));
    TwoLevelQueue<vid_t> activeVertices(static_cast<size_t>(vStepSize));

    gpu::allocate(d_countsPerPair, vStepSize*nV);
    cudaMemset(d_countsPerPair, 0, vStepSize*nV*sizeof(count_t));
    int* d_pairsVisited;
    gpu::allocate(d_pairsVisited, vStepSize*nV);
    cudaMemset(d_pairsVisited, UNSET, vStepSize*nV*sizeof(int));
    std::vector<mytuple> top_k;
    while (vStart < nV) {
        //std::cout << "vStart: " << vStart << ", " << "vEnd: " << vEnd << std::endl;
       forAll(static_cast<size_t>(vEnd-vStart), OPERATOR_InitVertexSubset { activeVertices, vStart });
       activeVertices.swap();
       //std::cout << "active vertices size: " << activeVertices.size() << std::endl;

       TM.start();
       forAllEdges(hornet, activeVertices, OPERATOR_InitPairsFromChains { queue, d_pairsVisited, vStart, nV }, load_balancing);
       queue.swap(); // needed here 
       TM.stop();
       if (verbose)
            TM.print("Creating pairs:");

       const vid2_t* d_vertexPairs = queue.device_input_ptr();
       queue_size = queue.size();
       if (verbose)
            std::cout << "queue_size: " << queue_size << std::endl;
       

       TM.start();
       forAllAdjUnions(hornet, d_vertexPairs, queue_size, OPERATOR_AdjIntersectionCountBalanced { d_countsPerPair, vStart, nV }, WORK_FACTOR); 
       TM.stop();

       if (verbose)
            TM.print("Intersection processing:");
        
       if (isTopK) {  
            count_t* h_countsPerPair;
            host::allocate(h_countsPerPair, (vEnd-vStart)*nV); // TODO: move allocation outside loop; should be 1-time cost
            gpu::copyToHost(d_countsPerPair, (vEnd-vStart)*nV, h_countsPerPair);
            //printResults(h_countsPerPair, vStart, vEnd, nV);
            // logic for top_k calculations
            std::vector<mytuple> partition_top_k = topK(h_countsPerPair, vStart, vEnd, nV, K);
            std::vector<mytuple> temp;
            temp.reserve(partition_top_k.size() + top_k.size());
            temp.insert(temp.end(), top_k.begin(), top_k.end());
            temp.insert(temp.end(), partition_top_k.begin(), partition_top_k.end());
            std::sort(temp.begin(), temp.end(), decreasingComparator);
            top_k.assign(temp.begin(), temp.begin()+K);
       }

       TM.start();
       forAllEdges(hornet, activeVertices, OPERATOR_InitPairsData { d_pairsVisited, d_countsPerPair, vStart, nV }, load_balancing);
       TM.stop();
       if (verbose)
            TM.print("Resetting memory:");

       vStart = vEnd;
       vEnd = min(vEnd+vStepSize, nV);

    }

    if (isTopK) {
        for (int i=0; i<K; i++) {
            mytuple t = top_k[i];
            std::cout << "(" << t.u << "," << t.v << "): " << t.count << std::endl;
        }
    }
    
    gpu::free(d_pairsVisited); // TODO: move into release()
    
}



} // namespace hornets_nest