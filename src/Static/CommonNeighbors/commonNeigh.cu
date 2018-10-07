
#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/CommonNeighbors/commonNeigh.cuh"
#include <iostream>
#include <fstream>
#include <math.h> 

namespace hornets_nest {

commonNeigh::commonNeigh(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet)

{                                       
}

commonNeigh::~commonNeigh(){
    release();
}

struct OPERATOR_InitTriangleCounts {
    triangle_t *d_triPerEdge;
    const eoff_t* d_offsets;
    // usage in ForAllEdges preferable, but unclear how to get dst index in adjacency
    OPERATOR (Vertex &vertex) {
        degree_t degree = vertex.degree();
        for (int i=0; i<degree; i++) {
            eoff_t src_offset = d_offsets[vertex.id()];
            //d_triPerEdge[src_offset+i] = 1; // test: total count should = |E|
            d_triPerEdge[src_offset+i] = 0; 
        }
    }
};

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
    triangle_t* d_triPerEdge;
    //vid2_t* d_vertexPairs;
    const eoff_t* d_offsets;

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
        printf("(%d, %d)\n", u.id(), v.id());
        /*
		vid_t dst_neigh_index = -1; 
		// search in smaller degree vertex
		indexBinarySearch(u.neighbor_ptr(), u.degree(), v.id(), dst_neigh_index);
        eoff_t src_offset = d_offsets[u.id()];
        atomicAdd(d_triPerEdge+src_offset+dst_neigh_index, count);
        */
    }
};


triangle_t commonNeigh::countTriangles(){

    triangle_t* h_triPerEdge;
    host::allocate(h_triPerEdge, hornet.nE());
    gpu::copyToHost(triPerEdge, hornet.nE(), h_triPerEdge);
    triangle_t sum=0;
    for(int i=0; i<hornet.nE(); i++){
        // printf("%d %ld\n", i,outputArray[i]);
        sum+=h_triPerEdge[i];
    }
    free(h_triPerEdge);
    //triangle_t sum=gpu::reduce(hd_triangleData().triPerEdge, hd_triangleData().nv+1);

    return sum;
}

/*
 * Writes triangle counts by edge to file
 */
void commonNeigh::writeToFile(char* outPath) {

    triangle_t* h_triPerEdge;
    host::allocate(h_triPerEdge, hornet.nE());
    gpu::copyToHost(triPerEdge, hornet.nE(), h_triPerEdge);

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
			triangles = h_triPerEdge[j];
			fout << src << " " << dst << " " << triangles << std::endl;
		}
    }
    fout.close();
    free(h_triPerEdge);
}

void commonNeigh::reset(){
    forAllVertices(hornet, OPERATOR_InitTriangleCounts { triPerEdge, hornet.device_csr_offsets() });
}

void commonNeigh::run() {
    return;
}

void commonNeigh::run(const int WORK_FACTOR=1){
    const unsigned int QUEUE_PAIRS_LIMIT = 1E9;
    const unsigned int nV = hornet.nV(); 
    vid2_t* vertexPairs = NULL;
    const unsigned int vStepSize = ceil(QUEUE_PAIRS_LIMIT/nV); // double to avoid underflow from division
    unsigned int vStart = 0;
    unsigned int vEnd = min(vStart + vStepSize, nV); 
    unsigned int queue_size;

    vertexPairs = new vid2_t[QUEUE_PAIRS_LIMIT];
    vid2_t* d_vertexPairs = nullptr; 
    gpu::allocate(d_vertexPairs, QUEUE_PAIRS_LIMIT); // could be smaller

    while (vStart < nV) {
        std::cout << "vStart: " << vStart << ", " << "vEnd: " << vEnd << std::endl;
       // fill array 
       for (unsigned int v = vStart; v < vEnd; v++) {
           for (unsigned int index = 0; index < nV; index++) {
              vertexPairs[v*nV+index] = xlib::make2<vid_t>(v, index);
           }
       }
       queue_size = (vEnd-vStart)*nV;
       cudaMemcpy(d_vertexPairs, vertexPairs, queue_size*sizeof(vid2_t), cudaMemcpyHostToDevice);
       forAllAdjUnions(hornet, d_vertexPairs, queue_size, OPERATOR_AdjIntersectionCountBalanced { triPerEdge, hornet.device_csr_offsets() }, WORK_FACTOR); 
       vStart = vEnd;
       vEnd = min(vEnd+vStepSize, nV);
    }

    delete [] vertexPairs;
    gpu::free(d_vertexPairs);
}


void commonNeigh::release(){
    gpu::free(triPerEdge);
    triPerEdge = nullptr;
}

void commonNeigh::init(){
    gpu::allocate(triPerEdge, hornet.nE());
    reset();
}

} // namespace hornets_nest
