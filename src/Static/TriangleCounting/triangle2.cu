
#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/TriangleCounting/triangle2.cuh"
#include <iostream>
#include <fstream>

namespace hornets_nest {

TriangleCounting2::TriangleCounting2(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet)

{                                       
}

TriangleCounting2::~TriangleCounting2(){
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
		vid_t dst_neigh_index = -1; 
		// search in smaller degree vertex
		indexBinarySearch(u.neighbor_ptr(), u.degree(), v.id(), dst_neigh_index);
        eoff_t src_offset = d_offsets[u.id()];
        atomicAdd(d_triPerEdge+src_offset+dst_neigh_index, count);
    }
};


triangle_t TriangleCounting2::countTriangles(){

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
void TriangleCounting2::writeToFile(char* outPath) {

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

void TriangleCounting2::reset(){
    forAllVertices(hornet, OPERATOR_InitTriangleCounts { triPerEdge, hornet.device_csr_offsets() });
}

void TriangleCounting2::run() {
    forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerEdge, hornet.device_csr_offsets() }, 1);
}

void TriangleCounting2::run(const int WORK_FACTOR=1){
    forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerEdge, hornet.device_csr_offsets() }, WORK_FACTOR);
}


void TriangleCounting2::release(){
    gpu::free(triPerEdge);
    triPerEdge = nullptr;
}

void TriangleCounting2::init(){
    gpu::allocate(triPerEdge, hornet.nE());
    reset();
}

} // namespace hornets_nest
