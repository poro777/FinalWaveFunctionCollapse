#pragma once
#include "WFC.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>



class CudaWFC: public WFC
{
private:
    /* data */
protected:
    shared_ptr<Rule> rules = nullptr;
    
    template <typename Set>
    Position impl_selectOneCell(Set& unobserved, RandomGen& random);

    template <typename Set>
    void impl_propogate(Set& unobserved, Position& position, bool print_process = false);

    // 4 * M, top_bottom_rules, bottom_top_rules, left_right_rules, right_left_rules
    thrust::device_vector<ull> d_rules;
    
    // H * W, row major
    thrust::device_vector<ull> d_grid;
    thrust::host_vector<ull> h_grid;
public:
    CudaWFC(){};
    CudaWFC(int H, int W,  shared_ptr<Rule> rules, int selection);
    ~CudaWFC(){

    };

    int getH(){return H;}
    int getW(){return W;}

    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override {
        return impl_selectOneCell(unobserved,random);
    };
    Position selectOneCell(unordered_set<Position, pair_hash>& unobserved, RandomGen& random) override {
        return impl_selectOneCell(unobserved,random);
    };

    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override;

    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override {
        return impl_propogate(unobserved, position, print_process);
    };
    void propogate(unordered_set<Position, pair_hash>& unobserved, Position& position, bool print_process = false) override {
        return impl_propogate(unobserved, position, print_process);
    };

    Grid getGrid() override{
        // copy from device to host
        h_grid = d_grid;

        Grid sp_grid = Grid(H, vector<Superposition>(W));
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                ull state = h_grid[h * W + w];
                bits_to_sp(state, sp_grid[h][w]);
            }
            
        }
        return sp_grid;
    }
};

