#pragma once
#include <queue>

#include "utils.h"
#include "setOperator.h"
#include "rules.h"
#include <cassert>
#include "myTimer.h"
#include <bit>
#include <cmath>
#include <numeric>
#include <queue>
#include <cuda_runtime.h>
enum RETURN_STATE {
    OK,
    FAILED
};

using Rule = Rules::Rule;


class CudaWFC
{
    public:
//private:
    ull* grid;
    ull* top_bottom_rules; 
    ull* bottom_top_rules;
    ull* left_right_rules; 
    ull* right_left_rules;

    bool* collapseOk;
    bool* propogateOk;

    __host__ ull sp_to_bits(const Superposition& sp);
    //Superposition bits_to_sp(ull value);
    //void bits_to_sp(ull state, Superposition& out_sp);

    //template <typename Set>
    //Position impl_selectOneCell(Set& unobserved, RandomGen& random);

    //template <typename Set>
    //void impl_propogate(Set& unobserved, Position& position, bool print_process = false);
    
    double* entropies;
    double* weights;
    double* weightLogweights;
    /* data */
//protected:
    int H = 0;
    int W = 0;
    int M = 0;
    shared_ptr<Rule> rules = nullptr;
    ull **cuda_rules;
    int selection = 0; // mode of selectOneCell

    __host__ CudaWFC(){};

    __host__ void CudaWFCinit(int _H, int _W, shared_ptr<Rule> _rules,  int _selection){
        H = _H;
        W = _W;
        M = _rules->M;
        rules = _rules;
        selection = _selection;

        
        //assume rules is transform ok
        cudaMallocManaged(&grid, H*W*sizeof(ull));
        ull init = 0;
        for(int i=0;i<rules->M;i++)
            init += (1<<i);
        for(int i=0;i<H*W;i++)
            grid[i] = init;

        cudaMallocManaged(&top_bottom_rules, M*sizeof(ull));
        cudaMallocManaged(&bottom_top_rules, M*sizeof(ull));
        cudaMallocManaged(&left_right_rules, M*sizeof(ull));
        cudaMallocManaged(&right_left_rules, M*sizeof(ull));

        
        for (int i = 0; i < rules->M; i++)
        {
            top_bottom_rules[i] = sp_to_bits(rules->top_bottom_rules[i]);
            bottom_top_rules[i] = sp_to_bits(rules->bottom_top_rules[i]);
            left_right_rules[i] = sp_to_bits(rules->left_right_rules[i]);
            right_left_rules[i] = sp_to_bits(rules->right_left_rules[i]);
        }
        cudaMallocManaged(&weights, M*sizeof(double));
        for(int i=0;i<M;i++)
            weights[i] = 1e1;
        //weights = vector<double>(rules->M, 1e1);
        cudaMallocManaged(&weightLogweights, M*sizeof(double));
        for(int i=0;i<M;i++)
            weightLogweights[i] = 0;
        //weightLogweights = vector<double>(rules->M, 0);
        double sumOfweights = 0;
        for(int i=0;i<rules->M;i++)
            sumOfweights += weights[i];

        double sumOfweightLogweights = 0;
        for(int i = 0; i < rules->M; i++){
            weightLogweights[i] = weights[i] * log(weights[i]);
            sumOfweightLogweights += weights[i] * log(weights[i]);
        }        
        double starting_entropy = log(sumOfweights) - sumOfweightLogweights / sumOfweights;
        
        cudaMallocManaged(&entropies,H*W*sizeof(double));
        cudaMallocManaged(&collapseOk,H*W*sizeof(bool));
        cudaMallocManaged(&propogateOk,H*W*sizeof(bool));
        for(int i = 0; i < H*W; i++){
            entropies[i] = starting_entropy;
            collapseOk[i] = false;
            propogateOk[i] = false;
        }        
        //entropies = vector<vector<double>>(H, vector<double>(W, starting_entropy));
        
    };
    __host__ ~CudaWFC(){
        cudaFree(grid);
        cudaFree(top_bottom_rules);
        cudaFree(bottom_top_rules);
        cudaFree(left_right_rules);
        cudaFree(right_left_rules);
        cudaFree(weights);
        cudaFree(weightLogweights);
        cudaFree(collapseOk);
        cudaFree(entropies);
        cudaFree(propogateOk);
    };

    __host__ int getH(){return H;}
    __host__ int getW(){return W;}

    __host__ void printGrid() {
        //print_grid(getGrid());
    };

    __host__ int selectOneCell(RandomGen& random);

    __host__ bool collapse(int position, bool print, RandomGen& random);

    __host__ void propogate(int positoin);
   //__host__ __device__  void propogateCuda(int *check);

    //__host__ __device__ void propogateCell(int *check);
    //virtual void propogate(unordered_set<Position, pair_hash>& unobserved, Position& position, bool print) = 0;

    //virtual void validateNeighbor() {};

    void bits_to_sp(ull state, Superposition& out_sp);
    Grid getGrid(){
        Grid sp_grid = Grid(H, vector<Superposition>(W));
        for (size_t h = 0; h < H; h++)
        {
            for (size_t w = 0; w < W; w++)
            {
                ull state = this->grid[h*W+w];
                bits_to_sp(state, sp_grid[h][w]);
            }
            
        }
        return sp_grid;
    }



};