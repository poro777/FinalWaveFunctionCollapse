#include "myKernel.cuh"
#include <cuda_runtime.h>

__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


CudaWFC::CudaWFC(int H, int W,  shared_ptr<Rule> rules, int selection):WFC(H, W, rules,selection){
    assert(rules->M <= 64);

    auto init = sp_to_bits(rules->initValue());
    h_grid = thrust::host_vector<ull>(H * W, init);
    thrust::host_vector<ull> h_rules(rules->M * 4, 0);

    int M = rules->M;
    for (int i = 0; i < rules->M; i++)
    {
        h_rules[M * 0 + i] = sp_to_bits(rules->top_bottom_rules[i]);
        h_rules[M * 1 + i] = sp_to_bits(rules->bottom_top_rules[i]);
        h_rules[M * 2 + i] = sp_to_bits(rules->left_right_rules[i]);
        h_rules[M * 3 + i] = sp_to_bits(rules->right_left_rules[i]);
    }
    
    // Copy the host vector to a Thrust device vector
    d_grid = h_grid;
    d_rules = h_rules;
}


RETURN_STATE CudaWFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    int row = position.first, col = position.second;
    int index = row * W + col;
    // Copy a single value back to the host
    ull state = d_grid[index];
    if(state == 0){ // There is no pattern available for a cell
        return FAILED;
    }

    int collapsed_state = -1;
    auto size = std::popcount(state); // count how many 1 in binary representation
    if(selection <= 2){
        // keep n-th 1, the other 1 to 0 
        auto n = findNthSetBit(state, 1 + (random.randomInt() % size));
        // collapse to one pattern
        state = 1ull << n;
        collapsed_state = n;
    }
    else{
        throw std::logic_error("Method not yet implemented");
    }

    // Copy a single value back to the device
    d_grid[index] = state;
    if(print_step){
        std::cout << position.first << " " << position.second;
        std::cout << " collapse to " << collapsed_state << "\n";
        printGrid();
        std::cout << "\n";
    }

    return OK;
};

template <typename Set>
Position CudaWFC::impl_selectOneCell(Set &unobserved, RandomGen &random){
    if(selection <= 1){  // first element of order_set, unorderd_set
        auto position_it = unobserved.begin();
        return *position_it;
    }
    else if (selection == 2){ // full random
        auto position_it = unobserved.begin();
        std::advance(position_it, random.randomInt() % unobserved.size());
        return *position_it;
    }
    else{
        // implement other methods e.g. min entropy selection
        // or cuda version
        throw std::logic_error("Method not yet implemented");
    }
}
template <typename Set>
void CudaWFC::impl_propogate(Set &unobserved, Position &position, bool print_process){
    // TODO
    d_grid;
    d_rules;
}