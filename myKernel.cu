#include "myKernel.cuh"
#include <cuda_runtime.h>
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA ERROR::" << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << " \n";
        cudaDeviceReset();
        exit(99);
    }
}
__global__
void add(int H, int W, ull* d_grid, ull* d_rules)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
}

__global__ 
void propogateCuda(int H, int W, int M, int center_row, int center_col, ull* d_grid, ull* d_rules)
{
    
    int index = threadIdx.x;
    int stride = blockDim.x;
    __shared__ unsigned long long d_rules_total[64*4];
    __shared__ int canexit;
    //__shared__ unsigned long long bottom_top_rules[64];
    //__shared__ unsigned long long left_right_rules[64];
    //__shared__ unsigned long long right_left_rules[64];
    //__shared__ double weights[64];
    //__shared__ double weightLogweights[64];
    for(int i=0;i<4*M;i++)
    {
        d_rules_total[i] = d_rules[i];
        //top_bottom_rules[i] = wfc_solver->top_bottom_rules[i];
        //bottom_top_rules[i] = wfc_solver->bottom_top_rules[i];
        //left_right_rules[i] = wfc_solver->left_right_rules[i];
        //right_left_rules[i] = wfc_solver->right_left_rules[i];
        //weights[i] = wfc_solver->weights[i];
        //weightLogweights[i] = wfc_solver->weightLogweights[i];
    }
    canexit = 0;
    int col = index % W;
    int row = index / W;
    int left = col - 1;
    int right = col + 1;
    int up = row - 1;
    int down = row + 1;
    bool leftcheck = left >= 0;
    bool rightcheck = right < W;
    bool upcheck = up >= 0 ;
    bool downcheck = down < H;
    ull my = d_grid[index];
    ull result = my;
    ull sp_left = 1<<M - 1, sp_right = 1<<M - 1, sp_up = 1<<M - 1, sp_down = 1<<M - 1;
    //__syncthreads();
    


    int c = H+W;

    
    while((c--))
    {
        for(int i=index;i<H*W;i+=stride)
        {
            col = i % W; row = i / W;
            left = col - 1; right = col + 1; up = row - 1; down = row + 1;
            leftcheck = left >= 0; rightcheck = right < W; upcheck = up >= 0; downcheck = down < H;
            my = d_grid[i];
            result = my;
            sp_left = leftcheck ? d_grid[row* W + left] : 1<<M - 1;
            sp_right = rightcheck ? d_grid[row* W + right] : 1<<M - 1;
            sp_up = upcheck ? d_grid[up*W + col] : 1<<M - 1;
            sp_down = downcheck ? d_grid[down*W + col] : 1<<M - 1;

            ull vaild_state = 0;
            for (ull j = 0; j < M && leftcheck; j++)
            {
                vaild_state = ((sp_left >> j) & 1ull) ? (vaild_state | d_rules_total[M * 2 + j]) : vaild_state;
            }
            result = leftcheck ? result & vaild_state : result;
            vaild_state = 0;
            for (ull j = 0; j < M && rightcheck; j++)
            {
                vaild_state = ((sp_right >> j) & 1ull) ? (vaild_state | d_rules_total[M * 3 + j]) : vaild_state;
            }
            result = rightcheck ? result & vaild_state : result;
            vaild_state = 0;
            for (ull j = 0; j < M && upcheck; j++)
            {   
                vaild_state = ((sp_up >> j) & 1ull) ? (vaild_state | d_rules_total[M * 0 + j]) : vaild_state;
            }
            result = upcheck ? result & vaild_state : result;
            vaild_state = 0;
            for (ull j = 0; j < M && downcheck; j++)
            {
                vaild_state = ((sp_down >> j) & 1ull) ? (vaild_state | d_rules_total[M * 1 + j]) : vaild_state;
            }
            result = downcheck ? result & vaild_state : result;
            d_grid[i] = result;
            if(my > result) canexit = 1;
            //canexit = my == result ? canexit : 1;
        }
        __syncthreads();
        if(canexit == 0)break;
        canexit = 0;
    }    
}

CudaWFC::CudaWFC(int H, int W,  shared_ptr<Rule> rules, int selection):WFC(H, W, rules,selection){
    assert(rules->M <= 64);

    auto init = sp_to_bits(rules->initValue());
    h_grid = (ull*)calloc(H * W, sizeof(ull));
    for(int i=0; i< H*W; i++)
        h_grid[i] = init;
    h_rules = (ull*)calloc(rules->M * 4, sizeof(ull));

    M = rules->M;
    for (int i = 0; i < rules->M; i++)
    {
        h_rules[M * 0 + i] = sp_to_bits(rules->top_bottom_rules[i]);
        h_rules[M * 1 + i] = sp_to_bits(rules->bottom_top_rules[i]);
        h_rules[M * 2 + i] = sp_to_bits(rules->left_right_rules[i]);
        h_rules[M * 3 + i] = sp_to_bits(rules->right_left_rules[i]);
    }
    
    // Copy the host vector to a Thrust device vector
    cudaMalloc((void**)&d_grid, sizeof(ull) * H*W);
    cudaMalloc((void**)&d_grid_backup, sizeof(ull) * H*W);
    cudaMalloc((void**)&d_rules, sizeof(ull) * rules->M * 4);
    cudaMemcpy(d_grid, h_grid, sizeof(ull) * H*W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules, h_rules, sizeof(ull) * rules->M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_backup, h_grid, sizeof(ull) * H*W, cudaMemcpyHostToDevice);
}


RETURN_STATE CudaWFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    int row = position.first, col = position.second;
    int index = row * W + col;
    // Copy a single value back to the host
    ull state = h_grid[index];
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
    h_grid[index] = state;
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
    cudaMemcpy(d_grid, h_grid, sizeof(ull) * H*W, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_grid, h_grid, sizeof(ull) * H*W, cudaMemcpyHostToDevice);
    int row = position.first, col = position.second;
    int index = row * W + col;
    propogateCuda<<<1,1024>>>(H,W, M, row, col,d_grid,d_rules);
    //add<<<1,1>>>(H,W,d_grid,d_rules);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
        return ;
    }
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(h_grid, d_grid, sizeof(ull) * H*W, cudaMemcpyDeviceToHost);
    for (auto it = unobserved.begin(); it != unobserved.end(); ) {
        Position pos = (*it);
        row = pos.first, col = pos.second;
        index = row * W + col;
        if (std::popcount(h_grid[index])==1) {
            it = unobserved.erase(it);
        }
        else {
            ++it;
        }
    }
}