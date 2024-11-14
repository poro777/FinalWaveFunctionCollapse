#include "CudaWFC.h"
#include "utils.h"

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
__host__ ull CudaWFC::sp_to_bits(const Superposition& sp)
{
    ull n = 0;
    for(ull pattern: sp){
        n += (1ull << pattern);
    }
    return n;
}
__host__ int CudaWFC::selectOneCell(RandomGen& random)
{
    //std::cout<<"select\n";
    double min_ = 1e+4;
    int argmin = -1;
    for(int i=0;i<H*W;i++)
    {     
        if(collapseOk[i]==true)continue;       
        double entropy = entropies[i];
        if (entropy <= min_){
            double noise = 1e-6 * random.randomDouble();
            if (entropy + noise < min_){
                min_ = entropy + noise;
                argmin = i;
            }
        }
    }
    if(argmin > -1)
        collapseOk[argmin] = true;
    return argmin;   
    
}
__host__ bool CudaWFC::collapse(int position, bool print, RandomGen& random)
{
    ull state = grid[position];
    if(state == 0){ // There is no pattern available for a cell
        return FAILED;
    }
    double sum = 0;
    auto bwidth = M;
    for (ull i = 0; i < bwidth; i++){
        if((state >> i) & 1ull){
            sum += weights[i];
        }
    }
    double threshold = random.randomDouble() * sum;
    double partialSum = 0;
    int collapsed_state = -1;
    for (ull i = 0; i < bwidth; i++){
        if((state >> i) & 1ull){
            //std::cout<<i<<" ";
            partialSum += weights[i];
            if(partialSum >= threshold){
                collapsed_state = i;
                state = 1ull << i;
                break;
            }
        }
    }
    //std::cout<<"\n";

    entropies[position] = -1;
    propogateOk[position] = true;
    if(print){
        std::cout << position;
        std::cout << " collapse to " << collapsed_state << "\n";
        printGrid();
        std::cout << "\n";
    }
    grid[position] = 1ull<<collapsed_state;
    return OK;
}

__global__ void propogateCuda(int * check, CudaWFC* wfc_solver, int *notok, int *count)
{
    int M = wfc_solver->M;
    int col = threadIdx.x;
    __shared__ unsigned long long top_bottom_rules[64];
    __shared__ unsigned long long bottom_top_rules[64];
    __shared__ unsigned long long left_right_rules[64];
    __shared__ unsigned long long right_left_rules[64];
    __shared__ double weights[64];
    __shared__ double weightLogweights[64];
    if(col < M)
    {
        //for(int i=0;i<M;i++)
        //{
        int i = col;
        top_bottom_rules[i] = wfc_solver->top_bottom_rules[i];
        bottom_top_rules[i] = wfc_solver->bottom_top_rules[i];
        left_right_rules[i] = wfc_solver->left_right_rules[i];
        right_left_rules[i] = wfc_solver->right_left_rules[i];
        weights[i] = wfc_solver->weights[i];
        weightLogweights[i] = wfc_solver->weightLogweights[i];
        //}
    }
    __syncthreads();
    //wfc_solver->propogateOk[0] = true;
    int row = blockIdx.x;
    int left = col - 1;
    int right = col + 1;
    int up = row - 1;
    int down = row + 1;
    bool checkcanpropogate = false;
    bool leftcheck = ((left >= 0 && wfc_solver->propogateOk[row* blockDim.x + left]) == true);
    bool rightcheck = ((right < wfc_solver->W && wfc_solver->propogateOk[row*blockDim.x + right]) == true);
    bool upcheck = ((up >= 0 && wfc_solver->propogateOk[up*blockDim.x + col]) == true);
    bool downcheck = ((down < wfc_solver->H && wfc_solver->propogateOk[down*blockDim.x + col]) == true);
    if(leftcheck == false && rightcheck == false && upcheck == false && downcheck == false)return;
    if(wfc_solver->collapseOk[row* blockDim.x + col] == true)
    {
        wfc_solver->propogateOk[row* blockDim.x + col] = true;
        return;
    }
    if(wfc_solver->propogateOk[row* blockDim.x + col] == true)return;
    

    ull my = wfc_solver->grid[row * blockDim.x + col];
    if(leftcheck)
    {
        *(check) = 1;
        ull vaild_state = 0;
        ull sp = wfc_solver->grid[row*blockDim.x + left];
        int bwidth = M;
        for (ull i = 0; i < bwidth; i++)
        {
            if((sp >> i) & 1ull){
                ull rule = left_right_rules[i];
                vaild_state |= rule;
            }
        }
        //ull my = wfc_solver->grid[row * blockDim.x + col];
        ull result = my & vaild_state;

        // remove at least one element, add to queue propogate later.
        if(result < my )
        {
            my = result;
            int bwidth = M;
            double sumOfweights = 0;
            double sumOfweightLogweights = 0;
            int index = 0;
            for (ull i = 0; i < bwidth; i++){
                if((result >> i) & 1ull){
                    index += 1;
                    sumOfweights += weights[i];
                    sumOfweightLogweights += weightLogweights[i];
                }
            }
            if(index == 1)
                wfc_solver->collapseOk[row * blockDim.x + col] = true;
            wfc_solver->entropies[row * blockDim.x + col] = logf(sumOfweights) - sumOfweightLogweights / sumOfweights;
            *(count) = 1;
        }
        if(my == 0){
            wfc_solver->entropies[row * blockDim.x + col] = -1;
            wfc_solver->collapseOk[row * blockDim.x + col] = false;
            (*notok) = 1;
            wfc_solver->grid[row * blockDim.x + col] = 0;
            return;
        }
        wfc_solver->propogateOk[row*blockDim.x + col] = true;
    }
    //wfc_solver->propogateOk[row* blockDim.x + col] = true;
    //return;
    if(rightcheck)
    {
        *(check) = 1;
        ull vaild_state = 0;
        ull sp = wfc_solver->grid[row*blockDim.x + right];
        int bwidth = M;
        for (ull i = 0; i < bwidth; i++)
        {
            if((sp >> i) & 1ull){
                ull rule = right_left_rules[i];
                vaild_state |= rule;
            }
        }
        //ull my = wfc_solver->grid[row * blockDim.x + col];
        ull result = my & vaild_state;

        // remove at least one element, add to queue propogate later.
        if(result < my )
        {
            my = result;
            int bwidth = M;
            double sumOfweights = 0;
            double sumOfweightLogweights = 0;
            int index = 0;
            for (ull i = 0; i < bwidth; i++){
                if((result >> i) & 1ull){
                    index += 1;
                    sumOfweights += weights[i];
                    sumOfweightLogweights += weightLogweights[i];
                }
            }
            if(index == 1)
                wfc_solver->collapseOk[row * blockDim.x + col] = true;
            wfc_solver->entropies[row * blockDim.x + col] = logf(sumOfweights) - sumOfweightLogweights / sumOfweights;
            *(count) = 1;
        }
        if(my == 0){
            wfc_solver->collapseOk[row * blockDim.x + col] = false;
            wfc_solver->entropies[row * blockDim.x + col] = -1;
            (*notok) = 1;
            wfc_solver->grid[row * blockDim.x + col] = 0;
            return;
        }
        wfc_solver->propogateOk[row*blockDim.x + col] = true;
    }
    if(upcheck)
    {
        *(check) = 1;
        ull vaild_state = 0;
        ull sp = wfc_solver->grid[up*blockDim.x + col];
        int bwidth =M;
        for (ull i = 0; i < bwidth; i++)
        {
            if((sp >> i) & 1ull){
                ull rule = top_bottom_rules[i];
                vaild_state |= rule;
            }
        }
        //ull my = wfc_solver->grid[row * blockDim.x + col];
        ull result = my & vaild_state;

        // remove at least one element, add to queue propogate later.
        if(result < my )
        {
            my = result;
            int bwidth = M;
            double sumOfweights = 0;
            double sumOfweightLogweights = 0;
            int index = 0;
            for (ull i = 0; i < bwidth; i++){
                if((result >> i) & 1ull){
                    index += 1;
                    sumOfweights += weights[i];
                    sumOfweightLogweights += weightLogweights[i];
                }
            }
            if(index == 1)
                wfc_solver->collapseOk[row * blockDim.x + col] = true;
            wfc_solver->entropies[row * blockDim.x + col] = logf(sumOfweights) - sumOfweightLogweights / sumOfweights;
            *(count) = 1;
        }
        if(my == 0){
            wfc_solver->collapseOk[row * blockDim.x + col] = false;
            wfc_solver->entropies[row * blockDim.x + col] = -1;
            wfc_solver->grid[row * blockDim.x + col] = 0;
            (*notok) = 1;
            return;
        }
        wfc_solver->propogateOk[row*blockDim.x + col] = true;
    }
    if(downcheck)
    {
        *(check) = 1;
        ull vaild_state = 0;
        ull sp = wfc_solver->grid[down*blockDim.x + col];

        int bwidth = M;
        for (ull i = 0; i < bwidth; i++)
        {
            if((sp >> i) & 1ull){
                ull rule = bottom_top_rules[i];
                vaild_state |= rule;
            }
        }
        //ull my = wfc_solver->grid[row * blockDim.x + col];
        ull result = my & vaild_state;

        // remove at least one element, add to queue propogate later.
        if(result < my )
        {
            my = result;
            int bwidth = M;
            double sumOfweights = 0;
            double sumOfweightLogweights = 0;
            int index = 0;
            for (ull i = 0; i < bwidth; i++){
                if((result >> i) & 1ull){
                    index += 1;
                    sumOfweights += weights[i];
                    sumOfweightLogweights += weightLogweights[i];
                }
            }
            if(index == 1)
                wfc_solver->collapseOk[row * blockDim.x + col] = true;
            wfc_solver->entropies[row * blockDim.x + col] = logf(sumOfweights) - sumOfweightLogweights / sumOfweights;
            *(count) = 1;
        }
        if(my == 0){
            wfc_solver->collapseOk[row * blockDim.x + col] = false;
            wfc_solver->entropies[row * blockDim.x + col] = -1;
            wfc_solver->grid[row * blockDim.x + col] = 0;
            (*notok) = 1;
            return;
        }
        wfc_solver->propogateOk[row*blockDim.x + col] = true;      
    }
    wfc_solver->grid[row * blockDim.x + col] = my;

} 

__host__ void CudaWFC::propogate(int position)
{
    //std::cout<<"propogate\n";

    for(int i=0;i<H*W;i++)
        propogateOk[i] = false;
    propogateOk[position] = true;
    int *check;
    int *notok;
    int *count;
    cudaMalloc(&check, sizeof(int));
    cudaMalloc(&notok, sizeof(int));
    cudaMalloc(&count, sizeof(int));
    int one= 1;
    int zero= 0;
    cudaMemcpy(check, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(notok, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    

    while(true)
    {
        cudaMemcpy(check, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(notok, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(count, &zero, sizeof(int), cudaMemcpyHostToDevice);
        //tmp<<<H,W>>>(&check);
        
        propogateCuda<<<H, W>>>(check, this, notok, count);
        //    weights, weightLogweights, entropies, H, W);
        checkCudaErrors(cudaDeviceSynchronize());
        int host_check;
        int host_ontok;
        int host_count;
        cudaMemcpy(&host_check, check, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_ontok, notok, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_count, count, sizeof(int), cudaMemcpyDeviceToHost);

        if(host_ontok == 1)break;
        if(host_check == 0)break;
        if(host_count == 0)break;
    
        /*for(int i=0;i<H;i++)
        {
            for(int j=0;j<W;j++)
                std::cout<<propogateOk[i*W+j]<<" ";
            std::cout<<"\n";
        }
        std::cout<<"next\n";*/
        
    }
    cudaFree(check);
    cudaFree(notok);
}

void CudaWFC::bits_to_sp(ull state, Superposition &out_sp)
{
    int bwidth = M;
    for (int i = 0; i < bwidth; i++)
    {
        if(state & 1ull){
            out_sp.insert(i);
        }
        state >>= 1ull;
    }
}