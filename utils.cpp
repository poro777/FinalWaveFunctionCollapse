#include <iostream>

#include "utils.h"

void print_grid_bits(vector<vector<unsigned long long>> &grid)
{
    for (size_t h = 0; h < grid.size(); h++)
    {
        for (size_t w = 0; w < grid[0].size(); w++)
        {
            auto state = grid[h][w];
            std::cout <<" "<< state << " ";
        }
        std::cout << "\n";
        
    }
}

void print_set(const set<int>& s) {
    for (auto elem : s) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

void print_grid(Grid& grid){
    for (size_t h = 0; h < grid.size(); h++)
    {
        for (size_t w = 0; w < grid[0].size(); w++)
        {
            auto state = grid[h][w];
            
            if(state.size() == 0){
                std::cout << " z ";
            }
            else if(state.size() != 1){
                //std::cout << " r ";
                std::cout <<"-"<< state.size() << "-";

            }
            else{
                //auto s = *state.begin();
                
                std::cout <<" "<< *state.begin() << " ";
            }
            
        }
        std::cout << "\n";
        
    }
}