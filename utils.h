#pragma once
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <random>
#include <memory>
#include <iostream>

using std::shared_ptr;
using std::set;
using std::vector;
using std::map;
using ull = unsigned long long;


using Superposition = std::set<int>;
using Cell = Superposition;
using Grid = std::vector<std::vector<Superposition>>;
using Position = std::pair<int,int>;

template <typename T>
int findNthSetBit(T bits, int N) {
    int position = 0;
    int count = 0;
    
    while (bits != 0) {
        if (bits & 1) {  // If the least significant bit is 1
            count++;  // Increment the count of '1' bits
            if (count == N) {
                return position;  // Return the position of the N-th '1'
            }
        }
        bits >>= 1;  // Shift right to check the next bit
        position++;  // Move to the next bit position
    }
    
    return -1;  // Return -1 if there are not enough '1' bits
}

class RandomGen
{
private:
    std::mt19937 gen; // Seed the generator
    std::uniform_int_distribution<> random;

public:
    RandomGen(long long seed = -1){
        if(seed == -1){
            std::random_device rd;  // Obtain a random seed from the hardware
            auto seed = rd();
            std::cout << "Seed: " << seed << "\n";
            gen = std::mt19937(seed);
        }  
        else{
            std::cout << "Seed: " << seed << "\n";
            gen = std::mt19937(seed);
        }

        random = std::uniform_int_distribution<>(0, 100000);
    };
    ~RandomGen(){};

    int randomInt(){
        return random(gen);
    }
};



void print_grid_bits(vector<vector<unsigned long long>>& grid);
void print_set(const set<int>& s);
void print_grid(Grid& grid);
inline void print_grid(Grid&& grid){print_grid(grid);}
