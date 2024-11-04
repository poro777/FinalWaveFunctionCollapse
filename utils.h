#pragma once
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <random>
#include <memory>
#include <iostream>
#include <unordered_set>

using std::unordered_set;
using std::shared_ptr;
using std::set;
using std::vector;
using std::map;
using ull = unsigned long long;


using Superposition = std::set<int>;
using Cell = Superposition;
using Grid = std::vector<std::vector<Superposition>>;
using Position = std::pair<int,int>;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1); // combine the two hash values
    }
};

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
    std::uniform_real_distribution<double> random_double;

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
        random_double = std::uniform_real_distribution<double>(0.0, 1.0);
    };
    ~RandomGen(){};

    int randomInt(){
        return random(gen);
    }
    double randomDouble(){
        return random_double(gen);
    }
};



void print_grid_bits(vector<vector<unsigned long long>>& grid);
void print_set(const set<int>& s);
void print_grid(Grid& grid);
inline void print_grid(Grid&& grid){print_grid(grid);}
