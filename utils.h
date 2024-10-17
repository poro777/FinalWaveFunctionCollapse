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


using Superposition = std::set<int>;
using Cell = Superposition;
using Grid = std::vector<std::vector<Superposition>>;
using Position = std::pair<int,int>;

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



void print_set(const set<int>& s);
void print_grid(Grid& grid);
inline void print_grid(Grid&& grid){print_grid(grid);}
