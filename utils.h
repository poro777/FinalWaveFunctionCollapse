#pragma once
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <random>

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
    RandomGen(/* args */){
        std::random_device rd;  // Obtain a random seed from the hardware
        gen = std::mt19937(rd());
        random = std::uniform_int_distribution<>(0, 100000);
    };
    ~RandomGen(){};

    int randomInt(){
        return random(gen);
    }
};



void print_set(const set<int>& s);
void print_grid(Grid& grid);