#pragma once
#include <queue>

#include "utils.h"
#include "setOperator.h"
#include "rules.h"
#include <cassert>
#include "myTimer.h"
#include <bit>

enum RETURN_STATE {
    OK,
    FAILED
};

using Rule = Rules::Rule;
class WFC
{
private:
    /* data */
protected:
    int H = 0;
    int W = 0;
    shared_ptr<Rule> rules = nullptr;
public:
    WFC(){};
    WFC(int H, int W,  shared_ptr<Rule> rules):H(H), W(W), rules(rules){

    };
    ~WFC(){

    };

    void printGrid() {
        print_grid(getGrid());
    };

    virtual Position selectOneCell(set<Position>& unobserved, RandomGen& random) = 0;
    virtual RETURN_STATE collapse(Position& position, RandomGen& random, bool print) = 0;
    virtual void propogate(set<Position>& unobserved, Position& position, bool print) = 0;
    virtual void validateNeighbor() {};

    virtual Grid getGrid() = 0;
};

class naive_WFC:public WFC
{
private:
    Grid grid;
    /* data */
public:
    naive_WFC(int H, int W, shared_ptr<Rule> rules): WFC(H,W,rules){
        grid = Grid(H, std::vector<Cell>(W, rules->initValue()));

    };
    ~naive_WFC(){

    };
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override ;
    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override ;
    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override;
    void validateNeighbor() override;

    Grid getGrid() override{
        return grid;
    }
};


/* Decorator of WFC base class, to measure the time of each function */
class profiling_WFC: public WFC
{
private:
    shared_ptr<WFC> component;
    /* data */
public:
    profiling_WFC(shared_ptr<WFC> component): component(component){

    };
    ~profiling_WFC(){

    };
    myTimer timer;
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override ;
    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override ;
    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override;
    void validateNeighbor() override {
        component->validateNeighbor();
    };

    Grid getGrid() override{
        return component->getGrid();
    };
};

class bit_WFC:public WFC
{
private:
    vector<vector<ull>> grid;
    vector<ull> top_bottom_rules; 
    vector<ull> bottom_top_rules;
    vector<ull> left_right_rules; 
    vector<ull> right_left_rules;

    /* data */
public:
    bit_WFC(int H, int W, shared_ptr<Rule> rules): WFC(H,W,rules){
        auto sp_to_bits = [](Superposition& sp){
            ull n = 0;
            for(auto pattern: sp){
                n += (1 << pattern);
            }
            return n;
        };
        assert(rules->M <= 64);

        auto init = rules->initValue();
        grid = vector<vector<ull>>(H, vector<ull>(W, sp_to_bits(init)));

        top_bottom_rules = vector<ull>(rules->M);
        bottom_top_rules = vector<ull>(rules->M);
        left_right_rules = vector<ull>(rules->M);
        right_left_rules = vector<ull>(rules->M);
        
        for (int i = 0; i < rules->M; i++)
        {
            top_bottom_rules[i] = sp_to_bits(rules->top_bottom_rules[i]);
            bottom_top_rules[i] = sp_to_bits(rules->bottom_top_rules[i]);
            left_right_rules[i] = sp_to_bits(rules->left_right_rules[i]);
            right_left_rules[i] = sp_to_bits(rules->right_left_rules[i]);
        }
        
    };
    ~bit_WFC(){

    };
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override ;
    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override ;
    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override;
    void validateNeighbor() override;

    Grid getGrid() override{
        Grid grid = Grid(H, vector<Superposition>(W));
        for (size_t h = 0; h < H; h++)
        {
            for (size_t w = 0; w < W; w++)
            {
                ull state = this->grid[h][w];
                auto bwidth = std::bit_width(state);
                auto& sp = grid[h][w];
                for (size_t i = 0; i < bwidth; i++)
                {
                    /* code */
                    if(state & 1u){
                        sp.insert(i);
                    }
                    state >>= 1u;
                }
            }
            
        }
        return grid;
    }
};