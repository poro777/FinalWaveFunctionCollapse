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

    int selection = 0; // mode of selectOneCell
public:
    WFC(){};
    WFC(int H, int W,  shared_ptr<Rule> rules, int selection):H(H), W(W), rules(rules), selection(selection){

    };
    ~WFC(){

    };

    int getH(){return H;}
    int getW(){return W;}

    void printGrid() {
        print_grid(getGrid());
    };

    virtual Position selectOneCell(set<Position>& unobserved, RandomGen& random) = 0;
    virtual Position selectOneCell(unordered_set<Position, pair_hash>& unobserved, RandomGen& random) = 0;

    virtual RETURN_STATE collapse(Position& position, RandomGen& random, bool print) = 0;

    virtual void propogate(set<Position>& unobserved, Position& position, bool print) = 0;
    virtual void propogate(unordered_set<Position, pair_hash>& unobserved, Position& position, bool print) = 0;

    virtual void validateNeighbor() {};

    virtual Grid getGrid() = 0;
};

class naive_WFC:public WFC
{
private:
    Grid grid;

    template <typename Set>
    Position impl_selectOneCell(Set& unobserved, RandomGen& random);

    template <typename Set>
    void impl_propogate(Set& unobserved, Position& position, bool print_process = false);

    vector<vector<double>> entropies;
    vector<double> weights;
    vector<double> weightLogweights;


public:
    naive_WFC(int H, int W, shared_ptr<Rule> rules, int selection): WFC(H,W,rules, selection){
        grid = Grid(H, std::vector<Cell>(W, rules->initValue()));

        weights = vector<double>(rules->M, 1e1);
        weightLogweights = vector<double>(rules->M, 0);
        double sumOfweights = std::accumulate(weights.begin(), weights.end(), 0);
        double sumOfweightLogweights = 0;
        for(int i = 0; i < rules->M; i++){
            weightLogweights[i] = weights[i] * log(weights[i]);
            sumOfweightLogweights += weights[i] * log(weights[i]);
        }        
        double starting_entropy = log(sumOfweights) - sumOfweightLogweights / sumOfweights;
        entropies = vector<vector<double>>(H, vector<double>(W, starting_entropy));
    };
    ~naive_WFC(){

    };
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override {
        return impl_selectOneCell(unobserved,random);
    };
    Position selectOneCell(unordered_set<Position, pair_hash>& unobserved, RandomGen& random) override {
        return impl_selectOneCell(unobserved,random);
    };

    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override ;

    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override {
        return impl_propogate(unobserved, position, print_process);
    };
    void propogate(unordered_set<Position, pair_hash>& unobserved, Position& position, bool print_process = false) override {
        return impl_propogate(unobserved, position, print_process);
    };

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
    profiling_WFC(shared_ptr<WFC> component, shared_ptr<myTimer> timer): component(component), timer(timer){
        H = component->getH();
        W = component->getW();
    };
    ~profiling_WFC(){

    };
    shared_ptr<myTimer> timer;
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override ;
    Position selectOneCell(unordered_set<Position, pair_hash>& unobserved, RandomGen& random) override ;

    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override ;

    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override;
    void propogate(unordered_set<Position, pair_hash>& unobserved, Position& position, bool print_process = false) override;

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
    template <typename Set>
    Position impl_selectOneCell(Set& unobserved, RandomGen& random);

    template <typename Set>
    void impl_propogate(Set& unobserved, Position& position, bool print_process = false);

protected:
    vector<vector<ull>> grid;
    vector<ull> top_bottom_rules; 
    vector<ull> bottom_top_rules;
    vector<ull> left_right_rules; 
    vector<ull> right_left_rules;

    vector<vector<double>> entropies;
    vector<double> weights;
    vector<double> weightLogweights;

public:
    bit_WFC(int H, int W, shared_ptr<Rule> rules, int selection): WFC(H,W,rules, selection){
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
        
        weights = vector<double>(rules->M, 1e1);
        weightLogweights = vector<double>(rules->M, 0);
        double sumOfweights = std::accumulate(weights.begin(), weights.end(), 0);
        double sumOfweightLogweights = 0;
        for(int i = 0; i < rules->M; i++){
            weightLogweights[i] = weights[i] * log(weights[i]);
            sumOfweightLogweights += weights[i] * log(weights[i]);
        }        
        double starting_entropy = log(sumOfweights) - sumOfweightLogweights / sumOfweights;
        entropies = vector<vector<double>>(H, vector<double>(W, starting_entropy));
    };
    ~bit_WFC(){

    };
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override {
        return impl_selectOneCell(unobserved,random);
    };
    Position selectOneCell(unordered_set<Position, pair_hash>& unobserved, RandomGen& random) override {
        return impl_selectOneCell(unobserved,random);
    };

    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override ;

    void propogate(set<Position>& unobserved, Position& position, bool print_process = false) override {
        return impl_propogate(unobserved, position, print_process);
    };
    void propogate(unordered_set<Position, pair_hash>& unobserved, Position& position, bool print_process = false) override {
        return impl_propogate(unobserved, position, print_process);
    };

    void validateNeighbor() override;

    Grid getGrid() override{
        Grid sp_grid = Grid(H, vector<Superposition>(W));
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                ull state = this->grid[h][w];
                bits_to_sp(state, sp_grid[h][w]);
            }
            
        }
        return sp_grid;
    }
};
