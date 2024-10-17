#pragma once
#include <queue>

#include "utils.h"
#include "setOperator.h"
#include "rules.h"
#include <cassert>
#include "myTimer.h"

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
    void propogate(set<Position>& unobserved, Position& position, bool print_process = false);

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
    void propogate(set<Position>& unobserved, Position& position, bool print_process = false);
    Grid getGrid() override{
        return component->getGrid();
    };
};