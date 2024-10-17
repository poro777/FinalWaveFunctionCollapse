#pragma once
#include <queue>

#include "utils.h"
#include "setOperator.h"
#include "rules.h"
#include <cassert>

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
    shared_ptr<Rule> rules;
public:
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
    Position selectOneCell(set<Position>& unobserved, RandomGen& random) override {
        auto position_it = unobserved.begin();
        std::advance(position_it, random.randomInt() % unobserved.size());

        return *position_it;
    };

    RETURN_STATE collapse(Position& position, RandomGen& random, bool print_step = false) override {
        Superposition& state = grid[position.first][position.second];
        if(state.size() == 0){ // There is no pattern available for a cell
            return FAILED;
        }

        auto it_pattern = state.begin();
        std::advance(it_pattern, random.randomInt() % state.size());

        // collapse to one pattern
        state = {*it_pattern};

        if(print_step){
            std::cout << position.first << " " << position.second;
            std::cout << " collapse to " <<*state.begin() << "\n";
            print_grid(grid);
            std::cout << "\n";
        }

        return OK;
    };

    void propogate(set<Position>& unobserved, Position& position, bool print_process = false){
        // BFS
        std::queue<Position> q;
        std::set<Position> processed;
        q.push(position);

        int max_distance = 0;
        Position distance_record;
        
        while (!q.empty()) {
            Position curr = q.front();
            q.pop();
            int h = curr.first;
            int w = curr.second;
            auto& ss = grid[h][w];
            processed.insert(curr);
            
            auto distance = abs(h - position.first) + abs(w - position.second);
            if(distance > max_distance){
                max_distance = distance;
                distance_record = std::make_pair(h - position.first, w - position.second);
            }

            auto propogate = [&](Position dir, vector<set<int>>& rules){
                int neighbor_h = h + dir.first;
                int neighbor_w = w + dir.second;
                auto neighbor_pos = std::make_pair(neighbor_h, neighbor_w);

                auto unobserved_it = unobserved.find(neighbor_pos);
                if(neighbor_h < 0 || neighbor_h >= H || neighbor_w < 0 || neighbor_w >= W
                    || unobserved_it == unobserved.end() || processed.find(neighbor_pos) != processed.end()){
                    return;
                }
                auto& neighbor_ss = grid[neighbor_h][neighbor_w];

                Superposition vaild_state;
                for (int state: ss)
                {
                    auto& rule = rules[state];
                    vaild_state.insert(rule.begin(), rule.end());
                }

                // remove elemnet not in vaild_state
                Superposition result = set_intersection(neighbor_ss, vaild_state);
                assert(neighbor_ss.size() >= result.size());

                // remove at least one element, add to queue propogate later.
                if(result.size() < neighbor_ss.size()){
                    q.push(neighbor_pos);
                    neighbor_ss = result;
                }
                
                if(neighbor_ss.size() == 1){
                    unobserved.erase(unobserved_it);
                }
            };

            propogate(std::make_pair(1, 0), rules->top_bottom_rules); // to bottom
            propogate(std::make_pair(-1, 0), rules->bottom_top_rules); // to top
            propogate(std::make_pair(0, 1), rules->left_right_rules); // to right
            propogate(std::make_pair(0, -1), rules->right_left_rules); // to left

        }

        if(print_process){
            std::cout << "BF search: " << processed.size() << "\tDistance: " <<max_distance<< " (" << 
                distance_record.first << "," << distance_record.second<<")\n\n";
        }
    };

    
    Grid getGrid(){
        return grid;
    }
};


