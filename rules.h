#pragma once
#include <vector>
#include <set>
#include <iostream>
#include <filesystem>

#include "utils.h"
#include "setOperator.h"

using std::set;
using std::vector;
namespace fs = std::filesystem;

namespace Rules{
struct ImageShape
{
    ImageShape(){};
    ImageShape(int width,
    int height,
    int channels):width(width), height(height), channels(channels){

    }
    int width = 0;
    int height = 0;
    int channels = 0;

    bool operator==(const ImageShape& that){
        return this->width == that.width && this->height == that.height && this->channels == that.channels;
    }

    bool operator!=(const ImageShape& that){
        return !(*this == that);
    }
};

class Rule
{

protected:
    fs::path patternImagesRoot = "";
    vector<unsigned char*> patterns;
    ImageShape patternShape;

    // mirror top_bottom_rules, left_right_rules to bottom_top_rules, right_left_rules
    void mirror(){
        for (int i = 0; i< M;i++)
        {
            auto& bottom_rules = top_bottom_rules[i];
            for(auto bottom: bottom_rules){
                bottom_top_rules[bottom].insert(i);
            }

            auto& right_rules = left_right_rules[i];
            for(auto right: right_rules){
                right_left_rules[right].insert(i);
            }
        }
    }
public:
    vector<set<int>> top_bottom_rules;
    vector<set<int>> bottom_top_rules;
    vector<set<int>> left_right_rules;
    vector<set<int>> right_left_rules;

    int M = 0;

    virtual std::string name() = 0;
    void writeImage(Grid& grid);

    Rule(/* args */){};
    ~Rule();

    
    Superposition initValue(){
        Superposition initial_state;
        for (size_t i = 0; i < M; i++)
        {
            initial_state.insert(i);
        }
        return initial_state;
    }

    void print(){
        for (size_t i = 0; i < M; i++)
        {
            std::cout << "\n->" << i << "\n";
            std::cout << "top:\t";
            print_set(bottom_top_rules[i]);

            std::cout << "bottom:\t" ;
            print_set(top_bottom_rules[i]);

            std::cout << "right:\t";
            print_set(left_right_rules[i]);

            std::cout << "left:\t" ;
            print_set(right_left_rules[i]);
        }
    }
};

class Road: public Rule
    {
    private:
    public:
        Road(/* args */){
            patternImagesRoot = "./data/road/";
            M = 16;
            top_bottom_rules = vector<set<int>>(M);
            bottom_top_rules = vector<set<int>>(M);
            left_right_rules = vector<set<int>>(M);
            right_left_rules = vector<set<int>>(M);

            Superposition all = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

            Superposition edge_at_bottom = {2, 6, 7, 8, 9, 11, 12, 14};
            Superposition edge_at_top = {0, 4, 5, 9, 10, 11, 12, 14};
            Superposition edge_at_right = {3, 4, 7, 8, 9, 10, 13, 14};
            Superposition edge_at_left = {1, 5, 6, 8, 10, 11, 13, 14};

            Superposition no_edge_at_bottom = set_difference(all, edge_at_bottom);
            Superposition no_edge_at_top  = set_difference(all, edge_at_top);
            Superposition no_edge_at_right  = set_difference(all, edge_at_right);
            Superposition no_edge_at_left  = set_difference(all, edge_at_left);

            for(auto pattern: no_edge_at_bottom){
                top_bottom_rules[pattern] = no_edge_at_top;
            }
            for (auto pattern: edge_at_bottom)
            {
                top_bottom_rules[pattern] = edge_at_top;
            }
            
            for(auto pattern: no_edge_at_right){
                left_right_rules[pattern] = no_edge_at_left;
            }
            for (auto pattern: edge_at_right)
            {
                left_right_rules[pattern] = edge_at_left;
            }

            mirror();
        };

    std::string name(){return "Road";}
};

class Example: public Rule
{

public:
    Example(){
        patternImagesRoot = "./data/example/";
        M = 16;
        top_bottom_rules = vector<set<int>>(M);
        bottom_top_rules = vector<set<int>>(M);
        left_right_rules = vector<set<int>>(M);
        right_left_rules = vector<set<int>>(M);

        top_bottom_rules[0] = {4, 8};
        top_bottom_rules[1] = {5};
        top_bottom_rules[2] = {6, 10};
        top_bottom_rules[3] = {0, 1, 2, 3, 7, 9};
        top_bottom_rules[4] = {4, 8};
        top_bottom_rules[5] = {5};
        top_bottom_rules[6] = {6, 10};
        top_bottom_rules[7] = {11};
        top_bottom_rules[8] = {5};
        top_bottom_rules[9] = {12,13,14,15};
        top_bottom_rules[10] = {5};
        top_bottom_rules[11] = {12,13,14,15};
        top_bottom_rules[12] = {5};
        top_bottom_rules[13] = {5};
        top_bottom_rules[14] = {5};
        top_bottom_rules[15] = {5};
        
        left_right_rules[0] = {1,2,8,12,15};
        left_right_rules[1] = {1,2,8,12,15};
        left_right_rules[2] = {0,3,4,7,9,11};
        left_right_rules[3] = {0,3,4,7,9,11};
        left_right_rules[4] = {5,6,10};
        left_right_rules[5] = {5,6,10};
        left_right_rules[6] = {0,3,4,7,9,11};
        left_right_rules[7] = {0,3,4,7,9,11};
        left_right_rules[8] = {5,6,10};
        left_right_rules[9] = {0,3,4,7,9,11};
        left_right_rules[10] = {1,2,8,12,15};
        left_right_rules[11] = {0,3,4,7,9,11};
        left_right_rules[12] = {13,14};
        left_right_rules[13] = {13,14};
        left_right_rules[14] = {1,2,8,12,15};
        left_right_rules[15] = {1,2,8,12,15};

        mirror();
    }
    std::string name(){return "Example";}
};




}
