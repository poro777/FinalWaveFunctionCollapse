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

class RPGMap: public Rule
{
/* TODO : Example for Rule, there are 7 TODO in this Class */
public:
    RPGMap(){
        // TODO 1 : assign image root directory
        patternImagesRoot = "./data/rpgmap/";

        // TODO 2 : assign total number of pattern
        M = 34;

        // TODO 3 : init all rules to M
        top_bottom_rules = vector<Superposition>(M);
        bottom_top_rules = vector<Superposition>(M);
        left_right_rules = vector<Superposition>(M);
        right_left_rules = vector<Superposition>(M);
        /* TODO 4 : 
        design 'top_bottom_rules' rules: 
            If the current pattern is at the top,
            What patterns can be connected to the bottom?
        example:
            top_bottom_rules[0] = {4, 8};
            there are two rules
            | 0 |  and  | 0 |
            | 4 |       | 8 |
        the index is same as in `patternImagesRoot`

        */
        top_bottom_rules[0] = {1, 2, 16};
        top_bottom_rules[1] = {1, 2, 16};
        top_bottom_rules[2] = {33};
        top_bottom_rules[3] = {4, 5, 9, 12};
        top_bottom_rules[4] = {4, 5, 9, 12, 15};
        top_bottom_rules[5] = {33};
        top_bottom_rules[6] = {7, 8, 10};
        top_bottom_rules[7] = {7, 8, 10};
        top_bottom_rules[8] = {33};
        top_bottom_rules[9] = {7, 8, 10};
        top_bottom_rules[10] = {7, 8, 10};
        top_bottom_rules[11] = {22, 23};
        top_bottom_rules[12] = {33};
        top_bottom_rules[13] = {33};
        top_bottom_rules[14] = {22, 23};
        top_bottom_rules[15] = {1, 2, 16};
        top_bottom_rules[16] = {1, 2, 16};
        top_bottom_rules[17] = {22, 23};
        top_bottom_rules[18] = {20};
        top_bottom_rules[19] = {17, 19, 20};
        top_bottom_rules[20] = {33};
        top_bottom_rules[21] = {22, 23};
        top_bottom_rules[22] = {22, 23};
        top_bottom_rules[23] = {33};
        top_bottom_rules[24] = {26, 27};
        top_bottom_rules[25] = {25, 26};
        top_bottom_rules[26] = {33};
        top_bottom_rules[27] = {11, 14, 17, 18, 21, 24, 28, 29, 32};
        top_bottom_rules[28] = {14, 18, 21, 24, 28, 29, 30, 32};
        top_bottom_rules[29] = {33};
        top_bottom_rules[30] = {14, 21, 31};
        top_bottom_rules[31] = {14, 21, 31};
        top_bottom_rules[32] = {33};
        top_bottom_rules[33] = {0, 3, 6, 13, 14, 18, 21, 24, 27, 30, 33};



        
        /* TODO 5 : 
        design 'left_right_rules' rules: 
            If the current pattern is at the left,
            What patterns can be connected to the right?
        */
        left_right_rules[0] = {3, 6};
        left_right_rules[1] = {4, 7, 9, 10};
        left_right_rules[2] = {5, 8, 12, 15};
        left_right_rules[3] = {3, 6};
        left_right_rules[4] = {4, 7, 9, 10};
        left_right_rules[5] = {5, 8, 12, 15};
        left_right_rules[6] = {33};
        left_right_rules[7] = {33};
        left_right_rules[8] = {33};
        left_right_rules[9] = {5, 8, 12, 15};
        left_right_rules[10] = {33};
        left_right_rules[11] = {14, 17, 21, 24};
        left_right_rules[12] = {5, 8, 12, 15};
        left_right_rules[13] = {33};
        left_right_rules[14] = {14, 17, 21, 24};
        left_right_rules[15] = {4, 7, 9, 10};
        left_right_rules[16] = {4, 7, 9, 10};
        left_right_rules[17] = {33};
        left_right_rules[18] = {14, 17, 21, 24};
        left_right_rules[19] = {11, 22, 25};
        left_right_rules[20] = {11, 23, 26};
        left_right_rules[21] = {14,17,21,24};
        left_right_rules[22] = {11,22,25};
        left_right_rules[23] = {23,26};
        left_right_rules[24] = {33};
        left_right_rules[25] = {33};
        left_right_rules[26] = {33};
        left_right_rules[27] = {33};
        left_right_rules[28] = {33};
        left_right_rules[29] = {2, 5, 12, 15, 16, 29};
        left_right_rules[30] = {33};
        left_right_rules[31] = {33};
        left_right_rules[32] = {32, 33};
        left_right_rules[33] = {0, 1, 2, 13, 16, 18, 19, 20, 27, 28, 30, 31, 32, 33};
        mirror();
    }
            /* name for this class */
    std::string name(){/* TODO 7 */ return "RPGMap";}
};

class Road: public Rule
    {
    private:
    public:
        Road(/* args */){
            patternImagesRoot = "./data/road/";
            M = 16;
            top_bottom_rules = vector<Superposition>(M);
            bottom_top_rules = vector<Superposition>(M);
            left_right_rules = vector<Superposition>(M);
            right_left_rules = vector<Superposition>(M);

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
/* TODO : Example for Rule, there are 7 TODO in this Class */
public:
    Example(){
        // TODO 1 : assign image root directory
        patternImagesRoot = "./data/example/";

        // TODO 2 : assign total number of pattern
        M = 16;

        // TODO 3 : init all rules to M
        top_bottom_rules = vector<Superposition>(M);
        bottom_top_rules = vector<Superposition>(M);
        left_right_rules = vector<Superposition>(M);
        right_left_rules = vector<Superposition>(M);

        /* TODO 4 : 
        design 'top_bottom_rules' rules: 
            If the current pattern is at the top,
            What patterns can be connected to the bottom?
        example:
            top_bottom_rules[0] = {4, 8};
            there are two rules
            | 0 |  and  | 0 |
            | 4 |       | 8 |
        the index is same as in `patternImagesRoot`

        */
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
        
        /* TODO 5 : 
        design 'left_right_rules' rules: 
            If the current pattern is at the left,
            What patterns can be connected to the right?
        */
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

        /* TODO 6 : call mirror()
        since top_bottom_rules / bottom_top_rules , left_right_rules / left_right_rules
        has corresponding relationship, mirror() function will handle it.
        */
        mirror();
    }

    
    /* name for this class */
    std::string name(){/* TODO 7 */ return "Example";}
};


class Summer: public Rule
{
public:
    Summer(){
        patternImagesRoot = "./data/Summer/";

        M = 40;

        top_bottom_rules = vector<Superposition>(M);
        bottom_top_rules = vector<Superposition>(M);
        left_right_rules = vector<Superposition>(M);
        right_left_rules = vector<Superposition>(M);

        Superposition sea = {13,14,15};
        Superposition grass = {12};

        Superposition grass_at_top =   {0,2,6,7,10,11,12, /* part 1*/ 
                                         18, 19, 22, /*part 2*/
                                        34, 38,39/*part 3*/ };      
        Superposition grass_at_bottom = {0,2,4,5,8,9,12,
                                        16,17,20,
                                        32,36, 37};
        Superposition grass_at_left =   {1,3,4,7,8,11,12,
                                        16,19, 23,
                                        35,36,39};
        Superposition grass_at_right =  {1,3,5,6,9,10,12,
                                        17,18, 21,
                                        33,37,38};


        Superposition water_at_top =    {13,14,15,20,24,25};
        Superposition water_at_bottom = {13,14,15,22,26,27};
        Superposition water_at_left =   {13,14,15,21,25,26};
        Superposition water_at_right =  {13,14,15,23,24,27};

        Superposition road_at_top =    {30,31,32};
        Superposition road_at_bottom = {28,29,34};
        Superposition road_at_left =   {28,31,33};
        Superposition road_at_right =  {29,30,35};

        top_bottom_rules[0]  = grass;
        top_bottom_rules[1]  = {1, 4, 9};
        top_bottom_rules[2]  = grass;
        top_bottom_rules[3]  = {3, 5, 8};
        top_bottom_rules[4]  = grass;
        top_bottom_rules[5]  = grass;
        top_bottom_rules[6]  = {3};
        top_bottom_rules[7]  = {1};
        top_bottom_rules[8]  = grass;
        top_bottom_rules[9]  = grass;
        top_bottom_rules[10] = {1};
        top_bottom_rules[11] = {3};
        top_bottom_rules[12] = grass_at_top;
        top_bottom_rules[13] = water_at_top;
        top_bottom_rules[14] = water_at_top;
        top_bottom_rules[15] = water_at_top;
        top_bottom_rules[16] = grass_at_top;
        top_bottom_rules[17] = grass_at_top;
        top_bottom_rules[18] = {21};
        top_bottom_rules[19] = {23};
        top_bottom_rules[20] = grass_at_top;
        top_bottom_rules[21] = {17,21,26};
        top_bottom_rules[22] = sea;
        top_bottom_rules[23] = {16,23,27};
        top_bottom_rules[24] = {23};
        top_bottom_rules[25] = {21};
        top_bottom_rules[26] = sea;
        top_bottom_rules[27] = sea;
        top_bottom_rules[28] = road_at_top;
        top_bottom_rules[29] = road_at_top;
        top_bottom_rules[30] = {35, 36};
        top_bottom_rules[31] = {33, 37};
        top_bottom_rules[32] = grass_at_top;
        top_bottom_rules[33] = {28,33,37};
        top_bottom_rules[34] = road_at_top;
        top_bottom_rules[35] = {29,35,36};
        top_bottom_rules[36] = grass_at_top;
        top_bottom_rules[37] = grass_at_top;
        top_bottom_rules[38] = {28,33};
        top_bottom_rules[39] = {29,35};

        left_right_rules[0]  = {0, 6, 9};
        left_right_rules[1]  = grass;
        left_right_rules[2]  = {2, 5, 10};
        left_right_rules[3]  = grass;
        left_right_rules[4]  = {2};
        left_right_rules[5]  = grass;
        left_right_rules[6]  = grass;
        left_right_rules[7]  = {0};
        left_right_rules[8]  = {0};
        left_right_rules[9]  = grass;
        left_right_rules[10] = grass;
        left_right_rules[11] = {2};
        left_right_rules[12] = grass_at_left;
        left_right_rules[13] = water_at_left;
        left_right_rules[14] = water_at_left;
        left_right_rules[15] = water_at_left;
        left_right_rules[16] = {20};
        left_right_rules[17] = grass_at_left;
        left_right_rules[18] = grass_at_left;
        left_right_rules[19] = {22};
        left_right_rules[20] = {17, 20,24};
        left_right_rules[21] = grass_at_left;
        left_right_rules[22] = {18, 22, 27};
        left_right_rules[23] = sea;
        left_right_rules[24] = sea;
        left_right_rules[25] = {20};
        left_right_rules[26] = {22};
        left_right_rules[27] = sea;
        left_right_rules[28] = {34,38};
        left_right_rules[29] = road_at_left;
        left_right_rules[30] = road_at_left;
        left_right_rules[31] = {32,37};
        left_right_rules[32] = {30,32,37};
        left_right_rules[33] = grass_at_left;
        left_right_rules[34] = {29,34,38};
        left_right_rules[35] = road_at_left;
        left_right_rules[36] = {30,32};
        left_right_rules[37] = grass_at_left;
        left_right_rules[38] = grass_at_left;
        left_right_rules[39] = {30,34};
        mirror();
    }

    
    /* name for this class */
    std::string name(){return "Summer";}
};



}
