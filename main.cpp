#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <utility> // For std::pair
#include <queue>
#include <algorithm>
#include <cassert>
#include <string>
#include <getopt.h>

#include "utils.h"
#include "WFC.h"
#include "setOperator.h"
#include "rules.h"

template <typename Set>
bool run(shared_ptr<WFC> wfc_solver, long long seed, shared_ptr<myTimer> timer, bool print_step, bool print_process){
    SCOPE_PROFILING(timer, "Run()")

    RandomGen random(seed);

    int H = wfc_solver->getH();
    int W = wfc_solver->getW();

    Set unobserved;
    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            unobserved.insert(std::make_pair(h, w));
        }
    }

    int i = -1;
    while (unobserved.size() > 0)
    {
        //wfc_solver->validateNeighbor();
        // SCOPE_PROFILING(timer, "Iterator")
        i++;
        if(print_process){
            std::cout << "Iter: " << i << "\n";
            std::cout <<"unobserved: " << unobserved.size() << "\n";
        }
        
        // collapse
        auto selected_position = wfc_solver->selectOneCell(unobserved, random);

        auto collapseState = wfc_solver->collapse(selected_position, random, print_step);

        if(collapseState == FAILED){
            std::cout << "-Failed- unobserved(rate):" << unobserved.size() << "("<<unobserved.size() / float(H*W) <<")"<< "\n";
            return false;
        }

        auto it_selected_position = unobserved.find(selected_position);
        assert(it_selected_position != unobserved.end());
        unobserved.erase(it_selected_position);

        // propogate
        wfc_solver->propogate(unobserved, selected_position, print_process);
    }

    return true;
}

int main(int argc, char *argv[]){
    // > make
    // > time ./a.out [long_options]
    // argument
    int W = 8, H = 8;
    int ruleType = 0;
    long long seed = -1;
    int bitOp = 1;
    int selection = 0;

    // flags
    int print_rules = false;
    int print_process = false;
    int print_step = false;
    int print_result = false;
    int save_result = false;
    int print_time = false;


    // Define long options
    static struct option long_options[] = {
        {"height", required_argument, 0, 'h'},
        {"width", required_argument, 0, 'w'},
        {"rule", required_argument, 0, 'r'},
        {"seed", required_argument, 0, 's'},
        {"bitOp", required_argument, 0, 'b'},
        {"selection", required_argument, 0, 'c'},

        {"print-time", no_argument, &print_time, 1},
        {"print-rules", no_argument, &print_rules, 1},
        {"print-process", no_argument, &print_process, 1},
        {"print-step", no_argument, &print_step, 1},
        {"print-result", no_argument, &print_result, 1},
        {"save-result", no_argument, &save_result, 1},

        {0, 0, 0, 0} // End of options
    };

    int option_index = 0;
    int opt;

    // Parse options
    while ((opt = getopt_long(argc, argv, "w:h:r:s:b:c:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'w':
                W = std::stoi(optarg); // Assign width
                break;
            case 'h':
                H = std::stoi(optarg); // Assign height
                break;
            case 'r':
                ruleType = std::stoi(optarg); // ruleType
                break;
            case 's':
                seed = std::stoll(optarg); // seed
                break;    
            case 'b':
                bitOp = std::stoi(optarg);
                break;
            case 'c':
                selection = std::stoi(optarg);
                break;                
            case 0: // long flag opt
                break;
            default:
                std::cerr << "Usage: see long_options\n" ;
                return 1;
        }
    }

    
    shared_ptr<Rules::Rule> rule;
    switch (ruleType)
    {
    case 0:
        rule = std::make_shared<Rules::Road>();
        break;
    case 1:
        rule = std::make_shared<Rules::Example>();
        break;
    case 2:
        rule = std::make_shared<Rules::Summer>();
        break;
    case 3:
        rule = std::make_shared<Rules::RPGMap>();
        break;
    default:
        rule = std::make_shared<Rules::Example>();
        break;
    } 
    
    std::cout << "Running\n" << "H="<<H << ", W="<<W  << ", Rule: " << rule->name() << "\n";

    shared_ptr<WFC> wfc_solver = nullptr;
    shared_ptr<myTimer> timer = std::make_shared<myTimer>();

    if(bitOp == 1){
        wfc_solver= std::make_shared<bit_WFC>(H, W, rule, selection);
    }
    else if(bitOp == 2){
        wfc_solver= std::make_shared<min_entropy_WFC>(H, W, rule, selection);
    }
    else{
        wfc_solver= std::make_shared<naive_WFC>(H, W, rule, selection);
    }

    if(print_time){
        wfc_solver = std::make_shared<profiling_WFC>(wfc_solver, timer);
    }

    if(print_rules){
        rule->print();
    }


    bool finish = false;
    
    if(selection == 1){
        // random selection by call set.begin()
        finish = run<unordered_set<Position, pair_hash>>(wfc_solver, seed, timer, print_step, print_process);
    }
    else{
        // order selection by call set.begin()
        // or use other implement
        finish = run<set<Position>>(wfc_solver, seed, timer, print_step, print_process);
    }


    if(finish){
        wfc_solver->validateNeighbor();
    }

    if(print_result){
        wfc_solver->printGrid();
    }
    
    if(save_result){
        Grid result = wfc_solver->getGrid();
        rule->writeImage(result);
    }

    if(print_time){
        timer->print();
    }
    return 0;
}