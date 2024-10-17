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

int main(int argc, char *argv[]){
    // > make
    // > time ./a.out [long_options]
    // argument
    int W = 8, H = 8;
    int ruleType = 0;
    int bitOp = 1;
    bool print_rules = false;
    bool print_process = false;
    bool print_step = false;
    bool print_result = false;
    bool save_result = false;
    bool print_time = false;
    long long seed = -1;

    // Define long options
    static struct option long_options[] = {
        {"height", required_argument, 0, 'h'},
        {"width", required_argument, 0, 'w'},
        {"rule", required_argument, 0, 'r'},
        {"seed", required_argument, 0, 's'},
        {"bitOp", required_argument, 0, 'b'},

        {"print-time", no_argument, 0, 'i'},
        {"print-rules", no_argument, 0, 'u'},
        {"print-process", no_argument, 0, 'p'},
        {"print-step", no_argument, 0, 't'},
        {"print-result", no_argument, 0, 'o'},
        {"save-result", no_argument, 0, 'a'},

        {0, 0, 0, 0} // End of options
    };

    int option_index = 0;
    int opt;

    // Parse options
    while ((opt = getopt_long(argc, argv, "w:h:r:p:u:s:o:a:t:b", long_options, &option_index)) != -1) {
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
            case 'u':
                print_rules = true; // Enable print_rules
                break;
            case 'p':
                print_process = true; // Enable print_process
                break;
            case 't':
                print_step = true; // Enable print_step
                break;
            case 'i':
                print_time = true; // Enable timing
                break;
            case 'o':
                print_result = true; // Enable print_result
                break;
            case 'a':
                save_result = true;
                break;
            default:
                std::cerr << "Usage: see long_options\n";
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
    default:
        rule = std::make_shared<Rules::Example>();
        break;
    } 
    
    std::cout << "Running\n" << "H="<<H << ", W="<<W  << ", Rule: " << rule->name() << "\n";
    RandomGen random(seed);

    shared_ptr<WFC> wfc_solver = nullptr;
    if(bitOp){
        wfc_solver= std::make_shared<bit_WFC>(H, W, rule);
    } 
    else{
        wfc_solver= std::make_shared<naive_WFC>(H, W, rule);
    }

    if(print_time){
        wfc_solver = std::make_shared<profiling_WFC>(wfc_solver);
    }

    set<Position> unobserved;
    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            unobserved.insert(std::make_pair(h, w));
        }
    }

    if(print_rules){
        rule->print();
    }

    int i = -1;
    while (unobserved.size() > 0)
    {
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
            break;
        }

        auto it_selected_position = unobserved.find(selected_position);
        assert(it_selected_position != unobserved.end());
        unobserved.erase(it_selected_position);

        // propogate
        wfc_solver->propogate(unobserved, selected_position, print_process);
    }
    
    
    if(print_result){
        wfc_solver->printGrid();
    }
    
    if(save_result){
        Grid result = wfc_solver->getGrid();
        rule->writeImage(result);
    }

    if(print_time){
        profiling_WFC* profilier = dynamic_cast<profiling_WFC*>(wfc_solver.get());
        profilier->timer.print();
    }
    return 0;
}