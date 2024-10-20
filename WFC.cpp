#include "WFC.h"
#include "utils.h"

Position naive_WFC::selectOneCell(set<Position> &unobserved, RandomGen &random)
{
    
    auto position_it = unobserved.begin();
    std::advance(position_it, random.randomInt() % unobserved.size());

    return *position_it;
    
}

RETURN_STATE naive_WFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    
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
    
}

void naive_WFC::propogate(set<Position> &unobserved, Position &position, bool print_process)
{
    // BFS
    std::queue<Position> q;
    set<Position> processed;
    int count = 0;
    q.push(position);

    int max_distance = 0;
    Position distance_record;
    
    bool stop = false;
    while (!q.empty() && !stop) {
        Position curr = q.front();
        q.pop();
        int h = curr.first;
        int w = curr.second;
        auto& sp = grid[h][w];
        processed.insert(curr);
        count++;
        
        auto distance = abs(h - position.first) + abs(w - position.second);
        if(distance > max_distance){
            max_distance = distance;
            distance_record = std::make_pair(h - position.first, w - position.second);
        }

        auto propogate_dir = [&](Position dir, vector<set<int>>& rules){
            int neighbor_h = h + dir.first;
            int neighbor_w = w + dir.second;
            auto neighbor_pos = std::make_pair(neighbor_h, neighbor_w);

            auto unobserved_it = unobserved.find(neighbor_pos);
            if(neighbor_h < 0 || neighbor_h >= H || neighbor_w < 0 || neighbor_w >= W
                || unobserved_it == unobserved.end()){
                return;
            }
            auto& neighbor_sp = grid[neighbor_h][neighbor_w];

            Superposition vaild_state;
            for (int state: sp)
            {
                auto& rule = rules[state];
                vaild_state.insert(rule.begin(), rule.end());
            }

            // remove elemnet not in vaild_state
            Superposition result = set_intersection(neighbor_sp, vaild_state);
            assert(neighbor_sp.size() >= result.size());

            // remove at least one element, add to queue propogate later.
            if(result.size() < neighbor_sp.size()){
                q.push(neighbor_pos);
                neighbor_sp = result;
            }
            
            if(neighbor_sp.size() == 1){
                unobserved.erase(unobserved_it);
            }
            else if(neighbor_sp.size() == 0){
                stop = true;
            }
        };

        propogate_dir(std::make_pair(1, 0), rules->top_bottom_rules); // to bottom
        propogate_dir(std::make_pair(-1, 0), rules->bottom_top_rules); // to top
        propogate_dir(std::make_pair(0, 1), rules->left_right_rules); // to right
        propogate_dir(std::make_pair(0, -1), rules->right_left_rules); // to left

    }

    if(print_process){
        std::cout << "BF search: " << count << "\tCells: " << processed.size() << "\tDistance: " <<max_distance<< " (" << 
            distance_record.first << "," << distance_record.second<<")\n\n";
    }
}

void naive_WFC::validateNeighbor()
{
    auto validate = [&](Position pos, Position dir, vector<set<int>>& rules){
        int h = pos.first;
        int w = pos.second;
        auto& sp = grid[h][w];
        
        int neighbor_h = h + dir.first;
        int neighbor_w = w + dir.second;
        auto neighbor_pos = std::make_pair(neighbor_h, neighbor_w);

        if(neighbor_h < 0 || neighbor_h >= H || neighbor_w < 0 || neighbor_w >= W){
            return true;
        }
        auto& neighbor_sp = grid[neighbor_h][neighbor_w];

        Superposition vaild_state;
        for (int state: sp)
        {
            auto& rule = rules[state];
            vaild_state.insert(rule.begin(), rule.end());
        }

        // all patterns in neighbor most in vaild state.
        // neighbor is subset of vaild state.
        // neighbor - vaild = 0
        Superposition result = set_difference(neighbor_sp, vaild_state);
        return result.size() == 0;
    };
    

    for (size_t h = 0; h < H; h++)
    {
        for (size_t w = 0; w < W; w++)
        {
            auto pos = std::make_pair(h, w);
            bool vaild = validate(pos, std::make_pair(1, 0), rules->top_bottom_rules) &&
                        validate(pos, std::make_pair(-1, 0), rules->bottom_top_rules) &&
                        validate(pos, std::make_pair(0, 1), rules->left_right_rules) && 
                        validate(pos, std::make_pair(0, -1), rules->right_left_rules);

            if (vaild == false){
                std::cout << "Invaild pattern at ("
                     << h << ", " << w  << ")\n";
            }
        }
        
    }    
}

Position profiling_WFC::selectOneCell(set<Position> &unobserved, RandomGen &random)
{
    const char* name = "SelectOneCell()";  

    timer.start(name);
    auto returnValue = component->selectOneCell(unobserved, random);
    timer.end(name);

    return returnValue;
}

RETURN_STATE profiling_WFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    const char* name = "Collapse()";

    timer.start(name);
    auto returnValue = component->collapse(position, random, print_step);
    timer.end(name);

    return returnValue;
}

void profiling_WFC::propogate(set<Position> &unobserved, Position &position, bool print_process)
{
    const char* name = "Propogate()";
    timer.start(name);
    component->propogate(unobserved, position, print_process);
    timer.end(name);
}

ull bit_WFC::sp_to_bits(const Superposition &sp)
{
    ull n = 0;
    for(ull pattern: sp){
        n += (1ull << pattern);
    }
    return n;
}

Superposition bit_WFC::bits_to_sp(ull state)
{
    Superposition sp = {};
    bits_to_sp(state, sp);
    return sp;
}

void bit_WFC::bits_to_sp(ull state, Superposition &out_sp)
{
    int bwidth = std::bit_width(state);
    for (int i = 0; i < bwidth; i++)
    {
        if(state & 1ull){
            out_sp.insert(i);
        }
        state >>= 1ull;
    }
}

Position bit_WFC::selectOneCell(set<Position> &unobserved, RandomGen &random)
{
    auto position_it = unobserved.begin();
    std::advance(position_it, random.randomInt() % unobserved.size());

    return *position_it;
}



RETURN_STATE bit_WFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    ull& state = grid[position.first][position.second];
    if(state == 0){ // There is no pattern available for a cell
        return FAILED;
    }


    auto size = std::popcount(state); // count how many 1 in binary representation
    // keep n-th 1, the other 1 to 0 
    auto n = findNthSetBit(state, 1 + (random.randomInt() % size));
    // collapse to one pattern
    state = 1ull << n;

    if(print_step){
        std::cout << position.first << " " << position.second;
        std::cout << " collapse to " << n << "\n";
        printGrid();
        std::cout << "\n";
    }

    return OK;
}

void bit_WFC::propogate(set<Position> &unobserved, Position &position, bool print_process)
{
    // BFS
    std::queue<Position> q;
    set<Position> processed;
    int count = 0;
    q.push(position);

    int max_distance = 0;
    Position distance_record;
    bool stop = false;
    while (!q.empty() && !stop) {
        Position curr = q.front();
        q.pop();
        int h = curr.first;
        int w = curr.second;
        auto sp = grid[h][w];
        processed.insert(curr);
        count++;

        auto distance = abs(h - position.first) + abs(w - position.second);
        if(distance > max_distance){
            max_distance = distance;
            distance_record = std::make_pair(h - position.first, w - position.second);
        }

        auto propogate_dir = [&](Position dir, vector<ull>& rules){
            int neighbor_h = h + dir.first;
            int neighbor_w = w + dir.second;
            auto neighbor_pos = std::make_pair(neighbor_h, neighbor_w);

            auto unobserved_it = unobserved.find(neighbor_pos);
            if(neighbor_h < 0 || neighbor_h >= H || neighbor_w < 0 || neighbor_w >= W
                || unobserved_it == unobserved.end()){
                return;
            }
            auto& neighbor_sp = grid[neighbor_h][neighbor_w];

            ull vaild_state = 0;
            auto bwidth = std::bit_width(sp);
            for (ull i = 0; i < bwidth; i++)
            {
                if((sp >> i) & 1ull){
                    auto rule = rules[i];
                    vaild_state |= rule;
                }
            }

            // remove elemnet not in vaild_state
            ull result = neighbor_sp & vaild_state;
            assert(neighbor_sp >= result);

            // remove at least one element, add to queue propogate later.
            if(result < neighbor_sp){
                q.push(neighbor_pos);
                neighbor_sp = result;
            }
            
            auto size = std::popcount(neighbor_sp);
            if(size == 1){  
                unobserved.erase(unobserved_it);
            }
            else if(size == 0){
                stop = true;
            }
        };

        propogate_dir(std::make_pair(1, 0),  top_bottom_rules); // to bottom
        propogate_dir(std::make_pair(-1, 0), bottom_top_rules); // to top
        propogate_dir(std::make_pair(0, 1),  left_right_rules); // to right
        propogate_dir(std::make_pair(0, -1), right_left_rules); // to left

    }

    if(print_process){
        std::cout << "BF search: " << count << "\tCells: " << processed.size() << "\tDistance: " <<max_distance<< " (" << 
            distance_record.first << "," << distance_record.second<<")\n\n";
    }
}

void bit_WFC::validateNeighbor()
{
    auto validate = [&](Position pos, Position dir, vector<ull>& rules){
        int h = pos.first;
        int w = pos.second;
        auto sp = grid[h][w];
        
        int neighbor_h = h + dir.first;
        int neighbor_w = w + dir.second;
        auto neighbor_pos = std::make_pair(neighbor_h, neighbor_w);

        if(neighbor_h < 0 || neighbor_h >= H || neighbor_w < 0 || neighbor_w >= W){
            return true;
        }
        auto neighbor_sp = grid[neighbor_h][neighbor_w];

        ull vaild_state = 0;
        auto bwidth = std::bit_width(sp);
        for (ull i = 0; i < bwidth; i++)
        {
            if((sp >> i) & 1ull){
                auto rule = rules[i];
                vaild_state |= rule;
            }
        }

        // all patterns in neighbor most in vaild state.
        // neighbor is subset of vaild state.
        // neighbor - vaild = 0
        ull result = (neighbor_sp ^ vaild_state) & neighbor_sp;
        return result == 0;
    };
    

    for (size_t h = 0; h < H; h++)
    {
        for (size_t w = 0; w < W; w++)
        {
            auto pos = std::make_pair(h, w);
            bool vaild = validate(pos, std::make_pair(1, 0), top_bottom_rules) &&
                        validate(pos, std::make_pair(-1, 0), bottom_top_rules) &&
                        validate(pos, std::make_pair(0, 1),  left_right_rules) && 
                        validate(pos, std::make_pair(0, -1), right_left_rules);

            if (vaild == false){
                std::cout << "Invaild pattern at ("
                     << h << ", " << w  << ")\n";
            }
        }
        
    }        
}
