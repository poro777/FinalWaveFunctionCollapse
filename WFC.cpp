#include "WFC.h"
#include "utils.h"
#include <omp.h>
#include <mutex>
#include <shared_mutex>

template Position naive_WFC::impl_selectOneCell<>(set<Position> &unobserved, RandomGen &random);
template Position naive_WFC::impl_selectOneCell<>(unordered_set<Position, pair_hash> &unobserved, RandomGen &random);

template <typename Set>
Position naive_WFC::impl_selectOneCell(Set &unobserved, RandomGen &random)
{
    if(selection <= 1){ // first element of order_set, unorderd_set
        auto position_it = unobserved.begin();
        return *position_it;
    }
    else if (selection == 2){ // full random
        auto position_it = unobserved.begin();
        std::advance(position_it, random.randomInt() % unobserved.size());
        return *position_it;
    }
    else if (selection == 3){
        double min_ = 1e+4;
        Position argmin;
        for (const Position& pos: unobserved)
        {
            double entropy = entropies[pos.first][pos.second];
            // if(unobserved.size() <= 9){
            //     std::cout<<entropy<<" "<<min_<<std::endl;
            // }
            if (entropy <= min_){
                double noise = 1e-6 * random.randomDouble();
                if (entropy + noise < min_){
                    min_ = entropy + noise;
                    argmin = pos;
                }
            }
        }
        return argmin;
    }
    else{
        // implement other methods e.g. min entropy selection
        throw std::logic_error("Method not yet implemented");
    }
    
}

RETURN_STATE naive_WFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    
    Superposition& state = grid[position.first][position.second];
    if(state.size() == 0){ // There is no pattern available for a cell
        return FAILED;
    }
    if(selection <= 2){
        auto it_pattern = state.begin();
        std::advance(it_pattern, random.randomInt() % state.size());
        // collapse to one pattern
        state = {*it_pattern};
    }
    else if(selection == 3){
        double sum = 0;
        for (auto iter = state.begin(); iter != state.end(); iter++){    
            sum += weights[(*iter)];   
        }
        double threshold = random.randomDouble() * sum;
        double partialSum = 0;
        
        for (auto iter = state.begin(); iter != state.end(); iter++){
            partialSum += weights[*(iter)];
            if(partialSum >= threshold){
                state = {*iter};
                break;
            }
        }
        entropies[position.first][position.second] = -1;
    }

    if(print_step){
        std::cout << position.first << " " << position.second;
        std::cout << " collapse to " <<*state.begin() << "\n";
        print_grid(grid);
        std::cout << "\n";
    }

    return OK;
    
}

template <typename Set>
void naive_WFC::impl_propogate(Set &unobserved, Position &position, bool print_process)
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
            if(result.size() < neighbor_sp.size() && selection <= 2){
                q.push(neighbor_pos);
                neighbor_sp = result;
            }
            else if(result.size() < neighbor_sp.size() && selection == 3){
                q.push(neighbor_pos);
                neighbor_sp = result;
                double sumOfweights = 0;
                double sumOfweightLogweights = 0;
                for (auto iter = result.begin(); iter != result.end(); iter++){
                        sumOfweights += weights[*(iter)];
                        sumOfweightLogweights += weightLogweights[*(iter)];
                }
                entropies[neighbor_h][neighbor_w] = log(sumOfweights) - sumOfweightLogweights / sumOfweights;
            }
            
            if(neighbor_sp.size() == 1){
                unobserved.erase(unobserved_it);
            }
            else if(neighbor_sp.size() == 0){
                stop = true;
            }
            if(neighbor_sp.size() == 0 && selection == 3){
                entropies[neighbor_h][neighbor_w] = -1;
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
    

    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
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
    SCOPE_PROFILING(timer, "SelectOneCell()")
    return component->selectOneCell(unobserved, random);
}

Position profiling_WFC::selectOneCell(unordered_set<Position, pair_hash> &unobserved, RandomGen &random)
{
    SCOPE_PROFILING(timer, "SelectOneCell()")
    return component->selectOneCell(unobserved, random);
}

RETURN_STATE profiling_WFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    SCOPE_PROFILING(timer, "Collapse()")
    return component->collapse(position, random, print_step);
}

void profiling_WFC::propogate(set<Position> &unobserved, Position &position, bool print_process)
{
    SCOPE_PROFILING(timer, "Propogate()")
    component->propogate(unobserved, position, print_process);
}

void profiling_WFC::propogate(unordered_set<Position, pair_hash> &unobserved, Position &position, bool print_process)
{
    SCOPE_PROFILING(timer, "Propogate()")
    component->propogate(unobserved, position, print_process);
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

template <typename Set>
Position bit_WFC::impl_selectOneCell(Set &unobserved, RandomGen &random)
{
    if(selection <= 1){  // first element of order_set, unorderd_set
        auto position_it = unobserved.begin();
        return *position_it;
    }
    else if (selection == 2){ // full random
        auto position_it = unobserved.begin();
        std::advance(position_it, random.randomInt() % unobserved.size());
        return *position_it;
    }
    else if( selection == 3){
        double min_ = 1e+4;
        Position argmin;
        for (const Position& pos: unobserved)
        {
            double entropy = entropies[pos.first][pos.second];
            // if(unobserved.size() <= 9){
            //     std::cout<<entropy<<" "<<min_<<std::endl;
            // }
            if (entropy <= min_){
                double noise = 1e-6 * random.randomDouble();
                if (entropy + noise < min_){
                    min_ = entropy + noise;
                    argmin = pos;
                }
            }
        }
        return argmin;
    }
    else{
        // implement other methods e.g. min entropy selection
        throw std::logic_error("Method not yet implemented");
    }
}



RETURN_STATE bit_WFC::collapse(Position &position, RandomGen &random, bool print_step)
{
    ull& state = grid[position.first][position.second];
    if(state == 0){ // There is no pattern available for a cell
        return FAILED;
    }

    int collapsed_state = -1;
    auto size = std::popcount(state); // count how many 1 in binary representation
    if(selection <= 2){
        // keep n-th 1, the other 1 to 0 
        auto n = findNthSetBit(state, 1 + (random.randomInt() % size));
        // collapse to one pattern
        state = 1ull << n;
        collapsed_state = n;
    }
    else if(selection == 3){
        double sum = 0;
        auto bwidth = std::bit_width(state);
        for (ull i = 0; i < bwidth; i++){
            if((state >> i) & 1ull){
                sum += weights[i];
            }
        }
        double threshold = random.randomDouble() * sum;
        double partialSum = 0;
        
        for (ull i = 0; i < bwidth; i++){
            if((state >> i) & 1ull){
                partialSum += weights[i];
                if(partialSum >= threshold){
                    collapsed_state = i;
                    state = 1ull << i;
                    break;
                }
            }
        }
        entropies[position.first][position.second] = -1;
    }

    if(print_step){
        std::cout << position.first << " " << position.second;
        std::cout << " collapse to " << collapsed_state << "\n";
        printGrid();
        std::cout << "\n";
    }

    return OK;
}

template <typename Set>
void bit_WFC::impl_propogate(Set &unobserved, Position &position, bool print_process)
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
            if(result < neighbor_sp && selection <= 2){
                q.push(neighbor_pos);
                neighbor_sp = result;
            }
            else if(result < neighbor_sp && selection == 3)
            {
                q.push(neighbor_pos);
                neighbor_sp = result;
                auto bwidth = std::bit_width(result);
                double sumOfweights = 0;
                double sumOfweightLogweights = 0;
                for (ull i = 0; i < bwidth; i++){
                    if((result >> i) & 1ull){
                        sumOfweights += weights[i];
                        sumOfweightLogweights += weightLogweights[i];
                    }
                }
                entropies[neighbor_h][neighbor_w] = log(sumOfweights) - sumOfweightLogweights / sumOfweights;
            }
            
            auto size = std::popcount(neighbor_sp);
            if(size == 1){  
                unobserved.erase(unobserved_it);
            }
            else if(size == 0){
                stop = true;
            }
            if(size == 0 && selection == 3){
                entropies[neighbor_h][neighbor_w] = -1;
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
    

    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
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

template void mp_WFC::impl_propogate<>(set<Position> &unobserved, Position &position, bool print_process);
template void mp_WFC::impl_propogate<>(unordered_set<Position, pair_hash> &unobserved, Position &position, bool print_process);


template <typename Set>
inline void mp_WFC::impl_propogate(Set &unobserved, Position &position, bool print_process)
{
    // BFS
    shared_ptr<Position[]> q(new Position[H * W]());
    int index_q = 0;

    shared_ptr<Position[]> next_q(new Position[H * W]());
    int index_next_q=0;

    shared_ptr<Position[]> re_q(new Position[H * W]());
    int index_re_q=0;

    std::shared_mutex rwMutex[H*W];

    set<Position> processed;
    int count = 0;
    {
        int h = position.first, w = position.second;
        q[index_q++] = std::make_pair( 1+h,w+  0);
        q[index_q++] = std::make_pair(-1+h,w+  0);
        q[index_q++] = std::make_pair( 0+h,w+  1);
        q[index_q++] = std::make_pair( 0+h,w+ -1);
    }
    int max_distance = 0;
    Position distance_record;

    bool stop = false;

    while ((index_q> 0 ) &&!stop) {

            #pragma omp parallel for num_threads(4) schedule(auto)
            for (int i = 0; i < index_q; i++)
            {
                bool remove = false;
                bool prop = false;

                Position curr = q[i];
                int h = curr.first;
                int w = curr.second;
                if(h < 0 || h >= H || w < 0 || w >= W){
                        continue;
                    }
                
                
                Superposition sp;
                { std::shared_lock<std::shared_mutex> readLock(rwMutex[h * W + w]);
                sp =grid[h][w];
                }
                auto propogate_dir = [&](Position dir, vector<set<int>>& rules, int id){
                    int neighbor_h = h + dir.first;
                    int neighbor_w = w + dir.second;

                    if(neighbor_h < 0 || neighbor_h >= H || neighbor_w < 0 || neighbor_w >= W){
                        return;
                    }

                    Superposition neighbor_sp ;
                    {std::shared_lock<std::shared_mutex> readLock(rwMutex[neighbor_h * W + neighbor_w]);
                    neighbor_sp = grid[neighbor_h][neighbor_w];
                    }
                    if(sp.size() == 0 || sp.size() == 1) return;
                    

                    Superposition vaild_state;
                    for (int state: neighbor_sp)
                    {
                        auto& rule = rules[state];
                        vaild_state.insert(rule.begin(), rule.end());
                    }                    

                    // remove elemnet not in vaild_state
                    Superposition result = set_intersection(sp, vaild_state);
                    assert(sp.size() >= result.size());
                                    
                    // remove at least one element, add to queue propogate later.
                    if(result.size() < sp.size()){
                        prop = true;
                        sp = result;
                    }

                    if(result.size() == 1){
                        remove = true;
                    }
                    else if(result.size() == 0){
                        stop = true;
                    }
                };


                propogate_dir(std::make_pair(1, 0), rules->bottom_top_rules,0); // to bottom
                propogate_dir(std::make_pair(-1, 0), rules->top_bottom_rules,1); // to top
                propogate_dir(std::make_pair(0, 1), rules->right_left_rules,2); // to right
                propogate_dir(std::make_pair(0, -1), rules->left_right_rules,3); // to left

                {std::unique_lock<std::shared_mutex> readLock(rwMutex[h*W+w]);
                grid[h][w] = sp;
                }
                if(prop){
                    int index = __sync_fetch_and_add(&index_next_q, 4);
                    next_q[index + 0] = std::make_pair( 1+h,w+  0);
                    next_q[index + 1] = std::make_pair(-1+h,w+  0);
                    next_q[index + 2] = std::make_pair( 0+h,w+  1);
                    next_q[index + 3] = std::make_pair( 0+h,w+ -1);
                }
                if(remove){
                    int index = __sync_fetch_and_add(&index_re_q, 1);
                    re_q[index] = curr;
                }
            } 
        

        next_q.swap(q);

        index_q = index_next_q;
        index_next_q = 0;
    }


    for (int i = 0; i < index_re_q; i++)
    {
        unobserved.erase(re_q[i]);
    }
    

    if(print_process){
        std::cout << "BF search: " << count << "\tCells: " << processed.size() << "\tDistance: " <<max_distance<< " (" << 
            distance_record.first << "," << distance_record.second<<")\n\n";
    }
}
