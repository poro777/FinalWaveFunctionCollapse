#pragma once

#include <set>
#include <algorithm>

template<typename T, typename P>
inline std::set<int> set_intersection(T&& A, P&& B){
    std::set<int> result;
    std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                        std::inserter(result, result.begin()));
    return result;
}

template<typename T, typename P>
inline std::set<int> set_union(T&& A, P&& B){
    std::set<int> result;
    std::set_union(A.begin(), A.end(), B.begin(), B.end(),
                        std::inserter(result, result.begin()));
    return result;
}

template<typename T, typename P>
inline std::set<int> set_difference(T&& A, P&& B){
    std::set<int> result;
    std::set_difference(A.begin(), A.end(), B.begin(), B.end(),
                        std::inserter(result, result.begin()));
    return result;
}