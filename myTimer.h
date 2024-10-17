#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <limits>

struct Time
{
    std::chrono::_V2::system_clock::time_point start;
    bool during;
    double total = 0;
    double average = 0;
    unsigned long long count = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
};

class myTimer
{
private:
    std::unordered_map<std::string, Time> data;
public:
    myTimer(/* args */){

    };

    ~myTimer(){};
    template <typename T>
    void start(const T && name){
        auto start = std::chrono::high_resolution_clock::now();
        auto it = data.find(name);
        if(it == data.end()){
            // create new one
            auto time = Time();
            time.start = start;
            time.during = true;
            data[name] = time;
        }
        else{
            it->second.start = start;
            it->second.during = true;
        }
    }

    template <typename T>
    void end(const T&& name){
        auto end = std::chrono::high_resolution_clock::now();
        auto it = data.find(name);
        if(it != data.end() && it->second.during == true){
            auto& time = it->second;
            time.during = false;
            std::chrono::duration<double, std::milli> duration = (end - time.start);
            double t = duration.count();
            time.total += t;
            time.count += 1;
            time.average = time.total / (double) time.count;
            time.max = time.max > t? time.max: t;
            time.min = time.min < t? time.min: t;
        }
        else{
            static bool print = false;
            if(!print){
                std::cerr << "Incorrect measure " << name << "\n";
                print = true;
            }
        }
    }
    
    void start(const char* name){
        start(std::string(name));
    }
    void end(const char* name){
        end(std::string(name));
    }  
    void print(){
        std::cout << "\nProfiling:\n";
        for (auto& p: data)
        {
            auto& time = p.second;
            std::cout << p.first << ":\n"
                << "\tTotal: " << time.total
                << "\tAverage: " << time.average << " (ms)\n"
                << "\tCount: " << time.count
                << "\tMin: " << time.min << "\tMax: " << time.max << "\n";
        }
    }
};
