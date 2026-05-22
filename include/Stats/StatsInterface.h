#ifndef STATS_INTERFACE_H
#define STATS_INTERFACE_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class StatsInterface {
    public:
        virtual ~StatsInterface(){};
        virtual void saveStats(const std::string &file_path) = 0;
};

#endif 