#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include "walltime.h"

#include "utils/mymalloc.h"
#include "utils/openmpsort.h"

static struct ClockTable * CT = NULL;

static double WallTimeClock;
static double LastReportTime;

static struct Clock& walltime_clock(const std::string& name);
static void walltime_clock_insert(const std::string& name);
static double seconds(void);

void walltime_init(struct ClockTable * ct) {
    CT = ct;
    CT->ElapsedTime = 0;
    walltime_reset();
    walltime_clock_insert("/");
    LastReportTime = seconds();
}

static void walltime_summary_clocks(std::map<std::string, struct Clock>& C, int root, MPI_Comm comm) {
    double * t = ta_malloc("clocks", double, 4 * C.size());
    double * min = t + C.size();
    double * max = t + 2 * C.size();
    double * sum = t + 3 * C.size();
    int i = 0;
    for (const auto& [key, clock] : C) {
        t[i] = clock.time;
        i++;
    }
    MPI_Reduce(t, min, C.size(), MPI_DOUBLE, MPI_MIN, root, comm);
    MPI_Reduce(t, max, C.size(), MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(t, sum, C.size(), MPI_DOUBLE, MPI_SUM, root, comm);

    int NTask;
    MPI_Comm_size(comm, &NTask);
    /* min, max and mean are good only on process 0 */
    i = 0;
    for (const auto& [name, clock] : C) {
        C[name].min = min[i];
        C[name].max = max[i];
        C[name].mean = sum[i] / NTask;
        i++;
    }
    ta_free(t);
}

static void walltime_update_parents() {
    /* returns the sum of every clock with the same prefix */
    for (const auto& [prefix, clock] : CT->C) {
        CT->Nchildren[prefix] = 0;
        double t = 0;
        for (const auto& [name, clock2] : CT->C) {
            /* The same clock entry*/
            if(name == prefix)
                continue;
            if(prefix == name.substr(0, prefix.length())) {
                t += clock2.time;
                CT->Nchildren[prefix] ++;
            }
        }
        /* update only if there are children */
        if (t > 0) CT->C[prefix].time = t;
    }
}

/* put min max mean of MPI ranks to rank 0*/
/* AC will have the total timing, C will have the current step information */
void walltime_summary(int root, MPI_Comm comm) {
    walltime_update_parents();
    /* add to the cumulative time */
    for (const auto& [name, clock] : CT->C) {
        CT->AC[name].time += clock.time;
    }
    walltime_summary_clocks(CT->C, root, comm);
    walltime_summary_clocks(CT->AC, root, comm);

    /* clear .time for next step */
    for (const auto& [name, clock] : CT->C) {
        CT->C[name].time = 0;
    }
    MPI_Barrier(comm);
    /* wo do this here because all processes are sync after summary_clocks*/
    double step_all = seconds() - LastReportTime;
    LastReportTime = seconds();
    CT->ElapsedTime += step_all;
    CT->StepTime = step_all;
}

static void walltime_clock_insert(const std::string& name) {
    if(name.length() > 1) {
        size_t end = name.find("/", 1);
        walltime_clock("/");
        while (end != std::string::npos) {
            walltime_clock(name.substr(0, end));
            end = name.find("/", end+1);
        }
    }
    struct Clock tmp = {0};
    CT->C[name] = tmp;
    CT->AC[name] = tmp;
}

struct Clock& walltime_clock(const std::string& name) {
    if(!CT->C.contains(name))
        walltime_clock_insert(name);
    return CT->C[name];
};

char walltime_get_symbol(const std::string& name) {
    return walltime_clock(name).symbol;
}

double walltime_get(const std::string& name, enum clocktype type) {
    struct Clock cc = walltime_clock(name);
    struct Clock ac = walltime_clock(name);
    /* only make sense on root */
    switch(type) {
        case CLOCK_STEP_MEAN:
            return cc.mean;
        case CLOCK_STEP_MIN:
            return cc.min;
        case CLOCK_STEP_MAX:
            return cc.max;
        case CLOCK_ACCU_MEAN:
            return ac.mean;
        case CLOCK_ACCU_MIN:
            return ac.min;
        case CLOCK_ACCU_MAX:
            return ac.max;
    }
    return 0;
}

double walltime_get_time(const std::string& name) {
    return walltime_clock(name).time;
}

void walltime_reset() {
    WallTimeClock = seconds();
}

double walltime_measure_full(const std::string& name, const char * file, const int line) {
    char fullname[128] = {0};
    const char * basename = file + strlen(file);
    while(basename >= file && *basename != '/') basename --;
    basename ++;
    snprintf(fullname, 128, "%s@%s:%04d", name.c_str(), basename, line);

    double t = seconds();
    double dt = t - WallTimeClock;
    WallTimeClock = seconds();
    if(name[0] != '.') { //WALLTIME_IGNORE
        if(!CT->C.contains(fullname))
            walltime_clock_insert(fullname);
        CT->C[fullname].time += dt;
    }
    return dt;
}

double walltime_add_full(const std::string& name, const double dt, const char * file, const int line) {
    char fullname[128] = {0};
    const char * basename = file + strlen(file);
    while(basename >= file && *basename != '/') basename --;
    basename ++;
    snprintf(fullname, 128, "%s@%s:%04d", name.c_str(), basename, line);
    if(!CT->C.contains(fullname))
            walltime_clock_insert(fullname);
    CT->C[fullname].time += dt;
    return dt;
}

/* returns the number of cpu-ticks in seconds that
 * have elapsed. (or the wall-clock time)
 */
static double seconds(void)
{
  return MPI_Wtime();
}

void walltime_report(FILE * fp, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if(rank != root) return;

    for (const auto& [name, clock] : CT->C) {
        size_t end = name.find("/");
        size_t start = 0;
        int level = 0;
        while (end != std::string::npos) {
            level++;
            start = end + 1;
            end = name.find("/", start);
        }
        auto subname = name.substr(start, end);
        /* if there is just one child, don't print it*/
        if(CT->Nchildren[subname] == 1) continue;
        fprintf(fp, "%*s%-26s  %10.2f %4.1f%%  %10.2f %4.1f%%  %10.2f %10.2f\n",
                level, "",  /* indents */
                subname.c_str(),   /* just the last seg of name*/
                CT->AC[name].mean,
                CT->AC[name].mean / CT->ElapsedTime * 100.,
                clock.mean,
                clock.mean / CT->StepTime * 100.,
                clock.min,
                clock.max
                );
    }
}
