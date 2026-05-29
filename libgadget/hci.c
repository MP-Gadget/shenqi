#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include "utils/endrun.h"
#include "hci.h"

static double
hci_now(HCIManager * manager)
{
    if(manager->OVERRIDE_NOW) {
        return manager->_now;
    }
    manager->_now = MPI_Wtime();
    /* must be consistent between all ranks. */
    MPI_Bcast(&manager->_now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return manager->_now;
}

void
hci_init(HCIManager * manager, char * prefix, const double WallClockTimeLimit, const double AutoCheckPointTime, const int FOFEnabled)
{
    manager->prefix = prefix;
    manager->timer_begin = hci_now(manager);
    manager->timer_query_begin = manager->timer_begin;

    manager->WallClockTimeLimit = WallClockTimeLimit;
    manager->AutoCheckPointTime = AutoCheckPointTime;
    manager->TimeLastCheckPoint = manager->timer_begin;
    manager->FOFEnabled = FOFEnabled;
    manager->LongestTimeBetweenQueries = 0;
}

void
hci_action_init(HCIAction * action)
{
    action->type = HCI_NO_ACTION;
    action->write_snapshot = 0;
    action->write_fof = 0;
    action->write_plane = 0;
}

/* override the result of hci_now; for unit testing -- we can't rely on MPI_Wtime there!
 * this function can be called before hci_init. */
void
hci_override_now(HCIManager * manager, double now)
{
    manager->_now = now;
    manager->OVERRIDE_NOW = 1;
}

static double
hci_get_elapsed_time(HCIManager * manager)
{
    return manager->timer_query_begin - manager->timer_begin;
}

static
void hci_update_query_timer(HCIManager * manager)
{
    double e = hci_now(manager);
    double g = e - manager->timer_query_begin;
    if(g > manager->LongestTimeBetweenQueries)
        manager->LongestTimeBetweenQueries = g;

    manager->timer_query_begin = e;
}

/*
 * query the filesystem for HCI commands;
 * returns 1 if the file is present; collectively
 * */
int
hci_query_filesystem(HCIManager * manager, std::string filename)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    int retval = 0;
    if(ThisTask == 0) {
        std::string fullname(std::string(manager->prefix) + "/" + filename);
        FILE * fp = fopen(fullname.c_str(), "r");
        if(fp) {
            fclose(fp);
            remove(fullname.c_str());
            retval = 1;
        }
    }
    MPI_Bcast(&retval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return retval;
}

static int
hci_query_timeout(HCIManager * manager)
{
    /* this function is collective because we take care to ensure manager is
     * collective */
    double now = hci_get_elapsed_time(manager);
    /*
     * factor 0.9 is a safety tolerance
     * for possible inconsistency between measured time and the true wallclock
     *
     * If there likely isn't time for a new query, then we shall timeout as well.
     * */

    if (now + manager->LongestTimeBetweenQueries < manager->WallClockTimeLimit * 0.95) {
        return 0;
    }

    /* any freeable string would work. */
    return 1;
}

static int
hci_query_auto_checkpoint(HCIManager * manager)
{
    /* this function is collective because we take care to ensure manager is
     * collective */
    if(manager->AutoCheckPointTime <= 0) return 0;

    /* How long since the last checkpoint? */
    double now = hci_get_elapsed_time(manager);
    if(now - manager->TimeLastCheckPoint >= manager->AutoCheckPointTime) {
        return 1;
    }
    return 0;
}

/*
 * the return value is non-zero if the mainloop shall break.
 * */
int
hci_query(HCIManager * manager, HCIAction * action)
{
    hci_action_init(action);

    /* measure time since last query */
    hci_update_query_timer(manager);

    /* Check whether we need to interrupt the run */
    /* Will we run out of time by the query ? highest priority.
     */
    if(hci_query_timeout(manager)) {
        message(0, "HCI: Stopping due to TimeLimitCPU, dumping a CheckPoint.\n");
        action->type = HCI_TIMEOUT;
        action->write_snapshot = 1;
        if(manager->FOFEnabled)
            action->write_fof = 1;
        return 1;
    }

    if(hci_query_filesystem(manager, "checkpoint"))
    {
        message(0, "HCI: human controlled stop with checkpoint at next PM.\n");
        action->type = HCI_CHECKPOINT;
        /* will write checkpoint in this PM timestep */
        action->write_snapshot = 1;
        /* Write fof as well*/
        if(manager->FOFEnabled)
            action->write_fof = 1;
        manager->TimeLastCheckPoint = hci_get_elapsed_time(manager);
        return 0;
    }

    /* Is the plane-file present? If yes, ask to write a plane file. */
    if(hci_query_filesystem(manager, "plane"))
    {
        /* will write a lensing plane in this PM timestep, then continue.*/
        action->type = HCI_PLANE;
        action->write_plane = 1;
        return 1;
    }

    /* Is the stop-file present? If yes, interrupt the run with a snapshot. */
    if(hci_query_filesystem(manager, "stop"))
    {
        /* will write checkpoint in this PM timestep, then stop */
        action->type = HCI_STOP;
        action->write_snapshot = 1;
        return 1;
    }

    /* Is the terminate-file present? If yes, interrupt the run immediately. */
    if(hci_query_filesystem(manager, "terminate"))
    {
        message(0, "HCI: human triggered termination.\n");
        /* the caller shall take care of immediate termination.
         * This action is better than KILL as it avoids corrupt/incomplete snapshot files.*/
        action->type = HCI_TERMINATE;
        action->write_snapshot = 0;
        return 1;
    }

    /* lower priority */
    if(hci_query_auto_checkpoint(manager))
    {
        message(0, "HCI: Auto checkpoint due to AutoCheckPointTime.\n");
        action->type = HCI_AUTO_CHECKPOINT;
        /* Write when the PM timestep completes*/
        action->write_snapshot = 1;
        if(manager->FOFEnabled)
            action->write_fof = 1;
        manager->TimeLastCheckPoint = hci_get_elapsed_time(manager);
        return 0;
    }

    message(0, "HCI: Nothing happened. \n");
    return 0;
}
