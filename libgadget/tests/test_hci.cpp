#define BOOST_TEST_MODULE hci

#include "booststub.h"
#include <string>
#include <fstream>
#include <iostream>

#include <libgadget/hci.h>

char prefix[1024] = ".";

static void
touch(char * prefix, std::string b)
{
    std::string fn = std::string(prefix)+ "/" +  b;
    std::ofstream tfile(fn, std::ofstream::out);
    tfile.close();
}

static int
exists(char * prefix, std::string b)
{
    std::string fn = std::string(prefix)+ "/" +  b;
    std::ifstream tfile(fn, std::ifstream::in);
    return tfile.good();
}

HCIManager manager[1] = {0};


BOOST_AUTO_TEST_CASE(test_hci_no_action)
{
    HCIAction action[1];

    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 0);

    hci_query(manager, action);
    BOOST_TEST(action->type == HCI_NO_ACTION);
    BOOST_TEST(action->write_snapshot == 0);

}

BOOST_AUTO_TEST_CASE(test_hci_auto_checkpoint)
{
    HCIAction action[1];

    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    hci_override_now(manager, 0.0);
    hci_query(manager, action);

    hci_override_now(manager, 1.0);
    hci_query(manager, action);

    BOOST_TEST(action->type == HCI_AUTO_CHECKPOINT);
    BOOST_TEST(action->write_snapshot == 1);
    BOOST_TEST(action->write_fof == 1);
    BOOST_TEST(manager->LongestTimeBetweenQueries == 1.0);
}

BOOST_AUTO_TEST_CASE(test_hci_auto_checkpoint2)
{

    HCIAction action[1];
    hci_override_now(manager, 1.0);
    hci_init(manager, prefix, 10.0, 1.0, 0);

    hci_override_now(manager, 2.0);
    hci_query(manager, action);

    hci_override_now(manager, 4.0);
    hci_query(manager, action);

    BOOST_TEST(manager->LongestTimeBetweenQueries == 2.0);
    BOOST_TEST(action->type == HCI_AUTO_CHECKPOINT);
    BOOST_TEST(action->write_snapshot == 1);
    BOOST_TEST(action->write_fof == 0);
}

BOOST_AUTO_TEST_CASE(test_hci_timeout)
{
    HCIAction action[1];
    hci_override_now(manager, 1.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    hci_override_now(manager, 5.0);
    hci_query(manager, action);

    BOOST_TEST(manager->LongestTimeBetweenQueries == 4.0);

    hci_override_now(manager, 8.5);
    hci_query(manager, action);
    BOOST_TEST(action->type == HCI_TIMEOUT);
    BOOST_TEST(action->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_hci_stop)
{
    HCIAction action[1];
    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    touch(prefix, "stop");
    hci_override_now(manager, 4.0);
    hci_query(manager, action);
    BOOST_TEST(!exists(prefix, "stop"));

    BOOST_TEST(action->type == HCI_STOP);
    BOOST_TEST(action->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_hci_checkpoint)
{
    HCIAction action[1];
    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    touch(prefix, "checkpoint");
    hci_override_now(manager, 4.0);
    hci_query(manager, action);
    BOOST_TEST(!exists(prefix, "checkpoint"));

    BOOST_TEST(action->type == HCI_CHECKPOINT);
    BOOST_TEST(action->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_hci_terminate)
{
    HCIAction action[1];
    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    touch(prefix, "terminate");
    hci_override_now(manager, 4.0);
    hci_query(manager, action);
    BOOST_TEST(!exists(prefix, "terminate"));

    BOOST_TEST(action->type == HCI_TERMINATE);
    BOOST_TEST(action->write_snapshot == 0);
}
