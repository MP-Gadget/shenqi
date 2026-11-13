/* Multi-rank tests for the top tree and the C++ ForceTree interface (forcetree2.h).
 * Particles are distributed with a real domain decomposition, so each rank owns
 * only some of the top leaves and the rest are pseudo particles. This checks that
 * the top tree mirrors the domain, that pseudo nodes point at the right leaves, and
 * that the pseudo particle exchange gives every rank the same top-level moments.*/
#define BOOST_TEST_MODULE forcetree2_mpi

#include "booststub.h"

#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <libgadget/forcetree2.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/domain.h>
#include <libgadget/walltime.h>

static struct ClockTable Clocks;

/* The tree internals are protected: these wrappers expose them read-only for testing.*/
class TestTopTree : public TopTree<MomentNODE> {
public:
    using TopTree::TopTree;
    int64_t get_numnodes() const { return this->numnodes; }
};

template <typename Tree>
class TestTree : public Tree {
public:
    using Tree::Tree;
    int64_t get_numnodes() const { return this->numnodes; }
    int64_t get_numparticles() const { return this->NumParticles; }
};

typedef TestTree<ForceTreeMoments> TestTreeMoments;
typedef TestTree<ForceTreeHmax> TestTreeHmax;

static int this_task(void)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    return ThisTask;
}

/* Particle inside node, allowing for rounding in the node geometry.*/
static bool inside(const double Pos[3], const auto * node)
{
    for(int j = 0; j < 3; j++)
        if(fabs(Pos[j] - node->center[j]) > node->len/2 * (1 + 1e-6))
            return false;
    return true;
}

/* Create particles on every rank: a uniform background and a cluster, so that the
 * domain is refined unevenly. Then decompose, which moves each particle to the rank
 * owning its top leaf and sets Part[i].TopLeaf.*/
static void setup_particles(const int numpart_per_task, DomainDecomp * ddecomp)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    const int ThisTask = this_task();
    memset(PartManager, 0, sizeof(PartManager[0]));
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));
    const double BoxSize = 8;
    /* Leave room for load imbalance after the exchange*/
    particle_alloc_memory(PartManager, BoxSize, 2 * numpart_per_task);
    /* DM only, so no slots are needed*/
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    PartManager->NumPart = numpart_per_task;

    auto r = boost::random::mt19937(17 + ThisTask);
    boost::random::uniform_real_distribution<double> dist(0, 1);
    for(int i = 0; i < numpart_per_task; i++) {
        particle_data& p = PartManager->Base[i];
        memset(&p, 0, sizeof(p));
        p.ID = 1 + i + (int64_t) numpart_per_task * ThisTask;
        p.Type = 1;
        /* Unequal masses so the center of mass is a real test*/
        p.Mass = 1 + dist(r);
        const double cluster[3] = {0.3, 0.6, 0.7};
        for(int j = 0; j < 3; j++) {
            if(i % 2)
                p.Pos[j] = BoxSize * dist(r);
            else
                p.Pos[j] = BoxSize * (cluster[j] + 0.05 * (dist(r) - 0.5));
        }
    }
    struct DomainParams dp = {0};
    /* Several top leaves per task, so each task merges and computes moments for more than one
     * local leaf, and local leaves are interleaved with pseudo leaves in the top tree.*/
    dp.DomainOverDecompositionFactor = 4;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    walltime_init(&Clocks);
    memset(ddecomp, 0, sizeof(*ddecomp));
    domain_decompose_full(ddecomp, MPI_COMM_WORLD);
}

static void teardown_particles(DomainDecomp * ddecomp)
{
    domain_free(ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}

/* Check the top tree structure and that it mirrors the domain decomposition.
 * Nodes is a tree whose first NTopNodes nodes are the top tree.
 * DependsOnLocalMass is set when moments (or hmax) are computed, so is only checked if updated is set.*/
static void check_toptree(const auto * Nodes, const int * TopLeafNodes, const DomainDecomp * ddecomp, const bool updated)
{
    const int ThisTask = this_task();
    const int NTopNodes = ddecomp->NTopNodes;

    /* Root covers the whole box*/
    BOOST_TEST(Nodes[0].father == -1);
    BOOST_TEST(Nodes[0].sibling == -1);
    BOOST_TEST(Nodes[0].TopLevel);
    BOOST_TEST(fabs(Nodes[0].len - PartManager->BoxSize * 1.001) < 1e-10);

    /* Every leaf has its own top-level leaf node, which points back to it*/
    std::vector<int> leafof(NTopNodes, -1);
    for(int l = 0; l < ddecomp->NTopLeaves; l++) {
        const int no = TopLeafNodes[l];
        BOOST_TEST_REQUIRE(no >= 0);
        BOOST_TEST_REQUIRE(no < NTopNodes);
        BOOST_TEST(leafof[no] == -1);
        leafof[no] = l;
        const auto * nop = &Nodes[no];
        BOOST_TEST(nop->TopLevel);
        BOOST_TEST(!nop->InternalTopLevel);
        if(ddecomp->TopLeaves[l].Task == ThisTask) {
            BOOST_TEST(nop->ChildType != PSEUDO_NODE_TYPE);
            BOOST_TEST(nop->DependsOnLocalMass == updated);
        }
        else {
            BOOST_TEST(nop->ChildType == PSEUDO_NODE_TYPE);
            /* Pseudo particle index is the leaf number*/
            BOOST_TEST(nop->s.suns[0] == l);
            BOOST_TEST(!nop->DependsOnLocalMass);
        }
    }

    /* Internal top-level nodes have 8 top-level children, which are either internal or leaves*/
    int ninternal = 0;
    for(int no = 0; no < NTopNodes; no++) {
        const auto * nop = &Nodes[no];
        BOOST_TEST(nop->TopLevel);
        if(!nop->InternalTopLevel) {
            BOOST_TEST(leafof[no] >= 0);
            continue;
        }
        ninternal++;
        BOOST_TEST(leafof[no] == -1);
        BOOST_TEST(nop->ChildType == NODE_NODE_TYPE);
        int seen = 0;
        bool anylocal = false;
        for(int j = 0; j < 8; j++) {
            const int c = nop->s.suns[j];
            BOOST_TEST_REQUIRE(c > no);
            BOOST_TEST_REQUIRE(c < NTopNodes);
            BOOST_TEST(Nodes[c].father == no);
            BOOST_TEST(Nodes[c].TopLevel);
            BOOST_TEST(fabs(Nodes[c].len / nop->len - 0.5) < 1e-10);
            const int sub = nop->get_subnode(Nodes[c].center);
            BOOST_TEST(!(seen & (1 << sub)));
            seen |= 1 << sub;
            /* Siblings run through the children, the last pointing to the parent's sibling*/
            BOOST_TEST(Nodes[c].sibling == (j < 7 ? nop->s.suns[j+1] : nop->sibling));
            anylocal = anylocal || Nodes[c].DependsOnLocalMass;
        }
        /* Local mass dependence propagates up from the local leaves*/
        BOOST_TEST(nop->DependsOnLocalMass == (updated && anylocal));
    }
    /* An oct-tree with L leaves has (L-1)/7 internal nodes*/
    BOOST_TEST(ninternal * 7 + 1 == ddecomp->NTopLeaves);

    /* Every local particle is inside the node for its top leaf*/
    int nbad = 0;
    for(int i = 0; i < PartManager->NumPart; i++) {
        const int l = PartManager->Base[i].TopLeaf;
        BOOST_TEST_REQUIRE(ddecomp->TopLeaves[l].Task == ThisTask);
        if(!inside(PartManager->Base[i].Pos, &Nodes[TopLeafNodes[l]]))
            nbad++;
    }
    BOOST_TEST(nbad == 0);
}

/* Node-type specific values that should agree between ranks on the top tree*/
static bool values_agree(const MomentNODE& a, const MomentNODE& b)
{
    bool good = fabs(a.mass - b.mass) <= 1e-10 * b.mass;
    for(int k = 0; k < 3; k++)
        good = good && fabs(a.cofm[k] - b.cofm[k]) <= 1e-10 * PartManager->BoxSize;
    return good;
}

static bool values_agree(const HmaxNODE& a, const HmaxNODE& b)
{
    return fabs(a.hmax - b.hmax) <= 1e-10;
}

/* Check the top tree is the same on every rank, comparing to rank 0:
 * the geometry and links, and the moments (mass and center of mass) or hmax,
 * depending on the node type. The top-level values of remote leaves come
 * from the pseudo particle exchange, so this checks the exchange.*/
template <typename NodeType>
static void check_toptree_consistent(const NodeType * Nodes, const DomainDecomp * ddecomp)
{
    const int NTopNodes = ddecomp->NTopNodes;
    std::vector<NodeType> root(Nodes, Nodes + NTopNodes);
    MPI_Bcast(root.data(), NTopNodes * sizeof(NodeType), MPI_BYTE, 0, MPI_COMM_WORLD);
    int nbad = 0;
    for(int no = 0; no < NTopNodes; no++) {
        const NodeType& a = Nodes[no];
        const NodeType& b = root[no];
        bool good = a.len == b.len && a.father == b.father && a.sibling == b.sibling
            && a.InternalTopLevel == b.InternalTopLevel;
        for(int k = 0; k < 3; k++)
            good = good && a.center[k] == b.center[k];
        if(a.InternalTopLevel)
            for(int j = 0; j < 8; j++)
                good = good && a.s.suns[j] == b.s.suns[j];
        good = good && values_agree(a, b);
        if(!good)
            nbad++;
    }
    BOOST_TEST(nbad == 0);
}

/* Walk the tree in treewalk order (suns[0] to open, sibling to skip) from the root,
 * checking it terminates and visits each top leaf once.*/
static void check_walk(const auto * Nodes, const int64_t numnodes, const DomainDecomp * ddecomp)
{
    int no = 0;
    int64_t count = 0;
    int nleaves = 0;
    while(no >= 0) {
        BOOST_TEST_REQUIRE(no < numnodes);
        BOOST_TEST_REQUIRE(count++ <= numnodes);
        const auto * nop = &Nodes[no];
        if(nop->TopLevel && !nop->InternalTopLevel)
            nleaves++;
        if(nop->ChildType == NODE_NODE_TYPE)
            no = nop->s.suns[0];
        else
            no = nop->sibling;
    }
    BOOST_TEST(nleaves == ddecomp->NTopLeaves);
}

/* Walk the subtree below a local node, recording the parent of every particle.*/
static void walk_local(const auto * Nodes, const int64_t numnodes, const int no, std::vector<int>& pfather)
{
    BOOST_TEST_REQUIRE(no >= 0);
    BOOST_TEST_REQUIRE(no < numnodes);
    const auto * nop = &Nodes[no];
    BOOST_TEST_REQUIRE(nop->ChildType != PSEUDO_NODE_TYPE);
    if(nop->ChildType == PARTICLE_NODE_TYPE) {
        for(int j = 0; j < nop->s.noccupied; j++) {
            const int p = nop->s.suns[j];
            BOOST_TEST_REQUIRE(p >= 0);
            BOOST_TEST_REQUIRE(p < PartManager->NumPart);
            BOOST_TEST(pfather[p] == -1);
            pfather[p] = no;
        }
        return;
    }
    for(int j = 0; j < NMAXCHILD && nop->s.suns[j] >= 0; j++) {
        BOOST_TEST(Nodes[nop->s.suns[j]].father == no);
        walk_local(Nodes, numnodes, nop->s.suns[j], pfather);
    }
}

/* Every local particle is in the subtree of its own top leaf, exactly once.*/
static std::vector<int> check_local_particles(const auto * Nodes, const int * TopLeafNodes, const int64_t numnodes, const DomainDecomp * ddecomp)
{
    const int ThisTask = this_task();
    std::vector<int> pfather(PartManager->NumPart, -1);
    std::vector<int> leafofp(PartManager->NumPart, -1);
    for(int l = ddecomp->Tasks[ThisTask].StartLeaf; l < ddecomp->Tasks[ThisTask].EndLeaf; l++) {
        const int no = TopLeafNodes[l];
        std::vector<int> before = pfather;
        walk_local(Nodes, numnodes, no, pfather);
        for(int i = 0; i < PartManager->NumPart; i++)
            if(before[i] == -1 && pfather[i] != -1)
                leafofp[i] = l;
    }
    int nmissing = 0, nwrongleaf = 0;
    for(int i = 0; i < PartManager->NumPart; i++) {
        if(pfather[i] < 0)
            nmissing++;
        else if(leafofp[i] != PartManager->Base[i].TopLeaf)
            nwrongleaf++;
    }
    BOOST_TEST(nmissing == 0);
    BOOST_TEST(nwrongleaf == 0);
    return pfather;
}

BOOST_AUTO_TEST_CASE(test_toptree_mpi)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    DomainDecomp ddecomp;
    setup_particles(32*32*32, &ddecomp);
    /* Leaves on other tasks become pseudo nodes, so there are some whenever NTask > 1.*/
    const int ThisTask = this_task();
    const int nlocal = ddecomp.Tasks[ThisTask].EndLeaf - ddecomp.Tasks[ThisTask].StartLeaf;
    const int npseudo = ddecomp.NTopLeaves - nlocal;
    message(0, "Domain has %d top nodes and %d top leaves on %d tasks. Task 0 has %d local leaves and %d pseudo leaves\n",
            ddecomp.NTopNodes, ddecomp.NTopLeaves, NTask, nlocal, npseudo);
    if(NTask > 1)
        BOOST_TEST(npseudo > 0);
    else
        message(0, "Single task: no pseudo nodes, so the pseudo particle checks are trivial\n");

    TestTopTree tree(&ddecomp, PartManager->BoxSize);
    BOOST_TEST(tree.get_numnodes() == ddecomp.NTopNodes);
    check_toptree(tree.get_nodes(), tree.get_topleaf_nodes(), &ddecomp, false);
    check_toptree_consistent(tree.get_nodes(), &ddecomp);
    check_walk(tree.get_nodes(), tree.get_numnodes(), &ddecomp);
    teardown_particles(&ddecomp);
}

BOOST_AUTO_TEST_CASE(test_forcetree_moments_mpi)
{
    DomainDecomp ddecomp;
    setup_particles(32*32*32, &ddecomp);
    const int NTopLeaves = ddecomp.NTopLeaves;

    TestTreeMoments tb(*PartManager, &ddecomp, NULL, false, "");
    const auto * Nodes = tb.get_nodes();
    const int * const TopLeafNodes = tb.get_topleaf_nodes();
    BOOST_TEST(tb.get_numparticles() == PartManager->NumPart);
    check_toptree(Nodes, tb.get_topleaf_nodes(), &ddecomp, true);
    check_toptree_consistent(Nodes, &ddecomp);
    check_walk(Nodes, tb.get_numnodes(), &ddecomp);
    check_local_particles(Nodes, tb.get_topleaf_nodes(), tb.get_numnodes(), &ddecomp);

    /* Global mass and mass-weighted position in each top leaf*/
    std::vector<double> leafmass(NTopLeaves, 0), leafcofm(3*NTopLeaves, 0);
    double totmass = 0;
    for(int i = 0; i < PartManager->NumPart; i++) {
        const particle_data& p = PartManager->Base[i];
        leafmass[p.TopLeaf] += p.Mass;
        for(int k = 0; k < 3; k++)
            leafcofm[3*p.TopLeaf + k] += p.Mass * p.Pos[k];
        totmass += p.Mass;
    }
    MPI_Allreduce(MPI_IN_PLACE, leafmass.data(), NTopLeaves, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, leafcofm.data(), 3*NTopLeaves, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totmass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    BOOST_TEST(fabs(Nodes[0].mass - totmass) < 1e-9 * totmass);
    /* Each leaf, local or pseudo, has the global moments of its particles*/
    int nbadmass = 0, nbadcofm = 0;
    for(int l = 0; l < NTopLeaves; l++) {
        const auto * nop = &Nodes[TopLeafNodes[l]];
        if(fabs(nop->mass - leafmass[l]) > 1e-9 * totmass)
            nbadmass++;
        for(int k = 0; k < 3; k++) {
            const double expected = leafmass[l] > 0 ? leafcofm[3*l+k] / leafmass[l] : nop->center[k];
            if(fabs(nop->cofm[k] - expected) > 1e-8 * PartManager->BoxSize)
                nbadcofm++;
        }
    }
    BOOST_TEST(nbadmass == 0);
    BOOST_TEST(nbadcofm == 0);
    /* Internal top-level nodes sum their children*/
    for(int no = 0; no < ddecomp.NTopNodes; no++) {
        const auto * nop = &Nodes[no];
        if(!nop->InternalTopLevel)
            continue;
        double mass = 0;
        for(int j = 0; j < 8; j++)
            mass += Nodes[nop->s.suns[j]].mass;
        BOOST_TEST(fabs(nop->mass - mass) < 1e-9 * totmass);
    }
    teardown_particles(&ddecomp);
}

/* Largest amount a particle peeks out of a node.*/
static double peek(const particle_data& p, const auto * node)
{
    double hmax = 0;
    for(int j = 0; j < 3; j++)
        hmax = DMAX(hmax, fabs(p.Pos[j] - node->center[j]) + p.Hsml - node->len/2.);
    return hmax;
}

/* Build a gas hmax tree. If active, hmax is set after the build with update_tree_hmax_father,
 * otherwise it is set during the build from the inactive particles.*/
static double do_hmax_test(DomainDecomp * ddecomp, const bool active)
{
    const int NTopLeaves = ddecomp->NTopLeaves;
    auto r = boost::random::mt19937(31 + this_task());
    boost::random::uniform_real_distribution<double> dist(0, 1);
    /* The domain is already decomposed, so changing the type to gas does not move particles.
     * No SPH slots are enabled: the tree does not need them.*/
    for(int i = 0; i < PartManager->NumPart; i++) {
        particle_data& p = PartManager->Base[i];
        p.Type = 0;
        p.Hsml = 0.1 * dist(r);
        /* Bin 0 is always active. Bin 2 is inactive at Ti = 1.*/
        p.TimeBinHydro = active ? 0 : 2;
        p.Ti_drift = 1;
    }

    TestTreeHmax tb(*PartManager, ddecomp, GASMASK, "");
    if(active) {
        #pragma omp parallel for
        for(int i = 0; i < PartManager->NumPart; i++)
            tb.update_tree_hmax_father(i, PartManager->Base[i].Pos, PartManager->Base[i].Hsml);
    }
    tb.force_tree_calc_hmax(ddecomp);
    BOOST_TEST(tb.hmax_computed_flag);
    const auto * Nodes = tb.get_nodes();
    const int * const TopLeafNodes = tb.get_topleaf_nodes();
    check_toptree(Nodes, tb.get_topleaf_nodes(), ddecomp, true);
    check_toptree_consistent(Nodes, ddecomp);
    check_walk(Nodes, tb.get_numnodes(), ddecomp);
    std::vector<int> pfather = check_local_particles(Nodes, tb.get_topleaf_nodes(), tb.get_numnodes(), ddecomp);

    /* hmax of every ancestor of a local particle, up to the root, covers that particle*/
    int nbadfather = 0, nbadhmax = 0;
    std::vector<double> leafpeek(NTopLeaves, 0);
    for(int i = 0; i < PartManager->NumPart; i++) {
        const particle_data& p = PartManager->Base[i];
        if(tb.get_father_particle(i) != pfather[i])
            nbadfather++;
        for(int no = tb.get_father_particle(i); no >= 0; no = tb.get_father_node(no))
            if(!inside(p.Pos, &Nodes[no]) || peek(p, &Nodes[no]) > Nodes[no].hmax + 1e-5)
                nbadhmax++;
        leafpeek[p.TopLeaf] = DMAX(leafpeek[p.TopLeaf], peek(p, &Nodes[TopLeafNodes[p.TopLeaf]]));
    }
    BOOST_TEST(nbadfather == 0);
    BOOST_TEST(nbadhmax == 0);
    /* hmax of each leaf, local or pseudo, covers all particles in that leaf on every rank*/
    MPI_Allreduce(MPI_IN_PLACE, leafpeek.data(), NTopLeaves, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int nbadleaf = 0;
    for(int l = 0; l < NTopLeaves; l++)
        if(leafpeek[l] > Nodes[TopLeafNodes[l]].hmax + 1e-5)
            nbadleaf++;
    BOOST_TEST(nbadleaf == 0);
    const double roothmax = Nodes[0].hmax;
    message(0, "Root hmax (active = %d): %g\n", active, roothmax);
    BOOST_TEST(roothmax > 0);
    return roothmax;
}

BOOST_AUTO_TEST_CASE(test_forcetree_hmax_mpi)
{
    DomainDecomp ddecomp;
    setup_particles(32*32*32, &ddecomp);
    const double inactive = do_hmax_test(&ddecomp, false);
    const double active = do_hmax_test(&ddecomp, true);
    /* Same particles and smoothing lengths, so both routes should agree*/
    BOOST_TEST(fabs(active - inactive) < 1e-6);
    teardown_particles(&ddecomp);
}

/* Trees which number their nodes differently may be alive at the same time:
 * each tree stores the node index of each top leaf itself, so building a second
 * tree must not change where the first tree's top leaves are.*/
BOOST_AUTO_TEST_CASE(test_forcetree_coexist_mpi)
{
    DomainDecomp ddecomp;
    setup_particles(32*32*32, &ddecomp);
    const int NTopLeaves = ddecomp.NTopLeaves;

    TestTreeMoments tb(*PartManager, &ddecomp, NULL, false, "");
    const std::vector<int> before(tb.get_topleaf_nodes(), tb.get_topleaf_nodes() + NTopLeaves);

    /* An old-style tree, whose nodes are numbered from firstnode, built while the first tree is alive.*/
    init_forcetree_params(0.9);
    ForceTree oldtree = {0};
    force_tree_rebuild_mask(&oldtree, &ddecomp, ALLMASK, "");
    /* And another new tree*/
    ForceTreeSimple simple(*PartManager, &ddecomp, ALLMASK, "");

    int nchanged = 0, nbadold = 0, nbadsimple = 0;
    for(int l = 0; l < NTopLeaves; l++) {
        if(tb.get_topleaf_nodes()[l] != before[l])
            nchanged++;
        /* The top trees are built identically, so differ only by the offset of the first node.*/
        if(oldtree.TopLeafNodes[l] != before[l] + oldtree.firstnode)
            nbadold++;
        if(simple.get_topleaf_nodes()[l] != before[l])
            nbadsimple++;
    }
    BOOST_TEST(nchanged == 0);
    BOOST_TEST(nbadold == 0);
    BOOST_TEST(nbadsimple == 0);
    /* The first tree is still valid*/
    check_toptree(tb.get_nodes(), tb.get_topleaf_nodes(), &ddecomp, true);
    check_local_particles(tb.get_nodes(), tb.get_topleaf_nodes(), tb.get_numnodes(), &ddecomp);
    /* The old tree can still exchange its moments using its own top leaf nodes.*/
    force_tree_calc_moments(&oldtree, &ddecomp);
    BOOST_TEST(fabs(oldtree.Nodes[oldtree.firstnode].mom.mass - tb.get_nodes()[0].mass) < 1e-9 * tb.get_nodes()[0].mass);

    force_tree_free(&oldtree);
    teardown_particles(&ddecomp);
}
